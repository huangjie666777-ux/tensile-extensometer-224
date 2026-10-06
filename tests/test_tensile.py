import io
import json
import zipfile
from pathlib import Path
import sys

import numpy as np
import pytest
from PIL import Image
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tensile_dic224.app import app
from tensile_dic224.tensile import fit_modulus, offset_yield, FrameResult
from tests.test_pipeline import _speckle_pair

client = TestClient(app)

E_TRUE = 70000.0
AREA = 25.0
STRAINS = np.linspace(0.002, 0.020, 8)
YIELD_STRAIN = 0.010
HARDENING = 2500.0

FORM = dict(scale_mm_per_px="0.05", roi_x="40", roi_y="40",
            roi_w="112", roi_h="112", subset_size="29",
            grid_step="16", search_radius="8", max_iterations="50",
            area_mm2=str(AREA), p1_x="56", p1_y="96", p2_x="136", p2_y="96",
            fit_strain_min="0.002", fit_strain_max="0.008")


def _png(arr):
    buf = io.BytesIO()
    Image.fromarray(arr, mode="L").save(buf, format="PNG")
    return buf.getvalue()


def _stress(eps):
    if eps <= YIELD_STRAIN:
        return E_TRUE * eps
    return E_TRUE * YIELD_STRAIN + HARDENING * (eps - YIELD_STRAIN)


def _tensile_set(n=None):
    """Reference + warped frames + force CSV with a bilinear material."""
    from scipy.ndimage import map_coordinates
    rng = np.random.default_rng(11)
    H = W = 192
    imp = np.zeros((H, W))
    ys = rng.integers(0, H, 2600)
    xs = rng.integers(0, W, 2600)
    np.add.at(imp, (ys, xs), rng.uniform(0.5, 1.0, 2600))
    from scipy.ndimage import gaussian_filter
    field = gaussian_filter(imp, 1.0)
    ref = np.clip(40 + 180 * field / field.max()
                  + rng.normal(0, 1.0, (H, W)), 0, 255).astype(np.uint8)
    strains = STRAINS if n is None else STRAINS[:n]
    frames = {}
    lines = ["frame_id,time_s,force_N"]
    x, y = np.meshgrid(np.arange(W, dtype=float), np.arange(H, dtype=float))
    for k, eps in enumerate(strains):
        fid = f"f{k:02d}"
        F = np.array([[1 + eps, 0.0], [0.0, 1 - 0.3 * eps]])
        rel = np.vstack([(x - 96).ravel(), (y - 96).ravel()])
        src = np.linalg.solve(F, rel) + np.array([[96.0], [96.0]])
        defo = np.clip(map_coordinates(ref.astype(float), [src[1], src[0]],
                                       order=3, mode="reflect").reshape(H, W),
                       0, 255).astype(np.uint8)
        frames[fid] = _png(defo)
        lines.append(f"{fid},{(k + 1) * 0.5},{_stress(float(eps)) * AREA:.3f}")
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as zf:
        for fid, png in frames.items():
            zf.writestr(f"{fid}.png", png)
    return ref, zip_buf.getvalue(), "\n".join(lines).encode()


def _post(ref, zip_bytes, csv_bytes, form=None, path="/tensile/analyze"):
    files = {"reference": ("r.png", _png(ref), "image/png"),
             "frames": ("frames.zip", zip_bytes, "application/zip"),
             "forces": ("forces.csv", csv_bytes, "text/csv")}
    return client.post(path, data=form or FORM, files=files)


def test_tensile_end_to_end_modulus_and_yield():
    ref, zb, csvb = _tensile_set()
    r = _post(ref, zb, csvb)
    assert r.status_code == 200, r.text
    body = r.json()
    assert all(fr["gauge_valid"] for fr in body["frames"])
    fit = body["fit"]
    assert fit["status"] == "ok"
    assert fit["E_MPa"] == pytest.approx(E_TRUE, rel=0.05)
    assert abs(fit["b_MPa"]) < 0.02 * E_TRUE * 0.01 + 5
    assert fit["r_squared"] > 0.99
    y = body["yield"]
    assert y["strain"] is not None
    # 0.2% offset yield of the bilinear material
    # 0.2% offset yield of the bilinear material: eps_y + 0.002*E/(E-H)
    assert y["strain"] == pytest.approx(
        YIELD_STRAIN + 0.002 * E_TRUE / (E_TRUE - HARDENING), abs=2e-3)
    assert y["stress_MPa"] == pytest.approx(
        fit["E_MPa"] * (y["strain"] - 0.002) + fit["b_MPa"], rel=1e-6)


def test_tensile_download_zip_links_files():
    ref, zb, csvb = _tensile_set()
    r = _post(ref, zb, csvb, path="/tensile/download")
    assert r.status_code == 200
    zf = zipfile.ZipFile(io.BytesIO(r.content))
    names = set(zf.namelist())
    assert {"curve.csv", "frames.csv", "tensile_result.json"} <= names
    meta = json.loads(zf.read("tensile_result.json"))
    seq = meta["sequence"]
    assert seq == [f"f{k:02d}" for k in range(len(STRAINS))]
    for fid in seq:
        assert f"frames/{fid}_points.csv" in names
    curve = zf.read("curve.csv").decode().splitlines()
    assert curve[0].startswith("frame_id,time_s,force_N,stress_MPa")
    assert [ln.split(",")[0] for ln in curve[1:]] == seq
    frames_csv = zf.read("frames.csv").decode().splitlines()
    assert [ln.split(",")[0] for ln in frames_csv[1:]] == seq
    assert meta["fit"]["status"] == "ok"
    assert meta["yield"]["strain"] is not None


def test_reject_duplicate_frame_ids():
    ref, zb, csvb = _tensile_set()
    bad = csvb.replace(b"f01,", b"f00,")
    assert _post(ref, zb, bad).status_code == 422


def test_reject_non_increasing_time():
    ref, zb, csvb = _tensile_set()
    bad = csvb.replace(b"f02,1.5,", b"f02,0.5,")
    assert _post(ref, zb, bad).status_code == 422


def test_reject_negative_force():
    ref, zb, csvb = _tensile_set()
    bad = csvb.replace(b"f03,2.0,17000.000", b"f03,2.0,-1.0")
    assert _post(ref, zb, bad).status_code == 422


def test_reject_csv_zip_mismatch():
    ref, zb, csvb = _tensile_set()
    bad = csvb.replace(b"f00,", b"g00,")
    assert _post(ref, zb, bad).status_code == 422


def test_reject_bad_fit_interval_and_endpoints():
    ref, zb, csvb = _tensile_set()
    form = dict(FORM, fit_strain_min="0.01", fit_strain_max="0.005")
    assert _post(ref, zb, csvb, form).status_code == 422
    form = dict(FORM, fit_strain_min="-0.1")
    assert _post(ref, zb, csvb, form).status_code == 422
    form = dict(FORM, p2_x=FORM["p1_x"], p2_y=FORM["p1_y"])
    assert _post(ref, zb, csvb, form).status_code == 422
    form = dict(FORM, p2_x="500")  # outside grid coverage
    assert _post(ref, zb, csvb, form).status_code == 422
    form = dict(FORM, area_mm2="0")
    assert _post(ref, zb, csvb, form).status_code == 422


def test_reject_16bit_png():
    ref, zb, csvb = _tensile_set()
    arr16 = (np.asarray(ref, dtype=np.uint16) * 257)
    buf = io.BytesIO()
    Image.fromarray(arr16, mode="I;16").save(buf, format="PNG")
    files = {"reference": ("r.png", buf.getvalue(), "image/png"),
             "frames": ("frames.zip", zb, "application/zip"),
             "forces": ("forces.csv", csvb, "text/csv")}
    assert client.post("/tensile/analyze", data=FORM, files=files).status_code == 422
    # old interface rejects 16-bit too
    files2 = {"reference": ("r.png", buf.getvalue(), "image/png"),
              "deformed": ("d.png", _png(ref), "image/png")}
    form2 = {k: FORM[k] for k in
             ("scale_mm_per_px", "roi_x", "roi_y", "roi_w", "roi_h",
              "subset_size", "grid_step", "search_radius", "max_iterations")}
    assert client.post("/analyze", data=form2, files=files2).status_code == 422


def test_reject_frame_count_out_of_range():
    ref, zb, csvb = _tensile_set(n=1)
    assert _post(ref, zb, csvb).status_code == 422


def test_gauge_invalid_when_corner_missing():
    ref, zb, csvb = _tensile_set()
    # flatten texture around p1 cell corner (56,96) -> nearest grid (54,94)
    ref[80:100, 40:60] = 128
    r = _post(ref, zb, csvb)
    assert r.status_code == 200
    body = r.json()
    assert not any(fr["gauge_valid"] for fr in body["frames"])
    assert all(fr["eng_strain"] is None for fr in body["frames"])
    assert body["fit"]["status"] == "failed"
    assert body["yield"]["strain"] is None
    assert body["yield"]["reason"] == "fit_failed"


def test_fit_modulus_requires_three_distinct_strains():
    def fr(eps, sig):
        return FrameResult(row=None, stress_MPa=sig, measurement=None,
                           gauge_valid=True, gauge_reason="", strain=eps)
    res = [fr(0.001, 70.0), fr(0.001, 70.0), fr(0.002, 140.0)]
    fit = fit_modulus(res, 0.0, 0.01)
    assert fit["status"] == "failed"
    res.append(fr(0.003, 210.0))
    fit = fit_modulus(res, 0.0, 0.01)
    assert fit["status"] == "ok"
    assert fit["E_MPa"] == pytest.approx(70000.0)
    assert fit["r_squared"] == pytest.approx(1.0)


def test_offset_yield_no_extrapolation_across_gaps():
    def fr(eps, sig, valid=True):
        return FrameResult(row=None, stress_MPa=sig, measurement=None,
                           gauge_valid=valid, gauge_reason="" if valid else "x",
                           strain=eps)
    fit = {"status": "ok", "E_MPa": 70000.0, "b_MPa": 0.0,
           "interval": [0.0, 0.005]}
    # line: 70000*(eps-0.002); measured above then below across a gap
    res = [fr(0.006, 300.0), fr(0.008, 400.0, valid=False), fr(0.010, 100.0)]
    y = offset_yield(res, fit)
    assert y["strain"] is None
    # contiguous pair crossing -> interpolated
    res = [fr(0.006, 300.0), fr(0.008, 400.0), fr(0.010, 100.0)]
    y = offset_yield(res, fit)
    # d at 0.008: 400-70000*0.006=-20 <=0 -> crossing between 0.006 and 0.008
    assert y["strain"] == pytest.approx(0.006 + 0.002 * 20 / 40)
