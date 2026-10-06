"""Build the result ZIP: per-point CSV, validity mask PNG, parameters JSON."""
from __future__ import annotations
import io
import json
import zipfile

import numpy as np
from PIL import Image

from tensile_dic224.pipeline import Measurement

CSV_NAME = "points.csv"
MASK_NAME = "valid_mask.png"
JSON_NAME = "result.json"


def _fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and not np.isfinite(value):
        return ""
    return repr(round(float(value), 8)) if isinstance(value, (float, np.floating)) \
        else str(value)


def points_csv(m: Measurement, scale: float) -> str:
    header = (
        "index,grid_x_mm,grid_y_mm,valid,u_mm,v_mm,zncc,converged,"
        "iterations,exx,eyy,exy,failure_reason\n")
    lines = [header]
    for i in range(m.n_points):
        row = [
            i,
            _fmt(m.grid_x_px[i] * scale),
            _fmt(m.grid_y_px[i] * scale),
            int(m.valid[i]),
            _fmt(m.u_mm[i]),
            _fmt(m.v_mm[i]),
            _fmt(m.correlation[i]),
            int(m.converged[i]),
            int(m.iterations[i]),
            _fmt(m.exx[i]),
            _fmt(m.eyy[i]),
            _fmt(m.exy[i]),
            m.reasons[i],
        ]
        lines.append(",".join(str(x) for x in row) + "\n")
    return "".join(lines)


def mask_png(m: Measurement) -> bytes:
    ys = np.unique(m.grid_y_px)
    xs = np.unique(m.grid_x_px)
    mask = np.zeros((ys.size, xs.size), dtype=np.uint8)
    pos = {(round(float(x), 6), round(float(y), 6)): (ix, iy)
           for iy, y in enumerate(ys) for ix, x in enumerate(xs)}
    for i in range(m.n_points):
        ix, iy = pos[(round(float(m.grid_x_px[i]), 6),
                      round(float(m.grid_y_px[i]), 6))]
        mask[iy, ix] = 255 if m.valid[i] else 0
    buf = io.BytesIO()
    Image.fromarray(mask, mode="L").save(buf, format="PNG")
    return buf.getvalue()


def result_json(m: Measurement, params: dict) -> str:
    reasons: dict[str, int] = {}
    for i, reason in enumerate(m.reasons):
        if not m.valid[i] and reason:
            reasons[reason] = reasons.get(reason, 0) + 1
    strain_valid = int(np.isfinite(m.exx).sum())
    payload = {
        "parameters": params,
        "n_points": m.n_points,
        "n_valid_displacement": m.n_valid,
        "n_valid_strain": strain_valid,
        "coordinate_system": {
            "x": "right, pixels from image origin, scaled to mm",
            "y": "down, pixels from image origin, scaled to mm",
            "grid": "reference configuration",
        },
        "outputs": {"csv": CSV_NAME, "valid_mask_png": MASK_NAME,
                   "json": JSON_NAME},
        "failure_counts": reasons,
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def build_zip(m: Measurement, scale: float, params: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(CSV_NAME, points_csv(m, scale))
        zf.writestr(MASK_NAME, mask_png(m))
        zf.writestr(JSON_NAME, result_json(m, params))
    return buf.getvalue()


# ---------------------------------------------------------------- tensile --

TENSILE_CURVE_CSV = "curve.csv"
TENSILE_FRAMES_CSV = "frames.csv"
TENSILE_JSON = "tensile_result.json"


def curve_csv(results) -> str:
    """Full stress/strain curve; one row per frame in sequence order."""
    lines = ["frame_id,time_s,force_N,stress_MPa,eng_strain,"
             "gauge_valid,gauge_reason\n"]
    for fr in results:
        lines.append(",".join([
            fr.row.frame_id,
            _fmt(fr.row.time_s),
            _fmt(fr.row.force_N),
            _fmt(fr.stress_MPa),
            _fmt(fr.strain) if fr.gauge_valid else "",
            str(int(fr.gauge_valid)),
            fr.gauge_reason,
        ]) + "\n")
    return "".join(lines)


def frames_csv(results) -> str:
    """Per-frame extensometer measurement summary."""
    lines = ["frame_id,n_grid_points,n_valid_points,"
             "p1_u_mm,p1_v_mm,p2_u_mm,p2_v_mm,"
             "gauge_len_mm,eng_strain,stress_MPa\n"]
    for fr in results:
        m = fr.measurement
        lines.append(",".join([
            fr.row.frame_id,
            str(m.n_points),
            str(m.n_valid),
            _fmt(fr.p1_u_mm), _fmt(fr.p1_v_mm),
            _fmt(fr.p2_u_mm), _fmt(fr.p2_v_mm),
            _fmt(fr.gauge_len_mm),
            _fmt(fr.strain) if fr.gauge_valid else "",
            _fmt(fr.stress_MPa),
        ]) + "\n")
    return "".join(lines)


def tensile_json(results, ext, fit, yield_point, params: dict,
                 gauge_len0_mm: float) -> str:
    payload = {
        "parameters": params,
        "extensometer": {
            "area_mm2": ext.area_mm2,
            "p1_px": list(ext.p1),
            "p2_px": list(ext.p2),
            "gauge_len0_mm": gauge_len0_mm,
            "fit_strain_interval": [ext.fit_lo, ext.fit_hi],
            "offset_strain": 0.002,
        },
        "sequence": [fr.row.frame_id for fr in results],
        "fit": fit,
        "yield": yield_point,
        "outputs": {
            "curve_csv": TENSILE_CURVE_CSV,
            "frames_csv": TENSILE_FRAMES_CSV,
            "per_frame_points": [f"frames/{fr.row.frame_id}_points.csv"
                                 for fr in results],
            "json": TENSILE_JSON,
        },
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def build_tensile_zip(results, ext, fit, yield_point, params: dict,
                      gauge_len0_mm: float, scale: float) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(TENSILE_CURVE_CSV, curve_csv(results))
        zf.writestr(TENSILE_FRAMES_CSV, frames_csv(results))
        for fr in results:
            zf.writestr(f"frames/{fr.row.frame_id}_points.csv",
                        points_csv(fr.measurement, scale))
        zf.writestr(TENSILE_JSON,
                    tensile_json(results, ext, fit, yield_point, params,
                                 gauge_len0_mm))
    return buf.getvalue()
