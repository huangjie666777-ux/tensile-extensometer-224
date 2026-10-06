"""Tensile-test virtual extensometer built on the DIC measurement pipeline.

Each deformed frame is correlated independently against the same reference
(no displacement accumulation).  The two gauge endpoints are located in the
reference grid; per frame their displacements are bilinearly interpolated
from the enclosing grid cell, only when all four cell corners are valid.
Engineering strain is the change of the endpoint Euclidean distance over the
original gauge length; engineering stress is force over the original
cross-section area (N/mm^2 = MPa).
"""
from __future__ import annotations

import csv
import io
import zipfile
from dataclasses import dataclass
from pathlib import PurePosixPath

import numpy as np

from tensile_dic224.pipeline import Measurement, run_measurement
from tensile_dic224.validation import (AnalysisParams, RequestError,
                                       decode_gray_png, parse_float)

MIN_FRAMES = 2
MAX_FRAMES = 12
OFFSET_STRAIN = 0.002          # 0.2% offset yield definition
MIN_FIT_POINTS = 3

REQUIRED_COLUMNS = ("frame_id", "time_s", "force_N")


@dataclass(frozen=True)
class ForceRow:
    frame_id: str
    time_s: float
    force_N: float


@dataclass(frozen=True)
class ExtensometerParams:
    area_mm2: float
    p1: tuple[float, float]    # reference pixel coordinates (x, y)
    p2: tuple[float, float]
    fit_lo: float              # closed fitting strain interval
    fit_hi: float


@dataclass
class FrameResult:
    row: ForceRow
    stress_MPa: float
    measurement: Measurement
    gauge_valid: bool
    gauge_reason: str          # "" when valid
    strain: float = np.nan
    p1_u_mm: float = np.nan
    p1_v_mm: float = np.nan
    p2_u_mm: float = np.nan
    p2_v_mm: float = np.nan
    gauge_len_mm: float = np.nan


def parse_force_csv(data: bytes) -> list[ForceRow]:
    """Parse and validate the frame/time/force series (sequence-defining)."""
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise RequestError(f"force CSV is not UTF-8 text: {exc}") from exc
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        raise RequestError("force CSV is empty")
    missing = [c for c in REQUIRED_COLUMNS if c not in reader.fieldnames]
    if missing:
        raise RequestError("force CSV missing columns: " + ", ".join(missing))
    rows: list[ForceRow] = []
    prev_time = None
    for line, raw in enumerate(reader, start=2):
        frame_id = (raw.get("frame_id") or "").strip()
        if not frame_id:
            raise RequestError(f"force CSV line {line}: empty frame_id")
        time_s = parse_float(f"time_s (line {line})", raw.get("time_s") or "")
        force_n = parse_float(f"force_N (line {line})", raw.get("force_N") or "")
        if force_n < 0:
            raise RequestError(f"force CSV line {line}: force_N must be non-negative")
        if prev_time is not None and time_s <= prev_time:
            raise RequestError(
                f"force CSV line {line}: time_s must be strictly increasing")
        prev_time = time_s
        rows.append(ForceRow(frame_id, time_s, force_n))
    if not MIN_FRAMES <= len(rows) <= MAX_FRAMES:
        raise RequestError(
            f"frame count must be in [{MIN_FRAMES}, {MAX_FRAMES}], got {len(rows)}")
    ids = [r.frame_id for r in rows]
    if len(set(ids)) != len(ids):
        raise RequestError("frame_id values must be unique")
    return rows


def parse_frames_zip(data: bytes) -> dict[str, bytes]:
    """Map frame_id -> PNG bytes from the uploaded ZIP of frame images."""
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise RequestError(f"frames archive is not a valid ZIP: {exc}") from exc
    frames: dict[str, bytes] = {}
    with zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = PurePosixPath(info.filename)
            if name.suffix.lower() != ".png":
                raise RequestError(
                    f"frames archive entry {info.filename!r} is not a .png image")
            frame_id = name.stem
            if not frame_id:
                raise RequestError(f"frames archive entry {info.filename!r} has no name")
            if frame_id in frames:
                raise RequestError(f"duplicate frame image for frame_id {frame_id!r}")
            frames[frame_id] = zf.read(info)
    if not MIN_FRAMES <= len(frames) <= MAX_FRAMES:
        raise RequestError(
            f"frames archive must hold [{MIN_FRAMES}, {MAX_FRAMES}] PNGs, "
            f"got {len(frames)}")
    return frames


def match_frames(rows: list[ForceRow], frames: dict[str, bytes]) -> None:
    """Frame ids must correspond one-to-one between CSV and image archive."""
    csv_ids = {r.frame_id for r in rows}
    zip_ids = set(frames)
    if csv_ids != zip_ids:
        only_csv = sorted(csv_ids - zip_ids)
        only_zip = sorted(zip_ids - csv_ids)
        parts = []
        if only_csv:
            parts.append("missing images for: " + ", ".join(only_csv))
        if only_zip:
            parts.append("no CSV rows for: " + ", ".join(only_zip))
        raise RequestError("frame_id mismatch between CSV and images ("
                           + "; ".join(parts) + ")")


def build_extensometer_params(form: dict[str, str]) -> ExtensometerParams:
    required = ("area_mm2", "p1_x", "p1_y", "p2_x", "p2_y",
                "fit_strain_min", "fit_strain_max")
    missing = [k for k in required if form.get(k) in (None, "")]
    if missing:
        raise RequestError("missing fields: " + ", ".join(missing))
    area = parse_float("area_mm2", form["area_mm2"])
    if area <= 0:
        raise RequestError("area_mm2 must be positive")
    p1 = (parse_float("p1_x", form["p1_x"]), parse_float("p1_y", form["p1_y"]))
    p2 = (parse_float("p2_x", form["p2_x"]), parse_float("p2_y", form["p2_y"]))
    if p1 == p2:
        raise RequestError("extensometer endpoints p1 and p2 must differ")
    lo = parse_float("fit_strain_min", form["fit_strain_min"])
    hi = parse_float("fit_strain_max", form["fit_strain_max"])
    if lo < 0:
        raise RequestError("fit_strain_min must be non-negative")
    if not hi > lo:
        raise RequestError("fit_strain_max must be greater than fit_strain_min")
    return ExtensometerParams(area, p1, p2, lo, hi)


class _Grid:
    """Axis-aligned lookup over the (possibly unevenly spaced) point grid."""

    def __init__(self, gx: np.ndarray, gy: np.ndarray):
        self.xs = np.unique(gx)
        self.ys = np.unique(gy)
        self.index = {(round(float(x), 6), round(float(y), 6)): i
                      for i, (x, y) in enumerate(zip(gx, gy))}

    def cell(self, x: float, y: float) -> tuple[int, int, list[int]] | None:
        """Corner point indices of the cell holding (x, y), or None outside."""
        if not (self.xs[0] - 1e-9 <= x <= self.xs[-1] + 1e-9
                and self.ys[0] - 1e-9 <= y <= self.ys[-1] + 1e-9):
            return None
        ix = int(np.clip(np.searchsorted(self.xs, x) - 1, 0, self.xs.size - 2))
        iy = int(np.clip(np.searchsorted(self.ys, y) - 1, 0, self.ys.size - 2))
        corners = []
        for j in (iy, iy + 1):
            for i in (ix, ix + 1):
                idx = self.index.get((round(float(self.xs[i]), 6),
                                      round(float(self.ys[j]), 6)))
                if idx is None:
                    return None
                corners.append(idx)
        return ix, iy, corners


def _bilinear(grid: _Grid, m: Measurement, ix: int, iy: int,
              corners: list[int], x: float, y: float):
    """Displacement (mm) at (x, y) px, or None if any corner is invalid."""
    if not all(m.valid[c] for c in corners):
        return None
    x0, x1 = grid.xs[ix], grid.xs[ix + 1]
    y0, y1 = grid.ys[iy], grid.ys[iy + 1]
    tx = 0.0 if x1 == x0 else (x - x0) / (x1 - x0)
    ty = 0.0 if y1 == y0 else (y - y0) / (y1 - y0)
    w = np.array([(1 - tx) * (1 - ty), tx * (1 - ty),
                  (1 - tx) * ty, tx * ty])
    u = float(np.dot(w, m.u_mm[corners]))
    v = float(np.dot(w, m.v_mm[corners]))
    return u, v


def run_tensile(ref: np.ndarray, rows: list[ForceRow],
                frame_png: dict[str, bytes], gx: np.ndarray, gy: np.ndarray,
                p: AnalysisParams, ext: ExtensometerParams) -> list[FrameResult]:
    grid = _Grid(gx, gy)
    cells = []
    for label, pt in (("p1", ext.p1), ("p2", ext.p2)):
        cell = grid.cell(pt[0], pt[1])
        if cell is None:
            raise RequestError(
                f"extensometer endpoint {label}={pt} lies outside the grid "
                f"coverage x:[{grid.xs[0]}, {grid.xs[-1]}] "
                f"y:[{grid.ys[0]}, {grid.ys[-1]}] px")
        cells.append(cell)
    scale = p.scale_mm_per_px
    a0 = np.array(ext.p1) * scale
    b0 = np.array(ext.p2) * scale
    len0 = float(np.linalg.norm(b0 - a0))

    results: list[FrameResult] = []
    for row in rows:
        img = decode_gray_png(frame_png[row.frame_id], f"frame {row.frame_id!r}")
        if img.shape != ref.shape:
            raise RequestError(
                f"frame {row.frame_id!r} size {img.shape[1]}x{img.shape[0]} "
                f"differs from reference {ref.shape[1]}x{ref.shape[0]}")
        m = run_measurement(ref, img, gx, gy, p)
        stress = row.force_N / ext.area_mm2
        disp = [_bilinear(grid, m, *cells[0], *ext.p1),
                _bilinear(grid, m, *cells[1], *ext.p2)]
        fr = FrameResult(row=row, stress_MPa=stress, measurement=m,
                         gauge_valid=False, gauge_reason="")
        if disp[0] is None or disp[1] is None:
            fr.gauge_reason = ("gauge cell corners not all valid for "
                               + ", ".join(
                                   lbl for lbl, d in (("p1", disp[0]), ("p2", disp[1]))
                                   if d is None))
        else:
            fr.gauge_valid = True
            fr.p1_u_mm, fr.p1_v_mm = disp[0]
            fr.p2_u_mm, fr.p2_v_mm = disp[1]
            a = a0 + np.array(disp[0])
            b = b0 + np.array(disp[1])
            fr.gauge_len_mm = float(np.linalg.norm(b - a))
            fr.strain = fr.gauge_len_mm / len0 - 1.0
        results.append(fr)
    return results


def fit_modulus(results: list[FrameResult], lo: float, hi: float) -> dict:
    """Least-squares sigma = E*eps + b over the closed strain interval."""
    pts = [(fr.strain, fr.stress_MPa) for fr in results
           if fr.gauge_valid and lo <= fr.strain <= hi]
    base = {"interval": [lo, hi], "n_points": len(pts)}
    if len({round(e, 12) for e, _ in pts}) < MIN_FIT_POINTS:
        return {**base, "status": "failed",
                "reason": "fewer_than_three_distinct_strains_in_interval"}
    eps = np.array([e for e, _ in pts])
    sig = np.array([s for _, s in pts])
    A = np.column_stack([eps, np.ones_like(eps)])
    (E, b), *_ = np.linalg.lstsq(A, sig, rcond=None)
    if not np.isfinite(E) or E <= 0:
        return {**base, "status": "failed", "reason": "non_positive_modulus"}
    resid = sig - (E * eps + b)
    ss_res = float(np.dot(resid, resid))
    ss_tot = float(np.dot(sig - sig.mean(), sig - sig.mean()))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else (1.0 if ss_res == 0 else 0.0)
    return {**base, "status": "ok", "E_MPa": float(E), "b_MPa": float(b),
            "r_squared": float(r2)}


def offset_yield(results: list[FrameResult], fit: dict) -> dict:
    """First 0.2% offset crossing after the fit interval, or null + reason."""
    if fit.get("status") != "ok":
        return {"strain": None, "stress_MPa": None, "reason": "fit_failed"}
    E = fit["E_MPa"]
    b = fit["b_MPa"]
    hi = fit["interval"][1]
    prev = None
    for fr in results:
        if not fr.gauge_valid:
            prev = None            # never bridge missing gauges
            continue
        cur = (fr.strain, fr.stress_MPa)
        if prev is not None and cur[0] > prev[0] and prev[0] >= hi:
            d0 = prev[1] - (E * (prev[0] - OFFSET_STRAIN) + b)
            d1 = cur[1] - (E * (cur[0] - OFFSET_STRAIN) + b)
            if d0 > 0 and d1 <= 0:
                t = d0 / (d0 - d1)
                eps_y = prev[0] + t * (cur[0] - prev[0])
                return {"strain": float(eps_y),
                        "stress_MPa": float(E * (eps_y - OFFSET_STRAIN) + b),
                        "reason": ""}
        prev = cur
    return {"strain": None, "stress_MPa": None,
            "reason": "no_positive_to_nonpositive_crossing_after_fit_interval"}
