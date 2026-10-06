"""FastAPI application: upload two 8-bit grayscale PNGs and analysis form."""
from __future__ import annotations

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from tensile_dic224.delivery import build_tensile_zip, build_zip
from tensile_dic224.pipeline import run_measurement
from tensile_dic224.tensile import (build_extensometer_params, fit_modulus,
                                    match_frames, offset_yield,
                                    parse_force_csv, parse_frames_zip,
                                    run_tensile)
from tensile_dic224.validation import (MAX_EDGE, MAX_GRID_POINTS,
                                          MAX_ITERATIONS, MAX_SEARCH_RADIUS,
                                          RequestError, build_params,
                                          decode_gray_png, grid_points,
                                          validate_pair)

app = FastAPI(
    title="tensile_dic224",
    description="2D subset DIC displacement and Green-Lagrange strain backend",
    version="0.1.0",
)


LIMITS = {
    "max_edge_px": MAX_EDGE,
    "max_grid_points": MAX_GRID_POINTS,
    "max_search_radius_px": MAX_SEARCH_RADIUS,
    "max_iterations": MAX_ITERATIONS,
}


@app.get("/health")
def health():
    return {"status": "ok", "limits": LIMITS}


async def _prepare(reference: UploadFile, deformed: UploadFile, form: dict):
    try:
        params = build_params(form)
        ref = decode_gray_png(await reference.read(), "reference")
        defo = decode_gray_png(await deformed.read(), "deformed")
        validate_pair(ref, defo, params)
        gx, gy = grid_points(params)
    except RequestError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    measurement = run_measurement(ref, defo, gx, gy, params)
    form_dump = {k: form.get(k) for k in (
        "scale_mm_per_px", "roi_x", "roi_y", "roi_w", "roi_h",
        "subset_size", "grid_step", "search_radius", "max_iterations")}
    return measurement, params, form_dump


@app.post("/analyze", summary="Run DIC and return per-point JSON")
async def analyze(
    reference: UploadFile = File(..., description="reference 8-bit gray PNG"),
    deformed: UploadFile = File(..., description="deformed 8-bit gray PNG"),
    scale_mm_per_px: str = Form(...),
    roi_x: str = Form(...),
    roi_y: str = Form(...),
    roi_w: str = Form(...),
    roi_h: str = Form(...),
    subset_size: str = Form(...),
    grid_step: str = Form(...),
    search_radius: str = Form(...),
    max_iterations: str = Form(...),
):
    form = dict(scale_mm_per_px=scale_mm_per_px, roi_x=roi_x, roi_y=roi_y,
                roi_w=roi_w, roi_h=roi_h, subset_size=subset_size,
                grid_step=grid_step, search_radius=search_radius,
                max_iterations=max_iterations)
    m, params, form_dump = await _prepare(reference, deformed, form)

    def cell(a):
        return None if not np.isfinite(a) else float(a)

    points = []
    for i in range(m.n_points):
        points.append({
            "index": i,
            "grid_x_mm": cell(m.grid_x_px[i] * params.scale_mm_per_px),
            "grid_y_mm": cell(m.grid_y_px[i] * params.scale_mm_per_px),
            "valid": bool(m.valid[i]),
            "u_mm": cell(m.u_mm[i]),
            "v_mm": cell(m.v_mm[i]),
            "zncc": cell(m.correlation[i]),
            "converged": bool(m.converged[i]),
            "iterations": int(m.iterations[i]),
            "exx": cell(m.exx[i]),
            "eyy": cell(m.eyy[i]),
            "exy": cell(m.exy[i]),
            "failure_reason": m.reasons[i],
        })
    return {
        "parameters": form_dump,
        "limits": LIMITS,
        "n_points": m.n_points,
        "n_valid_displacement": m.n_valid,
        "n_valid_strain": int(np.isfinite(m.exx).sum()),
        "points": points,
}


@app.post("/download", summary="Run DIC and download the result ZIP")
async def download(
    reference: UploadFile = File(...),
    deformed: UploadFile = File(...),
    scale_mm_per_px: str = Form(...),
    roi_x: str = Form(...),
    roi_y: str = Form(...),
    roi_w: str = Form(...),
    roi_h: str = Form(...),
    subset_size: str = Form(...),
    grid_step: str = Form(...),
    search_radius: str = Form(...),
    max_iterations: str = Form(...),
):
    form = dict(scale_mm_per_px=scale_mm_per_px, roi_x=roi_x, roi_y=roi_y,
                roi_w=roi_w, roi_h=roi_h, subset_size=subset_size,
                grid_step=grid_step, search_radius=search_radius,
                max_iterations=max_iterations)
    m, params, form_dump = await _prepare(reference, deformed, form)
    archive = build_zip(m, params.scale_mm_per_px, form_dump)
    return Response(
        content=archive,
        media_type="application/zip",
        headers={"Content-Disposition":
                 'attachment; filename="tensile224_result.zip"'},
)


TENSILE_FORM_KEYS = (
    "scale_mm_per_px", "roi_x", "roi_y", "roi_w", "roi_h",
    "subset_size", "grid_step", "search_radius", "max_iterations",
    "area_mm2", "p1_x", "p1_y", "p2_x", "p2_y",
    "fit_strain_min", "fit_strain_max",
)


async def _prepare_tensile(reference: UploadFile, frames: UploadFile,
                           forces: UploadFile, form: dict):
    try:
        params = build_params(form)
        ext = build_extensometer_params(form)
        ref = decode_gray_png(await reference.read(), "reference")
        rows = parse_force_csv(await forces.read())
        frame_png = parse_frames_zip(await frames.read())
        match_frames(rows, frame_png)
        gx, gy = grid_points(params)
        results = run_tensile(ref, rows, frame_png, gx, gy, params, ext)
    except RequestError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    fit = fit_modulus(results, ext.fit_lo, ext.fit_hi)
    yld = offset_yield(results, fit)
    form_dump = {k: form.get(k) for k in TENSILE_FORM_KEYS}
    gauge_len0 = float(np.linalg.norm(
        (np.array(ext.p2) - np.array(ext.p1)) * params.scale_mm_per_px))
    return results, ext, fit, yld, form_dump, gauge_len0, params


@app.post("/tensile/analyze", summary="Tensile virtual extensometer (JSON)")
async def tensile_analyze(
    reference: UploadFile = File(..., description="reference 8-bit gray PNG"),
    frames: UploadFile = File(..., description="ZIP of 2-12 frame PNGs"),
    forces: UploadFile = File(..., description="CSV: frame_id,time_s,force_N"),
    scale_mm_per_px: str = Form(...),
    roi_x: str = Form(...),
    roi_y: str = Form(...),
    roi_w: str = Form(...),
    roi_h: str = Form(...),
    subset_size: str = Form(...),
    grid_step: str = Form(...),
    search_radius: str = Form(...),
    max_iterations: str = Form(...),
    area_mm2: str = Form(...),
    p1_x: str = Form(...),
    p1_y: str = Form(...),
    p2_x: str = Form(...),
    p2_y: str = Form(...),
    fit_strain_min: str = Form(...),
    fit_strain_max: str = Form(...),
):
    form = {k: v for k, v in locals().items()
            if k in TENSILE_FORM_KEYS}
    results, ext, fit, yld, form_dump, gauge_len0, _ = \
        await _prepare_tensile(reference, frames, forces, form)

    def cell(a):
        return None if not np.isfinite(a) else float(a)

    frames_out = []
    for fr in results:
        frames_out.append({
            "frame_id": fr.row.frame_id,
            "time_s": fr.row.time_s,
            "force_N": fr.row.force_N,
            "stress_MPa": cell(fr.stress_MPa),
            "eng_strain": cell(fr.strain) if fr.gauge_valid else None,
            "gauge_valid": fr.gauge_valid,
            "gauge_reason": fr.gauge_reason,
            "gauge_len_mm": cell(fr.gauge_len_mm),
            "n_valid_points": fr.measurement.n_valid,
        })
    return {
        "parameters": form_dump,
        "extensometer": {
            "area_mm2": ext.area_mm2,
            "p1_px": list(ext.p1),
            "p2_px": list(ext.p2),
            "gauge_len0_mm": gauge_len0,
            "fit_strain_interval": [ext.fit_lo, ext.fit_hi],
            "offset_strain": 0.002,
        },
        "frames": frames_out,
        "fit": fit,
        "yield": yld,
    }


@app.post("/tensile/download", summary="Tensile virtual extensometer (ZIP)")
async def tensile_download(
    reference: UploadFile = File(...),
    frames: UploadFile = File(...),
    forces: UploadFile = File(...),
    scale_mm_per_px: str = Form(...),
    roi_x: str = Form(...),
    roi_y: str = Form(...),
    roi_w: str = Form(...),
    roi_h: str = Form(...),
    subset_size: str = Form(...),
    grid_step: str = Form(...),
    search_radius: str = Form(...),
    max_iterations: str = Form(...),
    area_mm2: str = Form(...),
    p1_x: str = Form(...),
    p1_y: str = Form(...),
    p2_x: str = Form(...),
    p2_y: str = Form(...),
    fit_strain_min: str = Form(...),
    fit_strain_max: str = Form(...),
):
    form = {k: v for k, v in locals().items()
            if k in TENSILE_FORM_KEYS}
    results, ext, fit, yld, form_dump, gauge_len0, params = \
        await _prepare_tensile(reference, frames, forces, form)
    archive = build_tensile_zip(results, ext, fit, yld, form_dump,
                                gauge_len0, params.scale_mm_per_px)
    return Response(
        content=archive,
        media_type="application/zip",
        headers={"Content-Disposition":
                 'attachment; filename="tensile224_extensometer.zip"'},
    )
