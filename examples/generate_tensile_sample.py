"""Generate a reproducible synthetic tensile series for demonstration.

A reference speckle image plus N_FRAMES deformed frames with increasing
uniaxial strain along x (Poisson contraction along y).  The material model is
bilinear: E_TRUE up to YIELD_STRAIN, then linear hardening, so the 0.2%
offset yield point exists inside the series.  Writes to examples/tensile/:
reference.png, frames.zip, forces.csv, truth.json.
Run: .venv/bin/python examples/generate_tensile_sample.py
"""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates

SEED = 97
SIZE = 256
N_DOTS = 4200
DOT_SIGMA = 1.1
SCALE_MM_PER_PX = 0.05
AREA_MM2 = 25.0
E_TRUE = 70000.0          # MPa
HARDENING = 2500.0        # MPa tangent after yield
YIELD_STRAIN = 0.010
POISSON = 0.3
N_FRAMES = 10
STRAINS = np.linspace(0.002, 0.020, N_FRAMES)
DT_S = 0.5


def make_reference(rng: np.random.Generator) -> np.ndarray:
    impulses = np.zeros((SIZE, SIZE))
    ys = rng.integers(0, SIZE, N_DOTS)
    xs = rng.integers(0, SIZE, N_DOTS)
    np.add.at(impulses, (ys, xs), rng.uniform(0.6, 1.0, N_DOTS))
    field = gaussian_filter(impulses, DOT_SIGMA)
    field /= field.max()
    img = 30.0 + 175.0 * field + rng.normal(0, 1.2, (SIZE, SIZE))
    return np.clip(img, 0, 255).astype(np.uint8)


def warp(ref: np.ndarray, exx: float) -> np.ndarray:
    x, y = np.meshgrid(np.arange(SIZE, dtype=float),
                       np.arange(SIZE, dtype=float))
    c = SIZE / 2.0
    F = np.array([[1.0 + exx, 0.0], [0.0, 1.0 - POISSON * exx]])
    rel = np.vstack([(x - c).ravel(), (y - c).ravel()])
    src = np.linalg.solve(F, rel) + np.array([[c], [c]])
    warped = map_coordinates(ref.astype(float), [src[1], src[0]],
                             order=3, mode="reflect")
    return np.clip(warped.reshape(SIZE, SIZE), 0, 255).astype(np.uint8)


def stress_of(eps: float) -> float:
    if eps <= YIELD_STRAIN:
        return E_TRUE * eps
    return E_TRUE * YIELD_STRAIN + HARDENING * (eps - YIELD_STRAIN)


def main() -> None:
    out = Path(__file__).resolve().parent / "tensile"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    ref = make_reference(rng)
    Image.fromarray(ref, mode="L").save(out / "reference.png")

    lines = ["frame_id,time_s,force_N"]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for k, eps in enumerate(STRAINS):
            frame_id = f"f{k:02d}"
            img = Image.fromarray(warp(ref, float(eps)), mode="L")
            png = io.BytesIO()
            img.save(png, format="PNG")
            zf.writestr(f"{frame_id}.png", png.getvalue())
            lines.append(f"{frame_id},{(k + 1) * DT_S:.3f},"
                         f"{stress_of(float(eps)) * AREA_MM2:.3f}")
    (out / "frames.zip").write_bytes(buf.getvalue())
    (out / "forces.csv").write_text("\n".join(lines) + "\n")

    meta = {
        "seed": SEED,
        "E_true_MPa": E_TRUE,
        "hardening_MPa": HARDENING,
        "yield_strain_true": YIELD_STRAIN,
        "area_mm2": AREA_MM2,
        "scale_mm_per_px": SCALE_MM_PER_PX,
        "strains": [round(float(e), 6) for e in STRAINS],
        "suggested_request": {
            "scale_mm_per_px": SCALE_MM_PER_PX,
            "roi": [32, 32, 192, 192],
            "subset_size": 31, "grid_step": 16,
            "search_radius": 8, "max_iterations": 50,
            "area_mm2": AREA_MM2,
            "p1": [64.0, 128.0], "p2": [192.0, 128.0],
            "fit_strain_interval": [0.002, 0.008],
        },
    }
    (out / "truth.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {out}/reference.png, frames.zip, forces.csv, truth.json")


if __name__ == "__main__":
    main()
