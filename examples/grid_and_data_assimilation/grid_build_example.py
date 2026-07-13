"""Data-assimilation example: build the conformal map and the computational mesh.

Reads a bubble boundary that the geometry stage produced (a text file of (x, y)
pixel coordinates, e.g. ``bubble_shape_tot.txt``), then:
  1. builds the conformal map between the unit disk and the bubble region, and
  2. builds a graded computational mesh on the bubble (dense near the interface and
     the boundary ends).
Both are saved (with preview plots) into a new folder under ``output/``.

Edit the CONFIG block below, then run:

    python examples/grid_and_data_assimilation/grid_build_example.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pyinnerbubble import use_matlab_style

from pyinnerbubble.assimilation import (
    build_bubble_mesh,
    build_conformal_map,
    plot_conformal_map,
    plot_mesh,
    save_conformal_map,
    save_mesh,
)

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
# Resolved from this file's own location so paths work regardless of the current
# working directory the script is launched from.
SCRIPT_DIR = Path(__file__).resolve().parent  # this folder (examples/grid_and_data_assimilation/), where the bubble-shape file lives
REPO_ROOT = SCRIPT_DIR.parent.parent  # up through examples/ to the repo root

# Bubble boundary from the geometry stage: (x, y) pixel coordinates, one point per row.
BUBBLE_SHAPE_FILE = SCRIPT_DIR / "bubble_shape_tot.txt"

# Reference bubble diameter in pixels (MATLAB Dref). The boundary is divided by this so the
# whole data-assimilation pipeline works in diameter-normalized units. Use the bubble's
# equivalent diameter -- for the bundled EXP8 shape that is 853 px (= 22 mm).
D_REF = 853.0

# --- conformal-map parameters ---
FOURIER_ORDER = 31   # harmonics in the Fourier fit of the boundary
N_BOUNDARY = 800     # Kerzman-Stein boundary correspondence points (M)

# --- mesh grid parameters (computational grid on the unit half-disk, mapped to the bubble) ---
# The disk is meshed over theta in [0, pi] and maps to the right half of the
# (axisymmetric) bubble. The grid is tanh-clustered where interfacial gradients are
# steepest: toward the interface (r=1) and toward the two ends of the boundary.
N_THETA_HALF = 31    # half the number of angular nodes; total angular nodes = 2*N_THETA_HALF
N_RADIAL = 25        # number of radial nodes (disk center r=0 -> interface r=1)
BETA_THETA = 3.0     # angular clustering strength; larger = denser toward the two ends
BETA_RADIAL = 4.0    # radial clustering strength; larger = denser toward the interface r=1
ALPHA = 0.705         # angular split: nodes cluster at ALPHA*pi, tuned so that lands on the
                     # bubble rim (widest point). Shape-specific: theta_rim/pi for this EXP8
                     # bubble is ~0.597 (rim at ~107 deg on the disk). Retune per bubble.
N_SMOOTH = 20        # Laplacian smoothing passes on the angular node distribution

# Output: a new folder under output/ for this data-assimilation stage.
OUT_DIR = REPO_ROOT / "output" / "conformal_map"
PLOT = True          # save preview plots alongside the .npz files
# ---------------------------------------------------------------------------


def main():
    use_matlab_style()   # MATLAB-style figures
    if not BUBBLE_SHAPE_FILE.exists():
        raise FileNotFoundError(
            f"Bubble shape file not found at {BUBBLE_SHAPE_FILE}. "
            f"Put a (x, y) boundary text file there (e.g. bubble_shape_tot.txt "
            f"from the geometry stage's output)."
        )
    bubble_shape = np.loadtxt(BUBBLE_SHAPE_FILE)
    print(f"[DA] loaded bubble shape: {bubble_shape.shape[0]} points from {BUBBLE_SHAPE_FILE.name}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- 1. conformal map: unit disk <-> bubble region ---
    conformal_map = build_conformal_map(
        bubble_shape, d_ref=D_REF,
        fourier_order=FOURIER_ORDER, n_boundary=N_BOUNDARY,
    )
    map_path = OUT_DIR / "conformal_map.npz"
    save_conformal_map(map_path, conformal_map)
    print(f"[DA] conformal map saved -> {map_path}")
    if PLOT:
        plot_conformal_map(conformal_map, save_path=OUT_DIR / "conformal_map.png")

    # --- 2. graded computational mesh on the bubble ---
    mesh = build_bubble_mesh(
        conformal_map,
        n_theta_half=N_THETA_HALF, n_radial=N_RADIAL,
        beta_theta=BETA_THETA, beta_radial=BETA_RADIAL,
        alpha=ALPHA, n_smooth=N_SMOOTH,
    )
    mesh_path = OUT_DIR / "mesh.npz"
    save_mesh(mesh_path, mesh)
    print(f"[DA] mesh saved -> {mesh_path}  (grid shape {mesh.z_mat.shape})")
    if PLOT:
        plot_mesh(mesh, save_path=OUT_DIR / "mesh.png")

    if PLOT:
        print(f"[DA] preview plots saved in {OUT_DIR}")


if __name__ == "__main__":
    main()
