"""Data-assimilation example: reconstruct the interior flow by inverse Navier-Stokes.

Fits the interfacial boundary velocity so the steady axisymmetric Navier-Stokes solution
inside the bubble best matches the measured (ray-traced optic-flow) velocity field.

Inputs (all in examples/grid_and_data_assimilation/):
  * mesh_low.npz              -- a low-resolution computational mesh (build_bubble_mesh)
  * x_mat.txt, y_mat.txt      -- the experimental grid (camera-plane physical coords, px)
  * u_phys.txt, v_phys.txt    -- ray-traced velocity on that grid (px/frame)
  * bubble_shape_tot.txt      -- the bubble interface in px (the mesh's normalization ref)

The solver is NON-DIMENSIONAL: lengths are normalized by the bubble diameter and
velocities by the bubble rise velocity (the calibration constants below), with the
physical scale carried by the Reynolds number.

Run (takes a few minutes on a laptop -- one steady solve ~5 s, Powell does tens of them):

    python examples/grid_and_data_assimilation/example_DA.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.interpolate import LinearNDInterpolator

from pyinnerbubble.assimilation import assimilate_boundary_velocity, plot_assimilation

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent

# --- physical calibration (MKS) -- normalize by bubble size and bubble velocity ---
BUBBLE_DIAMETER_PX = 853.0          # bubble equivalent diameter in pixels
BUBBLE_DIAMETER_M = 0.022           # = 22 mm
BUBBLE_VELOCITY_PX_PER_FRAME = 20.0  # bubble rise velocity in px/frame
BUBBLE_VELOCITY_M_S = 0.21          # = 21 cm/s
REYNOLDS = 301.0                    # = V * D / nu  (carries the physical scale)

# The conformal map / mesh_low was built normalized by this reference (grid_build_example
# D_REF). Rescaling by MESH_DREF_PX / BUBBLE_DIAMETER_PX puts the mesh in diameter units
# (this is the original code's "scale" ~ 1.216). Set to whatever D_REF built mesh_low.
MESH_DREF_PX = 1040.0

NUM_OF_POINTS = 2   # free spline control points for the boundary velocity (as requested)
MAXITER = 15        # optimizer iterations (raise for a tighter fit; each adds ~seconds)

# Weighting matrix in the cost. None -> np.ones_like(mesh) (uniform, the default). To weight
# by measurement confidence, build an (M, N) array on the mesh and assign it here instead.
WEIGHT_MAT = None

OUT_DIR = REPO_ROOT / "output" / "data_assimilation"
# ---------------------------------------------------------------------------

MESH_TO_DIAMETER = MESH_DREF_PX / BUBBLE_DIAMETER_PX  # mesh build-norm -> diameter units


def _load_mesh():
    m = np.load(SCRIPT_DIR / "mesh_low.npz")
    # x_mat = radial (r), y_mat = axial (z); rescale from build-normalization to diameter units.
    r_mat = m["x_mat"] * MESH_TO_DIAMETER
    z_mat = m["y_mat"] * MESH_TO_DIAMETER
    return r_mat, z_mat


def _measured_velocity_on_mesh(r_mat, z_mat):
    """Load the experimental grid + ray-traced velocity, put them in the mesh's
    non-dimensional frame, and interpolate onto the mesh nodes.

    NOTE (assumptions to verify against your convention): the pixel grid is mapped into
    the mesh frame with the same normalization the conformal map used (center in x, flip
    and shift in y), then scaled to diameter units. Velocity is normalized by the bubble
    rise velocity; u_z takes a sign flip because the y axis is flipped. Adjust the signs
    here if your circulation direction comes out reversed.
    """
    x_px = np.loadtxt(SCRIPT_DIR / "x_mat.txt")
    y_px = np.loadtxt(SCRIPT_DIR / "y_mat.txt")
    u_px = np.loadtxt(SCRIPT_DIR / "u_phys.txt")   # px/frame
    v_px = np.loadtxt(SCRIPT_DIR / "v_phys.txt")
    shape = np.loadtxt(SCRIPT_DIR / "bubble_shape_tot.txt")  # interface, px (normalization ref)

    x_center = 0.5 * (shape[:, 0].min() + shape[:, 0].max())
    y_top = shape[:, 1].min()

    def to_mesh_frame(x, y):
        # same transform the conformal preprocessing applied to the bubble shape, then
        # rescaled to diameter units -> equivalent to (px - ref) / BUBBLE_DIAMETER_PX.
        cf_x = (x - x_center) / MESH_DREF_PX
        cf_y = 0.25 - (y - y_top) / MESH_DREF_PX
        return cf_x * MESH_TO_DIAMETER, cf_y * MESH_TO_DIAMETER

    valid = (x_px >= 0) & (y_px >= 0) & np.isfinite(u_px) & np.isfinite(v_px)
    r_pts, z_pts = to_mesh_frame(x_px[valid], y_px[valid])
    ur = u_px[valid] / BUBBLE_VELOCITY_PX_PER_FRAME        # radial, same sign as x
    uz = -v_px[valid] / BUBBLE_VELOCITY_PX_PER_FRAME       # axial, sign-flipped (y flipped)

    pts = np.column_stack([r_pts, z_pts])
    query = np.column_stack([r_mat.ravel(), z_mat.ravel()])
    ur_meas = LinearNDInterpolator(pts, ur, fill_value=0.0)(query).reshape(r_mat.shape)
    uz_meas = LinearNDInterpolator(pts, uz, fill_value=0.0)(query).reshape(r_mat.shape)
    return ur_meas, uz_meas


def main():
    r_mat, z_mat = _load_mesh()
    print(f"[DA] mesh {r_mat.shape}, r in [{r_mat.min():.3f}, {r_mat.max():.3f}], "
          f"z in [{z_mat.min():.3f}, {z_mat.max():.3f}] (diameter units)")

    ur_meas, uz_meas = _measured_velocity_on_mesh(r_mat, z_mat)
    print(f"[DA] measured velocity on mesh: |u| max = "
          f"{np.nanmax(np.sqrt(ur_meas**2 + uz_meas**2)):.3f} (bubble-velocity units)")

    result = assimilate_boundary_velocity(
        r_mat, z_mat, ur_meas, uz_meas,
        reynolds=REYNOLDS, num_of_points=NUM_OF_POINTS,
        weight_mat=WEIGHT_MAT, maxiter=MAXITER, verbose=True,
    )
    print(f"[DA] done. optimized control points = {result.vel_opt}, final cost = {result.cost:.6g}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT_DIR / "assimilation.npz",
        vel_opt=result.vel_opt, cost=result.cost, s=result.s,
        r_mat=result.r_mat, z_mat=result.z_mat,
        psi=result.psi, phi=result.phi, u_r=result.u_r, u_z=result.u_z,
        cost_history=np.asarray(result.cost_history),
    )
    plot_assimilation(result, save_path=OUT_DIR / "assimilation.png")
    print(f"[DA] results saved -> {OUT_DIR}")


if __name__ == "__main__":
    main()
