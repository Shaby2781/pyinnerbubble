"""Data-assimilation example: reconstruct the interior flow by inverse Navier-Stokes.

Fits the interfacial boundary velocity so the steady axisymmetric Navier-Stokes solution
inside the bubble best matches the MEAN measured velocity field, weighted by the inverse
velocity variance (both produced by the exp8 validation run, validate_exp8.py).

Inputs (in examples/grid_and_data_assimilation/):
  * mesh_low.npz        -- low-resolution computational mesh (build_bubble_mesh)
  * mean_flow_exp8.npz  -- exp8 mean flow over the frame sequence:
        physical_x / physical_z  camera-plane physical grid (px)
        u_mean / v_mean          mean ray-traced velocity on that grid (px/frame)
        weight                   inverse velocity variance (confidence weighting)
        bubble_shape             the interface (px), used as the mesh normalization reference

The solver is NON-DIMENSIONAL: lengths are normalized by the bubble diameter and velocities
by the bubble rise velocity (the calibration constants below), with the physical scale
carried by the Reynolds number.

Run (takes a few minutes on a laptop -- one steady solve ~5 s, Powell does tens of them):

    python examples/grid_and_data_assimilation/example_DA.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pyinnerbubble.assimilation import (
    assimilate_boundary_velocity,
    interpolate_measured_field_to_mesh,
    plot_assimilation,
    plot_velocity_field_on_mesh,
)

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent

MESH_FILE = SCRIPT_DIR / "mesh.npz"          # computational mesh (use mesh.npz for the finer one)
FLOW_FILE = SCRIPT_DIR / "mean_flow_exp8.npz"    # mean velocity + inverse-variance weight (validate_exp8.py)

# --- physical calibration (MKS) -- normalize by bubble size and bubble velocity ---
BUBBLE_DIAMETER_PX = 853.0          # bubble equivalent diameter in pixels
BUBBLE_DIAMETER_M = 0.022           # = 22 mm
BUBBLE_VELOCITY_PX_PER_FRAME = 20.0  # bubble rise velocity in px/frame (matches u_b in the flow file)
BUBBLE_VELOCITY_M_S = 0.21          # = 21 cm/s
REYNOLDS = 301.0                    # = V * D / nu  (carries the physical scale)

# The conformal map / mesh was built normalized by this reference (grid_build_example D_REF).
# Rescaling by MESH_DREF_PX / BUBBLE_DIAMETER_PX puts the mesh in diameter units (the original
# code's "scale" ~ 1.216). Set to whatever D_REF built the mesh file.
MESH_DREF_PX = 1040.0

NUM_OF_POINTS = 4   # free spline control points for the boundary velocity
MAXITER = 400       # optimizer iterations (raise for a tighter fit; each adds ~seconds)

# True: weight the cost by the inverse velocity variance from the flow file (measurement
# confidence, as in the paper). False: uniform weight. Either way, nodes with NO measured
# data (outside the data hull) get weight 0 and are excluded from the cost.
USE_INVERSE_VARIANCE_WEIGHT = True

OUT_DIR = REPO_ROOT / "output" / "data_assimilation"
# ---------------------------------------------------------------------------

MESH_TO_DIAMETER = MESH_DREF_PX / BUBBLE_DIAMETER_PX  # mesh build-norm -> diameter units


def _load_mesh():
    m = np.load(MESH_FILE)
    # x_mat = radial (r), y_mat = axial (z); rescale from build-normalization to diameter units.
    return m["x_mat"] * MESH_TO_DIAMETER, m["y_mat"] * MESH_TO_DIAMETER


def _measured_on_mesh(r_mat, z_mat):
    """Load the exp8 mean flow + weight, map the pixel grid into the mesh's non-dimensional
    frame, and interpolate the mean velocity and the inverse-variance weight onto the mesh.

    NOTE (assumptions to verify against your convention): the pixel grid is mapped into the
    mesh frame with the same normalization the conformal map used (center in x, flip and shift
    in y), then scaled to diameter units. Velocity is normalized by the bubble rise velocity;
    u_z takes a sign flip because the y axis is flipped. Adjust the signs here if the
    circulation direction comes out reversed.
    """
    d = np.load(FLOW_FILE)
    x_px, y_px = d["physical_x"], d["physical_z"]     # camera-plane grid (px); -10 = outside
    u_px, v_px = d["u_mean"], d["v_mean"]             # mean velocity (px/frame)
    weight_px = d["weight"]                           # inverse velocity variance
    shape = d["bubble_shape"]                         # interface (px), normalization reference

    x_center = 0.5 * (shape[:, 0].min() + shape[:, 0].max())
    y_top = shape[:, 1].min()

    def to_mesh_frame(x, y):
        cf_x = (x - x_center) / MESH_DREF_PX
        cf_y = 0.25 - (y - y_top) / MESH_DREF_PX
        return cf_x * MESH_TO_DIAMETER, cf_y * MESH_TO_DIAMETER

    # Velocity samples: only points with real velocity data (the edge-removal band is NaN).
    valid = (x_px >= 0) & (y_px >= 0) & np.isfinite(u_px) & np.isfinite(v_px)
    r_pts, z_pts = to_mesh_frame(x_px[valid], y_px[valid])
    ur = u_px[valid] / BUBBLE_VELOCITY_PX_PER_FRAME        # radial (x), same sign
    uz = -v_px[valid] / BUBBLE_VELOCITY_PX_PER_FRAME       # axial (y), sign-flipped

    # Weight samples: ALL in-bubble points, INCLUDING the zero-weight edge-removal band -- the
    # library preserves those zeros on the mesh (see interpolate_measured_field_to_mesh).
    in_bubble = (x_px >= 0) & (y_px >= 0) & np.isfinite(weight_px)
    r_wpts, z_wpts = to_mesh_frame(x_px[in_bubble], y_px[in_bubble])

    return interpolate_measured_field_to_mesh(
        r_mat, z_mat, r_pts, z_pts, ur, uz,
        weight_r_pts=r_wpts, weight_z_pts=z_wpts, weight_vals=weight_px[in_bubble],
    )


def main():
    r_mat, z_mat = _load_mesh()
    print(f"[DA] mesh {r_mat.shape}, r in [{r_mat.min():.3f}, {r_mat.max():.3f}], "
          f"z in [{z_mat.min():.3f}, {z_mat.max():.3f}] (diameter units)")

    ur_meas, uz_meas, weight_meas, data_mask = _measured_on_mesh(r_mat, z_mat)
    print(f"[DA] measured velocity on mesh: |u| max = "
          f"{np.nanmax(np.sqrt(ur_meas**2 + uz_meas**2)):.3f} (bubble-velocity units); "
          f"{int(data_mask.sum())}/{data_mask.size} nodes have data")

    # Cost weight: inverse variance (already zeroed at no-data + edge band by the library),
    # or uniform (= the data mask: 1 where there is data, 0 elsewhere).
    weight = weight_meas if USE_INVERSE_VARIANCE_WEIGHT else data_mask
    print(f"[DA] weight: {'inverse-variance' if USE_INVERSE_VARIANCE_WEIGHT else 'uniform'}, "
          f"range [{weight[weight > 0].min():.3g}, {weight.max():.3g}] over "
          f"{int((weight > 0).sum())} weighted nodes")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Save the measured field + the weight on the mesh BEFORE the long solve (survives an interrupt).
    plot_velocity_field_on_mesh(
        r_mat, z_mat, ur_meas, uz_meas,
        title="interpolated measured velocity |u|", save_path=OUT_DIR / "measured_field_dense.png",
    )
    fig, ax = plt.subplots()
    pc = ax.pcolormesh(r_mat, z_mat, weight, shading="gouraud", cmap="viridis")
    fig.colorbar(pc, ax=ax, label="weight (inverse variance)")
    ax.plot(r_mat[:, -1], z_mat[:, -1], "k", linewidth=2)
    ax.set_aspect("equal"); ax.set_title("assimilation weight on mesh")
    ax.set_xlabel("r"); ax.set_ylabel("z")
    fig.savefig(OUT_DIR / "weight_on_mesh_dense.png", dpi=150); plt.close(fig)
    print(f"[DA] measured-field + weight plots saved in {OUT_DIR}")

    result = assimilate_boundary_velocity(
        r_mat, z_mat, ur_meas, uz_meas,
        reynolds=REYNOLDS, num_of_points=NUM_OF_POINTS,
        weight_mat=weight, maxiter=MAXITER, verbose=True,
    )
    print(f"[DA] done. optimized control points = {result.vel_opt}, final cost = {result.cost:.6g}")

    np.savez(
        OUT_DIR / "assimilation_dense.npz",
        vel_opt=result.vel_opt, cost=result.cost, s=result.s,
        r_mat=result.r_mat, z_mat=result.z_mat,
        psi=result.psi, phi=result.phi, u_r=result.u_r, u_z=result.u_z,
        weight=weight, cost_history=np.asarray(result.cost_history),
    )
    plot_assimilation(result, save_path=OUT_DIR / "assimilation_dense.png")
    plot_velocity_field_on_mesh(
        result.r_mat, result.z_mat, result.u_r, result.u_z,
        title="reconstructed velocity |u|", save_path=OUT_DIR / "reconstructed_field_dense.png",
    )
    print(f"[DA] results saved -> {OUT_DIR}")


if __name__ == "__main__":
    main()
