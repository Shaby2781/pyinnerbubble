"""Map camera-plane optic flow to physical velocity, and smooth it (paper Sec. III B/C boundary).

Converts the dense camera-plane flow field from :func:`compute_optic_flow`
into physical (x, z) velocity using the spatial derivatives of the ray-traced
camera<->physical mapping (``physical_x``/``physical_z`` from
:func:`pyinnerbubble.geometry.build_camera_plane_mapping`), via the chain rule
-- this is the single-frame version of the MATLAB ``calc_mean_and_steps``
mapping step, without its multi-frame averaging. Also provides the 2D
Savitzky-Golay smoothing filter (``sgsdf_2d`` + ``convolve2`` in the original)
applied to the mapped velocity afterward.

Ported directly from ``div_xsi.m``/``div_eta.m``/``sgsdf_2d.m``:

- ``div_xsi`` is a central-difference (forward/backward at the edges) derivative
  along axis 1 of its input; ``div_eta`` is the same along axis 0. This matches
  ``build_camera_plane_mapping``'s own documented convention for
  ``physical_x``/``physical_z``: axis 0 is "eta", axis 1 is "xsi".
- ``sgsdf_2d(-r:r, -r:r, order, order, d, 0)`` (the ``flag_coupling=0`` case used
  here) fits a *separable* polynomial basis {x^order..x^0, y^order..y^1} (no xy
  cross term) by least squares over the window, and returns the FIR kernel for
  the x-derivative of order d at the window center; the y-derivative kernel is
  this kernel's transpose (only valid because x/y ranges and orders match here).
"""

from __future__ import annotations

from math import factorial

import numpy as np
from ..plotting import matlab_styled
from scipy.ndimage import distance_transform_edt
from scipy.signal import convolve2d


def compute_mapping_derivatives(x_mat: np.ndarray, y_mat: np.ndarray):
    """Spatial derivatives of the camera<->physical mapping along each grid axis.

    Returns (x_xsi, x_eta, y_xsi, y_eta): central-difference derivatives of
    x_mat/y_mat along axis 1 ("xsi") and axis 0 ("eta") respectively -- ported
    from div_xsi.m/div_eta.m (forward/backward difference at the edges, central
    in the interior; np.gradient matches this along the interior but uses a
    slightly different one-sided formula at the very edge columns/rows).
    """
    x_eta, x_xsi = np.gradient(x_mat)
    y_eta, y_xsi = np.gradient(y_mat)
    return x_xsi, x_eta, y_xsi, y_eta


def map_camera_flow_to_physical(
    flow: np.ndarray, x_mat: np.ndarray, y_mat: np.ndarray, edge_removal: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Convert camera-plane optic flow to physical velocity via the chain rule.

    Parameters
    ----------
    flow : (H, W, 2) array from :func:`compute_optic_flow`.
    x_mat, y_mat : the physical mapping from
        :func:`pyinnerbubble.geometry.build_camera_plane_mapping`
        (``physical_x``/``physical_z``), same (H, W) shape as flow.
    edge_removal : int
        How many pixels to remove *inside* the bubble edge. Pixels outside the
        bubble (the -10 sentinel in ``x_mat``/``y_mat``) are always set to NaN;
        if ``edge_removal`` > 0, a band that many pixels wide just inside the
        interface is removed too, since there the Farneback optic-flow window and
        the 2D Savitzky-Golay smoothing window applied afterward reach across the
        edge and average in invalid outside-bubble data. A sensible value is the
        average of the two smoothing window sizes (Farneback ``filter_size`` and
        the S-G window ``2*radius+1``). 0 (default) removes only the outside.

    Returns
    -------
    u_phys, v_phys : physical velocity components, per frame, same units as
        flow's input pixels (no averaging across frames, no smoothing applied),
        with invalid pixels (outside, plus the edge band if requested) set to NaN.
    """
    x_xsi, x_eta, y_xsi, y_eta = compute_mapping_derivatives(x_mat, y_mat)
    flow_0, flow_1 = flow[..., 0], flow[..., 1]
    u_phys = flow_0 * x_xsi + flow_1 * x_eta
    v_phys = flow_0 * y_xsi + flow_1 * y_eta

    # Outside-bubble pixels are always invalid; optionally also strip an inside-edge band.
    inside = (x_mat >= 0) & (y_mat >= 0)
    valid = inside
    if edge_removal > 0:
        dist_to_edge = distance_transform_edt(inside)  # px from each interior pixel to the outside
        valid = inside & (dist_to_edge > edge_removal)
    u_phys = np.where(valid, u_phys, np.nan)
    v_phys = np.where(valid, v_phys, np.nan)
    return u_phys, v_phys


def savgol_2d_kernel(radius: int, order: int = 2, deriv: int = 0) -> np.ndarray:
    """Build a 2D Savitzky-Golay smoothing/differentiation kernel (x-direction).

    Direct port of ``sgsdf_2d(-radius:radius, -radius:radius, order, order, deriv, 0)``
    (the flag_coupling=0 case): fits the separable basis
    {x^order, ..., x^1, x^0, y^order, ..., y^1} (no xy cross term) by least
    squares over the (2*radius+1) square window, and returns the FIR kernel for
    the x-derivative of order ``deriv`` at the window center (deriv=0 for pure
    smoothing), where "x" is axis 0 of the returned kernel. Transpose the
    result to get the corresponding axis-1 ("y") derivative kernel.
    """
    coords = np.arange(-radius, radius + 1, dtype=float)
    xx, yy = np.meshgrid(coords, coords, indexing="ij")
    x, y = xx.ravel(), yy.ravel()

    x_terms = [x ** k for k in range(order, -1, -1)]  # x^order, ..., x^1, x^0
    y_terms = [y ** k for k in range(order, 0, -1)]  # y^order, ..., y^1 (no y^0: redundant with x^0)
    design = np.column_stack(x_terms + y_terms)

    pseudo_inverse = np.linalg.pinv(design)
    term_index = order - deriv  # position of the x^deriv column among x_terms
    kernel = pseudo_inverse[term_index] * factorial(deriv)
    return kernel.reshape(2 * radius + 1, 2 * radius + 1)


def savgol_2d_smooth(field: np.ndarray, radius: int = 7, order: int = 2) -> np.ndarray:
    """Smooth a 2D field with a 2D Savitzky-Golay filter (local polynomial fit).

    NaNs are filled with 0 before convolving and restored afterward -- values
    near a NaN region are influenced by those zero-filled neighbors, matching
    the boundary behavior of the original MATLAB convolve2-based smoothing.
    """
    kernel = savgol_2d_kernel(radius, order, deriv=0)
    nan_mask = np.isnan(field)
    filled = np.where(nan_mask, 0.0, field)
    smoothed = convolve2d(filled, kernel, mode="same")
    smoothed[nan_mask] = np.nan
    return smoothed


def velocity_to_physical_units(u: np.ndarray, v: np.ndarray, px_per_mm: float, frame_rate_hz: float):
    """Convert velocity from px/frame to m/s, given a pixel scale and frame rate."""
    scale = frame_rate_hz / px_per_mm / 1000.0  # (frame/s) / (px/mm) * (1 m / 1000 mm) = m/s per px/frame
    return u * scale, v * scale


@matlab_styled
def plot_velocity_field(
    x_mat: np.ndarray,
    y_mat: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    px_per_mm: float | None = None,
    frame_rate_hz: float | None = None,
    quiver_step: int = 15,
    quiver_scale: float | None = None,
    quiver_width: float | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    bubble_contour: np.ndarray | None = None,
    save_path=None,
) -> None:
    """Plot velocity magnitude as a flat-shaded pcolor (YlOrBr) with a black quiver
    overlay, in the physical (x_mat, y_mat) coordinate plane.

    Units are px/frame by default. Pass both px_per_mm and frame_rate_hz to plot
    in m/s instead (e.g. 38 px/mm at a 400 Hz frame rate).

    vmin/vmax fix the colorbar range (matching MATLAB's caxis/clim), e.g.
    vmin=0, vmax=0.5 for m/s. Defaults to auto-scaling from the data.

    quiver_scale : data units per arrow-length unit (matplotlib's quiver ``scale``).
        Smaller values draw LONGER arrows; None auto-scales from the data.
    quiver_width : arrow shaft width, as a fraction of the plot width (matplotlib's
        quiver ``width``); None uses matplotlib's default (~0.005).
    bubble_contour : optional (N, 2) array (e.g. CameraMappingResult.bubble_shape_closed)
        drawn as a thick black curve on top of the pcolor/quiver, in the same
        (x_mat, y_mat) coordinate plane.

    x_mat/y_mat's -10 sentinel for "outside the bubble" (see
    build_camera_plane_mapping) is masked to NaN here -- pcolor otherwise draws
    spurious quads connecting valid interior points to the collapsed (-10, -10)
    point, since it treats x_mat/y_mat as literal coordinates, not a mask.
    """
    import matplotlib.pyplot as plt

    if (px_per_mm is None) != (frame_rate_hz is None):
        raise ValueError("Provide both px_per_mm and frame_rate_hz, or neither.")

    outside = (x_mat < 0) | (y_mat < 0)
    x_mat = np.where(outside, np.nan, x_mat)
    y_mat = np.where(outside, np.nan, y_mat)

    if px_per_mm is not None:
        u, v = velocity_to_physical_units(u, v, px_per_mm, frame_rate_hz)
        unit_label = "m/s"
    else:
        unit_label = "px/frame"

    magnitude = np.sqrt(u ** 2 + v ** 2)

    fig, ax = plt.subplots()
    # shading="flat" treats x_mat/y_mat as cell vertices, so C must be one row/
    # column smaller (matplotlib raises on equal-sized inputs; MATLAB's pcolor
    # silently drops the last row/col in this situation instead).
    mesh = ax.pcolor(x_mat, y_mat, magnitude[:-1, :-1], cmap="YlOrBr", shading="flat", vmin=vmin, vmax=vmax)
    # shorter colorbar (half height) with larger (2x) tick/label font
    _cbfs = plt.rcParams["font.size"]   # same size as ticks/labels
    cbar = fig.colorbar(mesh, ax=ax, shrink=0.5)
    cbar.ax.set_title(rf"$|u|$ [{unit_label}]", fontsize=_cbfs, pad=12)   # math title on top
    cbar.ax.tick_params(labelsize=_cbfs)

    step = quiver_step
    # angles="xy" is required so the arrow direction is computed from the actual
    # data-coordinate displacement (and thus respects invert_yaxis() below);
    # the default angles="uv" ignores axis inversion and renders v upside down.
    ax.quiver(
        x_mat[::step, ::step], y_mat[::step, ::step],
        u[::step, ::step], v[::step, ::step],
        color="black", angles="xy", scale=quiver_scale, width=quiver_width,
    )
    if bubble_contour is not None:
        ax.plot(bubble_contour[:, 0], bubble_contour[:, 1], color="black", linewidth=2.5)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.axis("off")

    if save_path is not None:
        plt.savefig(save_path)
        plt.close(fig)
    else:
        plt.show()


def _parse_args(argv):
    import argparse

    parser = argparse.ArgumentParser(
        description="Map camera-plane optic flow to physical velocity, then smooth it."
    )
    parser.add_argument("--u-x", required=True, help="Path to u_x.txt (from compute_optic_flow).")
    parser.add_argument("--u-y", required=True, help="Path to u_y.txt (from compute_optic_flow).")
    parser.add_argument("--x-mat", required=True, help="Path to x_mat.txt (physical_x from build_camera_plane_mapping).")
    parser.add_argument("--y-mat", required=True, help="Path to y_mat.txt (physical_z from build_camera_plane_mapping).")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--smooth-radius", type=int, default=7)
    parser.add_argument("--smooth-order", type=int, default=2)
    parser.add_argument("--no-smooth", action="store_true", help="Skip the 2D Savitzky-Golay smoothing step.")
    parser.add_argument(
        "--edge-removal", type=int, default=0,
        help="Pixels to remove inside the bubble edge (0 = off). Discards vectors where the "
             "Farneback/Savitzky-Golay windows reach across the interface; try the average of "
             "the two smoothing windows.",
    )
    parser.add_argument("--plot", action="store_true", help="Save a velocity-magnitude + quiver plot.")
    parser.add_argument("--px-per-mm", type=float, default=None, help="e.g. 38 -- plot velocity in m/s instead of px/frame.")
    parser.add_argument("--frame-rate-hz", type=float, default=None, help="e.g. 400 -- required together with --px-per-mm.")
    parser.add_argument("--quiver-step", type=int, default=15)
    parser.add_argument("--quiver-scale", type=float, default=None, help="Data units per arrow length; smaller = longer arrows.")
    parser.add_argument("--quiver-width", type=float, default=None, help="Arrow shaft width, fraction of plot width.")
    parser.add_argument("--vmin", type=float, default=None, help="Colorbar lower bound, e.g. 0.")
    parser.add_argument("--vmax", type=float, default=None, help="Colorbar upper bound, e.g. 0.5.")
    parser.add_argument(
        "--bubble-contour", default=None,
        help="Path to a bubble contour txt (e.g. bubble_shape_tot.txt from build_camera_plane_mapping) "
             "to draw as a thick black curve on the plot.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    from pathlib import Path

    args = _parse_args(argv)

    u_x = np.loadtxt(args.u_x)
    u_y = np.loadtxt(args.u_y)
    x_mat = np.loadtxt(args.x_mat)
    y_mat = np.loadtxt(args.y_mat)
    flow = np.stack([u_x, u_y], axis=-1)

    # map_camera_flow_to_physical masks the -10 "outside the bubble" sentinel (see
    # build_camera_plane_mapping), plus an inside-edge band when --edge-removal > 0.
    u_phys, v_phys = map_camera_flow_to_physical(flow, x_mat, y_mat, edge_removal=args.edge_removal)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_dir / "u_phys.txt", u_phys)
    np.savetxt(out_dir / "v_phys.txt", v_phys)

    u_final, v_final = u_phys, v_phys
    if not args.no_smooth:
        u_final = savgol_2d_smooth(u_phys, radius=args.smooth_radius, order=args.smooth_order)
        v_final = savgol_2d_smooth(v_phys, radius=args.smooth_radius, order=args.smooth_order)
        np.savetxt(out_dir / "u_phys_smooth.txt", u_final)
        np.savetxt(out_dir / "v_phys_smooth.txt", v_final)

    if args.plot:
        bubble_contour = np.loadtxt(args.bubble_contour) if args.bubble_contour else None
        plot_velocity_field(
            x_mat, y_mat, u_final, v_final,
            px_per_mm=args.px_per_mm, frame_rate_hz=args.frame_rate_hz,
            quiver_step=args.quiver_step, quiver_scale=args.quiver_scale, quiver_width=args.quiver_width,
            vmin=args.vmin, vmax=args.vmax, bubble_contour=bubble_contour,
            save_path=out_dir / "velocity_field.png",
        )

    print(f"Saved outputs to {out_dir}")


if __name__ == "__main__":
    main()
