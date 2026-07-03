"""Camera-plane <-> physical bubble-plane mapping via ray tracing (paper Fig. 2, phi(xi,eta)).

For every pixel inside the bubble's reconstructed interior, ray-traces back
through the bubble surface (cap ellipse + base exponential, from
:mod:`pyinnerbubble.geometry.shape_reconstruction`) and Snell's law to find
the physical (x, z) point it corresponds to. This mapping is later used to
transform optic-flow velocities measured in the camera plane into the
physical bubble plane.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path as FilePath

import numpy as np
from matplotlib.path import Path as MplPath
from scipy.optimize import curve_fit

from pyinnerbubble.geometry.shape_reconstruction import (
    _exp_derivative,
    _exp_value,
    _find_norm,
    fit_cap_ellipse,
    generate_ellipse_points_shifted,
)
from pyinnerbubble.io import load_contour, save_contour


@dataclass
class CameraMappingResult:
    physical_x: np.ndarray
    """Physical (radial) x-coordinate for each camera-plane pixel; NaN outside the bubble."""
    physical_z: np.ndarray
    """Physical z-coordinate for each camera-plane pixel; NaN outside the bubble."""
    bubble_shape_closed: np.ndarray
    """Full, mirrored, closed bubble contour used to build the mapping (for reference/plotting)."""
    fit_params: dict


def _mirror_and_close(curve: np.ndarray, mean_x: float) -> np.ndarray:
    """Mirror a right-half curve about x=mean_x and close it, assuming axisymmetry."""
    mirrored = curve.copy()
    mirrored[:, 0] = -curve[:, 0] + 2 * mean_x
    mirrored = mirrored[::-1, :]
    closed = np.concatenate((curve, mirrored))
    return np.vstack([closed, closed[0]])


def _ray_trace_camera_point(point, e_up, r_up, z0_up, a_exp, b_exp, c_exp, mu_oil_gas, mean_x, e_off, z_transition):
    """Ray-trace one camera-plane point (eta, xsi) to its physical (x, y, z) position.

    This is deliberately kept separate from
    ``shape_reconstruction._calc_edge_position_one_ellipse_exp``: that function
    reconstructs the boundary curve itself (vectorized over an array, switching
    ellipse/exponential region exactly at z0_up), while this one is evaluated
    per-pixel over the bubble's interior and switches region at ``z_transition``,
    which includes a margin past z0_up to smooth the seam between regions.
    """
    z_pos = point[1]
    x_pos = point[0] - mean_x
    camera_vec = np.array((0, -1, 0))

    if z_pos <= z_transition:
        theta = np.arccos(np.sqrt(((z_pos - z0_up) / e_up / r_up) ** 2 + (x_pos / r_up) ** 2))
        y_pos = r_up * e_off * np.sin(theta)
        pos = np.array((x_pos, y_pos, z_pos - z0_up))
        norm_vec = _find_norm(pos, r_up, e_up, e_off)
    else:
        r_val = np.abs(_exp_value(z_pos, a_exp, b_exp, c_exp, z0_up))
        y_pos = np.sqrt(r_val ** 2 - x_pos ** 2)
        pos = np.array((x_pos, y_pos, z_pos))
        tet = np.arctan2(y_pos, x_pos)
        p_tag = _exp_derivative(z_pos, a_exp, b_exp, c_exp, z0_up)
        n_r = 1 / np.sqrt(1 + p_tag ** 2)
        n_z = p_tag / np.sqrt(1 + p_tag ** 2)
        norm_vec = np.hstack((n_r * np.cos(tet), n_r * np.sin(tet), n_z))
        norm_vec = norm_vec / np.linalg.norm(norm_vec)

    c = np.dot(camera_vec, norm_vec)
    disc = 1 - mu_oil_gas ** 2 * (1 - c ** 2)
    if c < 0:
        c = -c
        norm_vec = -norm_vec
    amp = np.sqrt(disc) - mu_oil_gas * c
    t = mu_oil_gas * camera_vec + amp * norm_vec
    s = -y_pos / t[1]
    camera_pos = pos + s * t
    if z_pos <= z_transition:
        camera_pos[2] += z0_up
    return camera_pos


def _plot_mapping(plif_image, bubble_shape_closed, inner_region, ellipse_up_points, exp_curve, exp_curve_z, mean_x, physical_x, physical_z):
    import matplotlib.pyplot as plt

    plt.figure()
    plt.imshow(plif_image, cmap="gray")
    plt.plot(bubble_shape_closed[:, 0], bubble_shape_closed[:, 1])
    plt.plot(inner_region[:, 0], inner_region[:, 1])
    plt.plot(ellipse_up_points[:, 0] + mean_x, ellipse_up_points[:, 1])
    plt.plot(exp_curve + mean_x, exp_curve_z, ":", color="k")

    x_plot, z_plot = physical_x.copy(), physical_z.copy()
    x_plot[x_plot < 0] = np.nan
    z_plot[z_plot < 0] = np.nan

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(x_plot[::3, ::3], z_plot[::3, ::3], color="black", linewidth=0.8)
    ax.plot(x_plot[::3, ::3].T, z_plot[::3, ::3].T, color="black", linewidth=0.8)
    ax.plot(bubble_shape_closed[:, 0], bubble_shape_closed[:, 1], "r")
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_title("2D deformed structured mesh")
    ax.set_xlabel(r"$\xi$ (X)")
    ax.set_ylabel(r"$\eta$ (Y)")
    plt.show()


def build_camera_plane_mapping(
    plif_image: np.ndarray,
    bubble_shape: np.ndarray,
    inner_region: np.ndarray,
    mean_x: float,
    mu_oil_gas: float = 1.405,
    e_off: float = 1.0,
    z_transition_margin: float = 45.0,
    plot: bool = False,
) -> CameraMappingResult:
    """Build the camera-plane -> physical-plane coordinate mapping for one frame.

    Parameters
    ----------
    plif_image : the cropped PLIF image for this frame (only its shape is used, to
        size the pixel grid, plus as the background for the optional plot).
    bubble_shape, inner_region : the right-half ``real_shape``/``inner_region``
        curves returned by :func:`pyinnerbubble.geometry.reconstruct_bubble_shape`
        for this frame, in the same cropped pixel frame as ``plif_image``.
    mean_x : the bubble centroid r, i.e. ``BubbleShapeResult.mean_r`` for this frame
        (same cropped pixel frame). Used to mirror the right-half curves into a
        closed, axisymmetric outline.
    z_transition_margin : the cap-ellipse fit is used for camera points up to
        ``z_0_up + z_transition_margin``, and the base exponential fit beyond that,
        to smooth the seam between the two regions.
    plot : if True, display the reconstructed mesh overlaid on the image.
    """
    bubble_shape_for_ellipse = bubble_shape.copy()
    bubble_shape_for_ellipse[:, 0] -= mean_x

    index_r_min = np.argmin(bubble_shape_for_ellipse[:, 0])
    index_max_z = np.argmax(bubble_shape_for_ellipse[:, 1])
    curve_up = bubble_shape_for_ellipse[:index_r_min, :]
    curve_down = bubble_shape_for_ellipse[index_r_min:index_max_z + 1, :]

    a_up, b_up, y_0_up = fit_cap_ellipse(curve_up)
    popt, _ = curve_fit(
        partial(_exp_value, z_ref=y_0_up), curve_down[:, 1], curve_down[:, 0], p0=[10, 0.01, -500]
    )
    a_down, b_down, c_down = popt
    e_up = b_up / a_up
    z_transition = y_0_up + z_transition_margin

    bubble_shape_closed = _mirror_and_close(bubble_shape, mean_x)
    inner_region_closed = _mirror_and_close(inner_region, mean_x)
    inner_path = MplPath(inner_region_closed)

    height, width = plif_image.shape[0], plif_image.shape[1]
    # NOTE: np.meshgrid(arange(H), arange(W)) with default 'xy' indexing returns
    # arrays of shape (W, H): xsi_grid[i, j] == j (a height/row index) and
    # eta_grid[i, j] == i (a width/column index), so (eta_grid[i,j], xsi_grid[i,j])
    # correctly gives (x, y) = (column, row) pixel coordinates for Path.contains_point.
    xsi_grid, eta_grid = np.meshgrid(np.arange(0, height, 1), np.arange(0, width, 1))
    physical_x = eta_grid.astype(float)
    physical_z = xsi_grid.astype(float)

    for ii in range(physical_x.shape[0]):
        for jj in range(physical_x.shape[1]):
            point = (eta_grid[ii, jj], xsi_grid[ii, jj])
            if inner_path.contains_point(point):
                camera_pos = _ray_trace_camera_point(
                    point, e_up, a_up, y_0_up, a_down, b_down, c_down, mu_oil_gas, mean_x, e_off, z_transition
                )
                physical_x[ii, jj] = camera_pos[0] + mean_x
                physical_z[ii, jj] = camera_pos[2]
            else:
                physical_x[ii, jj] = -10.0
                physical_z[ii, jj] = -10.0

    if plot:
        ellipse_up_points = generate_ellipse_points_shifted(a_up, b_up, y_0_up)
        exp_curve = _exp_value(curve_down[:, 1], a_down, b_down, c_down, y_0_up)
        _plot_mapping(
            plif_image, bubble_shape_closed, inner_region_closed,
            ellipse_up_points, exp_curve, curve_down[:, 1], mean_x, physical_x, physical_z,
        )

    return CameraMappingResult(
        physical_x=physical_x,
        physical_z=physical_z,
        bubble_shape_closed=bubble_shape_closed,
        fit_params={
            "a_up": a_up, "b_up": b_up, "y_0_up": y_0_up,
            "a_down": a_down, "b_down": b_down, "c_down": c_down,
            "z_transition": z_transition,
        },
    )


def _parse_args(argv):
    import argparse

    parser = argparse.ArgumentParser(description="Build the camera<->physical plane mapping for one frame.")
    parser.add_argument("--plif-image", required=True)
    parser.add_argument("--bubble-contour", required=True, help="Path to this frame's real_shape/bubble_contour txt.")
    parser.add_argument("--inner-region", required=True, help="Path to this frame's inner_region contour txt.")
    parser.add_argument("--mean-x", type=float, required=True, help="BubbleShapeResult.mean_r for this frame.")
    parser.add_argument("--mu-oil-gas", type=float, default=1.405)
    parser.add_argument("--e-off", type=float, default=1.0)
    parser.add_argument("--z-transition-margin", type=float, default=45.0)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--plot", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> CameraMappingResult:
    import cv2

    args = _parse_args(argv)

    plif_image = cv2.imread(args.plif_image)
    if plif_image is None:
        raise FileNotFoundError(args.plif_image)

    result = build_camera_plane_mapping(
        plif_image,
        load_contour(args.bubble_contour),
        load_contour(args.inner_region),
        mean_x=args.mean_x,
        mu_oil_gas=args.mu_oil_gas,
        e_off=args.e_off,
        z_transition_margin=args.z_transition_margin,
        plot=args.plot,
    )

    if args.out_dir:
        out_dir = FilePath(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        save_contour(result.physical_x, out_dir / "x_mat.txt")
        save_contour(result.physical_z, out_dir / "y_mat.txt")
        save_contour(result.bubble_shape_closed, out_dir / "bubble_shape_tot.txt")
        print(f"Saved outputs to {out_dir}")

    return result


if __name__ == "__main__":
    main()
