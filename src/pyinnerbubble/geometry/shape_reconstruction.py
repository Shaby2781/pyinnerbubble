"""Bubble shape reconstruction from shadowgraph + PLIF contours (paper Sec. III A).

Combines the shadowgraph-derived cap/rim contour (fit with an ellipse and an
exponential curve) and the PLIF-derived base contour, and projects them
through Snell's law to reconstruct the physical (r, z) bubble profile,
``real_shape``, for a single frame. Looping over a frame sequence, and
tracking the bubble position across frames, is left to the caller.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from scipy.optimize import curve_fit, fsolve
from scipy.signal import savgol_filter
from shapely.geometry import LineString, Point

from pyinnerbubble.io import load_contour, save_contour

_MATLAB_BLUE = (0, 0.4470, 0.7410)
_MATLAB_ORANGE = (0.8500, 0.3250, 0.0980)
_MATLAB_GREEN = (0.4660, 0.6740, 0.1880)


@dataclass
class CropParams:
    """Margins (in pixels) used to crop the rotated images around ``real_shape``.

    Set ``crop_size`` to the ``crop_size`` returned for a previous frame to
    force the same crop width/height across a sequence, matching only
    ``x_min_margin``/``y_min_margin`` to that frame's bubble position.
    """

    x_min_margin: float = 50
    y_min_margin: float = 10
    y_bottom_margin: float = 200
    x_width_factor: float = 2.0
    x_width_padding: float = 50
    crop_size: Optional[tuple[float, float]] = None


@dataclass
class BubbleShapeResult:
    real_shape: np.ndarray
    inner_region: np.ndarray
    rot_plif_contour: np.ndarray
    plif_image_cropped: np.ndarray
    shadowgraph_image_cropped: np.ndarray
    crop_size: tuple[float, float]
    mean_r: float
    """Contour centroid r, in the same cropped pixel frame as ``real_shape``/``inner_region``."""
    mean_z: float
    """Contour centroid z, in the same cropped pixel frame as ``real_shape``/``inner_region``."""
    shadow_leading_edge_y: float
    fit_params: dict


def rotate_curve(points: np.ndarray, angle: float) -> tuple[float, float, np.ndarray]:
    """Rotate a curve about its bounding-box center by ``angle`` radians."""
    points = np.asarray(points, dtype=float).copy()
    rot_points = np.zeros_like(points)
    mean_r = (points[:, 0].max() + points[:, 0].min()) / 2
    mean_z = (points[:, 1].max() + points[:, 1].min()) / 2
    points[:, 0] -= mean_r
    points[:, 1] -= mean_z
    rot_points[:, 0] = points[:, 0] * np.cos(angle) + points[:, 1] * np.sin(angle)
    rot_points[:, 1] = -points[:, 0] * np.sin(angle) + points[:, 1] * np.cos(angle)
    rot_points[:, 0] += mean_r
    rot_points[:, 1] += mean_z
    return mean_r, mean_z, rot_points


def find_intersections_shapely(curve1: np.ndarray, curve2: np.ndarray):
    """Find intersection points between two polylines, and their segment indices."""
    l1 = LineString(curve1)
    l2 = LineString(curve2)
    intersection = l1.intersection(l2)

    if intersection.is_empty:
        return [], [], []
    if isinstance(intersection, Point):
        points = [intersection]
    else:
        points = [p for p in intersection.geoms if isinstance(p, Point)]

    results, indices1, indices2 = [], [], []
    for pt in points:
        results.append((pt.x, pt.y))
        dist1 = l1.project(pt)
        indices1.append(np.where(np.array([l1.project(Point(p)) for p in curve1]) <= dist1)[0][-1])
        dist2 = l2.project(pt)
        indices2.append(np.where(np.array([l2.project(Point(p)) for p in curve2]) <= dist2)[0][-1])

    return np.array(results), np.array(indices1), np.array(indices2)


def fit_cap_ellipse(curve_up: np.ndarray) -> tuple[float, float, float]:
    """Fit the cap as an axis-aligned ellipse directly from its (x, y) extents.

    Returns (a_up, b_up, y_0_up): the x semi-axis, y semi-axis, and y-center.
    """
    a_up = -np.min(curve_up[:, 0])
    y_0_up = np.max(curve_up[:, 1])
    b_up = y_0_up - np.min(curve_up[:, 1])
    return a_up, b_up, y_0_up


def generate_ellipse_points_shifted(a: float, b: float, y_0: float, num_points: int = 100) -> np.ndarray:
    """Generate points on an ellipse of semi-axes (a, b) centered at (0, y_0)."""
    t = np.linspace(0, 2 * np.pi, num_points)
    return np.column_stack((a * np.cos(t), b * np.sin(t) + y_0))


def _exp_value(z, a, b, c, z_ref):
    return a * np.exp(b * (z - z_ref)) + c


def _exp_derivative(z, a, b, c, z_ref):
    return a * b * np.exp(b * (z - z_ref))


def _ellipsoid_line_intersection(t, point, direction, a, b, z0):
    line_point = point + t * direction
    return 1 - (line_point[0] ** 2 / a ** 2 + line_point[1] ** 2 / a ** 2 + (line_point[2] - z0) ** 2 / b ** 2)


def _find_intersection(point, direction, a, b, z0):
    t_intersection = fsolve(_ellipsoid_line_intersection, a, args=(point, direction, a, b, z0))
    return point + t_intersection * direction


def _exp_line_intersection(t, point, direction, a_exp, b_exp, c_exp, z_ref):
    line_point = point + t * direction
    r_val = np.abs(_exp_value(line_point[2], a_exp, b_exp, c_exp, z_ref))
    r = np.sqrt(line_point[0] ** 2 + line_point[1] ** 2)
    return r_val - r


def _find_intersection_exp(point, direction, a_exp, b_exp, c_exp, a_up, z_ref):
    t_intersection = fsolve(
        _exp_line_intersection, a_up, args=(point, direction, a_exp, b_exp, c_exp, z_ref)
    )
    return point + t_intersection * direction


def _find_norm(point, r, e, e_off):
    normal = np.array((2 * point[0] / r ** 2, 2 * point[1] / (r * e_off) ** 2, 2 * point[2] / (e * r) ** 2))
    return normal / np.linalg.norm(normal)


def _cost_direction(direction, point, a, b, z0, mu_oil_gas):
    mu_air_oil = 1 / mu_oil_gas
    direction = np.hstack((direction[0], np.sqrt(1 - direction[0] ** 2 - direction[1] ** 2), direction[1]))
    point_intersection = _find_intersection(point, direction, a, b, z0)
    normal = np.hstack((
        point_intersection[0] / (a ** 2),
        point_intersection[1] / (a ** 2),
        (point_intersection[2] - z0) / (b ** 2),
    ))
    normal = normal / np.linalg.norm(normal)
    c = np.dot(direction, normal)
    disc = 1 - mu_air_oil ** 2 * (1 - c ** 2)
    if disc < 0:
        return np.array((10000.0, 10000.0))
    amp = np.sqrt(disc) - mu_air_oil * c
    t = mu_air_oil * direction + amp * normal
    return np.hstack((t[0], t[2]))


def _cost_direction_exp(direction, point, a_exp, b_exp, c_exp, a_up, z_ref, mu_oil_gas):
    mu_air_oil = 1 / mu_oil_gas
    direction = np.hstack((direction[0], np.sqrt(1 - direction[0] ** 2 - direction[1] ** 2), direction[1]))
    point_intersection = _find_intersection_exp(point, direction, a_exp, b_exp, c_exp, a_up, z_ref)
    tet = np.arctan2(point_intersection[1], point_intersection[0])
    p_tag = _exp_derivative(point_intersection[2], a_exp, b_exp, c_exp, z_ref)
    n_r = 1 / np.sqrt(1 + p_tag ** 2)
    n_z = p_tag / np.sqrt(1 + p_tag ** 2)
    normal = np.hstack((n_r * np.cos(tet), n_r * np.sin(tet), n_z))
    normal = normal / np.linalg.norm(normal)
    c = np.dot(direction, normal)
    disc = 1 - mu_air_oil ** 2 * (1 - c ** 2)
    if disc < 0:
        return np.array((10000.0, 10000.0))
    amp = np.sqrt(disc) - mu_air_oil * c
    t = mu_air_oil * direction + amp * normal
    return np.hstack((t[0], t[2]))


def _find_direction(point, a, b, z0, mean_z, mu_oil_gas):
    point_move = np.hstack((point[0], 0, point[2] - mean_z))
    rad = np.linalg.norm(point_move)
    direction0 = np.hstack((-1 / np.sqrt(5) * point_move[0] / rad, 1, -1 / np.sqrt(5) * point_move[2] / rad))
    direction0 = direction0 / np.linalg.norm(direction0)
    direction = fsolve(_cost_direction, np.hstack((direction0[0], direction0[2])), args=(point, a, b, z0, mu_oil_gas))
    return np.hstack((direction[0], np.sqrt(1 - direction[0] ** 2 - direction[1] ** 2), direction[1]))


def _find_direction_exp(point, a_exp, b_exp, c_exp, a_up, mean_z, z_ref, mu_oil_gas):
    point_move = np.hstack((point[0], 0, point[2] - mean_z))
    rad = np.linalg.norm(point_move)
    direction0 = np.hstack((-1 / np.sqrt(2) * point_move[0] / rad, 1, 0))
    direction0 = direction0 / np.linalg.norm(direction0)
    direction = fsolve(
        _cost_direction_exp,
        np.hstack((direction0[0], direction0[2])),
        args=(point, a_exp, b_exp, c_exp, a_up, z_ref, mu_oil_gas),
    )
    return np.hstack((direction[0], np.sqrt(1 - direction[0] ** 2 - direction[1] ** 2), direction[1]))


def _calc_edge_position_one_ellipse_exp(
    points_camera, e_up, r_up, z0_up, a_exp, b_exp, c_exp, mu_oil_gas, e_off, z_ref
):
    z_vec = points_camera[:, 1]
    real_shape = np.zeros((z_vec.shape[0], 3))
    camera_vec = np.array((0, -1, 0))

    for ii, z_pos in enumerate(z_vec):
        x_pos = points_camera[ii, 0]
        if z_pos < z0_up:
            theta = np.arccos(np.sqrt(((z_pos - z0_up) / e_up / r_up) ** 2 + (x_pos / r_up) ** 2))
            y_pos = r_up * e_off * np.sin(theta)
            pos = np.array((x_pos, y_pos, z_pos - z0_up))
            norm_vec = _find_norm(pos, r_up, e_up, e_off)
        else:
            r_val = np.abs(_exp_value(z_pos, a_exp, b_exp, c_exp, z_ref))
            y_pos = np.sqrt(r_val ** 2 - x_pos ** 2)
            pos = np.array((x_pos, y_pos, z_pos))
            tet = np.arctan2(y_pos, x_pos)
            p_tag = _exp_derivative(z_pos, a_exp, b_exp, c_exp, z_ref)
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

        if z_pos < z0_up:
            if camera_pos[2] < z0_up:
                camera_pos[2] += z0_up
                real_shape[ii, :] = camera_pos
        else:
            real_shape[ii, :] = camera_pos

    return real_shape


def _plot_reconstruction(plif_image, shadowgraph_image, inner_region, real_shape, rot_plif_contour, save_dir=None):
    import matplotlib.pyplot as plt

    names = ("plif_overlay.png", "shadowgraph_overlay.png")
    for image, name in zip((plif_image, shadowgraph_image), names):
        plt.figure()
        plt.imshow(image, cmap="gray")
        plt.plot(inner_region[:, 0], inner_region[:, 1], color=_MATLAB_GREEN)
        plt.plot(real_shape[:, 0], real_shape[:, 1], color=_MATLAB_BLUE)
        plt.plot(rot_plif_contour[:, 0], rot_plif_contour[:, 1], color=_MATLAB_ORANGE, linestyle="--")
        plt.axis("off")
        if save_dir is not None:
            plt.savefig(Path(save_dir) / name)
            plt.close()
    if save_dir is None:
        plt.show()


def reconstruct_bubble_shape(
    shadowgraph_image: np.ndarray,
    plif_image: np.ndarray,
    shadowgraph_contour: np.ndarray,
    plif_contour: np.ndarray,
    velocity_px_per_frame: float,
    angle_to_rotate: float = 0.0,
    mu_oil_gas: float = 1.405,
    crop: CropParams | None = None,
    plot: bool | str | Path = False,
) -> BubbleShapeResult:
    """Reconstruct the physical bubble profile for a single frame.

    Parameters
    ----------
    shadowgraph_image, plif_image : raw (BGR) images for this frame.
    shadowgraph_contour, plif_contour : (N, 2) contours for this frame, as produced
        by :mod:`pyinnerbubble.geometry.segmentation`.
    velocity_px_per_frame : bubble rise speed in pixels/frame, used to correct for
        the sub-frame time offset between the paired shadowgraph and PLIF exposures.
    crop : cropping margins for the returned images; pass a previous frame's
        ``crop_size`` in to keep a fixed crop window across a sequence.
    plot : if True, display the reconstructed profile overlaid on both images.
        Pass a directory path instead to save the figures there rather than
        showing them -- required when calling this from a worker process (e.g.
        via ``ProcessPoolExecutor``), since interactive display doesn't work
        across process boundaries.
    """
    if crop is None:
        crop = CropParams()

    mean_r, _, rot_shadowgraph_contour = rotate_curve(shadowgraph_contour, angle_to_rotate)
    mean_r, mean_z, rot_plif_contour = rotate_curve(plif_contour, angle_to_rotate)

    h, w = shadowgraph_image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((mean_r, mean_z), angle_to_rotate * 180 / np.pi, 1.0)
    shadowgraph_rotated_img = cv2.warpAffine(
        shadowgraph_image, rotation_matrix, (w, h),
        borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255),
    )
    plif_rotated_img = cv2.warpAffine(
        plif_image, rotation_matrix, (w, h),
        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
    )
    gray_plif_image = cv2.cvtColor(plif_rotated_img, cv2.COLOR_BGR2GRAY)

    rot_shadowgraph_contour_org = rot_shadowgraph_contour.copy()
    rot_shadowgraph_contour[:, 1] -= velocity_px_per_frame / 2

    index_r_max = np.argmax(rot_shadowgraph_contour[:, 0])
    index_r_min = np.argmin(rot_shadowgraph_contour[:, 0])
    shadow_graph_curve1 = rot_shadowgraph_contour[:index_r_min, :].copy()
    shadow_graph_curve_base = rot_shadowgraph_contour[index_r_min:index_r_max, :]
    shadow_graph_curve_base_side1 = shadow_graph_curve_base[shadow_graph_curve_base[:, 0] < mean_r]
    shadow_graph_curve_base_side1 = shadow_graph_curve_base_side1[:-20, :]
    smoothed = savgol_filter(shadow_graph_curve_base_side1, 21, 2, axis=0)

    dr_dt = np.hstack((0, np.diff(smoothed[:, 0]) / np.sqrt(np.diff(smoothed[:, 0]) ** 2 + np.diff(smoothed[:, 1]) ** 2)))
    exceed_indices = np.where(dr_dt >= 0.95)[0]
    shadow_graph_curve2 = shadow_graph_curve_base_side1[: exceed_indices[0]] if len(exceed_indices) else shadow_graph_curve_base_side1[:]
    shadow_graph_curve2 = shadow_graph_curve2.copy()

    shadow_graph_curve1[:, 0] -= mean_r
    shadow_graph_curve2[:, 0] -= mean_r
    shadow_graph_curve_base_side1 = shadow_graph_curve_base_side1.copy()
    shadow_graph_curve_base_side1[:, 0] -= mean_r

    a_up, b_up, y_0_up = fit_cap_ellipse(shadow_graph_curve1)

    z_min_down = np.min(shadow_graph_curve2[:, 1])
    popt, _ = curve_fit(
        partial(_exp_value, z_ref=z_min_down),
        shadow_graph_curve2[:, 1], shadow_graph_curve2[:, 0], p0=[10, 0.01, -500],
        maxfev=10000,
    )
    a_down, b_down, c_down = popt

    plif_circle_points = np.zeros((shadow_graph_curve1.shape[0] + shadow_graph_curve_base_side1.shape[0], 3))
    for ii, point in enumerate(shadow_graph_curve1):
        point3 = np.hstack((point[0], 0, point[1]))
        direction = _find_direction(point3, a_up, b_up, y_0_up, mean_z, mu_oil_gas)
        plif_circle_points[ii, :] = _find_intersection(point3, direction, a_up, b_up, y_0_up)
    for ii, point in enumerate(shadow_graph_curve_base_side1):
        point3 = np.hstack((point[0], 0, point[1]))
        direction = _find_direction_exp(point3, a_down, b_down, c_down, a_up, mean_z, z_min_down, mu_oil_gas)
        kk = shadow_graph_curve1.shape[0] + ii
        plif_circle_points[kk, :] = _find_intersection_exp(point3, direction, a_down, b_down, c_down, a_up, z_min_down)

    plif_circle = np.column_stack((plif_circle_points[:, 0] + mean_r, plif_circle_points[:, 2]))

    rot_plif_contour = rot_plif_contour.copy()
    rot_plif_contour[:, 0] -= mean_r
    if rot_plif_contour[0, 0] < 0.0:
        rot_plif_contour = rot_plif_contour[rot_plif_contour[:, 0] > 0]
        rot_plif_contour[:, 0] *= -1
        rot_plif_contour = rot_plif_contour[::-1, :]
    else:
        rot_plif_contour = rot_plif_contour[rot_plif_contour[:, 0] < 0]
    rot_plif_contour[:, 0] += mean_r

    _, id_plif_circle, id_contour = find_intersections_shapely(plif_circle, rot_plif_contour)
    inner_region = np.vstack((
        plif_circle[: id_plif_circle[0], :],
        rot_plif_contour[id_contour[0] + 1:, :],
    ))
    inner_region_move = inner_region.copy()
    inner_region_move[:, 0] -= mean_r

    ar_b_up = b_up / a_up
    real_shape = _calc_edge_position_one_ellipse_exp(
        inner_region_move, ar_b_up, a_up, y_0_up, a_down, b_down, c_down, mu_oil_gas, 1.0, z_min_down
    )
    real_shape[:, 0] += mean_r
    real_shape = np.column_stack((real_shape[:, 0], real_shape[:, 2]))

    shadow_leading_edge_y = float(np.min(rot_shadowgraph_contour_org[:, 1]))

    y_min_lim = int(np.floor(np.min(real_shape[:, 1]))) - crop.y_min_margin
    x_min_lim = int(np.floor(np.min(real_shape[:, 0]) - crop.x_min_margin))
    if crop.crop_size is None:
        y_max_lim = int(np.floor(np.max(real_shape[:, 1])))
        x_max_lim = int(np.floor(np.max(real_shape[:, 0])))
        delta_y_frame = y_max_lim - y_min_lim
        delta_x_frame = crop.x_width_factor * (x_max_lim - x_min_lim) + crop.x_width_padding
    else:
        delta_x_frame, delta_y_frame = crop.crop_size
    x_min_lim, y_min_lim = int(x_min_lim), int(y_min_lim)
    x_max_lim, y_max_lim = int(x_min_lim + delta_x_frame), int(y_min_lim + delta_y_frame)

    real_shape[:, 0] -= x_min_lim
    real_shape[:, 1] -= y_min_lim
    move_vel = int(np.round(velocity_px_per_frame / 2))

    plif_image_cropped = gray_plif_image[y_min_lim:y_max_lim + int(crop.y_bottom_margin), x_min_lim:x_max_lim]
    shadowgraph_image_cropped = shadowgraph_rotated_img[
        y_min_lim + move_vel:y_max_lim + move_vel + int(crop.y_bottom_margin), x_min_lim:x_max_lim
    ]

    inner_region = inner_region.copy()
    inner_region[:, 0] -= x_min_lim
    inner_region[:, 1] -= y_min_lim
    rot_plif_contour[:, 0] -= x_min_lim
    rot_plif_contour[:, 1] -= y_min_lim

    if plot:
        save_dir = plot if isinstance(plot, (str, Path)) else None
        _plot_reconstruction(
            plif_image_cropped, shadowgraph_image_cropped, inner_region, real_shape, rot_plif_contour,
            save_dir=save_dir,
        )

    return BubbleShapeResult(
        real_shape=real_shape,
        inner_region=inner_region,
        rot_plif_contour=rot_plif_contour,
        plif_image_cropped=plif_image_cropped,
        shadowgraph_image_cropped=shadowgraph_image_cropped,
        crop_size=(delta_x_frame, delta_y_frame),
        mean_r=float(mean_r - x_min_lim),
        mean_z=float(mean_z - y_min_lim),
        shadow_leading_edge_y=shadow_leading_edge_y,
        fit_params={
            "a_up": a_up, "b_up": b_up, "y_0_up": y_0_up,
            "a_down": a_down, "b_down": b_down, "c_down": c_down,
            "z_min_down": z_min_down,
        },
    )


def _parse_args(argv):
    import argparse

    parser = argparse.ArgumentParser(description="Reconstruct a bubble profile for a single frame.")
    parser.add_argument("--shadowgraph-image", required=True)
    parser.add_argument("--plif-image", required=True)
    parser.add_argument("--shadowgraph-contour", required=True)
    parser.add_argument("--plif-contour", required=True)
    parser.add_argument("--velocity", type=float, required=True, help="Bubble rise speed, px/frame.")
    parser.add_argument("--angle-deg", type=float, default=-1.0)
    parser.add_argument("--mu-oil-gas", type=float, default=1.405)
    parser.add_argument("--x-min-margin", type=float, default=50)
    parser.add_argument("--y-min-margin", type=float, default=10)
    parser.add_argument("--y-bottom-margin", type=float, default=200)
    parser.add_argument("--x-width-factor", type=float, default=2.0)
    parser.add_argument("--x-width-padding", type=float, default=50)
    parser.add_argument(
        "--crop-size", type=float, nargs=2, metavar=("WIDTH", "HEIGHT"), default=None,
        help="Fix the crop window size, e.g. to a previous frame's reported crop_size.",
    )
    parser.add_argument("--out-dir", default=None, help="If given, write outputs here.")
    parser.add_argument("--plot", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> BubbleShapeResult:
    args = _parse_args(argv)

    shadowgraph_image = cv2.imread(args.shadowgraph_image)
    plif_image = cv2.imread(args.plif_image)
    if shadowgraph_image is None:
        raise FileNotFoundError(args.shadowgraph_image)
    if plif_image is None:
        raise FileNotFoundError(args.plif_image)

    crop = CropParams(
        x_min_margin=args.x_min_margin,
        y_min_margin=args.y_min_margin,
        y_bottom_margin=args.y_bottom_margin,
        x_width_factor=args.x_width_factor,
        x_width_padding=args.x_width_padding,
        crop_size=tuple(args.crop_size) if args.crop_size else None,
    )

    result = reconstruct_bubble_shape(
        shadowgraph_image, plif_image,
        load_contour(args.shadowgraph_contour), load_contour(args.plif_contour),
        velocity_px_per_frame=args.velocity,
        angle_to_rotate=np.deg2rad(args.angle_deg),
        mu_oil_gas=args.mu_oil_gas,
        crop=crop,
        plot=args.plot,
    )

    print(f"crop_size (width, height) = {result.crop_size}")
    print(f"mean_r={result.mean_r:.2f} mean_z={result.mean_z:.2f} shadow_leading_edge_y={result.shadow_leading_edge_y:.2f}")

    if args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        save_contour(result.real_shape, out_dir / "bubble_contour.txt")
        save_contour(result.inner_region, out_dir / "inner_region_contour.txt")
        cv2.imwrite(str(out_dir / "cut_plif.jpg"), result.plif_image_cropped)
        cv2.imwrite(str(out_dir / "cut_shadowgraph.jpg"), result.shadowgraph_image_cropped)
        print(f"Saved outputs to {out_dir}")

    return result


if __name__ == "__main__":
    main()
