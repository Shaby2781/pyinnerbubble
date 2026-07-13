"""Velocity eduction from PLIF image pairs via Farneback optic flow (paper Sec. III B).

Parameter names below match this project's original MATLAB configuration
(``opticalFlowFarneback``) rather than OpenCV's, since that's what was tuned
against the synthetic HR-solution validation described in the paper:

    MATLAB (opticalFlowFarneback)   OpenCV (calcOpticalFlowFarneback)
    NumPyramidLevels                levels
    PyramidScale                    pyr_scale
    NumIterations                   iterations
    NeighborhoodSize                poly_n
    FilterSize                      winsize (the paper's "15 pixel square window")

MATLAB's object doesn't expose a sigma-equivalent parameter, so ``poly_sigma``
here has no direct MATLAB counterpart -- the default below is OpenCV's own
suggested value for small poly_n and should be checked against your validation
data rather than trusted blindly.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from ..plotting import matlab_styled


@dataclass
class FarnebackParams:
    num_pyramid_levels: int = 3
    pyramid_scale: float = 0.5
    num_iterations: int = 3
    neighborhood_size: int = 1
    filter_size: int = 15
    poly_sigma: float = 1.1
    flags: int = 0


def compute_optic_flow(
    prev_image: np.ndarray,
    next_image: np.ndarray,
    params: FarnebackParams | None = None,
) -> np.ndarray:
    """Dense optic flow between two PLIF images via the Farneback algorithm.

    Parameters
    ----------
    prev_image, next_image : consecutive PLIF frames, grayscale or BGR (converted
        to grayscale automatically), same size.

    Returns
    -------
    flow : (H, W, 2) array; flow[..., 0] is u_x, flow[..., 1] is u_y, both in
        pixels/frame, in the camera plane (no ray-traced mapping applied).
    """
    if params is None:
        params = FarnebackParams()

    prev_gray = _to_gray(prev_image)
    next_gray = _to_gray(next_image)

    return cv2.calcOpticalFlowFarneback(
        prev_gray, next_gray, None,
        params.pyramid_scale, params.num_pyramid_levels, params.filter_size,
        params.num_iterations, params.neighborhood_size, params.poly_sigma, params.flags,
    )


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


@matlab_styled
def plot_flow_quiver(image: np.ndarray, flow: np.ndarray, step: int = 40, save_path=None) -> None:
    """Overlay a quiver plot of the flow field on the image, sampled every ``step`` pixels.

    Arrow lengths are auto-scaled by matplotlib for legibility -- they do not
    represent flow magnitude 1:1 in pixels. Use u_x/u_y (or the returned flow
    array from ``compute_optic_flow``) directly for anything quantitative.
    """
    import matplotlib.pyplot as plt

    h, w = flow.shape[:2]
    y, x = np.mgrid[0:h:step, 0:w:step]
    u = flow[::step, ::step, 0]
    v = flow[::step, ::step, 1]

    plt.figure(figsize=(10, 10))
    plt.imshow(image, cmap="gray")
    plt.quiver(x, y, u, v, color="red", angles="xy")
    plt.axis("off")
    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
