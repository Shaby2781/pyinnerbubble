"""Graded computational mesh on the bubble region (paper §IIIC).

Given a conformal map (unit disk -> bubble region), build a tanh-graded polar mesh
on the disk and map it into the bubble. The disk is meshed over a *half* disk
(theta in [0, pi]) which maps to the right half of the (axisymmetric) bubble -- the
domain the interfacial Navier-Stokes solve runs on.

The grid is clustered:
  * radially toward the interface (r = 1), and
  * angularly toward the two ends of the boundary,
so nodes are dense where the interfacial velocity gradients are steepest.

Ported from the research script ``create_mesh.py`` / ``create_mesh.m``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ..plotting import matlab_styled

from .conformal_mapping import ConformalMapResult, load_conformal_map


@dataclass
class BubbleMeshResult:
    """Output of :func:`build_bubble_mesh`.

    ``x_mat``/``y_mat`` are the physical mesh-node coordinates (bubble region), with
    ``x_mat`` the horizontal (radial) direction and ``y_mat`` the rise direction --
    matching the MATLAB convention ``Y_MAT = real(w)``, ``X_MAT = imag(w)``. All
    arrays are shaped ``(2*n_theta_half, n_radial)`` = (angular nodes, radial nodes).
    """

    x_mat: np.ndarray       # physical horizontal coordinate = imag(w_mat)
    y_mat: np.ndarray       # physical rise coordinate       = real(w_mat)
    z_mat: np.ndarray       # complex computational grid on the unit half-disk (R e^{i theta})
    w_mat: np.ndarray       # complex mapped grid in the bubble region = G(z_mat)
    r_vec: np.ndarray       # radial node positions in [0, 1] (graded toward 1)
    theta_vec: np.ndarray   # angular node positions in [0, pi] (graded toward the ends)


# ---------------------------------------------------------------------------
# Graded polar mesh on the unit half-disk
# ---------------------------------------------------------------------------
def build_disk_mesh(n_theta_half: int = 31, n_radial: int = 25,
                    beta_theta: float = 3.0, beta_radial: float = 4.0,
                    alpha: float = 0.615, n_smooth: int = 20):
    """Build the tanh-graded polar grid on the unit half-disk (theta in [0, pi]).

    Parameters
    ----------
    n_theta_half : int
        Half the number of angular nodes; the total is ``2 * n_theta_half`` (the
        distribution is built on [0, alpha*pi] and mirrored onto [alpha*pi, pi]).
    n_radial : int
        Number of radial nodes from the disk center (r=0) to the boundary (r=1).
    beta_theta : float
        Angular tanh-clustering strength -- larger packs nodes more tightly toward
        the two ends of the boundary.
    beta_radial : float
        Radial tanh-clustering strength -- larger packs nodes more tightly toward the
        interface r = 1.
    alpha : float
        Angular split fraction: the two graded halves meet at ``alpha * pi``.
    n_smooth : int
        Number of Laplacian smoothing passes applied to the angular node positions.

    Returns
    -------
    (z_mat, r_vec, theta_vec)
        ``z_mat`` is the complex grid ``R * exp(1j*theta)`` shaped
        ``(2*n_theta_half, n_radial)``; ``r_vec`` and ``theta_vec`` are the 1-D node
        coordinate vectors.
    """
    # --- angular grid: tanh clustering, mirrored about alpha*pi ---
    s = np.linspace(0.0, 1.0, n_theta_half)
    x = alpha * np.pi * (1.0 - np.tanh(beta_theta * (1.0 - s)) / np.tanh(beta_theta))
    y = (1.0 - alpha) * np.pi * (1.0 - np.tanh(beta_theta * (1.0 - s)) / np.tanh(beta_theta))
    theta_vec = np.concatenate([alpha * np.pi - x[::-1], y + alpha * np.pi])

    # light Laplacian smoothing of the interior nodes (simultaneous update)
    for _ in range(n_smooth):
        theta_vec[1:-1] = (0.25 * theta_vec[:-2]
                           + 0.5 * theta_vec[1:-1]
                           + 0.25 * theta_vec[2:])

    # --- radial grid: tanh clustering toward the boundary r = 1 ---
    s = np.linspace(0.0, 1.0, n_radial)
    r_vec = 1.0 - np.tanh(beta_radial * (1.0 - s)) / np.tanh(beta_radial)
    r_vec = 1.0 - r_vec[::-1]

    # --- polar grid (MATLAB meshgrid(r_vec, theta_vec) -> shape (n_theta, n_r)) ---
    r_grid, theta_grid = np.meshgrid(r_vec, theta_vec)   # 'xy' indexing
    z_mat = r_grid * np.cos(theta_grid) + 1j * r_grid * np.sin(theta_grid)
    return z_mat, r_vec, theta_vec


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def build_bubble_mesh(conformal_map: ConformalMapResult,
                      n_theta_half: int = 31, n_radial: int = 25,
                      beta_theta: float = 3.0, beta_radial: float = 4.0,
                      alpha: float = 0.615, n_smooth: int = 20) -> BubbleMeshResult:
    """Build a graded computational mesh on the bubble region from a conformal map.

    Meshes the unit half-disk (see :func:`build_disk_mesh` for the grid parameters)
    and maps it through ``conformal_map.G`` into the bubble region.

    Parameters
    ----------
    conformal_map : ConformalMapResult
        A map from :func:`~pyinnerbubble.assimilation.build_conformal_map` (or
        :func:`~pyinnerbubble.assimilation.load_conformal_map`).
    n_theta_half, n_radial, beta_theta, beta_radial, alpha, n_smooth
        Grid size and clustering parameters -- see :func:`build_disk_mesh`.

    Returns
    -------
    BubbleMeshResult
    """
    z_mat, r_vec, theta_vec = build_disk_mesh(
        n_theta_half=n_theta_half, n_radial=n_radial,
        beta_theta=beta_theta, beta_radial=beta_radial,
        alpha=alpha, n_smooth=n_smooth,
    )

    # Map disk -> bubble. Flatten/reshape so this works whether G is the scipy-AAA
    # callable (freshly built) or the barycentric callable (loaded from .npz).
    w_mat = np.asarray(conformal_map.G(z_mat.ravel())).reshape(z_mat.shape)

    return BubbleMeshResult(
        x_mat=np.imag(w_mat), y_mat=np.real(w_mat),
        z_mat=z_mat, w_mat=w_mat, r_vec=r_vec, theta_vec=theta_vec,
    )


# ---------------------------------------------------------------------------
# Serialize / restore
# ---------------------------------------------------------------------------
def save_mesh(path: str | Path, result: BubbleMeshResult) -> None:
    """Write a bubble mesh to a ``.npz`` file."""
    np.savez(
        str(path),
        x_mat=result.x_mat, y_mat=result.y_mat,
        z_mat=result.z_mat, w_mat=result.w_mat,
        r_vec=result.r_vec, theta_vec=result.theta_vec,
    )


def load_mesh(path: str | Path) -> BubbleMeshResult:
    """Load a bubble mesh saved by :func:`save_mesh`."""
    d = np.load(str(path))
    return BubbleMeshResult(
        x_mat=d["x_mat"], y_mat=d["y_mat"],
        z_mat=d["z_mat"], w_mat=d["w_mat"],
        r_vec=d["r_vec"], theta_vec=d["theta_vec"],
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
@matlab_styled
def plot_mesh(result: BubbleMeshResult, save_path: str | Path | None = None) -> None:
    """Plot the computational grid (unit half-disk) and the physical grid (bubble)."""
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    ax1.plot(result.z_mat.real, result.z_mat.imag, "x", ms=3)
    ax1.set_aspect("equal"); #ax1.set_title("Computational grid (unit disk)")
    ax1.set_xlabel("Re(z)"); ax1.set_ylabel("Im(z)")

    ax2.plot(result.x_mat, result.y_mat, "x", ms=3)
    ax2.set_aspect("equal"); #ax2.set_title("Physical grid (bubble region)")
    ax2.set_xlabel("Im(w)"); ax2.set_ylabel("Re(w)")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(str(save_path), dpi=150)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a graded bubble mesh from a saved conformal map (.npz)."
    )
    parser.add_argument("--map", required=True, help="Conformal map .npz (from save_conformal_map).")
    parser.add_argument("--n-theta-half", type=int, default=31, help="Half the angular nodes. Default 31.")
    parser.add_argument("--n-radial", type=int, default=25, help="Radial nodes. Default 25.")
    parser.add_argument("--beta-theta", type=float, default=3.0, help="Angular clustering. Default 3.0.")
    parser.add_argument("--beta-radial", type=float, default=4.0, help="Radial clustering (toward r=1). Default 4.0.")
    parser.add_argument("--alpha", type=float, default=0.615, help="Angular split fraction (alpha*pi). Default 0.615.")
    parser.add_argument("--n-smooth", type=int, default=20, help="Angular smoothing passes. Default 20.")
    parser.add_argument("--out", help="Where to write the mesh .npz file.")
    parser.add_argument("--plot", action="store_true", help="Show the disk + bubble grid plot.")
    args = parser.parse_args()

    conformal_map = load_conformal_map(args.map)
    mesh = build_bubble_mesh(
        conformal_map,
        n_theta_half=args.n_theta_half, n_radial=args.n_radial,
        beta_theta=args.beta_theta, beta_radial=args.beta_radial,
        alpha=args.alpha, n_smooth=args.n_smooth,
    )
    print(f"[mesh] grid shape {mesh.z_mat.shape} "
          f"(n_theta={mesh.theta_vec.size}, n_r={mesh.r_vec.size})")
    if args.out:
        save_mesh(args.out, mesh)
        print(f"[save] mesh written to {args.out}")
    if args.plot:
        plot_mesh(mesh)


if __name__ == "__main__":
    main()
