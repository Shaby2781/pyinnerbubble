"""Conformal map between the unit disk and a measured bubble shape (paper §IIIC).

The data-assimilation step works in a body-fitted coordinate system: the physical
bubble region is mapped to the unit disk by a conformal map, so a polar grid on the
disk becomes a boundary-following (n_hat, s_hat) grid on the bubble. This module
builds that map from a bubble boundary previously reconstructed by the geometry
stage (e.g. ``bubble_shape_tot.txt`` from ``build_camera_plane_mapping``).

Pipeline (ported from the research script ``creates_complex_bean_shape.py`` /
``creats_complex_bean_shape.m``):
  1. Normalize the pixel-coordinate boundary by a reference diameter and re-orient
     it (``_preprocess_bubble_shape``).
  2. Fit a truncated Fourier series to the ordered boundary points.
  3. Solve the Kerzman-Stein integral equation (Kerzman & Trummer 1986) for boundary
     correspondences, then AAA-rational-approximate (Gopal & Trefethen 2019) the map
     both ways. This replaces chebfun's ``conformal``.
        G      : unit disk    -> bubble region
        G_inv  : bubble region -> unit disk

The map can be serialized to ``.npz`` (its barycentric AAA data) and reloaded without
scipy via :func:`load_conformal_map`.

Requires scipy >= 1.15 for ``scipy.interpolate.AAA``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class ConformalMapResult:
    """Output of :func:`build_conformal_map`.

    G / G_inv are callables that accept complex scalars or arrays. In the complex
    plane used here the real axis is the bubble's rise (vertical) direction and the
    imaginary axis is the horizontal direction (``z = y + 1j*x``), matching the
    original MATLAB convention.
    """

    G: Callable[[np.ndarray], np.ndarray]      # unit disk    -> bubble region
    G_inv: Callable[[np.ndarray], np.ndarray]  # bubble region -> unit disk
    a: np.ndarray            # Fourier cosine coefficients of the fitted boundary (Re)
    b: np.ndarray            # Fourier sine coefficients of the fitted boundary (Im)
    rot: complex             # rotation applied so G_inv'(ctr) > 0
    ctr: complex             # interior point mapped to the disk center (0)
    z_norm: np.ndarray | None      # preprocessed (normalized) boundary points, or None if loaded
    gamma_curve: np.ndarray        # densely-sampled fitted boundary curve gamma(theta)
    raw: dict                # barycentric arrays for save_conformal_map()


# ---------------------------------------------------------------------------
# 1. Boundary normalization + re-orientation
# ---------------------------------------------------------------------------
def _preprocess_bubble_shape(bubble_shape: np.ndarray, d_ref: float) -> np.ndarray:
    """Normalize and re-orient a pixel-coordinate bubble boundary.

    Reproduces ``bubble_shape_norm = [x y] / Dref`` from the MATLAB pipeline
    (``calc_mean_and_steps_..._Farneback.m``) followed by the centering/flip/shift
    in ``creats_complex_bean_shape.m``. The whole data-assimilation pipeline works in
    these diameter-normalized coordinates (positions / d_ref), so the conformal map
    must be built in them too.

    Parameters
    ----------
    bubble_shape : (N, 2) array
        Closed bubble boundary as (x, y) pixel coordinates, e.g. the contents of
        ``bubble_shape_tot.txt`` from the geometry stage.
    d_ref : float
        Reference bubble diameter in pixels (the MATLAB ``Dref``, e.g. 1040 for EXP8).
        Typically close to the boundary's horizontal extent.

    Returns
    -------
    (N,) complex array
        The re-oriented boundary as ``z = y + 1j*x`` in normalized units.
    """
    b = np.asarray(bubble_shape, dtype=float).copy() / float(d_ref)

    # center horizontally about the mid-extent
    col0 = b[:, 0]
    b[:, 0] = col0 - (col0.min() + col0.max()) / 2.0
    # flip vertical axis, then put the top of the bubble at y = 0.25
    b[:, 1] = -b[:, 1]
    b[:, 1] = 0.25 + b[:, 1] - b[:, 1].max()
    # reverse point order (MATLAB bubble_shape(end:-1:1,:))
    b = b[::-1, :]

    return b[:, 1] + 1j * b[:, 0]


# ---------------------------------------------------------------------------
# 2. Fourier fit of the ordered boundary
# ---------------------------------------------------------------------------
def _fourier_fit(z: np.ndarray, order: int = 31) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares fit ``Re(z) ~ a0 + sum a_k cos(k*theta)`` and
    ``Im(z) ~ sum b_k sin(k*theta)``, with theta the equally-spaced index
    parameter (not arclength)."""
    n = z.size
    theta = np.linspace(0.0, 2.0 * np.pi, n + 1)[:-1]  # drop duplicate endpoint

    x_real = np.column_stack([np.ones(n)]
                             + [np.cos(k * theta) for k in range(1, order + 1)])
    x_imag = np.column_stack([np.sin(k * theta) for k in range(1, order + 1)])

    a, *_ = np.linalg.lstsq(x_real, z.real, rcond=None)   # a0..a_order
    b, *_ = np.linalg.lstsq(x_imag, z.imag, rcond=None)   # b1..b_order
    return a, b


def _make_z_curve(a: np.ndarray, b: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Evaluate the fitted boundary curve gamma(theta)."""
    order = b.size
    re = a[0] + sum(a[k] * np.cos(k * theta) for k in range(1, order + 1))
    im = sum(b[k - 1] * np.sin(k * theta) for k in range(1, order + 1))
    return re + 1j * im


def _make_z_curve_prime(a: np.ndarray, b: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Analytic derivative dgamma/dtheta of the fitted boundary curve."""
    order = b.size
    re = sum(-k * a[k] * np.sin(k * theta) for k in range(1, order + 1))
    im = sum(k * b[k - 1] * np.cos(k * theta) for k in range(1, order + 1))
    return re + 1j * im


# ---------------------------------------------------------------------------
# 3. Kerzman-Stein integral equation
# ---------------------------------------------------------------------------
def _kerzstein(a: np.ndarray, b: np.ndarray, m: int,
               ctr: complex = 0.0, n_dense: int = 8000) -> tuple[np.ndarray, np.ndarray]:
    """Return ``m`` boundary points Z (equally spaced in arclength) and their images
    W on the unit circle, for the map region -> disk that sends ``ctr`` to 0.

    The boundary is the fitted curve gamma(theta) described by (a, b)."""
    th = np.linspace(0.0, 2.0 * np.pi, n_dense, endpoint=False)
    pos = _make_z_curve(a, b, th)
    tan = _make_z_curve_prime(a, b, th)

    # Enforce counter-clockwise orientation (winding number +1 about ctr).
    pos_wrap = np.append(pos, pos[0])
    wind = np.sum(np.diff(pos_wrap) / (pos_wrap[:-1] - ctr)) / (2j * np.pi)
    if wind.real < 0:
        pos = pos[::-1].copy()
        tan = -tan[::-1].copy()

    # Cumulative arclength (curve closed by wrapping to the first point).
    pos_closed = np.append(pos, pos[0])
    seg = np.abs(np.diff(pos_closed))
    sc = np.concatenate([[0.0], np.cumsum(seg)])
    s_total = sc[-1]

    # Unit-tangent angle, unwrapped, closed with a +2*pi turn for a simple loop.
    phi = np.unwrap(np.angle(tan))
    phi_closed = np.append(phi, phi[0] + 2.0 * np.pi)

    # Sample m points equally spaced in arclength.
    svec = np.arange(m) * (s_total / m)
    d_vec = (np.interp(svec, sc, pos_closed.real)
             + 1j * np.interp(svec, sc, pos_closed.imag))
    phivec = np.interp(svec, sc, phi_closed)
    gamdot = np.exp(1j * phivec)          # unit tangents
    ds = s_total / m

    # Build and solve the Kerzman-Stein system (vectorized).
    d = 1.0 / (2j * np.pi)
    diff = d_vec[:, None] - d_vec[None, :]
    np.fill_diagonal(diff, 1.0)           # avoid /0 (diagonal overwritten below)
    term1 = np.conj(gamdot)[:, None] / np.conj(diff)
    term2 = -gamdot[None, :] / diff
    mat = -d * (term1 + term2) * ds
    np.fill_diagonal(mat, 1.0)

    gvec = d * np.conj(gamdot / (ctr - d_vec))
    fvec = np.linalg.solve(mat, gvec)

    r_prime = fvec ** 2
    r_vec = -1j * gamdot * (r_prime / np.abs(r_prime))   # points on the unit circle
    return d_vec, r_vec                                  # Z, W


# ---------------------------------------------------------------------------
# 4. Full conformal map (Kerzman-Stein + AAA)
# ---------------------------------------------------------------------------
def _conformal(a: np.ndarray, b: np.ndarray, ctr: complex = 0.0,
               m: int = 800, tol: float = 1e-6, verbose: bool = True):
    """Return ``(G, G_inv, raw)`` where G maps the unit disk to the bubble region and
    G_inv maps back, normalized so G_inv(ctr)=0 and G_inv'(ctr)>0.

    ``raw`` holds the barycentric AAA data for serialization."""
    from scipy.interpolate import AAA  # scipy >= 1.15

    z_bd, w_bd = _kerzstein(a, b, m, ctr=ctr)

    # Forward map region->disk:  f0(Z) ~ W
    f0 = AAA(z_bd, w_bd, rtol=tol)

    # Rotate so that f0'(ctr) > 0.
    scl = np.max(np.abs(z_bd - ctr))
    zz = 1e-4 * scl * np.array([1, 1j, -1, -1j])
    dwdz = np.sum(f0(ctr + zz) / zz)
    rot = np.exp(-1j * np.angle(dwdz))

    w_rot = rot * w_bd
    f = lambda z: rot * f0(np.asarray(z, dtype=complex))      # region -> disk
    finv_r = AAA(w_rot, z_bd, rtol=tol)                        # disk   -> region
    finv = lambda w: finv_r(np.asarray(w, dtype=complex))

    g, g_inv = finv, f
    if verbose:
        err = np.max(np.abs(z_bd - finv(f(z_bd))))
        print(f"[conformal] m={m}, |W| in [{np.abs(w_bd).min():.3e}, "
              f"{np.abs(w_bd).max():.3e}], back-and-forth boundary error = {err:.2e}")

    raw = dict(
        # G (disk -> region) == finv_r, no extra rotation
        G_points=np.asarray(finv_r.support_points),
        G_values=np.asarray(finv_r.support_values),
        G_weights=np.asarray(finv_r.weights),
        # G_inv (region -> disk) == rot * f0
        Ginv_points=np.asarray(f0.support_points),
        Ginv_values=np.asarray(f0.support_values),
        Ginv_weights=np.asarray(f0.weights),
        rot=np.complex128(rot), ctr=np.complex128(ctr),
        a=np.asarray(a), b=np.asarray(b),
    )
    return g, g_inv, raw


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def build_conformal_map(bubble_shape: np.ndarray, d_ref: float,
                        fourier_order: int = 31, n_boundary: int = 800,
                        center: complex = 0.0, tol: float = 1e-6,
                        verbose: bool = True) -> ConformalMapResult:
    """Build a conformal map between the unit disk and a measured bubble shape.

    Parameters
    ----------
    bubble_shape : (N, 2) array
        Closed bubble boundary as (x, y) pixel coordinates (e.g. the contents of
        ``bubble_shape_tot.txt`` produced by the geometry stage).
    d_ref : float
        Reference bubble diameter in pixels (MATLAB ``Dref``); the boundary is divided
        by this so all data-assimilation work is in diameter-normalized units. Roughly
        the boundary's horizontal extent (e.g. ~1040 px for EXP8).
    fourier_order : int
        Number of harmonics in the truncated Fourier fit of the boundary. Default 31.
    n_boundary : int
        Number of boundary correspondence points in the Kerzman-Stein solve (MATLAB
        ``M``). Default 800.
    center : complex
        Interior point (in normalized ``z = y + 1j*x`` coordinates) mapped to the disk
        center. Default 0.
    tol : float
        Relative tolerance for the AAA rational approximations. Default 1e-6.
    verbose : bool
        Print the back-and-forth boundary accuracy of the fitted map. Default True.

    Returns
    -------
    ConformalMapResult
        With callables ``G`` (disk -> bubble) and ``G_inv`` (bubble -> disk), the
        Fourier coefficients, and the barycentric data for serialization.
    """
    z_norm = _preprocess_bubble_shape(bubble_shape, d_ref)
    a, b = _fourier_fit(z_norm, order=fourier_order)
    g, g_inv, raw = _conformal(a, b, ctr=center, m=n_boundary, tol=tol, verbose=verbose)

    gamma_theta = np.linspace(0.0, 2.0 * np.pi, 2000)
    gamma_curve = _make_z_curve(a, b, gamma_theta)

    return ConformalMapResult(
        G=g, G_inv=g_inv, a=a, b=b,
        rot=complex(raw["rot"]), ctr=complex(raw["ctr"]),
        z_norm=z_norm, gamma_curve=gamma_curve, raw=raw,
    )


# ---------------------------------------------------------------------------
# Serialize / restore (scipy-independent load)
# ---------------------------------------------------------------------------
def _bary(z: np.ndarray, zj: np.ndarray, fj: np.ndarray, wj: np.ndarray) -> np.ndarray:
    """Evaluate a barycentric rational r(z) = S(wj*fj/(z-zj)) / S(wj/(z-zj)),
    reconstructing an AAA map from its arrays without needing scipy."""
    z = np.asarray(z, dtype=complex)
    shape = z.shape
    zf = z.ravel()
    with np.errstate(divide="ignore", invalid="ignore"):
        c = 1.0 / (zf[:, None] - zj[None, :])
        r = (c @ (wj * fj)) / (c @ wj)
    bad = ~np.isfinite(r)                       # z exactly on a support point
    if bad.any():
        for idx in np.nonzero(bad)[0]:
            r[idx] = fj[np.argmin(np.abs(zf[idx] - zj))]
    return r.reshape(shape)


def save_conformal_map(path: str | Path, result: ConformalMapResult | dict) -> None:
    """Write a conformal map to a ``.npz`` file.

    Accepts a :class:`ConformalMapResult` (or its ``.raw`` dict directly)."""
    raw = result.raw if isinstance(result, ConformalMapResult) else result
    np.savez(str(path), **raw)


def load_conformal_map(path: str | Path) -> ConformalMapResult:
    """Load a conformal map saved by :func:`save_conformal_map`.

    Rebuilds ``G``/``G_inv`` from the barycentric arrays (no scipy needed). The
    ``z_norm`` field is None (the original boundary points are not serialized);
    ``gamma_curve`` is recomputed from the stored Fourier coefficients."""
    d = np.load(str(path))
    rot = complex(d["rot"])
    gp, gv, gw = d["G_points"], d["G_values"], d["G_weights"]
    fp, fv, fw = d["Ginv_points"], d["Ginv_values"], d["Ginv_weights"]
    g = lambda z: _bary(z, gp, gv, gw)
    g_inv = lambda w: rot * _bary(w, fp, fv, fw)

    a, b = d["a"], d["b"]
    gamma_curve = _make_z_curve(a, b, np.linspace(0.0, 2.0 * np.pi, 2000))
    return ConformalMapResult(
        G=g, G_inv=g_inv, a=a, b=b, rot=rot, ctr=complex(d["ctr"]),
        z_norm=None, gamma_curve=gamma_curve, raw={k: d[k] for k in d.files},
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_conformal_map(result: ConformalMapResult, n_grid: int = 31,
                       save_path: str | Path | None = None) -> None:
    """Plot a polar grid on the unit disk and its image on the bubble region.

    Left panel: the unit disk with radial and circular grid lines. Right panel: the
    same grid mapped through ``G`` onto the bubble, with the fitted boundary (and the
    measured boundary, if available) overlaid.
    """
    import matplotlib.pyplot as plt

    theta_vals = np.linspace(0, 2 * np.pi, n_grid)
    r_vals = np.linspace(0, 1, n_grid)
    t = np.linspace(0, 2 * np.pi, n_grid)

    ray_lines = np.array([r_vals * np.exp(1j * th) for th in theta_vals])   # spokes
    circ_lines = np.array([r * np.exp(1j * t) for r in r_vals])             # rings
    mapped_rays = np.array([result.G(line) for line in ray_lines])
    mapped_circ = np.array([result.G(line) for line in circ_lines])

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(10, 5))
    axa.plot(circ_lines.imag.T, circ_lines.real.T, "r", lw=0.7)
    axa.plot(ray_lines.imag.T, ray_lines.real.T, "b", lw=0.7)
    axa.plot(np.cos(t), np.sin(t), "k", lw=2)
    axa.set_aspect("equal"); axa.set_title("(a) unit disk")
    axa.set_xlabel("Im(z)"); axa.set_ylabel("Re(z)")

    axb.plot(mapped_circ.imag.T, mapped_circ.real.T, "r", lw=0.7)
    axb.plot(mapped_rays.imag.T, mapped_rays.real.T, "b", lw=0.7)
    axb.plot(result.gamma_curve.imag, result.gamma_curve.real, "k-", lw=2, label="fitted")
    if result.z_norm is not None:
        axb.plot(result.z_norm.imag, result.z_norm.real, "m--", lw=1, label="measured")
    axb.set_aspect("equal"); axb.set_title("(b) bubble region")
    axb.set_xlabel("Im(w)"); axb.set_ylabel("Re(w)"); axb.legend()
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
        description="Build a conformal map (unit disk <-> bubble) from a bubble boundary."
    )
    parser.add_argument("--bubble-shape", required=True,
                        help="Text file of (x, y) boundary pixel coordinates (e.g. bubble_shape_tot.txt).")
    parser.add_argument("--d-ref", type=float, required=True,
                        help="Reference bubble diameter in pixels (MATLAB Dref, e.g. 1040 for EXP8).")
    parser.add_argument("--fourier-order", type=int, default=31, help="Fourier fit harmonics. Default 31.")
    parser.add_argument("--n-boundary", type=int, default=800,
                        help="Kerzman-Stein boundary correspondence points (M). Default 800.")
    parser.add_argument("--tol", type=float, default=1e-6, help="AAA relative tolerance. Default 1e-6.")
    parser.add_argument("--out", help="Where to write the conformal map .npz file.")
    parser.add_argument("--plot", action="store_true", help="Show the disk->bubble grid plot.")
    args = parser.parse_args()

    bubble_shape = np.loadtxt(args.bubble_shape)
    result = build_conformal_map(
        bubble_shape, d_ref=args.d_ref, fourier_order=args.fourier_order,
        n_boundary=args.n_boundary, tol=args.tol,
    )
    if args.out:
        save_conformal_map(args.out, result)
        print(f"[save] conformal map written to {args.out}")
    if args.plot:
        plot_conformal_map(result)


if __name__ == "__main__":
    main()
