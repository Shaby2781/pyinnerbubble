"""Scalar (mass) transport inside the bubble and the Sherwood number (paper §IV).

Given a reconstructed interior flow field (from the assimilation stage), solve the
axisymmetric advection-diffusion of a passive scalar on the body-fitted mesh, and from
the decaying scalar mass extract the time-dependent Sherwood number, for a range of
Peclet numbers.

Everything is non-dimensional: the mesh is normalized by the bubble diameter, the
velocity by the bubble rise velocity, and time by the advective scale; the diffusion
time is ``t / Pe``. The scalar is initialized to 1 in the interior and absorbed at the
interface (c = 0), so its total mass decays and ``Sh(t)`` measures the transfer rate.

Ported from the research script ``mass_transfer_solver_experment.py``. The finite-
difference operators and mesh metrics are shared with :mod:`pyinnerbubble.assimilation`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from ..assimilation.flow_reconstruction import (
    _div_eta,
    _div_eta2,
    _div_xsi,
    _div_xsi2,
    compute_mesh_metrics,
)


# ---------------------------------------------------------------------------
# Geometry helpers (axisymmetric volume / interface area)
# ---------------------------------------------------------------------------
def integrate_curvilinear_cylindrical(r_mat, z_mat, c_mat):
    """Axisymmetric volume integral of ``c`` over the curvilinear (r, z) mesh
    (dV = 2*pi*r dA, cell area from the quad diagonals' cross product)."""
    c_center = (c_mat[:-1, :-1] + c_mat[1:, :-1] + c_mat[1:, 1:] + c_mat[:-1, 1:]) / 4.0
    r_center = (r_mat[:-1, :-1] + r_mat[1:, :-1] + r_mat[1:, 1:] + r_mat[:-1, 1:]) / 4.0
    d1_r = r_mat[1:, 1:] - r_mat[:-1, :-1]
    d1_z = z_mat[1:, 1:] - z_mat[:-1, :-1]
    d2_r = r_mat[:-1, 1:] - r_mat[1:, :-1]
    d2_z = z_mat[:-1, 1:] - z_mat[1:, :-1]
    cell_areas = 0.5 * np.abs(d1_r * d2_z - d1_z * d2_r)
    return np.sum(c_center * cell_areas * 2 * np.pi * r_center)


def surface_area_of_revolution(r_vec, z_vec):
    """Surface area of the interface curve (r_vec, z_vec) revolved about the z axis."""
    r, z = np.asarray(r_vec), np.asarray(z_vec)
    ds = np.sqrt(np.diff(r) ** 2 + np.diff(z) ** 2)
    r_avg = (r[:-1] + r[1:]) / 2.0
    return np.sum(2 * np.pi * r_avg * ds)


def _div_t(x):
    """1-D central difference (forward/backward at the ends). Fixes a typo in the
    original (which set index 1 instead of the last index, leaving the final
    derivative -- and hence Sh at the last time -- zero/invalid)."""
    d = np.zeros_like(x)
    d[0] = x[1] - x[0]
    d[-1] = x[-1] - x[-2]
    d[1:-1] = (x[2:] - x[:-2]) / 2
    return d


# ---------------------------------------------------------------------------
# Boundary conditions + advection-diffusion RHS
# ---------------------------------------------------------------------------
def _reconstruct_c(vec, shape):
    """Rebuild the full (M, N) scalar field from the interior unknowns, applying the
    boundary conditions: zero-gradient top/bottom rows, uniform (mean) value on the axis
    column, and c = 0 (absorbing) on the interface column."""
    c_temp = vec.reshape(shape)
    c = np.vstack((c_temp[0, :], c_temp, c_temp[-1, :]))
    mean_c = np.mean(c[:, 0])
    c = np.column_stack((mean_c * np.ones_like(c[:, 0]), c, np.zeros_like(c[:, 0])))
    return c


def _ode_mass_transfer(t, vec, shape, metrics, u_r, u_z, r_mat, pe):
    xi_r, xi_z, eta_r, eta_z, xi_rr, xi_rz, xi_zz, eta_rr, eta_rz, eta_zz = metrics
    c = _reconstruct_c(vec, shape)

    c_eta = _div_eta(c)
    c_xi = _div_xsi(c)
    c_etaxi = (_div_eta(_div_xsi(c)) + _div_xsi(_div_eta(c))) / 2
    c_2eta = _div_eta2(c)
    c_2xi = _div_xsi2(c)
    r_safe = np.where(r_mat == 0, 1e-16, r_mat)
    laplace_c = ((xi_rr + xi_zz + xi_r / r_safe) * c_xi + (eta_rr + eta_zz + eta_r / r_safe) * c_eta
                 + (eta_r ** 2 + eta_z ** 2) * c_2eta + (xi_r ** 2 + xi_z ** 2) * c_2xi
                 + (2 * eta_r * xi_r + 2 * eta_z * xi_z) * c_etaxi)

    # Advection with central differences (the source's upwind branches both reduced to the
    # central stencil, i.e. upwinding was effectively disabled).
    c_z = c_xi * xi_z + c_eta * eta_z
    c_r = c_xi * xi_r + c_eta * eta_r
    delta_c = 1 / pe * laplace_c - u_r * c_r - u_z * c_z
    return delta_c[1:-1, 1:-1].flatten()


def _linear_operator(f_handle, n):
    """The advection-diffusion RHS is linear in c, so extract the constant matrix A with
    f(e_j) - f(0) columns. Passing this as ``jac`` makes the BDF solve fast and stable."""
    a = np.zeros((n, n))
    b = f_handle(0.0, np.zeros(n))
    for j in range(n):
        e_j = np.zeros(n)
        e_j[j] = 1.0
        a[:, j] = f_handle(0.0, e_j) - b
    return a


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class ScalarTransportResult:
    pe_vec: np.ndarray                    # (n_pe,) Peclet numbers
    t_diffusion: np.ndarray              # (n_time,) diffusion-time axis (= physical t / Pe), shared across Pe
    sh_matrix: np.ndarray                # (n_time, n_pe) Sherwood number vs (time, Pe)
    sh_final: np.ndarray                 # (n_pe,) Sherwood number at the last time point
    concentration_final: list            # list of (M, N) final scalar fields, one per Pe
    r_mat: np.ndarray
    z_mat: np.ndarray


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def simulate_scalar_transport(assimilation_file, t_end_diffusion: float, n_time: int,
                              pe_vec, dmax_deq: float = 1.0, rtol: float = 1e-6,
                              atol: float = 1e-8, verbose: bool = True) -> ScalarTransportResult:
    """Solve scalar transport for each Peclet number and compute the Sherwood number.

    Parameters
    ----------
    assimilation_file : str or Path
        A ``.npz`` from the assimilation stage, with the reconstructed flow field
        (``r_mat``/``z_mat``/``u_r``/``u_z``; upper-case ``R_mat`` etc. also accepted).
    t_end_diffusion : float
        Simulation end time in DIFFUSION-time units (t/Pe). For each Pe the physical
        integration runs to ``t_end_diffusion * Pe``.
    n_time : int
        Number of time points (shared diffusion-time axis, ``linspace(0, t_end_diffusion, n_time)``).
    pe_vec : array-like
        Peclet numbers to run.
    dmax_deq : float
        Optional shape factor multiplying the mesh (max-diameter / equivalent-diameter). Default 1.
    rtol, atol : float
        ``solve_ivp`` tolerances (BDF).
    verbose : bool
        Print per-Pe progress and elapsed time.

    Returns
    -------
    ScalarTransportResult
        With ``sh_matrix`` (Sh vs time & Pe), ``sh_final`` (Sh vs Pe at the last time),
        ``concentration_final`` (final scalar field per Pe), and the ``pe_vec`` /
        ``t_diffusion`` axes. Feed it to plot_sherwood_vs_time / plot_sherwood_vs_pe /
        plot_final_concentration.
    """
    d = np.load(str(assimilation_file))
    r_key = "r_mat" if "r_mat" in d.files else "R_mat"
    z_key = "z_mat" if "z_mat" in d.files else "Z_mat"
    r_mat = np.asarray(d[r_key], dtype=float) * dmax_deq
    z_mat = np.asarray(d[z_key], dtype=float) * dmax_deq
    u_r = np.asarray(d["u_r"], dtype=float)
    u_z = np.asarray(d["u_z"], dtype=float)

    pe_vec = np.atleast_1d(np.asarray(pe_vec, dtype=float))
    t_diffusion = np.linspace(0.0, t_end_diffusion, n_time)

    # geometry: volume/interface ratio (the length scale in Sh)
    vol = integrate_curvilinear_cylindrical(r_mat, z_mat, np.ones_like(r_mat))
    area = surface_area_of_revolution(r_mat[:, -1], z_mat[:, -1])
    ratio_vs = vol / area

    metrics = compute_mesh_metrics(r_mat, z_mat)
    shape = (r_mat.shape[0] - 2, r_mat.shape[1] - 2)   # interior unknowns
    n_var = shape[0] * shape[1]
    x0 = np.ones(n_var)

    sh_matrix = np.zeros((n_time, pe_vec.size))
    sh_final = np.zeros(pe_vec.size)
    concentration_final: list[np.ndarray] = []

    for j, pe in enumerate(pe_vec):
        t0 = time.perf_counter()
        tf = t_end_diffusion * pe
        t_eval = t_diffusion * pe   # physical time points

        def rhs(t, vec, pe=pe):
            return _ode_mass_transfer(t, vec, shape, metrics, u_r, u_z, r_mat, pe)

        a_matrix = _linear_operator(rhs, n_var)
        sol = solve_ivp(rhs, (0.0, tf), x0, method="BDF", jac=a_matrix,
                        t_eval=t_eval, rtol=rtol, atol=atol)

        mass = np.array([integrate_curvilinear_cylindrical(r_mat, z_mat, _reconstruct_c(sol.y[:, k], shape))
                         for k in range(sol.y.shape[1])])
        m_dot = _div_t(mass) / _div_t(t_eval)
        sh = -pe * ratio_vs * m_dot / mass

        sh_matrix[:, j] = sh
        sh_final[j] = sh[-1]
        concentration_final.append(_reconstruct_c(sol.y[:, -1], shape))
        if verbose:
            print(f"[transport] Pe={pe:g}: Sh_final={sh[-1]:.4g}  ({time.perf_counter() - t0:.1f} s)", flush=True)

    return ScalarTransportResult(
        pe_vec=pe_vec, t_diffusion=t_diffusion, sh_matrix=sh_matrix, sh_final=sh_final,
        concentration_final=concentration_final, r_mat=r_mat, z_mat=z_mat,
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_sherwood_vs_time(result: ScalarTransportResult, save_path=None) -> None:
    """Sh vs diffusion time (t/Pe), one line per Peclet number, with a legend."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    for j, pe in enumerate(result.pe_vec):
        ax.plot(result.t_diffusion, result.sh_matrix[:, j], label=f"Pe = {pe:g}")
    ax.set_xlabel("t / Pe  (diffusion time)")
    ax.set_ylabel("Sh")
    ax.set_title("Sherwood number vs diffusion time")
    ax.legend()
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(str(save_path), dpi=150); plt.close(fig)
    else:
        plt.show()


def plot_sherwood_vs_pe(result: ScalarTransportResult, save_path=None) -> None:
    """Sh at the last time point vs Peclet number."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot(result.pe_vec, result.sh_final, "o-")
    ax.set_xscale("log")
    ax.set_xlabel("Pe")
    ax.set_ylabel("Sh (final time)")
    ax.set_title("Sherwood vs Peclet")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(str(save_path), dpi=150); plt.close(fig)
    else:
        plt.show()


def plot_final_concentration(result: ScalarTransportResult, save_path=None) -> None:
    """Final scalar field for each Pe, normalized by its own maximum, side by side."""
    import matplotlib.pyplot as plt

    n = result.pe_vec.size
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4.5), squeeze=False)
    for j, pe in enumerate(result.pe_vec):
        ax = axes[0, j]
        c = result.concentration_final[j]
        c_norm = c / np.nanmax(c)
        pc = ax.pcolormesh(result.r_mat, result.z_mat, c_norm, shading="gouraud",
                           cmap="inferno", vmin=0, vmax=1)
        ax.plot(result.r_mat[:, -1], result.z_mat[:, -1], "k-", linewidth=1.5)
        ax.set_aspect("equal"); ax.set_title(f"Pe = {pe:g}"); ax.set_xlabel("r")
        if j == 0:
            ax.set_ylabel("z")
        fig.colorbar(pc, ax=ax, fraction=0.046, label="c / c_max")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(str(save_path), dpi=150); plt.close(fig)
    else:
        plt.show()
