"""Inverse axisymmetric Navier-Stokes assimilation of the interior flow (paper §IIIC).

Fits the interfacial (boundary) tangential velocity so that the steady axisymmetric
Navier-Stokes solution inside the bubble best matches the measured interior velocity
field, in a least-squares (optionally weighted) sense.

The flow is solved in stream-function / vorticity form (PSI, PHI) on the body-fitted
curvilinear mesh from :mod:`pyinnerbubble.assimilation.mesh`, marched to steady state
by pseudo-time integration (``solve_ivp``, BDF). The boundary velocity is parameterized
by ``num_of_points`` spline control points (zero at both ends) and optimized with
``scipy.optimize.minimize``.

Everything here is NON-DIMENSIONAL: the mesh (``r_mat``/``z_mat``) must be normalized
by the bubble diameter and the velocities by the bubble rise velocity, with the physical
scale carried entirely by ``reynolds``. (Report results in MKS by multiplying lengths by
the diameter and velocities by the rise velocity afterward.)

Ported from ``exp_optimization_NS_15_10_25_weighted.py``. The original also built a
numerical ODE Jacobian ``J`` that was never used by the cost function; it is dropped here.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import cumulative_trapezoid, solve_ivp
from scipy.interpolate import LinearNDInterpolator, splev, splrep
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# Finite-difference operators on the (xsi, eta) computational grid
# axis 1 = xsi (radial, disk center -> interface), axis 0 = eta (along the boundary)
# ---------------------------------------------------------------------------
def _div_xsi(x):
    d = np.zeros_like(x)
    d[:, 0] = x[:, 1] - x[:, 0]
    d[:, -1] = x[:, -1] - x[:, -2]
    d[:, 1:-1] = (x[:, 2:] - x[:, :-2]) / 2
    return d


def _div_eta(x):
    d = np.zeros_like(x)
    d[0, :] = x[1, :] - x[0, :]
    d[-1, :] = x[-1, :] - x[-2, :]
    d[1:-1, :] = (x[2:, :] - x[:-2, :]) / 2
    return d


def _div_xsi2(x):
    d = np.zeros_like(x)
    d[:, 0] = x[:, 2] - 2 * x[:, 1] + x[:, 0]
    d[:, -1] = x[:, -1] - 2 * x[:, -2] + x[:, -3]
    d[:, 1:-1] = x[:, 2:] - 2 * x[:, 1:-1] + x[:, :-2]
    return d


def _div_eta2(x):
    d = np.zeros_like(x)
    d[0, :] = x[2, :] - 2 * x[1, :] + x[0, :]
    d[-1, :] = x[-1, :] - 2 * x[-2, :] + x[-3, :]
    d[1:-1, :] = x[2:, :] - 2 * x[1:-1, :] + x[:-2, :]
    return d


def _div_xsi_forward(x):
    d = np.zeros_like(x)
    d[:, 0:-2] = (-x[:, 2:] + 4 * x[:, 1:-1] - 3 * x[:, 0:-2]) / 2
    d[:, -2] = x[:, -1] - x[:, -2]
    d[:, -1] = d[:, -2]
    return d


def _div_eta_forward(x):
    d = np.zeros_like(x)
    d[0:-2, :] = (-x[2:, :] + 4 * x[1:-1, :] - 3 * x[0:-2, :]) / 2
    d[-2, :] = x[-1, :] - x[-2, :]
    d[-1, :] = d[-2, :]
    return d


def _div_xsi_backward(x):
    d = np.zeros_like(x)
    d[:, 2:] = (3 * x[:, 2:] - 4 * x[:, 1:-1] + x[:, :-2]) / 2
    d[:, 1] = x[:, 1] - x[:, 0]
    d[:, 0] = d[:, 1]
    return d


def _div_eta_backward(x):
    d = np.zeros_like(x)
    d[2:, :] = (3 * x[2:, :] - 4 * x[1:-1, :] + x[:-2, :]) / 2
    d[1, :] = x[1, :] - x[0, :]
    d[0, :] = d[1, :]
    return d


# ---------------------------------------------------------------------------
# Curvilinear mesh metrics (grid -> physical derivative transforms)
# ---------------------------------------------------------------------------
def compute_mesh_metrics(r_mat, z_mat):
    """Metric terms of the (xsi, eta) -> (r, z) mapping used by the NS operator.

    Returns ``(xi_r, xi_z, eta_r, eta_z, xi_rr, xi_rz, xi_zz, eta_rr, eta_rz, eta_zz)``
    -- first derivatives and second derivatives of the computational coordinates with
    respect to physical (r, z). Ported from ``calc_jacobian``.
    """
    r_xsi, z_xsi = _div_xsi(r_mat), _div_xsi(z_mat)
    r_eta, z_eta = _div_eta(r_mat), _div_eta(z_mat)
    r_xsi2, z_xsi2 = _div_xsi2(r_mat), _div_xsi2(z_mat)
    r_eta2, z_eta2 = _div_eta2(r_mat), _div_eta2(z_mat)
    r_xsieta = (_div_eta(_div_xsi(r_mat)) + _div_xsi(_div_eta(r_mat))) / 2
    z_xsieta = (_div_eta(_div_xsi(z_mat)) + _div_xsi(_div_eta(z_mat))) / 2

    jacob = r_xsi * z_eta - z_xsi * r_eta
    jacob_xsi = r_xsi2 * z_eta + r_xsi * z_xsieta - r_xsieta * z_xsi - r_eta * z_xsi2
    jacob_eta = r_xsieta * z_eta + r_xsi * z_eta2 - r_eta2 * z_xsi - r_eta * z_xsieta
    jacob = jacob.copy()
    jacob[jacob == 0] = 1
    inv = 1 / jacob

    xi_r = z_eta * inv
    xi_z = -r_eta * inv
    eta_r = -z_xsi * inv
    eta_z = r_xsi * inv

    xi_rr = 1 / jacob ** 3 * (jacob * (z_xsieta * z_eta - z_eta2 * z_xsi) - z_eta * (jacob_xsi * z_eta - jacob_eta * z_xsi))
    xi_rz = 1 / jacob ** 3 * (jacob * (-z_xsieta * r_eta + z_eta2 * r_xsi) + z_eta * (jacob_xsi * r_eta - jacob_eta * r_xsi))
    xi_zz = 1 / jacob ** 3 * (jacob * (r_xsieta * r_eta - r_eta2 * r_xsi) - r_eta * (jacob_xsi * r_eta - jacob_eta * r_xsi))
    eta_rr = 1 / jacob ** 3 * (jacob * (-z_xsi2 * z_eta + z_xsieta * z_xsi) + z_xsi * (jacob_xsi * z_eta - jacob_eta * z_xsi))
    eta_rz = 1 / jacob ** 3 * (jacob * (z_xsi2 * r_eta - z_xsieta * r_xsi) - z_xsi * (jacob_xsi * r_eta - jacob_eta * r_xsi))
    eta_zz = 1 / jacob ** 3 * (jacob * (-r_xsi2 * r_eta + r_xsieta * r_xsi) + r_xsi * (jacob_xsi * r_eta - jacob_eta * r_xsi))

    return xi_r, xi_z, eta_r, eta_z, xi_rr, xi_rz, xi_zz, eta_rr, eta_rz, eta_zz


# ---------------------------------------------------------------------------
# Steady axisymmetric Navier-Stokes RHS (stream function PSI / vorticity PHI)
# ---------------------------------------------------------------------------
def _ode_ns(t, vec, psi, phi, variable_shape, metrics, r_mat, reynolds):
    xi_r, xi_z, eta_r, eta_z, xi_rr, xi_rz, xi_zz, eta_rr, eta_rz, eta_zz = metrics
    n_row, n_col = variable_shape
    psi[1:-2, 1:-2] = vec[:n_row * n_col].reshape(variable_shape)
    phi[1:-2, 1:-1] = vec[n_row * n_col:].reshape((n_row, n_col + 1))

    r_safe = np.where(r_mat == 0, 1e-16, r_mat)

    psi_eta = _div_eta(psi)
    psi_xi = _div_xsi(psi)
    psi_etaxi = (_div_eta(_div_xsi(psi)) + _div_xsi(_div_eta(psi))) / 2
    psi_2eta = _div_eta2(psi)
    psi_2xi = _div_xsi2(psi)
    laplace_psi = ((xi_rr + xi_zz - xi_r / r_safe) * psi_xi + (eta_rr + eta_zz - eta_r / r_safe) * psi_eta
                   + (eta_r ** 2 + eta_z ** 2) * psi_2eta + (xi_r ** 2 + xi_z ** 2) * psi_2xi
                   + (2 * eta_r * xi_r + 2 * eta_z * xi_z) * psi_etaxi)

    phi[:, -1:] = laplace_psi[:, -1:]

    phi_eta = _div_eta(phi)
    phi_xi = _div_xsi(phi)
    phi_etaxi = _div_eta(_div_xsi(phi))
    phi_2eta = _div_eta2(phi)
    phi_2xi = _div_xsi2(phi)
    laplace_phi = ((xi_rr + xi_zz - xi_r / r_safe) * phi_xi + (eta_rr + eta_zz - eta_r / r_safe) * phi_eta
                   + (eta_r ** 2 + eta_z ** 2) * phi_2eta + (xi_r ** 2 + xi_z ** 2) * phi_2xi
                   + (2 * eta_r * xi_r + 2 * eta_z * xi_z) * phi_etaxi)

    psi_r = psi_xi * xi_r + psi_eta * eta_r
    psi_z = psi_xi * xi_z + psi_eta * eta_z
    phi_r = phi_xi * xi_r + phi_eta * eta_r
    phi_z = phi_xi * xi_z + phi_eta * eta_z
    u_eta = -eta_r * psi_z + eta_z * psi_r
    u_xi = -xi_r * psi_z + xi_z * psi_r

    # first-order upwinding of the convective PHI gradients
    phi_eta_f, phi_eta_b = _div_eta_forward(phi), _div_eta_backward(phi)
    phi_xsi_f, phi_xsi_b = _div_xsi_forward(phi), _div_xsi_backward(phi)
    ueta_pos = (u_eta > 0).astype(float)
    uxi_pos = (u_xi > 0).astype(float)
    upwind_dphi_dxi = uxi_pos * phi_xsi_b + (1.0 - uxi_pos) * phi_xsi_f
    upwind_dphi_deta = ueta_pos * phi_eta_b + (1.0 - ueta_pos) * phi_eta_f
    phi_z_up = upwind_dphi_dxi * xi_z + upwind_dphi_deta * eta_z
    phi_r_up = upwind_dphi_dxi * xi_r + upwind_dphi_deta * eta_r

    delta_psi = laplace_psi - phi
    delta_phi = (r_mat ** 2 / reynolds * laplace_phi - r_mat * psi_r * phi_z_up
                 + r_mat * psi_z * phi_r_up - 2 * psi_z * phi)
    dvec = np.concatenate([delta_psi[1:-2, 1:-2].ravel(), delta_phi[1:-2, 1:-1].ravel()])
    return dvec


def _integrate_to_steady(vel_vec, r_mat, z_mat, metrics, reynolds, psi0, phi0):
    """March the NS system to steady state for a given boundary velocity, returning PSI, PHI."""
    r_xsi = _div_xsi(r_mat)
    z_xsi = _div_xsi(z_mat)
    curve_r, curve_z = r_mat[:, -1], z_mat[:, -1]
    ds = np.sqrt(np.concatenate([[0], np.diff(curve_r)]) ** 2
                 + np.concatenate([[0], np.diff(curve_z)]) ** 2)
    s = cumulative_trapezoid(ds, initial=0)

    n = len(vel_vec)
    s4interp = np.linspace(s[0], s[-1], n + 2)
    tck = splrep(s4interp, np.concatenate([[0], vel_vec, [0]]))
    vel_interp = splev(s, tck)

    psi = psi0.copy()
    phi = phi0.copy()
    g_xsi = np.sqrt(r_xsi[:, -1] ** 2 + z_xsi[:, -1] ** 2)
    psi[:, -2] = g_xsi * (vel_interp * curve_r)   # boundary condition from the tangential velocity

    variable_shape = psi[1:-2, 1:-2].shape
    x0 = np.concatenate([psi[1:-2, 1:-2].ravel(), phi[1:-2, 1:-1].ravel()])
    sol = solve_ivp(lambda t, v: _ode_ns(t, v, psi, phi, variable_shape, metrics, r_mat, reynolds),
                    (0.0, reynolds), x0, method="BDF")
    vec = sol.y[:, -1]
    n_row, n_col = variable_shape
    psi[1:-2, 1:-2] = vec[:n_row * n_col].reshape(variable_shape)
    phi[1:-2, 1:-1] = vec[n_row * n_col:].reshape((n_row, n_col + 1))
    return psi, phi


def _velocity_from_psi(psi, r_mat, z_mat, metrics, r_floor_zero_border=False):
    """Recover (u_r, u_z) from the stream function on the mesh."""
    xi_r, xi_z, eta_r, eta_z = metrics[0], metrics[1], metrics[2], metrics[3]
    psi_eta = _div_eta(psi)
    psi_xi = _div_xsi(psi)
    psi_z = psi_xi * xi_z + psi_eta * eta_z
    psi_r = psi_xi * xi_r + psi_eta * eta_r
    r_div = r_mat.copy()
    if r_floor_zero_border:
        r_div[:, 0] = 1e-16
        r_div[0, :] = 1e-16
        r_div[-1, :] = 1e-16
    r_div[r_div == 0] = 1e-6
    u_r = -1 / r_div * psi_z
    u_z = 1 / r_div * psi_r
    if r_floor_zero_border:
        # The 1/r factor blows up on the axis column (col 0) and the top/bottom rows, so the
        # velocity there is meaningless. Replace those edge values with the adjacent interior
        # row/column (the "second" row/column) instead of the garbage.
        u_r[:, 0], u_z[:, 0] = u_r[:, 1], u_z[:, 1]
        u_r[0, :], u_z[0, :] = u_r[1, :], u_z[1, :]
        u_r[-1, :], u_z[-1, :] = u_r[-2, :], u_z[-2, :]
    return u_r, u_z


def reconstruct_flow_field(vel_vec, r_mat, z_mat, metrics, reynolds, psi0, phi0):
    """Solve the NS flow for a boundary velocity vector and return ``(psi, phi, u_r, u_z)``."""
    psi, phi = _integrate_to_steady(vel_vec, r_mat, z_mat, metrics, reynolds, psi0, phi0)
    u_r, u_z = _velocity_from_psi(psi, r_mat, z_mat, metrics, r_floor_zero_border=True)
    return psi, phi, u_r, u_z


def _cost(vel_vec, r_mat, z_mat, metrics, ur_meas, uz_meas, weight_mat, reynolds, psi0, phi0):
    """Weighted sum-of-squares error between the NS solution and the measured velocity."""
    psi, _ = _integrate_to_steady(vel_vec, r_mat, z_mat, metrics, reynolds, psi0, phi0)
    u_r, u_z = _velocity_from_psi(psi, r_mat, z_mat, metrics, r_floor_zero_border=False)
    core = np.s_[2:-2, 2:-2]
    return (np.sum((u_r[core] - ur_meas[core]) ** 2 * weight_mat[core])
            + np.sum((u_z[core] - uz_meas[core]) ** 2 * weight_mat[core]))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def interpolate_measured_field_to_mesh(
    r_mat, z_mat, r_pts, z_pts, ur, uz,
    weight_r_pts=None, weight_z_pts=None, weight_vals=None,
):
    """Interpolate a scattered measured velocity field (and optional confidence weight),
    already expressed in the mesh's coordinate frame, onto the mesh nodes -- handling
    no-data and zero-weight regions correctly (so this logic is not re-implemented per example).

    Parameters
    ----------
    r_mat, z_mat : (M, N) arrays
        The mesh node coordinates.
    r_pts, z_pts, ur, uz : 1-D arrays
        Scattered measured velocity samples (radial/axial) at (r_pts, z_pts). Pass only points
        with real velocity data (e.g. drop the edge-removal band, where the velocity is NaN).
    weight_r_pts, weight_z_pts, weight_vals : 1-D arrays, optional
        Scattered confidence-weight samples. Pass the FULL in-region set INCLUDING any
        zero-weight band (e.g. the edge-removal band) so those zeros are preserved on the mesh
        instead of being smeared over by the nonzero interior -- this is the subtlety that is
        easy to get wrong. If omitted, a uniform weight of 1 is used.

    Returns
    -------
    (ur_meas, uz_meas, weight_mat, data_mask) : all (M, N)
        Velocity is 0 (and ``data_mask`` = 0) at mesh nodes outside the velocity-data hull;
        ``weight_mat`` is the interpolated weight, always forced to 0 where there is no
        velocity data.
    """
    query = np.column_stack([r_mat.ravel(), z_mat.ravel()])
    vpts = np.column_stack([r_pts, z_pts])
    ur_meas = LinearNDInterpolator(vpts, ur, fill_value=np.nan)(query).reshape(r_mat.shape)
    uz_meas = LinearNDInterpolator(vpts, uz, fill_value=np.nan)(query).reshape(r_mat.shape)
    data_mask = (np.isfinite(ur_meas) & np.isfinite(uz_meas)).astype(float)
    ur_meas = np.nan_to_num(ur_meas, nan=0.0)
    uz_meas = np.nan_to_num(uz_meas, nan=0.0)

    if weight_vals is None:
        weight_mat = np.ones_like(r_mat)
    else:
        wpts = np.column_stack([weight_r_pts, weight_z_pts])
        weight_mat = LinearNDInterpolator(wpts, weight_vals, fill_value=0.0)(query).reshape(r_mat.shape)
        weight_mat = np.clip(np.nan_to_num(weight_mat, nan=0.0), 0.0, None)
    weight_mat = weight_mat * data_mask   # no velocity data -> no weight
    return ur_meas, uz_meas, weight_mat, data_mask


@dataclass
class AssimilationResult:
    vel_opt: np.ndarray            # optimized boundary velocity control points
    cost: float                    # final cost value
    r_mat: np.ndarray
    z_mat: np.ndarray
    psi: np.ndarray                # reconstructed stream function
    phi: np.ndarray                # reconstructed vorticity variable
    u_r: np.ndarray                # reconstructed radial velocity field
    u_z: np.ndarray                # reconstructed axial velocity field
    s: np.ndarray                  # arclength coordinate of the control points
    cost_history: list = field(default_factory=list)   # cost per optimizer iteration


def assimilate_boundary_velocity(
    r_mat: np.ndarray,
    z_mat: np.ndarray,
    ur_meas: np.ndarray,
    uz_meas: np.ndarray,
    reynolds: float,
    num_of_points: int = 2,
    weight_mat: np.ndarray | None = None,
    u_scale: float = 1.0,
    bound_factor: float = 1.5,
    x0: np.ndarray | None = None,
    method: str = "Powell",
    maxiter: int = 400,
    tol: float = 1e-6,
    psi0: np.ndarray | None = None,
    phi0: np.ndarray | None = None,
    verbose: bool = True,
) -> AssimilationResult:
    """Fit the interfacial boundary velocity to the measured interior velocity field.

    All inputs are non-dimensional: ``r_mat``/``z_mat`` normalized by the bubble
    diameter, ``ur_meas``/``uz_meas`` by the bubble velocity; ``reynolds`` carries the
    physical scale.

    Parameters
    ----------
    r_mat, z_mat : (M, N) arrays
        Body-fitted curvilinear mesh (column -1 is the interface, column 0 the axis
        point) -- e.g. the ``x_mat``/``y_mat`` from :func:`~pyinnerbubble.assimilation.build_bubble_mesh`.
    ur_meas, uz_meas : (M, N) arrays
        Measured radial/axial velocity interpolated onto the mesh.
    reynolds : float
        Reynolds number (bubble velocity * diameter / viscosity).
    num_of_points : int
        Number of free spline control points for the boundary velocity (zero-clamped at
        both ends). Default 2.
    weight_mat : (M, N) array, optional
        Per-node weights in the cost. Default ``np.ones_like(r_mat)`` (uniform).
    u_scale, bound_factor : float
        Control-point bounds are ``+/- bound_factor * u_scale``.
    x0 : (num_of_points,) array, optional
        Initial control points. Default all zeros.
    method, maxiter, tol : optimizer settings for ``scipy.optimize.minimize``.
    psi0, phi0 : (M, N) arrays, optional
        Initial stream function / vorticity for the pseudo-time march. Default zeros.
    verbose : bool
        Print per-iteration cost.

    Returns
    -------
    AssimilationResult
    """
    r_mat = np.asarray(r_mat, dtype=float)
    z_mat = np.asarray(z_mat, dtype=float)
    if weight_mat is None:
        weight_mat = np.ones_like(r_mat)
    if psi0 is None:
        psi0 = np.zeros_like(r_mat)
    if phi0 is None:
        phi0 = np.zeros_like(r_mat)
    if x0 is None:
        x0 = np.zeros(num_of_points)

    metrics = compute_mesh_metrics(r_mat, z_mat)

    # arclength of the control points along the interface (for reference / plotting)
    curve_r, curve_z = r_mat[:, -1], z_mat[:, -1]
    ds = np.sqrt(np.concatenate([[0], np.diff(curve_r)]) ** 2
                 + np.concatenate([[0], np.diff(curve_z)]) ** 2)
    s_full = cumulative_trapezoid(ds, initial=0)
    s_ctrl = np.linspace(s_full[0], s_full[-1], num_of_points + 2)[1:-1]

    bounds = [(-bound_factor * u_scale, bound_factor * u_scale)] * num_of_points
    cost_history: list[float] = []
    last_cost = {"value": np.nan}
    n_eval = {"count": 0}

    def objective(vel_vec):
        # One objective call = one steady-state Navier-Stokes (CFD) solve. Print every call
        # with its elapsed time so it is obvious the solver is progressing -- on a real
        # (fine-mesh) problem each solve is slow, and without this the run looks stalled.
        n_eval["count"] += 1
        t0 = time.perf_counter()
        c = _cost(vel_vec, r_mat, z_mat, metrics, ur_meas, uz_meas, weight_mat, reynolds, psi0, phi0)
        dt = time.perf_counter() - t0
        last_cost["value"] = c
        if verbose:
            print(f"[assimilate] CFD solve {n_eval['count']}: cost={c:.6g}  ({dt:.2f} s)", flush=True)
        return c

    def on_iter(xk):
        cost_history.append(last_cost["value"])
        if verbose:
            print(f"[assimilate] --- iteration {len(cost_history)} complete: "
                  f"cost={last_cost['value']:.6g} after {n_eval['count']} CFD solves ---", flush=True)

    if verbose:
        print(f"[assimilate] mesh {r_mat.shape}, num_of_points={num_of_points}, Re={reynolds}, method={method}")
    result = minimize(objective, x0=x0, method=method, bounds=bounds,
                      options={"disp": verbose, "maxiter": maxiter}, callback=on_iter)

    psi, phi, u_r, u_z = reconstruct_flow_field(result.x, r_mat, z_mat, metrics, reynolds, psi0, phi0)
    return AssimilationResult(
        vel_opt=result.x, cost=float(result.fun), r_mat=r_mat, z_mat=z_mat,
        psi=psi, phi=phi, u_r=u_r, u_z=u_z, s=s_ctrl, cost_history=cost_history,
    )


def boundary_velocity_profile(result: AssimilationResult, n_dense: int = 300):
    """The fitted interfacial velocity profile, for plotting against point number.

    Returns ``(point_ctrl, vel_ctrl, point_dense, vel_dense)`` where ``vel_ctrl`` are the
    optimized control points with the two zero-clamped axis endpoints prepended/appended
    (so point 0 and point num_of_points+1 are the axis, both 0), and ``vel_dense`` is the
    spline evaluated between them. ``point_*`` are point-number coordinates.
    """
    vel = result.vel_opt
    n = len(vel)
    curve_r, curve_z = result.r_mat[:, -1], result.z_mat[:, -1]
    ds = np.sqrt(np.concatenate([[0], np.diff(curve_r)]) ** 2
                 + np.concatenate([[0], np.diff(curve_z)]) ** 2)
    s = cumulative_trapezoid(ds, initial=0)
    s4interp = np.linspace(s[0], s[-1], n + 2)
    vel_ctrl = np.concatenate([[0.0], vel, [0.0]])   # zero at both axis endpoints
    tck = splrep(s4interp, vel_ctrl)
    s_dense = np.linspace(s[0], s[-1], n_dense)
    vel_dense = splev(s_dense, tck)
    point_ctrl = np.arange(n + 2)                    # 0 (axis) .. n+1 (axis)
    point_dense = (s_dense - s[0]) / (s[-1] - s[0]) * (n + 1)
    return point_ctrl, vel_ctrl, point_dense, vel_dense


def plot_assimilation(result: AssimilationResult, save_path=None) -> None:
    """Plot the optimizer cost trace, the reconstructed field, and the boundary velocity."""
    import matplotlib.pyplot as plt

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
    if result.cost_history:
        ax1.plot(range(1, len(result.cost_history) + 1), result.cost_history, "o-")
    ax1.set_xlabel("iteration"); ax1.set_ylabel("cost"); ax1.set_title("optimization trace")
    ax1.set_yscale("log")

    ax2.contour(result.r_mat, result.z_mat, result.psi, linewidths=1.2)
    ax2.plot(result.r_mat[:, -1], result.z_mat[:, -1], "k", linewidth=2)
    ax2.set_aspect("equal"); ax2.set_title("reconstructed streamlines")
    ax2.set_xlabel("r"); ax2.set_ylabel("z")

    point_ctrl, vel_ctrl, point_dense, vel_dense = boundary_velocity_profile(result)
    ax3.plot(point_dense, vel_dense, "-", label="spline")
    ax3.plot(point_ctrl, vel_ctrl, "o", label="control points")
    ax3.axhline(0, color="0.7", linewidth=0.8)
    ax3.set_xlabel("point number"); ax3.set_ylabel("boundary velocity")
    ax3.set_title("interfacial velocity"); ax3.set_xticks(point_ctrl); ax3.legend()

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(str(save_path), dpi=150)
    else:
        plt.show()


def plot_velocity_field_on_mesh(r_mat, z_mat, u_r, u_z, title="velocity field",
                                quiver_step=1, quiver_scale=None, save_path=None) -> None:
    """Plot a velocity field on the mesh: |u| as a color map with a quiver overlay.

    Useful for the interpolated measured field (ur_meas, uz_meas) and the reconstructed
    field (result.u_r, result.u_z) alike, in the physical (r, z) plane. The interface
    (mesh column -1) is drawn as a thick black curve.

    Parameters
    ----------
    r_mat, z_mat : (M, N) arrays
        Mesh node coordinates.
    u_r, u_z : (M, N) arrays
        Radial / axial velocity components on the mesh.
    quiver_step : int
        Stride between plotted arrows (1 = every node).
    quiver_scale : float or None
        matplotlib quiver ``scale`` (smaller = longer arrows). None = auto.
    """
    import matplotlib.pyplot as plt

    mag = np.sqrt(u_r ** 2 + u_z ** 2)
    fig, ax = plt.subplots()
    # gouraud shading: treat (r_mat, z_mat) as node positions on the curvilinear body-fitted
    # mesh (the coordinates are not a monotonic rectilinear grid, so flat shading misplaces cells).
    pc = ax.pcolormesh(r_mat, z_mat, mag, shading="gouraud", cmap="viridis")
    fig.colorbar(pc, ax=ax, label="|u|")
    s = max(1, int(quiver_step))
    ax.quiver(r_mat[::s, ::s], z_mat[::s, ::s], u_r[::s, ::s], u_z[::s, ::s],
              color="white", scale=quiver_scale)
    ax.plot(r_mat[:, -1], z_mat[:, -1], "k-", linewidth=2)   # interface
    ax.set_aspect("equal"); ax.set_title(title)
    ax.set_xlabel("r"); ax.set_ylabel("z")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(str(save_path), dpi=150)
        plt.close(fig)
    else:
        plt.show()
