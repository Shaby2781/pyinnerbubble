import numpy as np
import pytest

from pyinnerbubble.assimilation import (
    compute_mesh_metrics,
    interface_arclength,
    interface_velocity_from_control_points,
    reconstruct_flow_field,
    reconstruct_flow_field_from_profile,
)


@pytest.fixture(scope="module")
def mesh():
    """A small hemispherical body-fitted mesh -- cheap enough for a steady solve in tests.

    Same layout as a real bubble mesh: rows run along the interface from the top axis to
    the bottom axis, column 0 is the collapsed center, column -1 is the interface.
    """
    phi = np.linspace(0.0, np.pi, 20)          # 0 = top axis, pi = bottom axis
    rho = np.linspace(0.0, 0.5, 8)             # center -> interface (radius 0.5)
    r_mat = np.outer(np.sin(phi), rho)
    z_mat = np.outer(np.cos(phi), rho)
    return r_mat, z_mat


def test_interface_arclength_starts_at_zero_and_increases(mesh):
    r_mat, z_mat = mesh
    s = interface_arclength(r_mat, z_mat)
    assert s.shape == (r_mat.shape[0],)
    assert s[0] == 0.0
    assert np.all(np.diff(s) > 0)


def test_control_points_to_profile_is_zero_clamped_at_the_axis(mesh):
    r_mat, z_mat = mesh
    prof = interface_velocity_from_control_points([0.4, -0.2], r_mat, z_mat)
    assert prof.shape == (r_mat.shape[0],)
    assert prof[0] == pytest.approx(0.0, abs=1e-12)
    assert prof[-1] == pytest.approx(0.0, abs=1e-12)


def test_profile_solve_matches_control_point_solve(mesh):
    """The two forward-solve entry points must agree when they prescribe the same profile.

    ``reconstruct_flow_field`` splines its control points internally; feeding that same
    spline sampled at the interface nodes into ``reconstruct_flow_field_from_profile``
    has to reproduce it exactly -- otherwise a modified profile cannot be compared
    against an assimilated one.
    """
    r_mat, z_mat = mesh
    metrics = compute_mesh_metrics(r_mat, z_mat)
    vel_vec = np.array([0.5, 0.3])
    reynolds = 50.0

    psi_a, phi_a, ur_a, uz_a = reconstruct_flow_field(
        vel_vec, r_mat, z_mat, metrics, reynolds, np.zeros_like(r_mat), np.zeros_like(r_mat))

    prof = interface_velocity_from_control_points(vel_vec, r_mat, z_mat)
    psi_b, phi_b, ur_b, uz_b = reconstruct_flow_field_from_profile(
        prof, r_mat, z_mat, metrics, reynolds)

    assert np.allclose(psi_a, psi_b, rtol=1e-10, atol=1e-14)
    assert np.allclose(phi_a, phi_b, rtol=1e-10, atol=1e-14)
    assert np.allclose(ur_a, ur_b, rtol=1e-10, atol=1e-14)
    assert np.allclose(uz_a, uz_b, rtol=1e-10, atol=1e-14)


def test_profile_solve_keeps_the_interface_impermeable(mesh):
    """A discontinuous profile is still a u_n = 0 boundary: PSI stays 0 on the interface."""
    r_mat, z_mat = mesh
    metrics = compute_mesh_metrics(r_mat, z_mat)
    prof = interface_velocity_from_control_points([0.5, 0.3], r_mat, z_mat)
    prof[r_mat.shape[0] // 2:] = 0.0     # truncate to zero past the midpoint

    psi, _, _, _ = reconstruct_flow_field_from_profile(prof, r_mat, z_mat, metrics, 50.0)
    assert np.allclose(psi[:, -1], 0.0, atol=1e-14)


def test_profile_solve_rejects_a_wrong_length_profile(mesh):
    r_mat, z_mat = mesh
    metrics = compute_mesh_metrics(r_mat, z_mat)
    with pytest.raises(ValueError, match="one value per interface node"):
        reconstruct_flow_field_from_profile(np.zeros(3), r_mat, z_mat, metrics, 50.0)


def test_interface_psi_zero_normal_velocity_is_impermeable(mesh):
    """u_n = 0 must reproduce the impermeable condition psi = 0 along the interface."""
    r_mat, z_mat = mesh
    from pyinnerbubble.assimilation import interface_psi_from_normal_velocity
    psi_s = interface_psi_from_normal_velocity(r_mat, z_mat, np.zeros(r_mat.shape[0]))
    assert np.allclose(psi_s, 0.0)


def test_interface_psi_closes_on_the_axis(mesh):
    """A volume-conserving u_n gives psi that starts AND ends at 0 on the symmetry axis."""
    r_mat, z_mat = mesh
    from pyinnerbubble.assimilation import interface_psi_from_normal_velocity
    phi = np.linspace(0.0, np.pi, r_mat.shape[0])
    u_n = 0.1 * (3 * np.cos(phi) ** 2 - 1)      # P2 deformation mode: zero net volume flux
    psi_s = interface_psi_from_normal_velocity(r_mat, z_mat, u_n, check=False)
    assert psi_s[0] == 0.0
    assert psi_s[-1] == pytest.approx(0.0, abs=1e-12)
    assert np.max(np.abs(psi_s)) > 1e-4         # non-trivial in between


def test_profile_solve_with_zero_psi_interface_matches_default(mesh):
    """Passing an all-zero psi_interface is identical to the impermeable default."""
    r_mat, z_mat = mesh
    metrics = compute_mesh_metrics(r_mat, z_mat)
    prof = interface_velocity_from_control_points([0.5, 0.3], r_mat, z_mat)
    a = reconstruct_flow_field_from_profile(prof, r_mat, z_mat, metrics, 50.0)
    b = reconstruct_flow_field_from_profile(prof, r_mat, z_mat, metrics, 50.0,
                                            psi_interface=np.zeros(r_mat.shape[0]))
    for x, y in zip(a, b):
        assert np.allclose(x, y)
