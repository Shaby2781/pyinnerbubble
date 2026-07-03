import numpy as np

from pyinnerbubble.opticflow import (
    map_camera_flow_to_physical,
    savgol_2d_kernel,
    savgol_2d_smooth,
    velocity_to_physical_units,
)


def test_map_camera_flow_to_physical_linear_mapping():
    # axis 0 is "eta", axis 1 is "xsi" (matches build_camera_plane_mapping's convention).
    # x_mat depends only on axis 1 ("xsi"): x_xsi=2, x_eta=0.
    # y_mat depends only on axis 0 ("eta"): y_xsi=0, y_eta=3.
    axis0, axis1 = np.mgrid[0:20, 0:20].astype(float)
    x_mat = 2.0 * axis1
    y_mat = 3.0 * axis0

    flow = np.zeros((20, 20, 2))
    flow[..., 0] = 5.0
    flow[..., 1] = 7.0

    u_phys, v_phys = map_camera_flow_to_physical(flow, x_mat, y_mat)

    interior = np.s_[2:-2, 2:-2]  # avoid edge effects from np.gradient
    np.testing.assert_allclose(u_phys[interior], 5.0 * 2.0 + 7.0 * 0, atol=1e-8)
    np.testing.assert_allclose(v_phys[interior], 5.0 * 0 + 7.0 * 3.0, atol=1e-8)


def test_savgol_2d_kernel_smoothing_sums_to_one():
    kernel = savgol_2d_kernel(radius=7, order=2, deriv=0)

    assert kernel.shape == (15, 15)
    assert abs(kernel.sum() - 1.0) < 1e-8


def test_savgol_2d_kernel_recovers_exact_separable_quadratic():
    # The filter fits a *separable* basis (no xy cross term, matching sgsdf_2d's
    # flag_coupling=0), so it must recover a noiseless separable quadratic
    # exactly at the center -- but NOT a surface with a genuine xy term.
    radius = 7
    x, y = np.mgrid[-radius:radius + 1, -radius:radius + 1].astype(float)
    surface = 2.0 + 3.0 * x - 1.5 * y + 0.5 * x ** 2 - 0.75 * y ** 2

    kernel = savgol_2d_kernel(radius, order=2)
    smoothed_center = np.sum(kernel * surface)

    assert abs(smoothed_center - surface[radius, radius]) < 1e-8


def test_savgol_2d_kernel_derivative_recovers_slope():
    # kernel's "x" is axis 0 of the returned array.
    radius = 7
    x, y = np.mgrid[-radius:radius + 1, -radius:radius + 1].astype(float)
    surface = 1.0 + 2.0 * x - 3.0 * y  # d/dx = 2, d/dy = -3 everywhere

    kernel_dx = savgol_2d_kernel(radius, order=2, deriv=1)
    dx_center = np.sum(kernel_dx * surface)
    assert abs(dx_center - 2.0) < 1e-8

    # sgsdf_2d obtains its y-derivative kernel via transposing the x-derivative
    # kernel (valid only because x/y ranges and orders match here).
    kernel_dy = kernel_dx.T
    dy_center = np.sum(kernel_dy * surface)
    assert abs(dy_center - (-3.0)) < 1e-8


def test_savgol_2d_smooth_preserves_nan_mask():
    field = np.random.default_rng(0).normal(size=(40, 40))
    field[10:15, 10:15] = np.nan

    smoothed = savgol_2d_smooth(field, radius=3, order=2)

    assert np.all(np.isnan(smoothed[10:15, 10:15]))
    assert not np.any(np.isnan(smoothed[20:25, 20:25]))


def test_velocity_to_physical_units():
    # 38 px/frame at 400 Hz, 38 px/mm -> 400 mm/s = 0.4 m/s.
    u = np.array([38.0])
    v = np.array([0.0])

    u_m_s, v_m_s = velocity_to_physical_units(u, v, px_per_mm=38.0, frame_rate_hz=400.0)

    np.testing.assert_allclose(u_m_s, [0.4])
    np.testing.assert_allclose(v_m_s, [0.0])
