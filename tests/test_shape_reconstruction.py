import numpy as np

from pyinnerbubble.geometry.shape_reconstruction import (
    _exp_derivative,
    _exp_value,
    find_intersections_shapely,
    rotate_curve,
)


def test_rotate_curve_zero_angle_preserves_points():
    points = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 4.0], [0.0, 4.0]])

    mean_r, mean_z, rotated = rotate_curve(points, angle=0.0)

    assert mean_r == 1.0
    assert mean_z == 2.0
    np.testing.assert_allclose(rotated, points, atol=1e-10)


def test_rotate_curve_90_degrees():
    points = np.array([[0.0, 0.0], [2.0, 0.0]])

    _, _, rotated = rotate_curve(points, angle=np.pi / 2)

    # A point to the right of center should move above center after a +90 deg rotation
    # (rotate_curve's convention: rot_r = r*cos(a) + z*sin(a), rot_z = -r*sin(a) + z*cos(a))
    np.testing.assert_allclose(rotated[1], [1.0, -1.0], atol=1e-10)


def test_find_intersections_shapely_crossing_lines():
    curve1 = np.array([[0.0, 0.0], [2.0, 2.0]])
    curve2 = np.array([[0.0, 2.0], [2.0, 0.0]])

    points, idx1, idx2 = find_intersections_shapely(curve1, curve2)

    np.testing.assert_allclose(points, [[1.0, 1.0]])
    assert idx1[0] == 0
    assert idx2[0] == 0


def test_find_intersections_shapely_no_crossing():
    curve1 = np.array([[0.0, 0.0], [1.0, 0.0]])
    curve2 = np.array([[0.0, 5.0], [1.0, 5.0]])

    points, idx1, idx2 = find_intersections_shapely(curve1, curve2)

    assert points == []
    assert idx1 == []
    assert idx2 == []


def test_exp_value_and_derivative_at_reference():
    # At z == z_ref, exp term is 1, so value == a + c and derivative == a * b
    assert _exp_value(z=5.0, a=2.0, b=0.5, c=1.0, z_ref=5.0) == 3.0
    assert _exp_derivative(z=5.0, a=2.0, b=0.5, c=1.0, z_ref=5.0) == 1.0
