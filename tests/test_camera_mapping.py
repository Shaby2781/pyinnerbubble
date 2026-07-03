import numpy as np

from pyinnerbubble.geometry.camera_mapping import _mirror_and_close


def test_mirror_and_close_reflects_about_mean_x():
    curve = np.array([[5.0, 0.0], [8.0, 1.0], [5.0, 2.0]])
    mean_x = 5.0

    closed = _mirror_and_close(curve, mean_x)

    # First point repeated at the end to close the loop
    np.testing.assert_allclose(closed[0], closed[-1])
    # The mirrored point for (8, 1) about x=5 is (2, 1)
    assert np.any(np.all(np.isclose(closed, [2.0, 1.0]), axis=1))
    # Original curve points are preserved
    for point in curve:
        assert np.any(np.all(np.isclose(closed, point), axis=1))
