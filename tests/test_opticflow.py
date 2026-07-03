import numpy as np

from pyinnerbubble.opticflow import FarnebackParams, compute_optic_flow


def _textured_image(size=200):
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, size=(size, size), dtype=np.uint8)
    # Smooth it so Farneback has gradients to lock onto, not pure noise.
    import cv2
    return cv2.GaussianBlur(image, (9, 9), sigmaX=3)


def test_compute_optic_flow_recovers_known_shift():
    prev = _textured_image()
    dx, dy = 4, -2
    next_image = np.roll(np.roll(prev, dy, axis=0), dx, axis=1)

    flow = compute_optic_flow(prev, next_image, FarnebackParams(filter_size=25))

    margin = 30
    interior = flow[margin:-margin, margin:-margin]
    assert np.abs(interior[..., 0].mean() - dx) < 1.0
    assert np.abs(interior[..., 1].mean() - dy) < 1.0


def test_compute_optic_flow_output_shape():
    prev = _textured_image(size=64)
    flow = compute_optic_flow(prev, prev)

    assert flow.shape == (64, 64, 2)
