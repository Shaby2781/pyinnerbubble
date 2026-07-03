import numpy as np

from pyinnerbubble.geometry.segmentation import mask_to_contour, save_contour


def test_mask_to_contour_square():
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5:15, 5:15] = 1

    contour = mask_to_contour(mask)

    assert contour is not None
    assert contour.shape[1] == 2
    assert contour[:, 0].min() >= 4
    assert contour[:, 0].max() <= 15
    assert contour[:, 1].min() >= 4
    assert contour[:, 1].max() <= 15


def test_mask_to_contour_empty():
    mask = np.zeros((20, 20), dtype=np.uint8)

    assert mask_to_contour(mask) is None


def test_save_contour_roundtrip(tmp_path):
    contour = np.array([[0.0, 0.0], [1.0, 2.0], [3.0, 4.0]])
    out_path = tmp_path / "contour.txt"

    save_contour(contour, out_path)
    loaded = np.loadtxt(out_path)

    np.testing.assert_allclose(loaded, contour)
