import numpy as np

from pyinnerbubble.io import load_contour, save_contour


def test_save_load_contour_roundtrip(tmp_path):
    contour = np.array([[0.0, 0.0], [1.0, 2.0], [3.0, 4.0]])
    out_path = tmp_path / "contour.txt"

    save_contour(contour, out_path)
    loaded = load_contour(out_path)

    np.testing.assert_allclose(loaded, contour)
