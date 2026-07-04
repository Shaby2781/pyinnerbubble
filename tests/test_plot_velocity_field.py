import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from pyinnerbubble.opticflow import plot_velocity_field  # noqa: E402


def test_quiver_arrow_direction_matches_data_after_yaxis_inversion():
    # Regression test: plot_velocity_field calls ax.invert_yaxis() (to match image
    # row-increases-downward convention), which requires angles="xy" on its quiver
    # call -- the default angles="uv" ignores axis inversion and renders v upside
    # down. This inspects the actual Quiver artist plot_velocity_field creates.
    h, w = 10, 10
    row_idx, col_idx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    x_mat = col_idx.astype(float)
    y_mat = row_idx.astype(float)

    u = np.zeros((h, w))
    v = np.full((h, w), 5.0)  # "downward" in image/row convention

    plot_velocity_field(x_mat, y_mat, u, v, quiver_step=5)  # no save_path -> plt.show() branch, no-op under Agg

    fig = plt.gcf()
    ax = fig.axes[0]
    quiver_artists = [c for c in ax.collections if type(c).__name__ == "Quiver"]
    assert len(quiver_artists) == 1
    quiver = quiver_artists[0]
    assert quiver.angles == "xy"

    # With angles="xy" and an inverted y-axis, a positive v (downward in data/image
    # terms) must render with the tip below the tail in display (screen) space.
    fig.canvas.draw()
    tail_display = ax.transData.transform((x_mat[0, 0], y_mat[0, 0]))
    tip_display = ax.transData.transform((x_mat[0, 0], y_mat[0, 0] + v[0, 0]))
    assert tip_display[1] < tail_display[1]  # smaller display-y == lower on screen

    plt.close(fig)


def test_bubble_contour_drawn_as_thick_black_line():
    h, w = 10, 10
    row_idx, col_idx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    x_mat = col_idx.astype(float)
    y_mat = row_idx.astype(float)
    u = np.zeros((h, w))
    v = np.zeros((h, w))
    contour = np.array([[1.0, 1.0], [8.0, 1.0], [8.0, 8.0], [1.0, 8.0]])

    plot_velocity_field(x_mat, y_mat, u, v, quiver_step=5, bubble_contour=contour)

    fig = plt.gcf()
    ax = fig.axes[0]
    lines = ax.get_lines()
    assert len(lines) == 1
    line = lines[0]
    assert line.get_color() == "black"
    assert line.get_linewidth() == 2.5
    np.testing.assert_allclose(line.get_xydata(), contour)

    plt.close(fig)
