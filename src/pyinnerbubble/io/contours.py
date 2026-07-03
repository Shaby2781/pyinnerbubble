"""Loading and saving (x, y) contour point arrays as plain text."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def save_contour(contour: np.ndarray, path: str | Path) -> None:
    np.savetxt(str(path), contour)


def load_contour(path: str | Path) -> np.ndarray:
    return np.loadtxt(str(path))
