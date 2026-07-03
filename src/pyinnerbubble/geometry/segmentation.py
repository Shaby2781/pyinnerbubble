"""SAM-based contour segmentation from foreground/background points.

Used to detect the bubble cap edge from shadowgraph images and the bubble
base edge from PLIF images (paper Sec. III A), by prompting the Segment
Anything Model (SAM) with foreground/background point clicks and extracting
the largest resulting contour.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def load_image_rgb(image_path: str | Path) -> np.ndarray:
    """Load an image from disk and convert it to RGB."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_predictor(checkpoint_path: str | Path, model_type: str = "vit_h", device: str | None = None):
    """Load a SAM model and return a ``SamPredictor`` for it.

    This is separated from ``segment_largest_contour`` because loading the
    model is expensive and should happen once per session, not once per image.
    """
    from segment_anything import sam_model_registry, SamPredictor

    sam = sam_model_registry[model_type](checkpoint=str(checkpoint_path))
    if device is not None:
        sam.to(device=device)
    return SamPredictor(sam)


def mask_to_contour(mask: np.ndarray) -> np.ndarray | None:
    """Extract the largest external contour from a binary mask.

    Returns an (N, 2) array of (x, y) pixel coordinates, or None if the mask
    contains no contours.
    """
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    return largest.reshape(-1, 2).astype(float)


def segment_largest_contour(
    image: np.ndarray,
    foreground_points: np.ndarray,
    background_points: np.ndarray,
    predictor,
) -> tuple[np.ndarray | None, np.ndarray, float]:
    """Run SAM on ``image`` with the given point prompts and return the largest contour.

    Parameters
    ----------
    image : (H, W, 3) RGB array.
    foreground_points, background_points : (K, 2) and (L, 2) arrays of (x, y) pixel coordinates.
    predictor : a ``SamPredictor`` already carrying a loaded model (see ``load_predictor``).

    Returns
    -------
    contour : (N, 2) array of (x, y) points on the largest contour of the best mask, or None.
    mask : the best (highest-score) predicted mask.
    score : the score of the best mask.
    """
    points = np.concatenate([foreground_points, background_points], axis=0)
    labels = np.concatenate(
        [np.ones(len(foreground_points)), np.zeros(len(background_points))]
    )

    predictor.set_image(image)
    masks, scores, _ = predictor.predict(
        point_coords=points,
        point_labels=labels,
        multimask_output=True,
    )
    best = int(np.argmax(scores))
    mask = masks[best]
    score = float(scores[best])
    contour = mask_to_contour(mask)
    return contour, mask, score


def save_contour(contour: np.ndarray, out_path: str | Path) -> None:
    np.savetxt(str(out_path), contour)


def plot_contour(image: np.ndarray, contour: np.ndarray, title: str = "Detected boundary") -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    if contour is not None:
        plt.plot(contour[:, 0], contour[:, 1], linewidth=2, color="red")
    plt.title(title)
    plt.axis("off")
    plt.show()


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Segment a bubble contour from an image using SAM point prompts."
    )
    parser.add_argument("--image", required=True, help="Path to the input image.")
    parser.add_argument("--checkpoint", required=True, help="Path to the SAM checkpoint (.pth).")
    parser.add_argument(
        "--model-type", default="vit_h", choices=["vit_h", "vit_l", "vit_b"],
        help="SAM model architecture matching the checkpoint (default: vit_h).",
    )
    parser.add_argument(
        "--foreground", action="append", nargs=2, type=float, metavar=("X", "Y"),
        required=True, help="A foreground (on-object) point, e.g. --foreground 1120 1219. Repeatable.",
    )
    parser.add_argument(
        "--background", action="append", nargs=2, type=float, metavar=("X", "Y"),
        required=True, help="A background (off-object) point, e.g. --background 1120 1401. Repeatable.",
    )
    parser.add_argument("--out", required=True, help="Path to write the contour (x, y) text file.")
    parser.add_argument("--plot", action="store_true", help="Show the detected contour overlaid on the image.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    image = load_image_rgb(args.image)
    predictor = load_predictor(args.checkpoint, model_type=args.model_type)

    foreground_points = np.array(args.foreground)
    background_points = np.array(args.background)

    contour, mask, score = segment_largest_contour(
        image, foreground_points, background_points, predictor
    )
    if contour is None:
        raise RuntimeError("No contour found in the predicted mask.")

    save_contour(contour, args.out)
    print(f"Saved contour ({contour.shape[0]} points, mask score {score:.3f}) to {args.out}")

    if args.plot:
        plot_contour(image, contour)


if __name__ == "__main__":
    main()
