"""Bubble shape reconstruction and ray-tracing (paper §IIIA).

Reconstructs the 3D bubble interface from shadowgraph (cap, via ellipse +
exponential fits) and PLIF (base, via Snell's-law projection) segmentations,
and builds the camera-plane <-> bubble-plane ray-traced mapping.
"""

from .segmentation import mask_to_contour, segment_largest_contour
from .shape_reconstruction import BubbleShapeResult, CropParams, reconstruct_bubble_shape

__all__ = [
    "mask_to_contour",
    "segment_largest_contour",
    "reconstruct_bubble_shape",
    "CropParams",
    "BubbleShapeResult",
]
