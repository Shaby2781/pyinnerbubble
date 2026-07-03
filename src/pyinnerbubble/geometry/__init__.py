"""Bubble shape reconstruction and ray-tracing (paper §IIIA).

Reconstructs the 3D bubble interface from shadowgraph (cap, via ellipse +
exponential fits) and PLIF (base, via Snell's-law projection) segmentations,
and builds the camera-plane <-> bubble-plane ray-traced mapping.
"""
