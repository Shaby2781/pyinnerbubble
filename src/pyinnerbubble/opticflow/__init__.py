"""Velocity eduction from PLIF imagery (paper §IIIB).

Applies optic flow (Farneback) to PLIF image time-series in the camera plane
and transforms the result into the physical (r, z) bubble-plane velocity
field using the geometry module's ray-traced mapping.
"""

from .farneback import FarnebackParams, compute_optic_flow, plot_flow_quiver

__all__ = ["FarnebackParams", "compute_optic_flow", "plot_flow_quiver"]
