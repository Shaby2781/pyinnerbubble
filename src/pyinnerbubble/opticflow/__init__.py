"""Velocity eduction from PLIF imagery (paper §IIIB).

Applies optic flow (Farneback) to PLIF image time-series in the camera plane
and transforms the result into the physical (r, z) bubble-plane velocity
field using the geometry module's ray-traced mapping.
"""

from .farneback import FarnebackParams, compute_optic_flow, plot_flow_quiver
from .velocity_mapping import (
    compute_mapping_derivatives,
    map_camera_flow_to_physical,
    plot_velocity_field,
    savgol_2d_kernel,
    savgol_2d_smooth,
    velocity_to_physical_units,
)

__all__ = [
    "FarnebackParams",
    "compute_optic_flow",
    "plot_flow_quiver",
    "compute_mapping_derivatives",
    "map_camera_flow_to_physical",
    "savgol_2d_kernel",
    "savgol_2d_smooth",
    "velocity_to_physical_units",
    "plot_velocity_field",
]
