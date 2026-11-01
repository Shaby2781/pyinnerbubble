"""Data assimilation for interfacial velocities (paper §IIIC).

Maps measured velocities onto body-fitted curvilinear (n_hat, s_hat)
coordinates and solves an inverse problem -- fitting a spline-parameterized
interfacial boundary velocity to an axisymmetric Navier-Stokes solve -- to
reconstruct velocities near the interface and interpolate the bulk field to
higher resolution.
"""

from .conformal_mapping import (
    ConformalMapResult,
    build_conformal_map,
    load_conformal_map,
    plot_conformal_map,
    save_conformal_map,
)
from .flow_reconstruction import (
    AssimilationResult,
    assimilate_boundary_velocity,
    assimilate_boundary_velocity_with_moving_surface,
    boundary_velocity_profile,
    interface_arclength,
    interface_psi_from_normal_velocity,
    interface_velocity_from_control_points,
    compute_mesh_metrics,
    interpolate_measured_field_to_mesh,
    plot_assimilation,
    plot_measured_vs_reconstructed_field,
    plot_velocity_field_on_mesh,
    reconstruct_flow_field,
    reconstruct_flow_field_from_profile,
)
from .mesh import (
    BubbleMeshResult,
    build_bubble_mesh,
    build_disk_mesh,
    load_mesh,
    plot_mesh,
    save_mesh,
)

__all__ = [
    "ConformalMapResult",
    "build_conformal_map",
    "save_conformal_map",
    "load_conformal_map",
    "plot_conformal_map",
    "BubbleMeshResult",
    "build_bubble_mesh",
    "build_disk_mesh",
    "save_mesh",
    "load_mesh",
    "plot_mesh",
    "AssimilationResult",
    "assimilate_boundary_velocity",
    "assimilate_boundary_velocity_with_moving_surface",
    "interface_arclength",
    "interface_psi_from_normal_velocity",
    "interface_velocity_from_control_points",
    "compute_mesh_metrics",
    "reconstruct_flow_field",
    "reconstruct_flow_field_from_profile",
    "interpolate_measured_field_to_mesh",
    "boundary_velocity_profile",
    "plot_assimilation",
    "plot_measured_vs_reconstructed_field",
    "plot_velocity_field_on_mesh",
]
