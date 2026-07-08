"""Scalar transport and Sherwood number calculation (paper §IV).

Solves the advection-diffusion equation on the assimilated flow field for
the gas-side-limiting scalar transport case, and computes the long-time
Sherwood number as a function of Peclet number.
"""

from .mass_transfer import (
    ScalarTransportResult,
    integrate_curvilinear_cylindrical,
    plot_final_concentration,
    plot_sherwood_vs_pe,
    plot_sherwood_vs_time,
    simulate_scalar_transport,
    surface_area_of_revolution,
)

__all__ = [
    "ScalarTransportResult",
    "simulate_scalar_transport",
    "plot_sherwood_vs_time",
    "plot_sherwood_vs_pe",
    "plot_final_concentration",
    "integrate_curvilinear_cylindrical",
    "surface_area_of_revolution",
]
