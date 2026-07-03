"""Data assimilation for interfacial velocities (paper §IIIC).

Maps measured velocities onto body-fitted curvilinear (n_hat, s_hat)
coordinates and solves an inverse problem -- fitting a spline-parameterized
interfacial boundary velocity to an axisymmetric Navier-Stokes solve -- to
reconstruct velocities near the interface and interpolate the bulk field to
higher resolution.
"""
