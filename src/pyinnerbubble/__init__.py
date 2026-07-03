"""pyInnerBubble: internal velocity fields and scalar transport for rising bubbles.

Pipeline stages (see submodules):
    geometry      -- bubble shape reconstruction and ray-tracing (camera <-> bubble plane)
    opticflow     -- velocity eduction from PLIF image sequences
    assimilation  -- curvilinear-coordinate data assimilation for interfacial velocities
    transport     -- scalar advection-diffusion and Sherwood number calculation
"""

__version__ = "0.1.0"
