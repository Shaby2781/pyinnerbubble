# pyInnerBubble

Measurement of internal velocity fields and scalar transport in rising bubbles from PLIF (planar laser-induced fluorescence) and shadowgraphy imagery.

This library implements the method described in:

> R. Shabtay and I. Jacobi, "Internal Velocity Measurements and Scalar Transport Simulations in Rising Bubbles" (under review).

## Pipeline

| Stage | Module | Paper section |
|---|---|---|
| Bubble shape reconstruction & ray-tracing | `pyinnerbubble.geometry` | §III A |
| Velocity eduction (optic flow) | `pyinnerbubble.opticflow` | §III B |
| Data assimilation of interfacial velocities | `pyinnerbubble.assimilation` | §III C |
| Scalar transport / Sherwood number | `pyinnerbubble.transport` | §IV |

## Status

Work in progress — scripts are being migrated from research code into this package one module at a time.

## Installation

```bash
pip install -e ".[dev]"
```

## License

MIT
