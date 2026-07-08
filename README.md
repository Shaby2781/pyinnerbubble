# pyInnerBubble

Measurement of internal velocity fields and scalar transport in rising bubbles from PLIF (planar laser-induced fluorescence) and shadowgraphy imagery.

This library implements the method described in:

> R. Shabtay and I. Jacobi, "Internal Velocity Measurements and Scalar Transport Simulations in Rising Bubbles" (under review).

## Pipeline

| Stage | Module | Paper section |
|---|---|---|
| Bubble shape reconstruction & ray-tracing | `pyinnerbubble.geometry` | §III A |
| Velocity eduction (optic flow) | `pyinnerbubble.opticflow` | §III B |
| Data assimilation — conformal map, mesh & inverse Navier-Stokes | `pyinnerbubble.assimilation` | §III C |
| Scalar transport & Sherwood number | `pyinnerbubble.transport` | §IV |

The data assimilation stage covers the body-fitted coordinate setup (the conformal map
between the unit disk and the bubble region, and the graded computational mesh built from
it) and the inverse solve: fitting the interfacial boundary velocity so a steady
axisymmetric Navier-Stokes solution best matches the measured interior velocity field.
The scalar transport stage then solves the axisymmetric advection-diffusion of a passive
scalar on that reconstructed flow field and extracts the time-dependent Sherwood number
over a range of Peclet numbers — completing the full method of the paper end to end.

## Status

Work in progress — scripts are being migrated from research code into this package one module at a time.

## Installation

For users, once this repo is on GitHub:

```bash
pip install git+https://github.com/Shaby2781/pyinnerbubble.git
```

Add the `segmentation` extra (torch + torchvision + segment-anything) if you need the
SAM-based contour segmentation in `pyinnerbubble.geometry.segmentation`. On a CPU-only
machine, install the CPU builds of torch/torchvision first to avoid pulling in several
GB of unneeded CUDA libraries:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

```bash
pip install "pyinnerbubble[segmentation] @ git+https://github.com/Shaby2781/pyinnerbubble.git"
```

Requires Python >=3.10.

**Note:** `pip install` only installs the importable `pyinnerbubble` package (the library
code under `src/`) -- it does **not** include `examples/`, `docs/`, or anything else in the
repo. To try the example script (below), you need to clone the full repo instead.

The SAM model checkpoint (e.g. `sam_vit_h_4b8939.pth`) is not included in this repo
(multi-GB file) -- download it into `checkpoints/` (see
[`checkpoints/README.md`](checkpoints/README.md) for direct links), which is where the
example scripts expect it by default.

## Example

To try the example, clone the full repo (`pip install` alone won't include it):

```bash
git clone https://github.com/Shaby2781/pyinnerbubble.git
cd pyinnerbubble
pip install -e ".[dev,segmentation]"
```

The `examples/` folder is organized by pipeline stage:

- [`examples/bubble_shape_and_optic_flow/run_multi_frame_pipeline.py`](examples/bubble_shape_and_optic_flow/run_multi_frame_pipeline.py)
  runs the shape + velocity pipeline end to end on a pair of frames (the four `.jpg` images live
  alongside it): SAM segmentation, shape reconstruction, camera-plane mapping, Farneback optic
  flow, and the final physical velocity field plot. Edit the `CONFIG` block at the top (frame
  image paths, SAM point prompts, calibration constants) and download the SAM checkpoint into
  `checkpoints/` (see above) before running.
- [`examples/grid_and_data_assimilation/grid_build_example.py`](examples/grid_and_data_assimilation/grid_build_example.py)
  takes a reconstructed bubble boundary (`bubble_shape_tot.txt`, included alongside it) and builds
  the conformal map (unit disk <-> bubble) and the graded computational mesh for the data
  assimilation stage. Edit its `CONFIG` block (reference diameter, grid size/clustering) then run.
- [`examples/grid_and_data_assimilation/example_DA.py`](examples/grid_and_data_assimilation/example_DA.py)
  runs the inverse Navier-Stokes assimilation: it loads a coarse mesh plus the experimental grid and
  ray-traced velocity (all included alongside it), and fits the interfacial boundary velocity to the
  measured interior field, plotting the optimization trace, reconstructed streamlines, and the fitted
  edge-velocity profile. **The bundled mesh is intentionally coarse so it runs on a laptop in a few
  minutes — it is a demonstration, not a research-grade grid** (see the note in the documentation).
- [`examples/scalar_transport/example_scalar_transport.py`](examples/scalar_transport/example_scalar_transport.py)
  takes an assimilation result (`assimilation.npz`, the reconstructed flow field, included alongside it)
  and solves the passive-scalar advection-diffusion for a range of Peclet numbers, producing the
  Sherwood-number-vs-diffusion-time curves, Sherwood-vs-Peclet, and the final scalar-concentration
  fields. Edit its `CONFIG` block (Peclet vector, end time, number of time points) then run.

Run `pytest` from the repo root to run the test suite too.

Each pipeline stage also has its own CLI, installed by `pip install` (no clone needed):
`pyinnerbubble-segment`, `pyinnerbubble-reconstruct-shape`, `pyinnerbubble-camera-mapping`,
`pyinnerbubble-map-velocity`, `pyinnerbubble-conformal-map`, `pyinnerbubble-build-mesh`. Run any
of them with `--help` for its full argument list.

## Documentation

[`docs/documentation.docx`](docs/documentation.docx) has the full function-by-function
reference (signatures, parameters, return values) for every module, plus setup/installation
instructions.

## License

MIT
