# pyInnerBubble

Measurement of internal velocity fields and scalar transport in rising bubbles from PLIF (planar laser-induced fluorescence) and shadowgraphy imagery.

This library implements the method described in:

> R. Shabtay and I. Jacobi, "Internal Velocity Measurements and Scalar Transport Simulations in Rising Bubbles" (under review).

## Pipeline

| Stage | Module | Paper section |
|---|---|---|
| Bubble shape reconstruction & ray-tracing | `pyinnerbubble.geometry` | §III A |
| Velocity eduction (optic flow) | `pyinnerbubble.opticflow` | §III B |

Data assimilation of interfacial velocities (§III C) and scalar transport / Sherwood
number (§IV) are described in the paper but not yet implemented in this library.

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

[`examples/run_multi_frame_pipeline.py`](examples/run_multi_frame_pipeline.py) runs the full
pipeline end to end on a pair of frames (the four `.jpg` images are included in `examples/`):
SAM segmentation, shape reconstruction, camera-plane mapping, Farneback optic flow, and the
final physical velocity field plot. Edit the `CONFIG` block at the top (frame image paths, SAM
point prompts, calibration constants), download the SAM checkpoint into `checkpoints/` (see
above), then run it directly. Run `pytest` from the repo root to run the test suite too.

Each pipeline stage also has its own CLI, installed by `pip install` (no clone needed):
`pyinnerbubble-segment`, `pyinnerbubble-reconstruct-shape`, `pyinnerbubble-camera-mapping`,
`pyinnerbubble-map-velocity`. Run any of them with `--help` for its full argument list.

## Documentation

[`docs/documentation.docx`](docs/documentation.docx) has the full function-by-function
reference (signatures, parameters, return values) for every module, plus setup/installation
instructions.

## License

MIT
