# pyInnerBubble

Measurement of internal velocity fields and scalar transport in rising bubbles from PLIF (planar laser-induced fluorescence) and shadowgraphy imagery.

This library implements the method of the accepted article:

> R. Shabtay and I. Jacobi, "Internal Velocity Measurements and Scalar Transport Simulations in Rising Bubbles," accepted for publication (American Physical Society).

The rising bubbles are seeded with a fluorescent aerosol, and their internal velocity fields are
educed by optic flow. Image distortions are corrected by ray tracing through a reconstructed
three-dimensional bubble geometry. Physics-based data assimilation then extrapolates the
missing velocity data near the interface and interpolates the interior field to high
resolution. The measured fields drive scalar-transport simulations that give the Sherwood
number for gas-side-limited transfer.

The method was demonstrated on dimpled spherical-cap bubbles in silicone oil (Morton number
Mo ≈ 1000, gas Reynolds number Re_g = 107–418). The measured fields revealed primary and
secondary toroidal vortices. Transfer grew up to 3.5× between the low- and high-Péclet limits,
compared with 2.7× in classical theory for spherical bubbles. The bundled example data is the
paper's reference Re_g = 301 bubble (d_eq ≈ 22.5 mm, U_b ≈ 21 cm/s). This is the bubble used for
the method figures in §III.

## Pipeline

| Stage | Module | Paper section |
|---|---|---|
| Bubble shape reconstruction & ray-tracing | `pyinnerbubble.geometry` | §III A — Geometric Reconstruction of the Bubble |
| Velocity eduction (Farneback optic flow + Savitzky–Golay smoothing) | `pyinnerbubble.opticflow` | §III B — Velocity Field Measurement in the Bubble Plane |
| Data assimilation — conformal map (Kerzman–Stein), mesh & inverse Navier-Stokes (Powell) | `pyinnerbubble.assimilation` | §III C — Data Assimilation for the Interfacial Velocities |
| Scalar transport & Sherwood number | `pyinnerbubble.transport` | §IV — Scalar Transport |

Validation of each stage (shape reconstruction, optic flow, data assimilation, mass transfer)
is reported in the paper's Appendix A, and the parameters of all measured bubbles are listed in
Appendix B (Table IV).

The data assimilation stage covers the body-fitted coordinate setup (the conformal map
between the unit disk and the bubble region, and the graded computational mesh built from
it) and the inverse solve: fitting the interfacial boundary velocity so a steady
axisymmetric Navier-Stokes solution best matches the measured interior velocity field.
The scalar transport stage then solves the axisymmetric advection-diffusion of a passive
scalar on that reconstructed flow field and extracts the time-dependent Sherwood number
over a range of Peclet numbers — completing the full method of the paper end to end.

The assimilation stage also supports a **deforming (moving) interface**: a measured normal
velocity can be prescribed at the surface (`assimilate_boundary_velocity_with_moving_surface`).
Forward solves can also use an arbitrary interfacial velocity profile
(`reconstruct_flow_field_from_profile`), e.g. to ask what flow a partially mobile interface
would drive.

A workflow diagram of the full pipeline is in [`docs/workflow.tex`](docs/workflow.tex) (TikZ source;
`pdflatex workflow.tex` renders it).

## Status

Alpha (v0.1.0). The full method of the paper is implemented and every tutorial in
`examples/` runs end to end on the bundled data. APIs may still change before 1.0.

## Installation

For users:

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

## Tutorials

To run the tutorials, clone the full repo (`pip install` alone won't include them):

```bash
git clone https://github.com/Shaby2781/pyinnerbubble.git
cd pyinnerbubble
pip install -e ".[dev,segmentation]"
```

The `examples/` folder has one tutorial per pipeline stage. Every input each tutorial needs is
bundled next to it, and paths resolve from the script's own location, so you can launch them from
any directory. Results go to `output/<stage>/` (gitignored). Each stage ships with precomputed
inputs, so any tutorial can also be run on its own.

| # | Tutorial | Typical runtime | Main outputs |
|---|---|---|---|
| 1 | `bubble_shape_and_optic_flow/run_multi_frame_pipeline.py` | dominated by SAM (needs the checkpoint) | `output/frame_00xx/`, `output/opticflow_0024_0025/velocity_field.png` |
| 2 | `grid_and_data_assimilation/grid_build_example.py` | seconds | `output/conformal_map/` (`conformal_map.npz`, `mesh.npz`, previews) |
| 3 | `grid_and_data_assimilation/example_DA.py` | ~4 min | `output/data_assimilation/assimilation.npz`, `measured_vs_reconstructed.png` |
| 4 | `grid_and_data_assimilation/example_DA_no_weighting.py` | ~4 min | `output/data_assimilation/*_no_mat.*` |
| 5 | `scalar_transport/example_scalar_transport.py` | < 1 min | `output/scalar_transport/sherwood_vs_time.png`, `sherwood_vs_pe.png` |

Run them from the repo root, e.g.:

```bash
python examples/grid_and_data_assimilation/grid_build_example.py
python examples/grid_and_data_assimilation/example_DA.py
python examples/scalar_transport/example_scalar_transport.py
```

[`examples/PyInnerBubble_examples.pdf`](examples/PyInnerBubble_examples.pdf) shows the figures each
tutorial is expected to produce. To use your own data, edit only the `CONFIG` block at the top of
each script.

Details:

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
- [`examples/grid_and_data_assimilation/example_DA_no_weighting.py`](examples/grid_and_data_assimilation/example_DA_no_weighting.py)
  runs the same solve with a uniform weight on a single-frame field in the optic-flow stage's own
  text-file format (`x_mat.txt`, `y_mat.txt`, `u_phys.txt`, `v_phys.txt`, all bundled), which shows how to
  chain the optic-flow output straight into the assimilation.
- [`examples/scalar_transport/example_scalar_transport.py`](examples/scalar_transport/example_scalar_transport.py)
  takes an assimilation result (`assimilation.npz`, the reconstructed flow field, included alongside it)
  and solves the passive-scalar advection-diffusion for a range of Peclet numbers, producing the
  Sherwood-number-vs-diffusion-time curves, Sherwood-vs-Peclet, and the final scalar-concentration
  fields. Edit its `CONFIG` block (Peclet vector, end time, number of time points) then run.

Run `pytest` from the repo root to check your installation (the suite runs in a few seconds and
does not need the SAM checkpoint).

Each pipeline stage also has its own CLI, installed by `pip install` (no clone needed):
`pyinnerbubble-segment`, `pyinnerbubble-reconstruct-shape`, `pyinnerbubble-camera-mapping`,
`pyinnerbubble-map-velocity`, `pyinnerbubble-conformal-map`, `pyinnerbubble-build-mesh`. Run any
of them with `--help` for its full argument list.

## Documentation

[`docs/documentation.docx`](docs/documentation.docx) has the full function-by-function
reference (signatures, parameters, return values) for every module, a tutorial overview, and
setup/installation instructions (including running on AWS EC2).

## Citation

If you use pyInnerBubble in your work, please cite the accompanying article:

> R. Shabtay and I. Jacobi, "Internal Velocity Measurements and Scalar Transport Simulations in Rising Bubbles," accepted for publication (American Physical Society).

## Acknowledgments

This work was supported by Israel Science Foundation grant 219/21.

## License

MIT
