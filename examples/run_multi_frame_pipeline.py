"""Multi-frame pipeline: segmentation (sequential) -> reconstruction + mapping (parallel).

SAM segmentation shares one loaded model instance and runs sequentially across
frames -- loading a second vit_h model in a parallel worker would blow memory.
Shape reconstruction and camera mapping are pure CPU-bound numerics with no
shared model, so those run one process per frame via ProcessPoolExecutor.

Edit the CONFIG block below (in particular FRAMES, one entry per frame), then run:

    python examples/run_multi_frame_pipeline.py
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from pyinnerbubble.geometry import (
    CropParams,
    build_camera_plane_mapping,
    load_predictor,
    reconstruct_bubble_shape,
    segment_largest_contour,
)
from pyinnerbubble.opticflow import FarnebackParams, compute_optic_flow, plot_flow_quiver

# ---------------------------------------------------------------------------
# CONFIG -- one entry per frame
# ---------------------------------------------------------------------------
SAM_CHECKPOINT = r"C:\royi\temp_python_prog\meta\sam_vit_h_4b8939.pth"
SAM_MODEL_TYPE = "vit_h"
OUT_DIR = r"C:\royi\Clude_python_lib\output"
MAX_WORKERS = 4

FRAMES = {
    24: dict(
        shadowgraph_image=r"C:\royi\Clude_python_lib\shadowgraph0024.jpg",
        plif_image=r"C:\royi\Clude_python_lib\piv0024.jpg",
        shadowgraph_foreground=np.array([[614, 640]]),
        shadowgraph_background=np.array([[320, 192]]),
        plif_foreground=np.array([[995, 850]]),
        plif_background=np.array([[995, 995]]),
    ),
    25: dict(
        shadowgraph_image=r"C:\royi\Clude_python_lib\shadowgraph0025.jpg",
        plif_image=r"C:\royi\Clude_python_lib\piv0025.jpg",
        shadowgraph_foreground=np.array([[614, 617]]),
        shadowgraph_background=np.array([[320, 169]]),
        plif_foreground=np.array([[995, 827]]),
        plif_background=np.array([[995, 972]]),
    ),
}

VELOCITY_PX_PER_FRAME = 23.0
ANGLE_TO_ROTATE_DEG = -1.0
MU_OIL_GAS = 1.405

# Matches the MATLAB opticalFlowFarneback config this project was validated with:
# NumPyramidLevels=3, PyramidScale=0.5, NumIterations=3, NeighborhoodSize=1, FilterSize=15.
# poly_sigma has no MATLAB-side equivalent that was set -- double check this value.
FARNEBACK_PARAMS = FarnebackParams(
    num_pyramid_levels=3, pyramid_scale=0.5, num_iterations=3,
    neighborhood_size=1, filter_size=15, poly_sigma=1.1,
)

PLOT = True  # each worker SAVES figures to its frame folder -- can't show() across processes
# ---------------------------------------------------------------------------


def _segment_frame(frame_number, config, predictor):
    """SAM segmentation for one frame. Sequential only: predictor.set_image() is
    stateful and the model is too large to duplicate across worker processes."""
    print(f"[frame {frame_number}] segmenting shadowgraph + PLIF...")
    shadowgraph_image = cv2.imread(config["shadowgraph_image"])
    plif_image = cv2.imread(config["plif_image"])
    if shadowgraph_image is None:
        raise FileNotFoundError(config["shadowgraph_image"])
    if plif_image is None:
        raise FileNotFoundError(config["plif_image"])

    shadowgraph_rgb = cv2.cvtColor(shadowgraph_image, cv2.COLOR_BGR2RGB)
    shadowgraph_contour, _, shadow_score = segment_largest_contour(
        shadowgraph_rgb, config["shadowgraph_foreground"], config["shadowgraph_background"], predictor
    )
    plif_rgb = cv2.cvtColor(plif_image, cv2.COLOR_BGR2RGB)
    plif_contour, _, plif_score = segment_largest_contour(
        plif_rgb, config["plif_foreground"], config["plif_background"], predictor
    )
    print(f"[frame {frame_number}] shadowgraph score={shadow_score:.3f}, plif score={plif_score:.3f}")

    return {
        "shadowgraph_image": shadowgraph_image,
        "plif_image": plif_image,
        "shadowgraph_contour": shadowgraph_contour,
        "plif_contour": plif_contour,
    }


def _process_frame_geometry(frame_number, frame_data, velocity, angle_deg, mu_oil_gas, out_dir, plot, crop_size=None):
    """Reconstruction + camera mapping for one frame -- pure numerics, no SAM/torch,
    safe to run in a worker process.

    If crop_size is given, the crop window is fixed to that size instead of being
    auto-computed from this frame alone, so this frame's cropped images stay
    pixel-aligned with whichever frame produced crop_size (needed before optic
    flow can meaningfully compare two frames' cropped PLIF images).
    """
    frame_dir = Path(out_dir) / f"frame_{frame_number:04d}"
    frame_dir.mkdir(parents=True, exist_ok=True)
    plot_arg = frame_dir if plot else False
    crop = CropParams(crop_size=crop_size) if crop_size is not None else None

    shape_result = reconstruct_bubble_shape(
        frame_data["shadowgraph_image"], frame_data["plif_image"],
        frame_data["shadowgraph_contour"], frame_data["plif_contour"],
        velocity_px_per_frame=velocity,
        angle_to_rotate=np.deg2rad(angle_deg),
        mu_oil_gas=mu_oil_gas,
        crop=crop,
        plot=plot_arg,
    )
    np.savetxt(frame_dir / "bubble_contour.txt", shape_result.real_shape)
    np.savetxt(frame_dir / "inner_region_contour.txt", shape_result.inner_region)
    cv2.imwrite(str(frame_dir / "cut_plif.jpg"), shape_result.plif_image_cropped)
    cv2.imwrite(str(frame_dir / "cut_shadowgraph.jpg"), shape_result.shadowgraph_image_cropped)

    mapping_result = build_camera_plane_mapping(
        shape_result.plif_image_cropped, shape_result.real_shape, shape_result.inner_region,
        mean_x=shape_result.mean_r, mu_oil_gas=mu_oil_gas, plot=plot_arg,
    )
    np.savetxt(frame_dir / "x_mat.txt", mapping_result.physical_x)
    np.savetxt(frame_dir / "y_mat.txt", mapping_result.physical_z)
    np.savetxt(frame_dir / "bubble_shape_tot.txt", mapping_result.bubble_shape_closed)

    print(f"[frame {frame_number}] done -> {frame_dir}")
    return frame_number, shape_result


def _compute_optic_flow_between_frames(frame_a, frame_b, plif_a_cropped, plif_b_cropped, out_dir):
    """Optic flow between two consecutive PLIF frames' cropped, pixel-aligned images."""
    print(f"[opticflow {frame_a}->{frame_b}] computing Farneback flow...")
    flow = compute_optic_flow(plif_a_cropped, plif_b_cropped, FARNEBACK_PARAMS)

    flow_dir = Path(out_dir) / f"opticflow_{frame_a:04d}_{frame_b:04d}"
    flow_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(flow_dir / "u_x.txt", flow[..., 0])
    np.savetxt(flow_dir / "u_y.txt", flow[..., 1])
    plot_flow_quiver(plif_a_cropped, flow, save_path=flow_dir / "quiver.png")
    print(f"[opticflow {frame_a}->{frame_b}] done -> {flow_dir}")


def main():
    predictor = load_predictor(SAM_CHECKPOINT, model_type=SAM_MODEL_TYPE)

    frame_data = {
        frame_number: _segment_frame(frame_number, config, predictor)
        for frame_number, config in FRAMES.items()
    }

    frame_numbers = sorted(frame_data)
    reference_frame, other_frames = frame_numbers[0], frame_numbers[1:]

    # Reference frame runs first (sequentially) to establish the shared crop window
    # that every other frame will reuse, so all frames' cropped images stay aligned.
    _, reference_shape_result = _process_frame_geometry(
        reference_frame, frame_data[reference_frame],
        VELOCITY_PX_PER_FRAME, ANGLE_TO_ROTATE_DEG, MU_OIL_GAS, OUT_DIR, PLOT,
    )
    shared_crop_size = reference_shape_result.crop_size
    shape_results = {reference_frame: reference_shape_result}

    if other_frames:
        with ProcessPoolExecutor(max_workers=min(MAX_WORKERS, len(other_frames))) as executor:
            futures = {
                frame_number: executor.submit(
                    _process_frame_geometry, frame_number, frame_data[frame_number],
                    VELOCITY_PX_PER_FRAME, ANGLE_TO_ROTATE_DEG, MU_OIL_GAS, OUT_DIR, PLOT, shared_crop_size,
                )
                for frame_number in other_frames
            }
            for frame_number, future in futures.items():
                _, shape_result = future.result()
                shape_results[frame_number] = shape_result

    for frame_a, frame_b in zip(frame_numbers, frame_numbers[1:]):
        _compute_optic_flow_between_frames(
            frame_a, frame_b,
            shape_results[frame_a].plif_image_cropped, shape_results[frame_b].plif_image_cropped,
            OUT_DIR,
        )

    print("All frames processed.")


if __name__ == "__main__":
    main()
