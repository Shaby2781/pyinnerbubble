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
    build_camera_plane_mapping,
    load_predictor,
    reconstruct_bubble_shape,
    segment_largest_contour,
)

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


def _process_frame_geometry(frame_number, frame_data, velocity, angle_deg, mu_oil_gas, out_dir, plot):
    """Reconstruction + camera mapping for one frame -- pure numerics, no SAM/torch,
    safe to run in a worker process."""
    frame_dir = Path(out_dir) / f"frame_{frame_number:04d}"
    frame_dir.mkdir(parents=True, exist_ok=True)
    plot_arg = frame_dir if plot else False

    shape_result = reconstruct_bubble_shape(
        frame_data["shadowgraph_image"], frame_data["plif_image"],
        frame_data["shadowgraph_contour"], frame_data["plif_contour"],
        velocity_px_per_frame=velocity,
        angle_to_rotate=np.deg2rad(angle_deg),
        mu_oil_gas=mu_oil_gas,
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
    return frame_number


def main():
    predictor = load_predictor(SAM_CHECKPOINT, model_type=SAM_MODEL_TYPE)

    frame_data = {
        frame_number: _segment_frame(frame_number, config, predictor)
        for frame_number, config in FRAMES.items()
    }

    with ProcessPoolExecutor(max_workers=min(MAX_WORKERS, len(FRAMES))) as executor:
        futures = [
            executor.submit(
                _process_frame_geometry, frame_number, data,
                VELOCITY_PX_PER_FRAME, ANGLE_TO_ROTATE_DEG, MU_OIL_GAS, OUT_DIR, PLOT,
            )
            for frame_number, data in frame_data.items()
        ]
        for future in futures:
            future.result()

    print("All frames processed.")


if __name__ == "__main__":
    main()
