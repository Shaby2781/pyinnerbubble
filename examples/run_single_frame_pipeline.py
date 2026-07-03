"""End-to-end single-frame pipeline: segmentation -> shape reconstruction -> camera mapping.

Edit the CONFIG block below to point at your own images/checkpoint/points, then run:

    python examples/run_single_frame_pipeline.py

See the "How to run" instructions from Claude for details on picking the
foreground/background points and the bubble velocity.
"""

import cv2
import numpy as np

from pyinnerbubble.geometry import (
    build_camera_plane_mapping,
    load_predictor,
    reconstruct_bubble_shape,
    segment_largest_contour,
)

# ---------------------------------------------------------------------------
# CONFIG -- edit these for your own images
# ---------------------------------------------------------------------------
SHADOWGRAPH_IMAGE = r"C:\royi\Clude_python_lib\shadowgraph0024.jpg"
PLIF_IMAGE = r"C:\royi\Clude_python_lib\piv0024.jpg"
SAM_CHECKPOINT = r"C:\royi\temp_python_prog\meta\sam_vit_h_4b8939.pth"
SAM_MODEL_TYPE = "vit_h"
OUT_DIR = r"C:\royi\Clude_python_lib\output"

# One point inside the object, one clearly outside it, per image: (x, y) pixels.
SHADOWGRAPH_FOREGROUND = np.array([[614, 640]])
SHADOWGRAPH_BACKGROUND = np.array([[320, 192]])
PLIF_FOREGROUND = np.array([[995, 850]])
PLIF_BACKGROUND = np.array([[995, 995]])

VELOCITY_PX_PER_FRAME = 23.0  # bubble rise speed in px/frame -- measure from consecutive frames, this is a placeholder
ANGLE_TO_ROTATE_DEG = -1.0
MU_OIL_GAS = 1.405

PLOT = True
# ---------------------------------------------------------------------------


def main():
    from pathlib import Path

    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    shadowgraph_image = cv2.imread(SHADOWGRAPH_IMAGE)
    plif_image = cv2.imread(PLIF_IMAGE)
    if shadowgraph_image is None:
        raise FileNotFoundError(SHADOWGRAPH_IMAGE)
    if plif_image is None:
        raise FileNotFoundError(PLIF_IMAGE)

    print("Loading SAM model (this can take a while on CPU)...")
    predictor = load_predictor(SAM_CHECKPOINT, model_type=SAM_MODEL_TYPE)

    print("Segmenting shadowgraph contour...")
    shadowgraph_rgb = cv2.cvtColor(shadowgraph_image, cv2.COLOR_BGR2RGB)
    shadowgraph_contour, _, shadow_score = segment_largest_contour(
        shadowgraph_rgb, SHADOWGRAPH_FOREGROUND, SHADOWGRAPH_BACKGROUND, predictor
    )
    print(f"  mask score={shadow_score:.3f}, {shadowgraph_contour.shape[0]} contour points")

    print("Segmenting PLIF contour...")
    plif_rgb = cv2.cvtColor(plif_image, cv2.COLOR_BGR2RGB)
    plif_contour, _, plif_score = segment_largest_contour(
        plif_rgb, PLIF_FOREGROUND, PLIF_BACKGROUND, predictor
    )
    print(f"  mask score={plif_score:.3f}, {plif_contour.shape[0]} contour points")

    print("Reconstructing bubble shape...")
    shape_result = reconstruct_bubble_shape(
        shadowgraph_image, plif_image, shadowgraph_contour, plif_contour,
        velocity_px_per_frame=VELOCITY_PX_PER_FRAME,
        angle_to_rotate=np.deg2rad(ANGLE_TO_ROTATE_DEG),
        mu_oil_gas=MU_OIL_GAS,
        plot=PLOT,
    )
    np.savetxt(out_dir / "bubble_contour.txt", shape_result.real_shape)
    np.savetxt(out_dir / "inner_region_contour.txt", shape_result.inner_region)
    cv2.imwrite(str(out_dir / "cut_plif.jpg"), shape_result.plif_image_cropped)
    cv2.imwrite(str(out_dir / "cut_shadowgraph.jpg"), shape_result.shadowgraph_image_cropped)

    print("Building camera <-> physical plane mapping...")
    mapping_result = build_camera_plane_mapping(
        shape_result.plif_image_cropped,
        shape_result.real_shape,
        shape_result.inner_region,
        mean_x=shape_result.mean_r,
        mu_oil_gas=MU_OIL_GAS,
        plot=PLOT,
    )
    np.savetxt(out_dir / "x_mat.txt", mapping_result.physical_x)
    np.savetxt(out_dir / "y_mat.txt", mapping_result.physical_z)
    np.savetxt(out_dir / "bubble_shape_tot.txt", mapping_result.bubble_shape_closed)

    print(f"Done. Outputs written to {out_dir}")


if __name__ == "__main__":
    main()
