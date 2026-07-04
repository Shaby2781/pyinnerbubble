# checkpoints/

Place downloaded SAM model checkpoints here (not tracked by git -- these are
multi-GB files). The example scripts expect `sam_vit_h_4b8939.pth` in this
folder by default.

Download from Meta's segment-anything repo:

| Model type | Direct download URL |
|---|---|
| vit_h (default, most accurate) | https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth |
| vit_l (medium) | https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth |
| vit_b (smallest, fastest) | https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth |

See also: https://github.com/facebookresearch/segment-anything#model-checkpoints
for the authoritative, up-to-date list.
