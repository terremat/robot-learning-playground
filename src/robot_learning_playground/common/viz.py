import numpy as np
from PIL import Image


def save_depth(depth, filename):
    # Normalize depth for visualization
    depth_vis = (depth - depth.min()) / ( depth.max() - depth.min() + 1e-8 )

    Image.fromarray( (depth_vis * 255).astype(np.uint8) ).save(filename)

def save_segmentation(segmentation, filename):
    # Segmentation using random colors to distinguish IDs
    rng = np.random.default_rng(42)

    object_ids = segmentation[..., 0]
    unique_ids = np.unique(object_ids)

    seg_vis = np.zeros((*object_ids.shape, 3), dtype=np.uint8)

    for obj_id in unique_ids:
        if obj_id >= 0:
            seg_vis[object_ids == obj_id] = rng.integers(
                0, 256, size=3, dtype=np.uint8
            )
    Image.fromarray(seg_vis).save(filename)