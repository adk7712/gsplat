"""NeRF-Synthetic (Blender) loader for research/train.py.

NEW file: gsplat's examples/ only parse COLMAP / NCore data, so there is no
transforms.json loader upstream. Conventions follow the 3DGS reference code
(scene/dataset_readers.py::readCamerasFromTransforms):
  * RGBA PNGs composited onto a WHITE background: rgb * a + 1 * (1 - a).
  * camera_angle_x -> fx = fy = 0.5 * W / tan(0.5 * camera_angle_x), principal
    point at the image centre.
  * Blender/OpenGL camera-to-world -> OpenCV by negating the y and z camera axes.
  * Splits: transforms_train.json (100 views) for training,
    transforms_test.json (200 views) for evaluation. Val is unused, as in 3DGS.
  * Full 800x800 resolution.
"""

import json
import math
import os
from typing import Dict

import imageio.v2 as imageio
import numpy as np
import torch


def load_split(scene_dir: str, split: str, white_bg: bool = True) -> Dict[str, torch.Tensor]:
    """Returns {"images": uint8 [N, H, W, 3], "camtoworlds": [N, 4, 4] (OpenCV),
    "K": [3, 3], "names": list[str]}."""
    with open(os.path.join(scene_dir, f"transforms_{split}.json")) as f:
        meta = json.load(f)

    images, c2ws, names = [], [], []
    for frame in meta["frames"]:
        path = os.path.join(scene_dir, frame["file_path"])
        if not path.endswith(".png"):
            path += ".png"
        img = imageio.imread(path).astype(np.float32) / 255.0  # [H, W, 4]
        if img.shape[-1] == 4:
            rgb, a = img[..., :3], img[..., 3:4]
            bg = 1.0 if white_bg else 0.0
            img = rgb * a + bg * (1.0 - a)
        # Round back to uint8 to keep 100 x 800 x 800 images cheap on the GPU;
        # 3DGS does the same (PIL RGBA -> uint8 RGB) before computing the loss.
        images.append(np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8))
        c2w = np.array(frame["transform_matrix"], dtype=np.float32)
        c2w[:3, 1:3] *= -1  # OpenGL (x right, y up, z back) -> OpenCV
        c2ws.append(c2w)
        names.append(os.path.basename(path))

    H, W = images[0].shape[:2]
    focal = 0.5 * W / math.tan(0.5 * float(meta["camera_angle_x"]))
    K = torch.tensor(
        [[focal, 0.0, W / 2.0], [0.0, focal, H / 2.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    return {
        "images": torch.from_numpy(np.stack(images)),
        "camtoworlds": torch.from_numpy(np.stack(c2ws)),
        "K": K,
        "names": names,
    }


def scene_extent(camtoworlds: torch.Tensor) -> float:
    """3DGS ``cameras_extent``: 1.1 x max distance of camera centres from their mean.
    Scales the means lr and the densification thresholds (gsplat's scene_scale)."""
    centers = camtoworlds[:, :3, 3]
    return float((centers - centers.mean(0)).norm(dim=-1).max()) * 1.1
