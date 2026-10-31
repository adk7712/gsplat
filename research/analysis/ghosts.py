"""Test the 'ghost' hypothesis from cnn_fluctuation_report.md on saved Gaussians (CPU).

Hypothesis H1: the CNN's extra error in chair/ship comes from faint (low-opacity) Gaussians in
empty space under the object, which the CNN-smoothed planes tint with the object's colour.

For every <variant>_<scene>_<seed>.npz written by train.py --save_gaussians, this script
classifies Gaussians as inside / outside the object's visual hull (carved from the training
silhouettes, dilated by a margin), and reports for the outside ones: how many, how opaque,
how many sit below the object, and how large their field colour residual is.

  python research/analysis/ghosts.py <dir with npz files> <nerf_synthetic root>
"""

import glob
import json
import math
import os
import sys

import numpy as np
from PIL import Image

G, LO, HI = 128, -1.5, 1.5  # same box and resolution as the field's planes


def visual_hull(scene_dir: str, downsample: int = 4) -> np.ndarray:
    """Occupancy grid [G, G, G] (x, y, z): voxels inside every training silhouette."""
    c = (np.arange(G) + 0.5) / G * (HI - LO) + LO
    X, Y, Z = np.meshgrid(c, c, c, indexing="ij")
    P = np.stack([X, Y, Z, np.ones_like(X)], -1).reshape(-1, 4).astype(np.float32)
    meta = json.load(open(os.path.join(scene_dir, "transforms_train.json")))
    W = 800 // downsample
    f = 0.5 * W / math.tan(0.5 * meta["camera_angle_x"])
    occ = np.ones(len(P), bool)
    for fr in meta["frames"]:
        im = np.asarray(Image.open(os.path.join(scene_dir, fr["file_path"] + ".png")).reduce(downsample))
        mask = im[..., 3] > 12  # alpha > ~0.05
        c2w = np.array(fr["transform_matrix"], np.float32)
        c2w[:3, 1:3] *= -1  # OpenGL -> OpenCV, as in research/blender.py
        pc = P @ np.linalg.inv(c2w).T
        z = pc[:, 2]
        u = f * pc[:, 0] / z + W / 2
        v = f * pc[:, 1] / z + W / 2
        inside = (z > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < W)
        ok = np.zeros(len(P), bool)
        ok[inside] = mask[v[inside].astype(int), u[inside].astype(int)]
        occ &= ok | ~inside
    return occ.reshape(G, G, G)


def dilate(V: np.ndarray, r: int) -> np.ndarray:
    out = V.copy()
    for _ in range(r):
        p = np.pad(out, 1)
        out = out | p[2:, 1:-1, 1:-1] | p[:-2, 1:-1, 1:-1] | p[1:-1, 2:, 1:-1] | p[1:-1, :-2, 1:-1] \
            | p[1:-1, 1:-1, 2:] | p[1:-1, 1:-1, :-2]
    return out


def classify(means: np.ndarray, hull: np.ndarray):
    idx = np.floor((means - LO) / (HI - LO) * G).astype(int)
    in_box = np.all((idx >= 0) & (idx < G), 1)
    inside = np.zeros(len(means), bool)
    ii = idx[in_box]
    inside[in_box] = hull[ii[:, 0], ii[:, 1], ii[:, 2]]
    # "below the object": outside the hull, under the hull's lowest occupied z in that (x, y)
    # column (columns with no hull voxels count as not-below).
    occ_z = np.where(hull.any(2), hull.argmax(2), G)  # first occupied z index per (x, y)
    below = np.zeros(len(means), bool)
    below[in_box] = (~inside[in_box]) & (ii[:, 2] < occ_z[ii[:, 0], ii[:, 1]]) & (occ_z[ii[:, 0], ii[:, 1]] < G)
    return inside, below


def main():
    npz_dir, data_root = sys.argv[1], sys.argv[2]
    margin = int(sys.argv[3]) if len(sys.argv) > 3 else 2  # voxels (~4.7 cm at 128^3 over 3 m)
    files = sorted(glob.glob(os.path.join(npz_dir, "**", "*.npz"), recursive=True))
    hulls = {}
    rows = []
    for fpath in files:
        tag = os.path.basename(fpath)[:-4]
        variant, scene, seed = tag.rsplit("_", 2)
        if scene not in hulls:
            hulls[scene] = dilate(visual_hull(os.path.join(data_root, scene)), margin)
        z = np.load(fpath)
        means = z["means"].astype(np.float32)
        opa = z["opacity"].astype(np.float32)
        inside, below = classify(means, hulls[scene])
        out = ~inside
        res = np.abs(z["field_sh0"].astype(np.float32)).mean(1) if "field_sh0" in z.files else None
        row = dict(
            run=tag, n=len(means), outside_frac=float(out.mean()), below_frac=float(below.mean()),
            opa_inside_median=float(np.median(opa[inside])) if inside.any() else float("nan"),
            opa_outside_median=float(np.median(opa[out])) if out.any() else float("nan"),
            outside_faint_frac=float((opa[out] < 0.2).mean()) if out.any() else float("nan"),
            # "visible mass" outside: sum of opacities, a rough proxy for how much they show up
            opa_mass_outside=float(opa[out].sum()), opa_mass_below=float(opa[below].sum()),
            field_res_inside=float(res[inside].mean()) if res is not None and inside.any() else float("nan"),
            field_res_outside=float(res[out].mean()) if res is not None and out.any() else float("nan"),
            field_res_below=float(res[below].mean()) if res is not None and below.any() else float("nan"),
        )
        rows.append(row)
    keys = list(rows[0]) if rows else []
    print("| " + " | ".join(keys) + " |")
    print("|" + "---|" * len(keys))
    for r in rows:
        print("| " + " | ".join(f"{r[k]:.3f}" if isinstance(r[k], float) else str(r[k]) for k in keys) + " |")
    json.dump(rows, open(os.path.join(npz_dir, "ghosts_summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
