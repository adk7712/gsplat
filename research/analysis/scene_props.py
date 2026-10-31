"""Per-scene properties from the NeRF-Synthetic data (CPU). Visual hull on a 128^3 grid
over the field's AABB [-1.5,1.5]^3 (same box and same R=128 as the field's planes)."""
import json, math, sys, numpy as np
from PIL import Image
R = sys.argv[1]; out = sys.argv[2]
G = 128; lo, hi = -1.5, 1.5
c = (np.arange(G) + 0.5) / G * (hi - lo) + lo
X, Y, Z = np.meshgrid(c, c, c, indexing="ij")
P = np.stack([X, Y, Z, np.ones_like(X)], -1).reshape(-1, 4).astype(np.float32)
res = {}
for s in ["chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship"]:
    meta = json.load(open(f"{R}/{s}/transforms_train.json"))
    ds = 4; W = 800 // ds
    f = 0.5 * W / math.tan(0.5 * meta["camera_angle_x"])
    occ = np.ones(len(P), bool)
    hl, sat = [], []
    for fr in meta["frames"]:
        im = np.asarray(Image.open(f"{R}/{s}/{fr['file_path']}.png").reduce(ds)).astype(np.float32) / 255
        m = im[..., 3] > 0.05
        c2w = np.array(fr["transform_matrix"], np.float32); c2w[:3, 1:3] *= -1
        w2c = np.linalg.inv(c2w)
        pc = P @ w2c.T
        z = pc[:, 2]; u = f * pc[:, 0] / z + W / 2; v = f * pc[:, 1] / z + W / 2
        inside = (z > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < W)
        ok = np.zeros(len(P), bool)
        ok[inside] = m[v[inside].astype(int), u[inside].astype(int)]
        occ &= ok | ~inside
        # highlight proxy (specular): near-white, low-saturation object pixels
        fg = im[..., 3] > 0.9; rgb = im[..., :3][fg]
        if len(rgb):
            mx, mn = rgb.max(1), rgb.min(1)
            hl.append(np.mean((mx > 0.92) & (mx - mn < 0.08)))
    V = occ.reshape(G, G, G)
    # surface voxels: occupied with an empty 6-neighbour
    pad = np.pad(V, 1)
    nb = sum(np.roll(pad, sh, ax) for ax in range(3) for sh in (1, -1))[1:-1, 1:-1, 1:-1]
    surf = V & (nb < 6)
    layers = []
    for ax in range(3):  # projection along ax -> plane of the other two axes
        S_ = np.moveaxis(surf, ax, -1)
        # count separate surface runs along the projection direction
        starts = S_ & ~np.concatenate([np.zeros_like(S_[..., :1]), S_[..., :-1]], -1)
        n = starts.sum(-1); layers.append(n[n > 0].mean())
    idx = np.argwhere(V) * (hi - lo) / G + lo
    ext = idx.max(0) - idx.min(0)
    # how much of each plane the object covers (plane cells actually used)
    cover = np.mean([np.moveaxis(V, ax, -1).any(-1).mean() for ax in range(3)])
    near_edge = np.mean(np.abs(idx).max(1) > 1.3)
    # axis alignment: share of surface normal energy along the 3 axes (from voxel gradients)
    gx, gy, gz = np.gradient(V.astype(np.float32))
    gm = np.sqrt(gx**2 + gy**2 + gz**2) + 1e-9
    sel = surf
    al = np.mean(np.max(np.abs(np.stack([gx, gy, gz]))[:, sel], 0) / gm[sel])
    res[s] = dict(volume_frac=float(V.mean()), extent=[round(float(e), 2) for e in ext],
                  plane_coverage=float(cover), surface_layers=float(np.mean(layers)),
                  surface_layers_per_axis=[round(float(l), 2) for l in layers],
                  near_box_edge=float(near_edge), axis_alignment=float(al),
                  highlight_frac=float(np.mean(hl)))
    print(s, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res[s].items()}, flush=True)
json.dump(res, open(out, "w"), indent=1)
