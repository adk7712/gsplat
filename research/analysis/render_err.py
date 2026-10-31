"""CNN vs no-CNN error split by pixel type, from the saved test renders (render|GT, 1600x800)."""
import glob, json, sys, numpy as np
from PIL import Image
root = sys.argv[1]; out = sys.argv[2]
dirs = [d for d in sorted(glob.glob(f"{root}/*/results/renders")) ]
res = {}
for s in ["chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship"]:
    acc = {k: [] for k in ("edge", "flat", "bg", "all")}
    for k in (0, 1, 2):
        for v in ("000", "050", "100", "150"):
            def load(var):
                for d in dirs:
                    p = f"{d}/{var}_{s}_{k}/test_{v}.png"
                    try:
                        a = np.asarray(Image.open(p)).astype(np.float32) / 255
                        return a[:, :800, :3], a[:, 800:, :3]
                    except FileNotFoundError:
                        pass
                raise FileNotFoundError(f"{var}_{s}_{k}/{v}")
            rc, gt = load("triplane_cnn"); rt, _ = load("triplane")
            g = gt.mean(-1)
            gy, gx = np.gradient(g); gm = np.hypot(gx, gy)
            fg = np.abs(gt - 1).sum(-1) > 0.03  # not white background
            thr = np.quantile(gm[fg], 0.8) if fg.any() else 1
            edge = fg & (gm >= thr); flat = fg & (gm < thr); bg = ~fg
            dmse = ((rc - gt) ** 2).mean(-1) - ((rt - gt) ** 2).mean(-1)  # >0: CNN worse
            for name, m in (("edge", edge), ("flat", flat), ("bg", bg), ("all", np.ones_like(bg))):
                acc[name].append(dmse[m].sum() / gt[..., 0].size)  # contribution to whole-image MSE
    res[s] = {kk: float(np.mean(vv)) * 1e5 for kk, vv in acc.items()}
    tot = res[s]["all"]
    print(f"{s:9s} dMSE(cnn-tp) x1e5: total {tot:+.2f} | edges {res[s]['edge']:+.2f} flat {res[s]['flat']:+.2f} bg {res[s]['bg']:+.2f}")
json.dump(res, open(out, "w"), indent=1)
