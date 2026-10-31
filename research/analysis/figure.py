import glob, sys, numpy as np
from PIL import Image, ImageDraw
root, out = sys.argv[1], sys.argv[2]
dirs = sorted(glob.glob(f"{root}/*/results/renders"))
def load(var, s, k, v):
    for d in dirs:
        try:
            a = np.asarray(Image.open(f"{d}/{var}_{s}_{k}/test_{v}.png")).astype(np.float32) / 255
            return a[:, :800, :3], a[:, 800:, :3]
        except FileNotFoundError: pass
rows = []
for s in ["ship", "chair", "hotdog"]:
    best = None
    for k in (0, 1, 2):
        for v in ("000", "050", "100", "150"):
            rc, gt = load("triplane_cnn", s, k, v); rt, _ = load("triplane", s, k, v)
            d = ((rc - gt) ** 2).mean(-1) - ((rt - gt) ** 2).mean(-1)
            score = d.sum() if s != "hotdog" else -d.sum()
            if best is None or score > best[0]: best = (score, rc, rt, gt, d, k, v)
    _, rc, rt, gt, d, k, v = best
    amp = 30.0
    diff = np.ones_like(gt)
    pos, neg = np.clip(d * amp, 0, 1), np.clip(-d * amp, 0, 1)
    diff[..., 1] -= pos + neg; diff[..., 2] -= pos; diff[..., 0] -= neg  # red: CNN worse, blue: CNN better
    # brighten residual haze so it is visible: |render - GT| x8
    hz = lambda r: np.clip(1 - np.abs(r - gt).mean(-1, keepdims=True).repeat(3, -1) * 8, 0, 1)
    row = np.concatenate([gt, rt, rc, hz(rt), hz(rc), np.clip(diff, 0, 1)], 1)
    img = Image.fromarray((row * 255).astype(np.uint8)).resize((6 * 300, 300))
    ImageDraw.Draw(img).text((5, 5), f"{s} (seed {k}, view {v})", fill=(0, 0, 0))
    rows.append(np.asarray(img))
hdr = Image.new("RGB", (1800, 24), "white"); dr = ImageDraw.Draw(hdr)
for i, t in enumerate(["ground truth", "field, no CNN", "field + CNN", "error x8 (no CNN)", "error x8 (CNN)", "red = CNN worse, blue = better"]):
    dr.text((i * 300 + 8, 6), t, fill=(0, 0, 0))
Image.fromarray(np.concatenate([np.asarray(hdr)] + rows, 0)).save(out)
print("saved", out)
