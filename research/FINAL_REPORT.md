# Tri-plane attribute field with a plane CNN for static 3DGS

All numbers are computed from the result JSONs in `research/results/`.

## Summary
- I added a tri-plane field to gsplat that predicts a per-Gaussian colour residual, optionally
  refining the planes with a small residual CNN.
- **100 training views (8 scenes × 3 seeds):** no variant beats vanilla gsplat. All averages are
  within 0.02 dB. The field adds ~60% training time.
- **Plane CNN:** consistent per-scene effects that cancel out. It helps mic, hotdog and lego, and
  hurts chair, ship and materials.
- **Opacity-gated field:** removes the CNN's losses and its gains, giving roughly vanilla.
- **12 training views (1 seed):** the field alone matches vanilla (+0.03 dB). With the CNN it is
  unstable (−0.45 dB).
- **Conclusion:** this colour-only field, with or without the CNN, does not improve static 3DGS
  on NeRF-Synthetic.

## Method
`attr = attr_param + field(mu)`. Three planes (xy, xz, yz), each 32×128×128, are sampled at the
Gaussian means and fused by product. A 2-layer MLP then outputs an SH-DC residual. Its last layer is
zero-initialised, so training starts exactly at vanilla.

```python
# research/triplane_field.py, sampling and product fusion
f = F.grid_sample(planes[r * n_planes + k], grid, mode="bilinear",
                  padding_mode="border", align_corners=True)  # [1, C, N, 1]
per_plane.append(f[0, :, :, 0])                               # [C, N]
...
feats.append(per_plane[0] * per_plane[1] * per_plane[2])
```

The plane CNN is residual, with `conv2` zero-initialised, so it is the identity at the start:

```python
# research/triplane_field.py, PlaneCNN
def forward(self, plane):
    return plane + self.conv2(F.relu(self.conv1(plane)))
```

TV regularisation acts on the post-CNN planes. The CNN runs once per step and the result is shared
with the field query:

```python
# research/train.py, training loop
planes = fld.effective_planes()  # CNN runs once per step
attrs = effective_attributes(splats, fld, planes, cfg.field_opacity_gate)
...
reg = fld.regularization(cfg.tv_weight, cfg.l1_weight, planes)
```

Planes, CNN and MLP share one Adam optimiser and one exponential decay to 0.01× (same shape as
gsplat's means lr):

```python
# research/triplane_field.py, make_field_optimizer
opt = torch.optim.Adam(fld.param_groups(plane_lr, cnn_lr, mlp_lr), eps=1e-15)
gamma = final_lr_ratio ** (1.0 / max_steps)
sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=gamma)
```

## Setup
- gsplat 1.6.0, torch 2.11.0+cu128, CUDA 12.8, torchmetrics 1.9.0, Kaggle Tesla T4.
- NeRF-Synthetic, 800×800, white background, 100K random init points in [-1.3, 1.3]³, 30K iterations,
  200 test views. This matches the 3DGS setup (arXiv 2308.04079, Table 2, and its code).
- Commit IDs below are those recorded in the result JSONs. `research/COMMIT_MAP.md` maps them to the
  current history.

## 1. 100 training views (3 seeds, commit 13c58fc)
PSNR, mean ± std over seeds:

| Scene | 3DGS paper | vanilla | triplane | triplane_cnn | triplane_cnn_off25k |
|---|---|---|---|---|---|
| chair | 35.83 | 36.01 ± 0.05 | 35.91 ± 0.04 | 35.83 ± 0.07 | 35.73 ± 0.11 |
| drums | 26.15 | 26.14 ± 0.01 | 26.15 ± 0.01 | 26.16 ± 0.01 | 26.14 ± 0.00 |
| ficus | 34.87 | 34.85 ± 0.01 | 34.86 ± 0.01 | 34.85 ± 0.02 | 34.81 ± 0.02 |
| hotdog | 37.72 | 37.71 ± 0.02 | 37.72 ± 0.04 | 37.77 ± 0.07 | 37.75 ± 0.06 |
| lego | 35.78 | 35.72 ± 0.03 | 35.75 ± 0.01 | 35.76 ± 0.02 | 35.75 ± 0.01 |
| materials | 30.00 | 29.97 ± 0.03 | 29.99 ± 0.03 | 29.94 ± 0.02 | 29.93 ± 0.01 |
| mic | 35.36 | 35.38 ± 0.06 | 35.36 ± 0.02 | 35.54 ± 0.09 | 35.49 ± 0.07 |
| ship | 30.80 | 30.84 ± 0.02 | 30.79 ± 0.03 | 30.74 ± 0.06 | 30.89 ± 0.04 |
| **average** | **33.32** | **33.33** | **33.31** | **33.32** | **33.31** |

My vanilla run is within 0.2 dB of the 3DGS paper on every scene.

| Mean over 8 scenes × 3 seeds | vanilla | triplane | triplane_cnn | triplane_cnn_off25k |
|---|---|---|---|---|
| ssim | 0.9689 | 0.9691 | 0.9690 | 0.9690 |
| lpips_vgg | 0.0379 | 0.0377 | 0.0377 | 0.0379 |
| lpips_alex | 0.0234 | 0.0232 | 0.0230 | 0.0232 |
| train_minutes | 16.9 | 26.8 | 28.4 | 28.0 |
| peak_vram_gb | 0.70 | 0.97 | 0.99 | 0.99 |
| fps | 263 | 262 | 259 | 267 |

PSNR change vs vanilla (3-seed mean):

| Scene | triplane − vanilla | triplane_cnn − vanilla | same sign in all 3 seeds? (cnn) |
|---|---|---|---|
| chair | -0.10 | -0.18 | yes (−) |
| drums | +0.00 | +0.02 | yes (+) |
| ficus | +0.01 | -0.00 | no |
| hotdog | +0.01 | +0.07 | yes (+) |
| lego | +0.03 | +0.05 | yes (+) |
| materials | +0.02 | -0.04 | yes (−) |
| mic | -0.02 | +0.16 | yes (+) |
| ship | -0.06 | -0.11 | yes (−) |

`triplane_cnn_off25k` drops the CNN abruptly at step 25K. Training PSNR dips sharply at that point,
so this variant does not cleanly test late over-smoothing.

## 2. Why the CNN helps some scenes and hurts others
Details are in `research/analysis/cnn_fluctuation_report.md`.
- In chair and ship, 95–97% of the CNN's extra error lies in the empty background, as a tinted smudge
  under the object. It does not blur object detail.
- Where the CNN helps, it shrinks the field's colour residual. Where it hurts, it enlarges it
  (Spearman ρ = −0.76 over 8 scenes, p = 0.04).
- **Unverified:** that the smudges are faint "ghost" Gaussians. My visual-hull check cannot see the
  hidden space under the chair seat.

## 3. Opacity-gated field (seed 0, commit 76c9214)
The residual is scaled by each Gaussian's detached opacity:

```python
# research/train.py, effective_attributes
if opacity_gate:
    sh0 = sh0 + torch.sigmoid(splats["opacities"]).detach()[:, None, None] * res["sh0"]
```

PSNR change vs vanilla:

| Variant (seed 0) | mic | hotdog | lego | chair | ship | materials | mean |
|---|---|---|---|---|---|---|---|
| triplane | +0.02 | +0.02 | +0.05 | -0.14 | -0.05 | +0.01 | -0.014 |
| triplane_cnn | +0.27 | +0.11 | +0.08 | -0.30 | -0.19 | -0.06 | -0.016 |
| triplane_gated | +0.12 | +0.02 | +0.02 | -0.14 | +0.07 | -0.01 | +0.013 |
| triplane_cnn_gated | +0.07 | +0.05 | +0.01 | -0.10 | +0.02 | -0.01 | +0.008 |

Object Gaussians have a median opacity of only ~0.2–0.3, so the gate weakens the field almost
everywhere. Rerunning identical code and seed changed PSNR by up to 0.11 dB, so single-run
differences of ~0.1 dB are noise.

## 4. 12 training views (seed 0, commit 6b93dbb)
I trained on views 0, 9, 18, …, 99 (`--train_views 12`) and tested on the same 200 views.

| Scene | vanilla | triplane − vanilla | triplane_cnn − vanilla |
|---|---|---|---|
| chair | 26.55 | +0.25 | -0.19 |
| drums | 19.16 | -0.06 | +0.73 |
| ficus | 26.49 | -0.01 | -1.75 |
| hotdog | 28.03 | -0.09 | -0.13 |
| lego | 24.53 | +0.04 | +0.51 |
| materials | 22.18 | -0.09 | -3.09 |
| mic | 27.76 | -0.02 | -0.13 |
| ship | 16.28 | +0.22 | +0.49 |
| **average** | **23.87** | **+0.03** | **-0.45** |

Mean SSIM: vanilla: 0.8854, triplane: 0.8858, triplane_cnn: 0.8804. Mean LPIPS-VGG: vanilla: 0.1145, triplane: 0.1143, triplane_cnn: 0.1292.
The CNN variant shows streak artifacts on materials (`research/analysis/sparse12_examples.png`).

## Limits
- The gate and 12-view tests use one seed each.
- This is not a reproduction of SplatFields (arXiv 2409.11211). SplatFields predicts all splat
  attributes, initialises inside the visual hull, uses a mask loss, and selects views differently.
  My 12-view numbers are not comparable to theirs.
- I did not test opacity or geometry residuals, multi-resolution planes, concat fusion, other TV
  weights or learning rates, fewer than 12 views, or real-world scenes.

## Reproduce
```bash
python research/aggregate.py research/results/phase1   # dense-view tables
python -m pytest research/tests -q                      # CPU tests
kaggle/launch.sh kaggle/configs/sweep_seed0.json        # also: fix_gate.json, sparse12.json
```
