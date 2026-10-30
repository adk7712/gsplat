# Tri-plane attribute field on gsplat (MLDA x EEE, NTU) — Phase 1

Static-scene port of our 4DGaussians plane-CNN idea: a tri-plane field sampled at
Gaussian centres predicts per-Gaussian attribute residuals, optionally with a light
residual CNN refining the stored planes. Question for Phase 1: **does the plane CNN help?**

Upstream gsplat is untouched: all new code lives in `research/` and `kaggle/`.

## Layout

| Path | What |
|---|---|
| `research/train.py` | Compact NeRF-Synthetic trainer (gsplat `rasterization` + `DefaultStrategy`/`MCMCStrategy`) with an optional field |
| `research/triplane_field.py` | `TriplaneField`, `PlaneCNN`, regularisers, optimiser/schedule |
| `research/blender.py` | NeRF-Synthetic `transforms_*.json` loader (gsplat's examples have none) |
| `research/eval.py` | Metrics + FPS protocol |
| `research/aggregate.py` | Collect result JSONs → markdown mean ± std tables |
| `research/configs/*.json` | Variants: `vanilla`, `triplane`, `triplane_cnn`, `triplane_cnn_off25k` |
| `research/tests/` | CPU unit tests + smoke test of the field |
| `kaggle/` | Script kernel runner, `launch.sh`, `status.sh`, `fetch.sh`, job configs (`kaggle/configs/`) |

## Versions targeted

* **gsplat**: this fork (base upstream `512d366`, `gsplat.__version__ == "1.6.0"`), built as a
  wheel for **sm_75 only** (`TORCH_CUDA_ARCH_LIST=7.5 BUILD_EXPERIMENTAL=0`). Research code uses
  only the public API (`gsplat.rasterization`, `gsplat.strategy`, `gsplat.losses`).
* **torch / CUDA**: Kaggle's preinstalled torch (not reinstalled, to save session time). The
  wheel is stored under a `torch<version>` folder so a Kaggle image upgrade forces a rebuild
  instead of an ABI mismatch. The exact torch, CUDA and gsplat versions are recorded in every
  result JSON (`extra.torch`, `extra.torch_cuda`, `extra.gsplat`) and in `_summary.json`.
  gsplat needs torch ≥ 2.7.
* **Pinned (as run on Kaggle, 2026-10-30)**: Python 3.13, torch **2.11.0+cu128**, CUDA **12.8**,
  gsplat **1.6.0** (this fork), wheel `gsplat-1.6.0-cp313-cp313-linux_x86_64.whl` stored in
  `adk7712/gsplat-wheels` under `torch2.11.0_cu128/`. Building it from source on Kaggle takes
  ~65 min, so keep the dataset in sync if the Kaggle image's torch changes.
* **GPU**: Tesla T4 (16 GB, sm_75). The runner refuses to run on anything else.

## Running locally (CUDA machine)

```bash
pip install -e . --no-build-isolation   # or a prebuilt wheel
pip install "torchmetrics[image]>=1.0" lpips imageio tqdm
python research/train.py --config research/configs/triplane_cnn.json \
    --data_dir data/nerf_synthetic/lego --scene lego --seed 0 --result_dir results/local
python research/aggregate.py results/local
```

Any `Config` field in `train.py` can be set in the JSON or overridden on the CLI
(`--iters 7000`, `--plane_cnn_off_after 20000`, `--no-plane_cnn`, `--field_multires`, ...).

## Running on Kaggle

One-time setup (datasets, wheel) is in `kaggle/README_kaggle.md`. Then:

```bash
kaggle/launch.sh kaggle/configs/checkpoint_a.json   # push + run the kernel on a T4
kaggle/status.sh --wait                             # polls every 5 min
kaggle/fetch.sh                                     # -> results/raw/<timestamp>/
python research/aggregate.py results/raw
```

A job config lists `(variant, scene, seed)` jobs; with T4 x2 two jobs run in parallel,
one per GPU. Commit **and push** before launching: the kernel clones `ref` from GitHub.

## Evaluation protocol (matches the 3DGS paper's NeRF-Synthetic setup)

* **Data**: `transforms_train.json` (100 views) for training, all 200 `transforms_test.json`
  views for testing, full 800×800, RGBA composited on a **white** background.
* **Init**: 100k random points uniform in [-1.3, 1.3]³, grey colour, opacity 0.1, kNN scales (as 3DGS).
* **Training**: 30k iters, batch 1, views drawn without replacement per epoch, loss
  0.8·L1 + 0.2·(1−SSIM), SH degree +1 every 1k iters up to 3, white background.
  gsplat defaults for lrs and densification (`DefaultStrategy`: grow_grad2d 2e-4, refine
  500–15k every 100, opacity reset every 3k), scene scale = 1.1 × max camera distance from
  the camera centroid (3DGS `cameras_extent`).
* **Metrics** (`research/eval.py`): renders clamped to [0, 1]; PSNR and SSIM (11×11 Gaussian,
  σ=1.5) per image, then averaged; **LPIPS-VGG** and **LPIPS-Alex** with torchmetrics
  (`normalize=True`, so [0, 1] inputs are mapped to [−1, 1]). Note that the 3DGS repo's LPIPS
  skips that rescaling. This doesn't affect the comparison, because 3DGS reports only PSNR on
  NeRF-Synthetic.
* **FPS**: rasterisation only, 800×800, batch 1, all 200 test views after 10 warm-up renders,
  `cuda.synchronize`d. The field is evaluated once before timing (means are fixed at test
  time); that one-off cost is `extra.field_query_ms`.
* **Peak VRAM**: `torch.cuda.max_memory_allocated` during training (reserved in `extra`).
* **Train time**: wall clock of the 30k-iteration loop, excluding data loading and eval.
* **Reference**: 3DGS (Kerbl et al. 2023), lego PSNR **35.78**, NeRF-Synthetic average 33.32.
* **Sanity check (Checkpoint A)**: vanilla, lego, seed 0, 30k iters on a T4: PSNR 35.67,
  SSIM 0.9822, LPIPS-VGG 0.0199, LPIPS-Alex 0.0106, 225 FPS, 314k Gaussians, 0.74 GB peak
  allocated VRAM, 18.4 min training.

## The field (`research/triplane_field.py`)

`attr = attr_param + field(mu)`. The field gives residuals for SH DC and (with `field_opacity`)
the opacity logit.

* Planes xy, xz, yz, each C×R×R (C=32, R=128; `field_multires` → {64,128,256}), initialised
  U(0.1, 0.5) (K-Planes). Means are normalised into `field_aabb` (default [−1.5, 1.5]³) and
  sampled with `grid_sample(bilinear, align_corners=True, padding_mode="border")`.
* Fusion: product (default) or concat (`field_fusion`), then concatenated across resolutions.
* MLP: Linear(C·n_res → 64) + ReLU + Linear(64 → 3[+1]). The last layer is zero-initialised,
  so step 0 is exactly vanilla gsplat.
* **Plane CNN** (`plane_cnn`): per plane `plane + conv2(relu(conv1(plane)))`, 3×3 with
  padding 1. conv2 is zero-initialised, so it starts as the identity. `plane_cnn_off_after N`
  uses the raw planes from step N onwards.
* **Regularisers**: TV (`tv_weight`, default 1e-4) and L1 (`l1_weight`, default 0) act on the
  **effective (post-CNN)** planes. `reg_on_raw` gives the old 4D behaviour for an ablation.
* **Optimiser**: one Adam with param groups planes / CNN / MLP (initial lrs `plane_lr` 1.6e-3,
  `cnn_lr` 1.6e-4, `mlp_lr` 1.6e-4, taken from the 4DGaussians grid/deformation lrs), and
  **one** ExponentialLR that decays every group to 0.01× by the final iteration, the same
  shape as gsplat's means lr.
* **Means are detached** when querying the field (`field_grad_to_means` false by default), so
  geometry is optimised exactly as in vanilla and the comparison isolates the appearance model.
* **Densification**: the field is not part of `splats`, so neither strategy sees it. New,
  split or relocated Gaussians just query the field at their new means on the next forward
  pass, and no extra state is needed. Caveat if `field_opacity` is on: both strategies
  prune, reset and relocate using the **raw** opacity parameter, not the effective one.
  That is why opacity residuals are off by default.

## Tests

```bash
python -m pytest research/tests -q        # uses research/pytest.ini
python research/tests/smoke_cpu.py
```
