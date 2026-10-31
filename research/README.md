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
* **Reference** (verified against arXiv 2308.04079, Table 2, p. 8): 3DGS-30K PSNR on Synthetic NeRF
  (white background, 100K random initial points) is mic 35.36, chair 35.83, ship 30.80, materials 30.00,
  lego 35.78, drums 26.15, ficus 34.87, hotdog 37.72, average **33.32**. The paper reports only PSNR for
  this dataset. Our vanilla 3-seed means: 35.38 / 36.01 / 30.84 / 29.97 / 35.72 / 26.14 / 34.85 / 37.71,
  average **33.33** (all scenes within 0.2 dB).
  Verified in the 3DGS code (graphdeco-inria/gaussian-splatting, main @ 54c035f): Blender init is
  `xyz = np.random.random((100_000, 3)) * 2.6 - 1.3`, i.e. uniform in [-1.3, 1.3]^3
  (`scene/dataset_readers.py`). Its `metrics.py` loads images with `to_tensor` ([0, 1]) and passes them
  straight to `lpipsPyTorch`, whose network applies the LPIPS shift/scale constants
  (-.030,-.088,-.188)/(.458,.448,.450). The original LPIPS code (richzhang/PerceptualSimilarity @ 082bb24,
  `lpips/lpips.py`) applies the same constants and says to rescale [0, 1] inputs to [-1, 1]
  (`normalize=True`), so 3DGS's LPIPS skips that rescaling. Ours uses torchmetrics with `normalize=True`,
  which does rescale (checked in torchmetrics 1.9.0 here; **the torchmetrics version installed on Kaggle
  was not checked**).
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

## Performance notes (field overhead, measured on a T4)

At Checkpoint B the field nearly doubled training time (lego: 33 vs 18 min). Profiling
(`research/bench_field.py`, `kaggle/configs/profile_overhead.json`):

* The field's own training step (query, loss, backward, Adam) costs ~8 / 18 / 36 ms at
  50k / 150k / 313k Gaussians, which is the whole overhead. It is linear in N. The plane
  CNN adds ~2.5 ms; the 16 GB T4 is nowhere near its memory limit.
* Collisions in the scatter-add are **not** the cause: clustered and uniform means cost
  the same, and an exactly-equivalent `F.embedding_bag` sampler (sorted segment-sum
  backward) gave identical end-to-end training time (4.53 vs 4.54 min for 7k iters), so it
  was removed again.
* Where the time goes (grid_sample path, per step at 313k): grid_sample backward ~9 ms,
  strided elementwise copies from the [C, N] -> [N, C] layout changes ~10 ms, grid_sample
  forward ~3 ms, MLP GEMMs ~2.5 ms, plus many small kernels. Generic PyTorch kernels run
  far below the T4's memory bandwidth for this access pattern. The bandwidth floor for
  N=313k, 3 planes x 4 corners x C=32 fp32 is a few ms per step.
* **Fix (option B)**: `TriplaneField.forward` keeps grid_sample's channels-first [C, N]
  layout through fusion and the MLP (applied as `W @ h + b`), removing the strided copies.
  The original computation is kept as `forward_reference`; a CPU test checks outputs and
  all gradients match, and `bench_field.py` times both.

## Phase 1 results (NeRF-Synthetic, 4 variants × 8 scenes × 3 seeds, T4, commit 13c58fc)

Raw per-run JSONs: `research/results/phase1/` (96 runs, all exit code 0). Generate the tables with
`python research/aggregate.py research/results/phase1`.

PSNR, mean ± sample std over seeds 0–2:

| Scene | vanilla | triplane | triplane_cnn | triplane_cnn_off25k |
|---|---|---|---|---|
| chair | 36.01 ± 0.05 | 35.91 ± 0.04 | 35.83 ± 0.07 | 35.73 ± 0.11 |
| drums | 26.14 ± 0.01 | 26.15 ± 0.01 | 26.16 ± 0.01 | 26.14 ± 0.00 |
| ficus | 34.85 ± 0.01 | 34.86 ± 0.01 | 34.85 ± 0.02 | 34.81 ± 0.02 |
| hotdog | 37.71 ± 0.02 | 37.72 ± 0.04 | 37.77 ± 0.07 | 37.75 ± 0.06 |
| lego | 35.72 ± 0.03 | 35.75 ± 0.01 | 35.76 ± 0.02 | 35.75 ± 0.01 |
| materials | 29.97 ± 0.03 | 29.99 ± 0.03 | 29.94 ± 0.02 | 29.93 ± 0.01 |
| mic | 35.38 ± 0.06 | 35.36 ± 0.02 | 35.54 ± 0.09 | 35.49 ± 0.07 |
| ship | 30.84 ± 0.02 | 30.79 ± 0.03 | 30.74 ± 0.06 | 30.89 ± 0.04 |
| **Average** | **33.33** ± 0.01 | **33.31** ± 0.01 | **33.32** ± 0.02 | **33.31** ± 0.03 |

| Metric (mean over 8 scenes × 3 seeds) | vanilla | triplane | triplane_cnn | triplane_cnn_off25k |
|---|---|---|---|---|
| ssim | 0.9689 | 0.9691 | 0.9690 | 0.9690 |
| lpips_vgg | 0.0379 | 0.0377 | 0.0377 | 0.0379 |
| lpips_alex | 0.0234 | 0.0232 | 0.0230 | 0.0232 |
| fps | 263 | 262 | 259 | 267 |
| num_gaussians | 283888 | 283996 | 283964 | 284152 |
| peak_vram_gb | 0.70 | 0.97 | 0.99 | 0.99 |
| train_minutes | 16.9 | 26.8 | 28.4 | 28.0 |

**Conclusions**
- None of the variants improves on vanilla gsplat on average (8-scene mean within 0.02 dB). The field
  adds ~60% training time and ~0.3 GB peak VRAM, and leaves FPS unchanged.
- The plane CNN has scene-dependent effects that hold in all 3 seeds: vs vanilla it is better on mic,
  hotdog, lego and drums, and worse on chair, ship and materials. These cancel out on average.
- `triplane_cnn_off25k` removes the CNN abruptly, which causes a large transient drop at step 25k
  (seen in training logs), so it does not cleanly test late over-smoothing.
- Why the CNN helps some scenes and hurts others: see `research/analysis/cnn_fluctuation_report.md`.
