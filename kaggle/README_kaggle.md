# Running on Kaggle

Training runs as a private Kaggle *script* kernel (`kaggle/run.py`) on T4 x2.
It needs two private datasets (per account): `nerf-synthetic` and `gsplat-wheels`.
Set `KAGGLE_USERNAME` / `KAGGLE_KEY` in your environment first.

## 1. nerf-synthetic dataset

Download `nerf_synthetic.zip` from the official NeRF Google Drive link at
https://github.com/bmild/nerf, then unzip so you have
`nerf_synthetic/{chair,drums,ficus,hotdog,lego,materials,mic,ship}`:

```bash
unzip nerf_synthetic.zip
kaggle datasets init -p nerf_synthetic
# edit nerf_synthetic/dataset-metadata.json:
#   "title": "nerf-synthetic", "id": "<user>/nerf-synthetic", "licenses": [{"name":"other"}]
kaggle datasets create -p nerf_synthetic --dir-mode zip   # private by default
```

The runner finds scenes by searching `/kaggle/input` for `<scene>/transforms_train.json`,
so the exact mount path does not matter.

## 2. gsplat-wheels dataset

Kaggle requires every dataset in `dataset_sources` to exist before a push, so create it
first with a placeholder:

```bash
mkdir -p wheels_ds && echo "gsplat wheels" > wheels_ds/README.md
kaggle datasets init -p wheels_ds        # set "id": "<user>/gsplat-wheels", title "gsplat-wheels"
kaggle datasets create -p wheels_ds --dir-mode zip
```

The first launch finds no wheel and builds one into `/kaggle/working/wheels/<tag>/`
(`<tag>` is e.g. `torch2.6.0_cu124`). After `kaggle/fetch.sh`, publish it so later runs skip the build:

```bash
cp -r results/raw/<ts>/wheels/<tag> wheels_ds/
kaggle datasets version -p wheels_ds -m "add wheel <tag>" --dir-mode zip
```

## 3. Launch

```bash
cat > job.json <<'JSON'
{"repo_url": "https://github.com/adk7712/gsplat.git", "ref": "<branch or sha>", "iters": 30000,
 "jobs": [{"variant": "vanilla", "scene": "lego", "seed": 0}]}
JSON
kaggle/launch.sh job.json        # pushes with --accelerator NvidiaTeslaT4
kaggle/status.sh --wait          # polls every 300 s
kaggle/fetch.sh                  # -> results/raw/<UTC timestamp>/
python research/aggregate.py results/raw
```

Push your branch to GitHub first; the kernel clones from there. For a private repo, add a
Kaggle secret `GH_TOKEN` (Add-ons -> Secrets, attached to the kernel).

## Teammates

Private datasets are per account, so each teammate must create their own copies of both
datasets (or share them via the Kaggle UI). `launch.sh` rewrites the dataset owner prefixes
to `$KAGGLE_USERNAME`.

## Notes
- New-style Kaggle API tokens (`KGAT_...`) must be passed as `KAGGLE_API_TOKEN`; the scripts
  map `KAGGLE_KEY=KGAT_...` to it automatically.
- `adk7712/nerf-synthetic` was created as an unmodified private copy of the public Kaggle dataset
  `nguyenhung1903/nerf-synthetic-dataset` (verified: 8 scenes, 100/100/200 frames, 800x800 RGBA).
  Teammates can instead put `nguyenhung1903/nerf-synthetic-dataset` in `dataset_sources`.
