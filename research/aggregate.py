#!/usr/bin/env python
"""Aggregate research result JSONs into markdown tables.

Usage: python research/aggregate.py [root=results/raw] [--metrics a,b,...] [--csv out.csv] [--selftest]
"""
import argparse
import csv
import json
import os
import statistics as st
import sys
import tempfile
from collections import defaultdict

DEFAULT_METRICS = ["psnr", "ssim", "lpips_vgg", "lpips_alex", "fps", "num_gaussians", "peak_vram_gb", "train_minutes"]
LABEL = {"psnr": "PSNR", "ssim": "SSIM", "lpips_vgg": "LPIPS-VGG", "lpips_alex": "LPIPS-Alex", "fps": "FPS",
         "num_gaussians": "#Gaussians", "peak_vram_gb": "VRAM (GB)", "train_minutes": "Train (min)"}
DECIMALS = {"psnr": 2, "ssim": 4, "lpips_vgg": 4, "lpips_alex": 4, "fps": 1, "peak_vram_gb": 2, "train_minutes": 1}


def fmt(metric, v):
    if v is None:
        return "-"
    if metric == "num_gaussians":
        return f"{v / 1000:.0f}k"
    return f"{v:.{DECIMALS.get(metric, 3)}f}"


def load_runs(root):
    runs = {}
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if not fn.endswith(".json") or fn == "_summary.json":
                continue
            path = os.path.join(dp, fn)
            try:
                d = json.load(open(path))
            except (OSError, ValueError):
                continue
            if not isinstance(d, dict) or "variant" not in d:
                continue
            d["_path"], d["_mtime"] = path, os.path.getmtime(path)
            key = (d["variant"], d.get("scene"), d.get("seed"))
            if key in runs:
                old = runs[key]
                print(f"WARNING: duplicate {key}: {old['_path']} vs {path}; keeping newest", file=sys.stderr)
                if old["_mtime"] >= d["_mtime"]:
                    continue
            runs[key] = d
    return list(runs.values())


def vals(rows, m):
    return [r[m] for r in rows if isinstance(r.get(m), (int, float))]


def cell(metric, xs):
    if not xs:
        return "-"
    m = st.mean(xs)
    if len(xs) < 2:
        return f"{fmt(metric, m)} ± – (n=1)"
    s = st.stdev(xs)
    if metric == "num_gaussians":
        return f"{fmt(metric, m)} ± {s / 1000:.0f}k (n={len(xs)})"
    return f"{fmt(metric, m)} ± {s:.{DECIMALS.get(metric, 3)}f} (n={len(xs)})"


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def report(runs, metrics):
    by = defaultdict(lambda: defaultdict(list))  # scene -> variant -> runs
    for r in runs:
        by[r.get("scene")][r["variant"]].append(r)
    head = ["Variant"] + [LABEL.get(m, m) for m in metrics]
    for scene in sorted(by, key=str):
        print(f"\n### {scene}\n")
        print(table(head, [[v] + [cell(m, vals(rs, m)) for m in metrics] for v, rs in sorted(by[scene].items())]))
    # overall: mean of per-scene means
    variants = sorted({r["variant"] for r in runs})
    rows = []
    for v in variants:
        row = [v]
        scenes = [s for s in by if v in by[s]]
        for m in metrics:
            means = [st.mean(x) for s in scenes if (x := vals(by[s][v], m))]
            if not means:
                row.append("-")
            elif m == "num_gaussians":
                row.append(f"{fmt(m, st.mean(means))} (n_scenes={len(means)})")
            else:
                row.append(f"{fmt(m, st.mean(means))} (n_scenes={len(means)})")
        rows.append(row)
    print("\n### Overall (mean of per-scene means)\n")
    print(table(head, rows))


def dump_csv(runs, metrics, path):
    cols = ["variant", "scene", "seed", "git_commit", "iters", "gpu_name"] + metrics + ["_path"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in sorted(runs, key=lambda r: (str(r["variant"]), str(r.get("scene")), str(r.get("seed")))):
            w.writerow([r.get(c, "") for c in cols])
    print(f"\nWrote {path}")


def selftest():
    import random
    random.seed(0)
    with tempfile.TemporaryDirectory() as d:
        for variant, base in (("vanilla", 35.0), ("mcmc", 35.5)):
            for scene in ("lego", "ship"):
                for seed in range(3 if variant == "vanilla" else 1):
                    rec = {"variant": variant, "scene": scene, "seed": seed, "git_commit": "abc", "iters": 30000,
                           "psnr": base + random.random(), "ssim": 0.97 + random.random() / 100,
                           "lpips_vgg": 0.05 + random.random() / 100, "lpips_alex": 0.03 + random.random() / 100,
                           "fps": 100 + random.random() * 20, "num_gaussians": 290000 + random.randint(0, 20000),
                           "peak_vram_gb": 3.1, "train_minutes": 20 + random.random(), "gpu_name": "Tesla T4",
                           "config": {}}
                    json.dump(rec, open(os.path.join(d, f"{variant}_{scene}_{seed}.json"), "w"))
        json.dump({"jobs": []}, open(os.path.join(d, "_summary.json"), "w"))
        report(load_runs(d), DEFAULT_METRICS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default="results/raw")
    ap.add_argument("--metrics", default=",".join(DEFAULT_METRICS))
    ap.add_argument("--csv")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    metrics = [m.strip() for m in a.metrics.split(",") if m.strip()]
    runs = load_runs(a.root)
    if not runs:
        sys.exit(f"No result JSONs found under {a.root}")
    print(f"Loaded {len(runs)} runs from {a.root}")
    report(runs, metrics)
    if a.csv:
        dump_csv(runs, metrics, a.csv)


if __name__ == "__main__":
    main()
