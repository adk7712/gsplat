#!/usr/bin/env python
"""Kaggle script kernel: clone repo, install gsplat, run train.py jobs on all GPUs."""
import glob
import json
import os
import queue
import subprocess
import sys
import threading
import time

_RAW_CONFIG = r'''__RUN_CONFIG__'''

EXTRA_PIP = ["torchmetrics[image]>=1.0", "lpips", "imageio", "tqdm"]
GSPLAT_DEPS = ["jaxtyping", "rich", "nvtx", "ninja"]

REPO_DIR = "/kaggle/tmp/repo"
WORK = "/kaggle/working"
LOGS = f"{WORK}/logs"
RESULTS = f"{WORK}/results"


def default_config():
    return {
        "repo_url": "https://github.com/adk7712/gsplat.git",
        "ref": "main",
        "iters": 30000,
        "allow_non_t4": False,
        "jobs": [{"variant": "vanilla", "scene": "lego", "seed": 0}],
        "extra_args": [],
    }


try:
    CONFIG = json.loads(_RAW_CONFIG)
except ValueError:  # launch.sh did not substitute the placeholder
    print("WARNING: no run config substituted; using default (vanilla/lego/0)", flush=True)
    CONFIG = default_config()

_T0 = time.time()
_STEP_T = [time.time()]


def step(n, title):
    now = time.time()
    if n > 1:
        print(f"--- previous step took {now - _STEP_T[0]:.1f}s (total {now - _T0:.1f}s) ---", flush=True)
    _STEP_T[0] = now
    print(f"\n=== STEP {n}: {title} ===", flush=True)


def die(msg, code=1):
    print(f"\nFATAL: {msg}", flush=True)
    sys.exit(code)


SECRETS = []


def mask(s):
    for sec in SECRETS:
        if sec:
            s = s.replace(sec, "***")
    return s


def run(cmd, env=None, cwd=None, log=None, check_msg=None, show=True):
    """Run a command; returns CompletedProcess. Output goes to log file if given else captured+printed."""
    if show:
        print("$ " + mask(" ".join(cmd)), flush=True)
    e = dict(os.environ)
    if env:
        e.update(env)
    if log:
        os.makedirs(os.path.dirname(log), exist_ok=True)
        with open(log, "a") as f:
            p = subprocess.run(cmd, env=e, cwd=cwd, stdout=f, stderr=subprocess.STDOUT, check=False)
    else:
        p = subprocess.run(cmd, env=e, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True, check=False)
        if show and p.stdout:
            print(mask(p.stdout), flush=True)
    if p.returncode != 0 and check_msg:
        die(f"{check_msg} (exit code {p.returncode})")
    return p


def tail(path, n):
    try:
        with open(path, errors="replace") as f:
            return "".join(f.readlines()[-n:])
    except OSError as ex:
        return f"<cannot read {path}: {ex}>"


def main():
    cfg = CONFIG
    repo_url = cfg["repo_url"]
    ref = cfg["ref"]
    iters = cfg.get("iters")
    allow_non_t4 = bool(cfg.get("allow_non_t4", False))
    jobs = cfg["jobs"]
    extra_args = list(cfg.get("extra_args") or [])
    os.makedirs(LOGS, exist_ok=True)
    os.makedirs(RESULTS, exist_ok=True)
    print("Config:", json.dumps(cfg), flush=True)

    # ---- STEP 1: GPU check
    step(1, "GPU check")
    try:
        run(["nvidia-smi"])
        p = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        gpu_names = [l.strip() for l in p.stdout.splitlines() if l.strip()] if p.returncode == 0 else []
    except FileNotFoundError:
        gpu_names = []
    if not gpu_names or (not allow_non_t4 and any("T4" not in g for g in gpu_names)):
        print("#" * 78)
        print(f"FATAL: expected Tesla T4 (sm_75) but got {gpu_names or 'no GPU'}; "
              f"re-launch with --accelerator NvidiaTeslaT4")
        print("#" * 78, flush=True)
        sys.exit(1)
    print("GPUs:", gpu_names, flush=True)

    # ---- STEP 2: clone
    step(2, "Clone repo")
    token = None
    try:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret("GH_TOKEN")
    except Exception as ex:  # secret absent or module missing
        print(f"(no GH_TOKEN secret: {type(ex).__name__})", flush=True)
    clone_url = repo_url
    if token:
        SECRETS.append(token)
        if clone_url.startswith("https://"):
            clone_url = "https://x-access-token:" + token + "@" + clone_url[len("https://"):]
    os.makedirs("/kaggle/tmp", exist_ok=True)
    if os.path.isdir(REPO_DIR):
        run(["rm", "-rf", REPO_DIR])
    run(["git", "clone", clone_url, REPO_DIR], check_msg=f"git clone of {repo_url} failed")
    p = run(["git", "checkout", ref], cwd=REPO_DIR)
    if p.returncode != 0:
        print(f"checkout {ref} failed, trying fetch origin {ref}", flush=True)
        run(["git", "fetch", "origin", ref], cwd=REPO_DIR, check_msg=f"git fetch of ref {ref} failed")
        run(["git", "checkout", "FETCH_HEAD"], cwd=REPO_DIR, check_msg=f"git checkout FETCH_HEAD ({ref}) failed")
    p = run(["git", "rev-parse", "HEAD"], cwd=REPO_DIR, check_msg="git rev-parse failed")
    git_commit = p.stdout.strip()
    run(["git", "submodule", "update", "--init", "--recursive"], cwd=REPO_DIR)  # best effort
    print("Commit:", git_commit, flush=True)

    # ---- STEP 3: install gsplat
    step(3, "Install gsplat")
    p = run([sys.executable, "-c", "import torch; print(torch.__version__); print(torch.version.cuda)"],
            check_msg="cannot import torch", show=False)
    torch_version, torch_cuda = (p.stdout.split() + ["?", "?"])[:2]
    tag = "torch" + torch_version.replace("+", "_")
    print("torch tag:", tag, "(cuda", torch_cuda + ")", flush=True)
    wheels = sorted(glob.glob(f"/kaggle/input/**/{tag}/gsplat-*.whl", recursive=True))
    built = False
    if wheels:
        wheel = wheels[-1]
        print("Found prebuilt wheel:", wheel, flush=True)
        wheel_info = wheel
    else:
        print(f"No prebuilt wheel for {tag}; building (this takes a while)...", flush=True)
        outdir = f"{WORK}/wheels/{tag}"
        os.makedirs(outdir, exist_ok=True)
        blog = f"{LOGS}/wheel_build.log"
        p = run([sys.executable, "-m", "pip", "wheel", ".", "--no-build-isolation", "--no-deps", "-w", outdir],
                cwd=REPO_DIR, log=blog,
                env={"TORCH_CUDA_ARCH_LIST": "7.5", "BUILD_EXPERIMENTAL": "0", "MAX_JOBS": "4"})
        if p.returncode != 0:
            print(tail(blog, 40))
            die(f"wheel build failed (exit {p.returncode}); full log: {blog}")
        found = sorted(glob.glob(f"{outdir}/gsplat-*.whl"))
        if not found:
            print(tail(blog, 40))
            die(f"wheel build produced no gsplat-*.whl in {outdir}")
        wheel = found[-1]
        built = True
        wheel_info = "built"
    run([sys.executable, "-m", "pip", "install", "--no-deps", wheel], check_msg="pip install of wheel failed")
    run([sys.executable, "-m", "pip", "install", "-q"] + GSPLAT_DEPS, check_msg="pip install of gsplat deps failed")
    if built:
        print("\n" + "#" * 78)
        print(f"# WHEEL BUILT: {wheel}")
        print(f"# It is in this kernel's output under wheels/{tag}/ . To avoid rebuilding next time,")
        print(f"# run kaggle/fetch.sh, then add wheels/{tag}/ to your 'gsplat-wheels' dataset")
        print("# (see kaggle/README_kaggle.md: `kaggle datasets version -p wheels_ds -m ... --dir-mode zip`).")
        print("#" * 78 + "\n", flush=True)
    run([sys.executable, "-m", "pip", "install", "-q"] + EXTRA_PIP, check_msg="pip install of extra packages failed")
    # Record the exact Python package versions of this run (e.g. torchmetrics), so results
    # can be tied to library versions later (added after Phase 1; they were not recorded).
    os.makedirs(RESULTS, exist_ok=True)
    run([sys.executable, "-m", "pip", "freeze"], log=f"{RESULTS}/_pip_freeze.txt", show=False)
    p = run([sys.executable, "-c",
             "import gsplat, torch; from gsplat.cuda._backend import _C; "
             "print(gsplat.__version__, _C is not None); "
             "import sys; sys.exit(0 if _C is not None else 3)"],
            cwd="/kaggle/working")
    if p.returncode != 0:
        die("gsplat import failed or CUDA extension _C is None")
    gv = p.stdout.split()
    gsplat_version = gv[0] if gv else "?"

    # ---- STEP 4: data
    step(4, "Locate data")
    scenes = sorted({j["scene"] for j in jobs})
    scene_dirs = {}
    for sc in scenes:
        hits = sorted(glob.glob(f"/kaggle/input/**/{sc}/transforms_train.json", recursive=True))
        if not hits:
            print("Contents of /kaggle/input (depth 3):", flush=True)
            run(["find", "/kaggle/input", "-maxdepth", "3"])
            die(f"scene '{sc}' not found: no <dir>/{sc}/transforms_train.json under /kaggle/input")
        scene_dirs[sc] = os.path.dirname(hits[0])
        print(f"{sc}: {scene_dirs[sc]}", flush=True)

    # ---- STEP 5: run jobs
    step(5, "Run jobs")
    p = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                       stdout=subprocess.PIPE, text=True, check=False)
    n_gpu = max(1, min(2, len([l for l in p.stdout.splitlines() if l.strip()])))
    print(f"Using {n_gpu} worker(s)", flush=True)
    q = queue.Queue()
    for j in jobs:
        q.put(j)
    records = []
    lock = threading.Lock()

    def worker(gpu):
        while True:
            try:
                j = q.get_nowait()
            except queue.Empty:
                return
            # Optional per-job keys: "name" (log/record name), "args" (extra CLI tokens for
            # this job only, appended last so they override), "script" (run this script
            # with "args" + --result_dir instead of train.py, e.g. research/bench_field.py).
            name = j.get("name") or f"{j['variant']}_{j['scene']}_{j['seed']}"
            log = f"{LOGS}/{name}.log"
            if "script" in j:
                cmd = [sys.executable, j["script"], "--result_dir", RESULTS] + list(j.get("args", []))
            else:
                cmd = [sys.executable, "research/train.py",
                       "--config", f"research/configs/{j['variant']}.json",
                       "--data_dir", scene_dirs[j["scene"]],
                       "--scene", j["scene"], "--seed", str(j["seed"]),
                       "--result_dir", RESULTS]
                if iters:
                    cmd += ["--iters", str(iters)]
                cmd += extra_args + list(j.get("args", []))
            print(f"[gpu{gpu}] START {name}", flush=True)
            t0 = time.time()
            try:
                with open(log, "w") as f:
                    rc = subprocess.run(cmd, cwd=REPO_DIR, stdout=f, stderr=subprocess.STDOUT, check=False,
                                        env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}).returncode
            except Exception as ex:
                rc = -1
                print(f"[gpu{gpu}] could not launch {name}: {ex}", flush=True)
            mins = (time.time() - t0) / 60
            print(f"[gpu{gpu}] DONE  {name} rc={rc} {mins:.1f} min", flush=True)
            if rc != 0:
                print(f"--- last 60 lines of {log} ---\n{tail(log, 60)}--- end ---", flush=True)
            with lock:
                records.append({"variant": j["variant"], "scene": j["scene"], "seed": j["seed"],
                                "returncode": rc, "minutes": round(mins, 2), "log": log})

    threads = [threading.Thread(target=worker, args=(g,)) for g in range(n_gpu)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # ---- STEP 6: summary
    step(6, "Summary")
    order = {(j["variant"], j["scene"], j["seed"]): i for i, j in enumerate(jobs)}
    records.sort(key=lambda r: order.get((r["variant"], r["scene"], r["seed"]), 0))
    summary = {"git_commit": git_commit, "ref": ref, "gpu_names": gpu_names, "torch_version": torch_version,
               "torch_cuda": torch_cuda, "gsplat_version": gsplat_version, "wheel": wheel_info, "jobs": records}
    with open(f"{RESULTS}/_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"{'variant':<20}{'scene':<12}{'seed':<6}{'rc':<5}{'min':>8}")
    for r in records:
        print(f"{r['variant']:<20}{r['scene']:<12}{r['seed']:<6}{r['returncode']:<5}{r['minutes']:>8.1f}")
    nfail = sum(1 for r in records if r["returncode"] != 0)
    print(f"\n{len(records) - nfail}/{len(records)} jobs succeeded; total {(time.time() - _T0) / 60:.1f} min", flush=True)
    sys.exit(1 if nfail else 0)


if __name__ == "__main__":
    main()
