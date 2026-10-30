#!/usr/bin/env bash
# Usage: kaggle/launch.sh <config.json>
set -euo pipefail
# New-style Kaggle tokens (KGAT_...) must be passed as KAGGLE_API_TOKEN, not KAGGLE_KEY.
if [[ "${KAGGLE_KEY:-}" == KGAT_* && -z "${KAGGLE_API_TOKEN:-}" ]]; then
  export KAGGLE_API_TOKEN="$KAGGLE_KEY"; unset KAGGLE_KEY
fi

CFG="${1:-}"
[ -n "$CFG" ] && [ -f "$CFG" ] || { echo "usage: $0 <config.json>" >&2; exit 2; }
command -v kaggle >/dev/null 2>&1 || { echo "ERROR: kaggle CLI not found (pip install kaggle)" >&2; exit 1; }
[ -n "${KAGGLE_USERNAME:-}" ] || { echo "ERROR: KAGGLE_USERNAME is not set" >&2; exit 1; }
[ -n "${KAGGLE_KEY:-}" ] || [ -f "$HOME/.kaggle/kaggle.json" ] || [ -n "${KAGGLE_API_TOKEN:-}" ] \
  || echo "WARNING: no Kaggle credentials found (KAGGLE_KEY / ~/.kaggle/kaggle.json)" >&2

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python3 - "$CFG" "$ROOT" <<'PY'
import json, os, subprocess, sys
cfg_path, root = sys.argv[1:3]
try:
    cfg = json.load(open(cfg_path))
except Exception as e:
    sys.exit(f"ERROR: {cfg_path} is not valid JSON: {e}")
for k in ("repo_url", "ref", "jobs"):
    if k not in cfg:
        sys.exit(f"ERROR: config missing '{k}'")
if not cfg["jobs"]:
    sys.exit("ERROR: config has no jobs")
for i, j in enumerate(cfg["jobs"]):
    for k in ("variant", "scene", "seed"):
        if k not in j:
            sys.exit(f"ERROR: job {i} missing '{k}'")
    if "script" in j:  # non-training job (e.g. research/bench_field.py)
        if not os.path.isfile(os.path.join(root, j["script"])):
            sys.exit(f"ERROR: job {i}: script {j['script']} does not exist")
        continue
    p = os.path.join(root, "research", "configs", j["variant"] + ".json")
    if not os.path.isfile(p):
        sys.exit(f"ERROR: job {i}: {p} does not exist")
if "'''" in json.dumps(cfg):
    sys.exit("ERROR: config contains ''' which cannot be embedded in run.py")
# warn about unpushed commits
def git(*a):
    r = subprocess.run(["git", "-C", root, *a], capture_output=True, text=True)
    return r.returncode, r.stdout.strip()
ref = cfg["ref"]
rc, _ = git("rev-parse", "--verify", "--quiet", f"refs/heads/{ref}")
if rc == 0:
    rc, out = git("rev-list", "--count", f"origin/{ref}..{ref}")
    if rc != 0:
        print(f"WARNING: branch '{ref}' has no origin/{ref}; has it been pushed?", file=sys.stderr)
    elif out != "0":
        print(f"WARNING: local branch '{ref}' has {out} unpushed commit(s); Kaggle will not see them", file=sys.stderr)
print(f"Config OK: {len(cfg['jobs'])} job(s), ref={ref}")
PY

rm -rf kaggle/build
mkdir -p kaggle/build
# DATASET_OWNER: account that owns nerf-synthetic / gsplat-wheels. Teammates launching
# from their own accounts set DATASET_OWNER=adk7712 once those datasets are shared with them.
python3 - "$CFG" "$KAGGLE_USERNAME" "${DATASET_OWNER:-$KAGGLE_USERNAME}" <<'PY'
import json, sys
cfg_path, user, owner = sys.argv[1:4]
cfg = json.load(open(cfg_path))
meta = json.load(open("kaggle/kernel-metadata.json"))
meta["id"] = f"{user}/splat-runner"
meta["dataset_sources"] = [f"{owner}/{d.split('/', 1)[1]}" for d in meta["dataset_sources"]]
json.dump(meta, open("kaggle/build/kernel-metadata.json", "w"), indent=2)
src = open("kaggle/run.py").read()
needle = "r'''__RUN_CONFIG__'''"
assert src.count(needle) == 1, "placeholder not found exactly once in run.py"
compact = json.dumps(cfg, separators=(",", ":"))
open("kaggle/build/run.py", "w").write(src.replace(needle, "r'''" + compact + "'''"))
json.dump(cfg, open("kaggle/build/run_config.json", "w"), indent=2)
PY
python3 -m py_compile kaggle/build/run.py
rm -rf kaggle/build/__pycache__

kaggle kernels push -p kaggle/build --accelerator NvidiaTeslaT4

cat <<MSG

Pushed. Next:
  kaggle/status.sh --wait      # poll until finished
  kaggle/fetch.sh              # download outputs to results/raw/<timestamp>/
  python research/aggregate.py results/raw
MSG
