#!/usr/bin/env bash
# Usage: kaggle/fetch.sh [dest]
set -euo pipefail
# New-style Kaggle tokens (KGAT_...) must be passed as KAGGLE_API_TOKEN, not KAGGLE_KEY.
if [[ "${KAGGLE_KEY:-}" == KGAT_* && -z "${KAGGLE_API_TOKEN:-}" ]]; then
  export KAGGLE_API_TOKEN="$KAGGLE_KEY"; unset KAGGLE_KEY
fi
[ -n "${KAGGLE_USERNAME:-}" ] || { echo "ERROR: KAGGLE_USERNAME is not set" >&2; exit 1; }
command -v kaggle >/dev/null 2>&1 || { echo "ERROR: kaggle CLI not found" >&2; exit 1; }
cd "$(dirname "${BASH_SOURCE[0]}")/.."
DEST="${1:-results/raw/$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$DEST"
kaggle kernels output "$KAGGLE_USERNAME/splat-runner" -p "$DEST"
echo "Downloaded to $DEST"

S="$(find "$DEST" -name _summary.json | head -n1)"
if [ -n "$S" ]; then
  echo "--- $S ---"
  python3 - "$S" "$DEST" <<'PY'
import json, os, sys
s = json.load(open(sys.argv[1]))
print(json.dumps(s, separators=(",", ":")))
dest = sys.argv[2]
for j in s.get("jobs", []):
    if j.get("returncode", 0) != 0:
        name = f"{j['variant']}_{j['scene']}_{j['seed']}.log"
        cands = [os.path.join(r, name) for r, _, fs in os.walk(dest) if name in fs]
        print(f"\n--- FAILED {name} (rc={j['returncode']}): last 30 lines ---")
        if cands:
            print("".join(open(cands[0], errors="replace").readlines()[-30:]))
        else:
            print("(log not found in download)")
PY
else
  echo "(no _summary.json found)"
fi
echo "--- result JSONs ---"
find "$DEST" -name '*.json' ! -name '_summary.json' -path '*results*' | sort || true
