#!/usr/bin/env bash
# Usage: kaggle/status.sh [--wait]
set -euo pipefail
[ -n "${KAGGLE_USERNAME:-}" ] || { echo "ERROR: KAGGLE_USERNAME is not set" >&2; exit 1; }
command -v kaggle >/dev/null 2>&1 || { echo "ERROR: kaggle CLI not found" >&2; exit 1; }
K="$KAGGLE_USERNAME/splat-runner"
if [ "${1:-}" != "--wait" ]; then
  kaggle kernels status "$K"
  exit 0
fi
P="${POLL_SECONDS:-300}"
[ "$P" -ge 120 ] 2>/dev/null || P=120
while true; do
  out="$(kaggle kernels status "$K" 2>&1 || true)"
  echo "[$(date -u +%FT%TZ)] $out"
  low="$(echo "$out" | tr 'A-Z' 'a-z')"
  case "$low" in
    *error*|*cancel*) exit 1 ;;
    *complete*) exit 0 ;;
  esac
  sleep "$P"
done
