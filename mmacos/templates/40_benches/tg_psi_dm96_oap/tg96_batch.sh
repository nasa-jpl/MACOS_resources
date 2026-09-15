#!/usr/bin/env bash
# Headless, memory-capped launcher for tg96_run (model 1024 ~ 11 GB).
# Usage: ./tg96_batch.sh <tag> [MATLAB-syntax name/value args]
#   ./tg96_batch.sh lens
#   ./tg96_batch.sh oap "'bench.optics','oap'"
set -u
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
tag="${1:?usage: tg96_batch.sh <tag> [args]}"; shift || true
args="$*"
export MACOS_HOME="${MACOS_HOME:-$HOME/dev/macos/macos_f90}"
mkdir -p "$here/runs"
call="tg96_run_batch('tag','$tag'${args:+, $args})"
cd "$here"
log="runs/${tag}.log"
echo "[tg96_batch] $call  (MemoryMax=${TG96_MEMMAX:-14G})" | tee "$log"
# One DM-gauge batch MATLAB at a time on this box (the same wait + lock as
# zwfs_batch.sh / pdi_batch.sh): on 2026-09-15 a clear22 run at model 1024
# started beside a zwfs gate run at model 1024 and systemd-oomd killed
# VS Code and the gate run.  TG96_NOWAIT=1 bypasses (dev-resolution jobs
# that fit beside a model-1024 run; a 64 GB box).
if [ -z "${TG96_NOWAIT:-}" ]; then
    while pgrep -f 'MATLAB -batch (zwfs|tg96|pdi)_run_batch' >/dev/null 2>&1; do
        echo "[$(date '+%F %T')] waiting: another DM-gauge batch MATLAB is running" >> "$log"
        sleep $((20 + RANDOM % 20))
    done
    lockcmd="flock $here/../zwfs_dm96/runs/.batch.lock"
else
    lockcmd=""
fi
if command -v systemd-run >/dev/null 2>&1; then
    $lockcmd systemd-run --user --scope -p MemoryMax="${TG96_MEMMAX:-14G}" \
        matlab -batch "$call" >>"$log" 2>&1
else
    $lockcmd matlab -batch "$call" >>"$log" 2>&1
fi
rc=$?
echo "[tg96_batch] exit $rc" | tee -a "$log"
exit $rc
