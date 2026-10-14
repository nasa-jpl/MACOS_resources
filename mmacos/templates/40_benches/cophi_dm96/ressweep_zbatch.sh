#!/usr/bin/env bash
# Headless, memory-capped launcher for ressweep_zrun (the resolution sweep, ZWFS family; model 2048 runs) (model 1024 ~ 11 GB).
# Usage: ./ressweep_batch.sh <logtag> [MATLAB-syntax name/value args]
#   ./ressweep_zbatch.sh ressweep_zwfs
set -u
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
tag="${1:?usage: cophi_batch.sh <tag> [args]}"; shift || true
args="$*"
export MACOS_HOME="${MACOS_HOME:-$HOME/dev/macos/macos_f90}"
mkdir -p "$here/runs"
call="ressweep_zrun_batch($args)"
cd "$here"
log="runs/${tag}.log"
echo "[cophi_batch] $call  (MemoryMax=${COPHI_MEMMAX:-20G})" | tee "$log"
# MemoryMax 16G, not 14: a model-1024 / 385 px descent with a 7548-column J'J
# peaked at 13.67 GB on 2026-09-18, 0.33 GB under the old default, and the
# failure mode is a systemd OOM kill HOURS into a run.  The box has 30 GB, so
# 16G still leaves headroom for one job plus the desktop -- but NOT for two, and
# that is what the serialization below is for.  A lane that sets COPHI_NOWAIT=1
# to run beside another model-1024 job is the case to watch.
# One DM-gauge batch MATLAB at a time on this box (the same wait + lock as
# zwfs_batch.sh / pdi_batch.sh): on 2026-09-15 a clear22 run at model 1024
# started beside a zwfs gate run at model 1024 and systemd-oomd killed
# VS Code and the gate run.  COPHI_NOWAIT=1 bypasses (dev-resolution jobs
# that fit beside a model-1024 run; a 64 GB box).
if [ -z "${COPHI_NOWAIT:-}" ]; then
    while pgrep -f 'MATLAB -batch (zwfs|tg96|pdi|oap|cophi|ressweep)[a-z0-9_]*batch' >/dev/null 2>&1; do
        echo "[$(date '+%F %T')] waiting: another DM-gauge batch MATLAB is running" >> "$log"
        sleep $((20 + RANDOM % 20))
    done
    if command -v flock >/dev/null 2>&1; then lockcmd="flock $here/../zwfs_dm96/runs/.batch.lock"; else lockcmd=""; fi   # macOS has no flock (2026-09-16: the Mac run died with exit 127); the wait loop above still serializes
else
    lockcmd=""
fi
if command -v systemd-run >/dev/null 2>&1; then
    $lockcmd systemd-run --user --scope -p MemoryMax="${COPHI_MEMMAX:-20G}" \
        matlab -batch "$call" >>"$log" 2>&1
else
    $lockcmd matlab -batch "$call" >>"$log" 2>&1
fi
rc=$?
echo "[cophi_batch] exit $rc" | tee -a "$log"
exit $rc
