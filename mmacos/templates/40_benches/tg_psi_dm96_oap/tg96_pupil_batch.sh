#!/usr/bin/env bash
# Headless launcher for the pupil-image tools (tg96_pupilq + tg96_pupilsim), model 512, ~6 min per rig.
# Usage: ./tg96_pupil_batch.sh <lens|oap|both> [MATLAB-syntax name/value args]
#   ./tg96_pupil_batch.sh both
#   ./tg96_pupil_batch.sh lens "'tool','sim','fourier',false"
set -u
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
rig="${1:?usage: tg96_pupil_batch.sh <lens|oap|both> [args]}"; shift || true
args="$*"
export MACOS_HOME="${MACOS_HOME:-$HOME/dev/macos/macos_f90}"
mkdir -p "$here/runs"
call="tg96_pupil_batch('rig','$rig'${args:+, $args})"
cd "$here"
log="runs/pupil_${rig}.log"
echo "[tg96_pupil_batch] $call  (MemoryMax=${TG96_MEMMAX:-8G})" | tee "$log"
# One DM-gauge batch MATLAB at a time on the 32 GB box (the same wait + lock as tg96_batch.sh);
# TG96_NOWAIT=1 bypasses on a 64 GB box (these are model-512 jobs, ~3 GB).
if [ -z "${TG96_NOWAIT:-}" ]; then
    while pgrep -f 'MATLAB -batch (zwfs|tg96|pdi|oap)[a-z0-9_]*batch' >/dev/null 2>&1; do
        echo "[$(date '+%F %T')] waiting: another DM-gauge batch MATLAB is running" >> "$log"
        sleep $((20 + RANDOM % 20))
    done
    if command -v flock >/dev/null 2>&1; then lockcmd="flock $here/../zwfs_dm96/runs/.batch.lock"; else lockcmd=""; fi
else
    lockcmd=""
fi
if command -v systemd-run >/dev/null 2>&1; then
    $lockcmd systemd-run --user --scope -p MemoryMax="${TG96_MEMMAX:-8G}" matlab -batch "$call" >>"$log" 2>&1
else
    $lockcmd matlab -batch "$call" >>"$log" 2>&1
fi
rc=$?
echo "[tg96_pupil_batch] exit $rc" | tee -a "$log"
exit $rc
