#!/usr/bin/env bash
# Headless, memory-capped launcher for oap_fold_solve (the OAP fold-angle
# clearance sweep).  Geometry only -- model 512 / 65 rays -- so it is light,
# but it takes the SAME one-at-a-time wait + lock as tg96_batch / zwfs_batch /
# pdi_batch: a sweep is many builds and the box is 32 GB.
# Usage: ./oap_fold_batch.sh <tag> [MATLAB-syntax name/value args]
#   OAP_ENTRY=<fn>  runs a different batch entry (default oap_fold_solve_batch);
#                   e.g. OAP_ENTRY=oap_conj_probe_batch ./oap_fold_batch.sh conj
set -u
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
tag="${1:?usage: oap_fold_batch.sh <tag> [args]}"; shift || true
args="$*"
export MACOS_HOME="${MACOS_HOME:-$HOME/dev/macos/macos_f90}"
mkdir -p "$here/runs"
entry="${OAP_ENTRY:-oap_fold_solve_batch}"
call="${entry}('tag','$tag'${args:+, $args})"
cd "$here"
log="runs/${tag}.log"
echo "[oap_fold_batch] $call  (MemoryMax=${TG96_MEMMAX:-14G})" | tee "$log"
if [ -z "${TG96_NOWAIT:-}" ]; then
    while pgrep -f 'MATLAB -batch (zwfs|tg96|pdi|oap)[a-z0-9_]*batch' >/dev/null 2>&1; do
        echo "[$(date '+%F %T')] waiting: another DM-gauge batch MATLAB is running" >> "$log"
        sleep $((20 + RANDOM % 20))
    done
    if command -v flock >/dev/null 2>&1; then lockcmd="flock $here/../zwfs_dm96/runs/.batch.lock"; else lockcmd=""; fi   # macOS has no flock (2026-09-16: the Mac run died with exit 127); the wait loop above still serializes
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
echo "[oap_fold_batch] exit $rc" | tee -a "$log"
exit $rc
