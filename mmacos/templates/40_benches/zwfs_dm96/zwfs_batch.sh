#!/usr/bin/env bash
# zwfs_batch.sh -- run the ZWFS runner headless, memory-capped, logged.
#   ./zwfs_batch.sh <tag> [zwfs_run name/value args, MATLAB syntax]
#   ./zwfs_batch.sh ng385 "'NGRID',385, 'stages',{'battery','figs'}"
# Log: runs/<tag>.log (the report itself lands in runs/<tag>/).  MODEL 1024
# runs need ~10 GB; run them ONE AT A TIME (two model-1024 MATLABs have
# taken this box down).  MACOS_HOME must point at the engine tree.
set -u
here="$(cd "$(dirname "$0")" && pwd)"
tag="${1:?usage: zwfs_batch.sh <tag> [args]}";  shift
args="${*:-}"
export MACOS_HOME="${MACOS_HOME:-$HOME/dev/macos/macos_f90}"
mkdir -p "$here/runs"
log="$here/runs/$tag.log"
: > "$log"
if [ -n "$args" ]; then call="zwfs_run_batch('tag','$tag', $args)"; else call="zwfs_run_batch('tag','$tag')"; fi
cd "$here"
# ONE engine MATLAB at a time (2026-09-12 near-miss: a sequence script from an
# earlier session chained into a loop run while a second sequence started --
# 27 of 30 GB, 0 free).  Serialize: wait while any batch MATLAB of the DM-gauge
# runners is alive (pattern excludes this script's own command line), then hold
# a lock for the run itself so two waiting launchers cannot start together.
# ZWFS_NOWAIT=1 bypasses (dev-res jobs that fit beside a model-1024 run).
if [ -z "${ZWFS_NOWAIT:-}" ]; then
    while pgrep -f 'MATLAB -batch (zwfs|tg96|pdi|oap)[a-z0-9_]*batch' >/dev/null 2>&1; do
        echo "[$(date '+%F %T')] waiting: another DM-gauge batch MATLAB is running" >> "$log"
        sleep $((20 + RANDOM % 20))
    done
    if command -v flock >/dev/null 2>&1; then lockcmd="flock $here/runs/.batch.lock"; else lockcmd=""; fi   # macOS has no flock (2026-09-16: the Mac run died with exit 127); the wait loop above still serializes
else
    lockcmd=""
fi
echo "[$(date '+%F %T')] $call" | tee -a "$log"
if command -v systemd-run >/dev/null 2>&1; then
    $lockcmd systemd-run --user --scope -q -p MemoryMax=${ZWFS_MEMMAX:-14G} \
        matlab -batch "$call" >> "$log" 2>&1
else
    $lockcmd matlab -batch "$call" >> "$log" 2>&1
fi
rc=$?
echo "[$(date '+%F %T')] exit $rc" | tee -a "$log"
exit $rc
