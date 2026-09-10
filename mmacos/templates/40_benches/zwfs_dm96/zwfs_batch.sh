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
if [ -n "$args" ]; then call="zwfs_run_batch('tag','$tag', $args)"; else call="zwfs_run_batch('tag','$tag')"; fi
echo "[$(date '+%F %T')] $call" | tee "$log"
cd "$here"
if command -v systemd-run >/dev/null 2>&1; then
    systemd-run --user --scope -q -p MemoryMax=${ZWFS_MEMMAX:-14G} \
        matlab -batch "$call" >> "$log" 2>&1
else
    matlab -batch "$call" >> "$log" 2>&1
fi
rc=$?
echo "[$(date '+%F %T')] exit $rc" | tee -a "$log"
exit $rc
