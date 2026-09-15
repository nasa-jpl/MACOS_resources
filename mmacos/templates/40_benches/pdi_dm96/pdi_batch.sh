#!/usr/bin/env bash
# pdi_batch.sh -- run the point-diffraction runner headless, memory-capped, logged.
#   ./pdi_batch.sh <tag> [pdi_run name/value args, MATLAB syntax]
#   ./pdi_batch.sh pfdeck "pdi_params, 'pdi.bench','psri', 'readings',{'PF'}"
# Log: runs/<tag>.log (the report itself lands in runs/<tag>/).
#
# SERIALIZED WITH THE ZWFS AND TG96 RUNNERS: one engine MATLAB at a time on
# this box (a model-1024 run needs ~10 GB; two have taken it down).  Same
# process pattern and the SAME lock file as ../zwfs_dm96/zwfs_batch.sh, so a
# PDI job and a ZWFS job cannot start together.
set -u
here="$(cd "$(dirname "$0")" && pwd)"
lockdir="$here/../zwfs_dm96/runs"
tag="${1:?usage: pdi_batch.sh <tag> [args]}";  shift
args="${*:-}"
export MACOS_HOME="${MACOS_HOME:-$HOME/dev/macos/macos_f90}"
mkdir -p "$here/runs" "$lockdir"
log="$here/runs/$tag.log"
: > "$log"
if [ -n "$args" ]; then call="pdi_run_batch('tag','$tag', $args)"; else call="pdi_run_batch('tag','$tag')"; fi
cd "$here"
if [ -z "${ZWFS_NOWAIT:-}" ]; then
    while pgrep -f 'MATLAB -batch (zwfs|tg96|pdi|oap)[a-z0-9_]*batch' >/dev/null 2>&1; do
        echo "[$(date '+%F %T')] waiting: another DM-gauge batch MATLAB is running" >> "$log"
        sleep $((20 + RANDOM % 20))
    done
    lockcmd="flock $lockdir/.batch.lock"
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
