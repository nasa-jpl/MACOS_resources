#!/usr/bin/env bash
# BRIEF_to_gauge_close -- the tail of the queue, after chain5 drains.
# item2bseq needs tg96_run.m PATCHED first (the saturating wrap meter replaced,
# the camera line moved to dmg_cam_line, the wrap stage added); the patch is
# staged and is applied by hand once the runs in flight release the file, so
# this chain refuses to start if the stage is not there rather than running two
# jobs that would print nothing new.
set -u
T=/home/dcr/dev/MACOS_resources/mmacos/templates/40_benches/tg_psi_dm96_oap/runs
n=0
while ! grep -q '\[closechain5\] queue drained' "$T/closechain5.nohup" 2>/dev/null; do
    n=$((n+1)); [ $n -gt 1440 ] && { echo "[closechain6] ABORT: chain5 never drained (12 h)"; exit 1; }
    sleep 30
done
if ! grep -q 'function stage_wrap_' "$T/../tg96_run.m"; then
    echo "[closechain6] STOP: tg96_run.m has no wrap stage -- apply the staged patch first"
    exit 1
fi
echo "[closechain6] chain5 drained -> item 2's wrap stage on both rigs"
"$T/item2bseq.sh" > "$T/item2bseq.nohup" 2>&1
echo "[closechain6] queue drained"
