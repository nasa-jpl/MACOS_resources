#!/usr/bin/env bash
# Continue the chain after gseq1, CHEAPEST COMPLETE DELIVERABLE FIRST:
# 5 (gseq4, ~75 min), then 3 (gseq2, ~4 h -- its value is all in the
# model-2048 leg), then 4 (gseq3, the descent and the within-scan drift,
# ~7 h).  If the box runs out of day, the deliverables that landed should
# be whole ones.
#
# gmaster.sh was edited while it was running (its loop list was already
# parsed, so the edit could not take effect and the byte offsets moved).
# Its loop was stopped; this script picks up from gseq1's end.
cd "$(dirname "$0")"
while pgrep -x -f 'bash ./gseq1.sh' >/dev/null 2>&1 || pgrep -f 'MATLAB -batch (zwfs|tg96|pdi)_run_batch' >/dev/null 2>&1; do
    sleep 60
done
echo "[$(date '+%F %T')] gseq1 finished; continuing"
for s in gseq4 gseq2 gseq3; do
    echo "[$(date '+%F %T')] === $s ==="
    ./$s.sh || { echo "[$(date '+%F %T')] $s FAILED -- chain stopped"; exit 1; }
done
echo "[$(date '+%F %T')] gmaster2 done"
