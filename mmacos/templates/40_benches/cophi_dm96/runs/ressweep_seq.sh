#!/usr/bin/env bash
# The resolution sweep (BRIEF_to_resolution_sweep, 2026-10-13): IFO, then COPHI's
# four-bucket camera, then its lock-in reading; five samplings each (3.277 = the
# anchor, 2, 1.5, 1 via the camera's pixels; 4 via a finer source grid).  Serial:
# ressweep_batch.sh waits on and locks against every other DM-gauge batch.
set -u
cd "$(dirname "$0")/.."
LOG=runs/ressweep_seq.log
for g in ifo het het_iq; do
    echo "[$(date '+%F %T')] start $g" >> $LOG
    ./ressweep_batch.sh ressweep_$g "'gauge','$g'"
    echo "[$(date '+%F %T')] end $g rc=$?" >> $LOG
done
echo "[$(date '+%F %T')] ALL DONE" >> $LOG
touch runs/ressweep_seq.done
