#!/usr/bin/env bash
# the ZWFS sweep's remaining samplings, one model size per process (2026-10-13: the
# first sweep's 3.30 -> 1.65 step in one process corrupted the heap)
set -u
cd "$(dirname "$0")/.."
LOG=runs/ressweep_seq.log
echo "[$(date '+%F %T')] start zwfs 1.65" >> $LOG
./ressweep_zbatch.sh ressweep_zwfs165 "'s',1.65"
echo "[$(date '+%F %T')] end zwfs 1.65 rc=$?; start zwfs 4" >> $LOG
./ressweep_zbatch.sh ressweep_zwfs400 "'s',4"
echo "[$(date '+%F %T')] end zwfs 4 rc=$?" >> $LOG
touch runs/ressweep_zseq2.done
