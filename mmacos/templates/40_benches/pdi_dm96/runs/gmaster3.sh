#!/usr/bin/env bash
# The chain after gseq1, re-ordered by BRIEF_to_capture.md (2026-09-13):
#   gcap   deliverable 9 -- the start-rms ladder BOTH WAYS (the deck's
#          capture slide); the unwrapper is deliverable 8, already gated
#   gseq3  the descent at 100 nm for P's shutter form, the fast-recal
#          probe, and the WITHIN-SCAN drift ("then the intra runs")
#   gseq4  the reference-arm walk ("and the reference-arm walk")
#   gseq2  the pinhole diameter of record (deliverable 3), last
# Supersedes gmaster2.sh, which was stopped mid-wait.
cd "$(dirname "$0")"
while pgrep -x -f 'bash ./gseq1.sh' >/dev/null 2>&1 || pgrep -f 'MATLAB -batch (zwfs|tg96|pdi)_run_batch' >/dev/null 2>&1; do
    sleep 60
done
echo "[$(date '+%F %T')] gseq1 finished; continuing"
for s in gcap gseq3 gseq4 gseq2; do
    echo "[$(date '+%F %T')] === $s ==="
    ./$s.sh || { echo "[$(date '+%F %T')] $s FAILED -- chain stopped"; exit 1; }
done
echo "[$(date '+%F %T')] gmaster3 done"
