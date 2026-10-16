#!/usr/bin/env bash
# The gauge-deck run chain: smoke first, then the four sequences in brief
# order.  ONE engine MATLAB at a time (pdi_batch.sh serializes); a failing
# sequence stops the chain so the rest of the box's day is not spent on it.
cd "$(dirname "$0")"
# Order: the brief's 1-2 (gseq1), then 3 (gseq2), then 5 (gseq4 -- a whole
# deliverable for ~75 min), then 4 (gseq3 -- the descent and the
# within-scan drift, the longest by far).  Deliverable 5 is moved ahead of
# 4 deliberately: if the box runs out of day, the cheap complete answer
# should be the one that landed.
for s in gsmoke gseq1 gseq2 gseq4 gseq3; do
    echo "[$(date '+%F %T')] === $s ==="
    ./$s.sh || { echo "[$(date '+%F %T')] $s FAILED -- chain stopped"; exit 1; }
done
echo "[$(date '+%F %T')] gmaster done"
