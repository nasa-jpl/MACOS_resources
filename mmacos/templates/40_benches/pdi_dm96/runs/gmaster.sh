#!/usr/bin/env bash
# The gauge-deck run chain: smoke first, then the four sequences in brief
# order.  ONE engine MATLAB at a time (pdi_batch.sh serializes); a failing
# sequence stops the chain so the rest of the box's day is not spent on it.
cd "$(dirname "$0")"
for s in gsmoke gseq1 gseq2 gseq3 gseq4; do
    echo "[$(date '+%F %T')] === $s ==="
    ./$s.sh || { echo "[$(date '+%F %T')] $s FAILED -- chain stopped"; exit 1; }
done
echo "[$(date '+%F %T')] gmaster done"
