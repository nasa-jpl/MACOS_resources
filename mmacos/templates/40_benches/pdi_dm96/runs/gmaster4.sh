#!/usr/bin/env bash
# The chain's tail, after gseq3 completed and gseq4's first run landed:
# the reference-arm walk (deliverable 5, with its run tags fixed -- `tr -d
# '-.'` read the leading '-' as an option and collapsed all three onto one
# name), then the pinhole diameter of record (deliverable 3).
cd "$(dirname "$0")"
for s in gseq4 gseq2; do
    echo "[$(date '+%F %T')] === $s ==="
    ./$s.sh || { echo "[$(date '+%F %T')] $s FAILED -- chain stopped"; exit 1; }
done
echo "[$(date '+%F %T')] gmaster4 done"
