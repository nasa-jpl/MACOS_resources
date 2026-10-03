#!/usr/bin/env bash
# Repeatability of the seed-relative ratio on the TIGHTEST leg.
#
# lens_tail passes at ratio 0.9467 against gate_rel 0.90 -- a margin of only
# 0.047.  Whether that margin is comfortable depends on the run-to-run spread
# of the ratio, which is TWO independently placed row measurements and has
# never been measured.  If the spread is a few 1e-3 the margin is ample; if it
# is a few 1e-2 then 0.90 is close to the noise and the constant needs to move
# DOWN (never the measure up to meet it).
set -u
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$here"
for i in 1 2; do
  OAP_ENTRY=tg96_tail_batch ./oap_fold_batch.sh rpt_lens$i "'bench.optics','lens','verify_tail','lens_tail.mat'"
done
echo "[repeatseq] done"
