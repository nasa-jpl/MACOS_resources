#!/usr/bin/env bash
# Re-gate the fixed tail objective STRICTLY SEQUENTIALLY.
#
# The first two attempts at this gate were contaminated: tg96_tail wrote fixed
# scratch deck names into the template directory and I ran each pair
# concurrently, so the two processes read and wrote each other's decks.  The
# filenames are now per-process (tail_<pid>_<tag>_*), which removes the hazard
# at the source -- but this script ALSO runs them one at a time, because a gate
# whose result decides whether a fix is proven should not depend on a race
# having been fixed correctly.  Two independent guards, deliberately.
#
# Each probe reports only its FIRST TAILEVAL: the cost of a pinned parameter
# set under the new objective.  What the gate must show is that the fixed cost
# RANKS them correctly -- the geometric seed (which reads at gain 0.99,
# runs/tailB) BELOW the old tuned winner (which reads at 0.03, runs/tailA).
#   objseed3 -- the geometric seed
#   objwin3  -- the old winner: FL_F 38.0937, FL_Kc -2.64620,
#               D_MASK_FL 1.9836, DET_TRIM 45.9613 (passed unscaled by s)
# No TG96_NOWAIT: each call takes the shared one-at-a-time lock, so this also
# queues behind the tune that is running now.
set -u
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$here"
s=$(python3 -c "print(96/56)")
G="'bench.optics','oap','bench.POL_IN','source','bench.SRC_AT_FOCUS',true,'bench.D_RC_L2',125,'oap.OAP1_AOI',20,'oap.OAP2_AOI',25,'oap.OAP1_SIDE',1,'oap.OAP2_SIDE',-1"
W="'bench.FL_F',38.0937/(96/56),'bench.FL_Kc',-2.64620,'bench.D_MASK_FL',1.9836/(96/56),'bench.DET_TRIM',45.9613/(96/56)"
OAP_ENTRY=tg96_tail_batch ./oap_fold_batch.sh objseed3 "$G,'tag','objseed3'"
OAP_ENTRY=tg96_tail_batch ./oap_fold_batch.sh objwin3  "$G,$W,'tag','objwin3'"
echo "[gateseq2] done"
