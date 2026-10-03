#!/usr/bin/env bash
# Item 4, thicknesses: RE-RUN the retune and its gate run.
#
# WHY.  thk22_tail ran in the few seconds before the advisory-gate edit reached
# a freshly-started MATLAB, so it used the ENFORCING gate -- whose measure the
# two-leg test had just shown to be wrong -- and it REFUSED a converged winner:
#
#   TAIL GATE REFUSED the winner: actuator-space gain -0.8742 < 0.95.
#   Falling back to the GEOMETRIC SEED (gain -0.8803).
#
# Both readings are ~0.87; the threshold, not a difference between the tails,
# decided it.  So thk22_tail.mat holds the SEED (null 75.242 nm) and thk22 ran
# on it -- its report says "RE-TUNED set ... (null 75.242 nm; seed 75.242)",
# which is the tell: a retune whose null equals its seed's did not retune.
# The tuned winner was null 20.09 nm, 3.7x better.
#
# sub22_tail is NOT affected: it starts long after the edit, so it gets the
# advisory gate and keeps its winner.
#
# Cost: one retune (~15 min) plus one gate run (~25 min).
set -u
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$here"
THK_TG="'bench.BS_T',5.8333,'bench.EDGE_MARGIN',4.0"
OAP_ENTRY=tg96_tail_batch ./oap_fold_batch.sh thk22_tail "'bench.optics','lens',$THK_TG,'tag','thk22'"
./tg96_batch.sh thk22 "'bench.optics','lens',$THK_TG,'battery.rows',{'base/single'},'stages',{'bench','battery'}"
echo "[item4bseq] done"
