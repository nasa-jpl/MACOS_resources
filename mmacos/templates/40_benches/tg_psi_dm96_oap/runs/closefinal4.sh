#!/usr/bin/env bash
# Item 4: the tail gate's two-leg test, run against the LATTICE measure and
# with the act_lam sweep in place -- plus a THIRD leg the brief does not name
# but which is the decisive one.
#
# WHY A THIRD LEG.  On the thk22 bench the BATTERY reports Stage C single
# actuator gain 0.9885, and the gate's lattice measure reported 0.9104 for the
# SAME bench and the SAME tail (thk22_tail's winner, kept because the gate is
# advisory).  A 7.9 % deficit on a tail the battery certifies -- so the 0.95
# threshold, which was set against the battery's est_matrix_tg, would REFUSE a
# good tail.  That is the point-sample defect in a new coat, and it is why
# enforcement is still not flipped.
#
# Re-verifying thk22_tail on its own bench, with the sweep, asks the question
# directly: does the gain climb toward the battery's 0.9885 as the Tikhonov
# weight falls?  A PLATEAU means the tail sets the number; still CLIMBING means
# the regularizer does, and then act_lam is what is wrong, not the tail and not
# the threshold.  verify_tail runs the gate ALONE, so this costs one placement
# plus one row, not a 150-evaluation tune.
#
#   gate3_win   objwin3_tail.mat  OAP design  -> must be REFUSED  (battery 0.0338)
#   gate3_lens  lens_tail.mat     lens rig    -> must be ACCEPTED (battery 0.9968)
#   gate3_thk   thk22_tail.mat    thk22 bench -> the calibration (battery 0.9885)
set -u
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$here"
OAP="'bench.optics','oap','bench.POL_IN','source','bench.SRC_AT_FOCUS',true,'bench.D_RC_L2',125,'oap.OAP1_AOI',20,'oap.OAP2_AOI',25,'oap.OAP1_SIDE',1,'oap.OAP2_SIDE',-1"
THK="'bench.optics','lens','bench.BS_T',5.8333,'bench.EDGE_MARGIN',4.0"

# wait for thk22 (step 3's gate run) -- bounded, on the wrapper's own marker
deadline=$(( $(date +%s) + 3600 ))
while ! grep -q '] exit' runs/thk22.log 2>/dev/null; do
    [ "$(date +%s)" -gt "$deadline" ] && { echo "[closefinal4] STOP: thk22 did not finish in 1 h"; exit 1; }
    sleep 30
done
echo "[closefinal4] thk22: $(grep -h '] exit' runs/thk22.log | tail -1)"
for i in $(seq 1 60); do pgrep -x MATLAB >/dev/null || break; sleep 30; done

echo "[closefinal4] $(date '+%F %T') leg 1/3: gate3_win (must be REFUSED)"
OAP_ENTRY=tg96_tail_batch ./oap_fold_batch.sh gate3_win  "$OAP,'verify_tail','objwin3_tail.mat'"
echo "[closefinal4] $(date '+%F %T') leg 2/3: gate3_lens (must be ACCEPTED)"
OAP_ENTRY=tg96_tail_batch ./oap_fold_batch.sh gate3_lens "'bench.optics','lens','verify_tail','lens_tail.mat'"
echo "[closefinal4] $(date '+%F %T') leg 3/3: gate3_thk (calibration vs battery 0.9885)"
OAP_ENTRY=tg96_tail_batch ./oap_fold_batch.sh gate3_thk  "$THK,'verify_tail','thk22_tail.mat'"
echo "[closefinal4] $(date '+%F %T') verdicts:"
grep -h 'TAIL VERIFY' runs/gate3_win.log runs/gate3_lens.log runs/gate3_thk.log 2>/dev/null
grep -h 'act_lam sweep' runs/gate3_win.log runs/gate3_lens.log runs/gate3_thk.log 2>/dev/null
echo "[closefinal4] queue drained"
