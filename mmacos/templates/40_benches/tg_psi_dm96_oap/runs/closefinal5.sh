#!/usr/bin/env bash
# The SEED-RELATIVE gate (Dave 2026-09-16), on the same three legs.
#
# The criterion is now |winner| / |seed| >= gate_rel, both measured through
# the same estimator on the same bench, so the estimator's systematic scale
# cancels instead of having to be calibrated.  Each leg therefore costs TWO
# rows (winner and seed) rather than one.
#
#   gate3r_win   objwin3_tail.mat  OAP design  -> must be REFUSED
#   gate3r_lens  lens_tail.mat     lens rig    -> must be ACCEPTED (battery 0.9968)
#   gate3r_thk   thk22_tail.mat    thk22 bench -> should be ACCEPTED (battery 0.9885)
#
# The absolute gate refused all three; the ratio should keep the two the
# battery certifies and still refuse the one it does not.
set -u
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$here"
OAP="'bench.optics','oap','bench.POL_IN','source','bench.SRC_AT_FOCUS',true,'bench.D_RC_L2',125,'oap.OAP1_AOI',20,'oap.OAP2_AOI',25,'oap.OAP1_SIDE',1,'oap.OAP2_SIDE',-1"
THK="'bench.optics','lens','bench.BS_T',5.8333,'bench.EDGE_MARGIN',4.0"
OAP_ENTRY=tg96_tail_batch ./oap_fold_batch.sh gate3r_win  "$OAP,'verify_tail','objwin3_tail.mat'"
OAP_ENTRY=tg96_tail_batch ./oap_fold_batch.sh gate3r_lens "'bench.optics','lens','verify_tail','lens_tail.mat'"
OAP_ENTRY=tg96_tail_batch ./oap_fold_batch.sh gate3r_thk  "$THK,'verify_tail','thk22_tail.mat'"
echo "[closefinal5] verdicts:"
grep -h 'TAIL VERIFY' runs/gate3r_*.log 2>/dev/null
echo "[closefinal5] queue drained"
