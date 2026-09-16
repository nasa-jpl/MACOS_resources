#!/usr/bin/env bash
# BRIEF_to_gauge_close item 2, part (b) settled properly.
#
# oapuw2 / lensuw2 answered as far as the wrap meter could, and that turned out
# to be not far: max|h|/(lambda/4) SATURATES (measured 1.00 at 60, 120, 240 and
# 480 nm alike on the OAP rig), so it cannot separate a rung that holds from one
# that breaks.  The patched ladder prints the MEASURED base rms and the
# beyond-fold FRACTION instead, and the new `wrap` stage reports exactly those
# with NO matrix and NO rows -- minutes rather than the ~25 the battery costs,
# because the question is about the base READING, not about the rows.
#
# The two rigs, same code, same rungs.  What it decides:
#   - if both rigs break at the same FOLD FRACTION, the ladder is measuring its
#     own wrapped-absolute subtraction and "the reflective rig has a smaller
#     capture range" comes off the deck;
#   - if the lens rig reaches a given fraction at a HIGHER base rms, its
#     measurement attenuates more (meas/cmd is printed), and the difference is
#     real but is about measurement amplitude, not about sampling -- which is
#     excluded twice over already (the lens rig has 7% MORE px per actuator,
#     the wrong direction, and 7% cannot move a break on a ladder that doubles).
set -u
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$here"
D="'bench.optics','oap','bench.POL_IN','source','bench.SRC_AT_FOCUS',true,'bench.D_RC_L2',125,'oap.OAP1_AOI',20,'oap.OAP2_AOI',25,'oap.OAP1_SIDE',1,'oap.OAP2_SIDE',-1,'bench.tail_from_mat',false"
./tg96_batch.sh wrapoap  "$D,'stages',{'bench','wrap'}"
./tg96_batch.sh wraplens "'bench.optics','lens','stages',{'bench','wrap'}"
echo "[item2bseq] done"
