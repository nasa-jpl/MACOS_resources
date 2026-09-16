#!/usr/bin/env bash
# BRIEF_to_gauge_close item 2 -- what remains of item 4 on the SEED TAIL.
#
# All four runs carry 'bench.tail_from_mat',false, which forces the GEOMETRIC
# SEED tail.  That is stronger than copying a mat under the tag: there is no
# file to fall back to, so a run cannot silently inherit another bench's tail
# (the failure runs/ifoseq.sh guarded against by hand).
#
#   oapifol2  bench + loop  -- the servo: photons per cycle for a 3 pm hold
#                             under noise and the 2 pm per-actuator walk.
#   oapdesc2  bench + loop  -- the descent from 100 / 200 nm rms
#                             (loop.start_rms; the record's descent_oap stalls).
#   oapuw2    bench+battery -- the SAME break ladder as oapifo2 with the
#                             unwrapper ON (battery.unwrap).  CCMac's lens_deck
#                             captured from 150 nm with unwrap alone, so the
#                             120 nm break must be re-read like with like.
#   lensuw2   bench+battery -- the LENS rig's ladder through TODAY's code, which
#                             prints the wrap fraction at every rung.  The
#                             record's lens run predates that column, so the
#                             "OAP wraps at 120, lens never flags" comparison
#                             cannot be made from the committed reports: the
#                             wrap meter used to be printed only where the
#                             estimator broke.  This is the control.
#
# Together oapuw2 + lensuw2 decide the brief's question with ONE number per
# rung: the wrapped fraction, against the detector px per actuator each rig
# gets (also printed now, next to the ray affine).
set -u
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$here"
D="'bench.optics','oap','bench.POL_IN','source','bench.SRC_AT_FOCUS',true,'bench.D_RC_L2',125,'oap.OAP1_AOI',20,'oap.OAP2_AOI',25,'oap.OAP1_SIDE',1,'oap.OAP2_SIDE',-1,'bench.tail_from_mat',false"
B="'clear.BODY',struct('Baffle',50,'Detector',50,'TestOptic',90,'PZT',60)"

./tg96_batch.sh oapuw2   "$D,$B,'battery.unwrap',true,'battery.rows',{'base/single'},'stages',{'bench','battery'}"
./tg96_batch.sh lensuw2  "'bench.optics','lens','battery.rows',{'base/single'},'stages',{'bench','battery'}"
./tg96_batch.sh oapifol2 "$D,$B,'stages',{'bench','loop','figs'}"
./tg96_batch.sh oapdesc2 "$D,'loop.start_rms',[100 200],'stages',{'bench','loop','figs'}"
echo "[item2seq] done"
