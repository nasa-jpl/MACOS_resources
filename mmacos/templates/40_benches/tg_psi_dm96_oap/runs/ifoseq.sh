#!/usr/bin/env bash
# Item 4: the interferometer on the DESIGNED reflective bench.  The same stages
# as the lens record, on the geometry item 2 settled (OAP1 20 deg / OAP2 25 deg,
# sides +1/-1, the input polarizer in the source leg, the output optics 125 mm
# ahead of OAP2, the collimator fed at its TRUE focus) and on its own retuned
# tail.
#   oapifo  -- bench + battery + figs: the rows on the 30 nm surface, matrix
#              measured ON that surface (battery.calib_mode 'matrix', S10).
#   oapifol -- bench + loop + figs: the closed-loop hold metric (hour-class).
# The record's oap_tail.mat was fit on the 7-deg bench WITH the conjugate error,
# so it must not be reused: copy the retuned oap22d_tail.mat under each tag.
set -u
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$here"
D="'bench.optics','oap','bench.POL_IN','source','bench.SRC_AT_FOCUS',true,'bench.D_RC_L2',125,'oap.OAP1_AOI',20,'oap.OAP2_AOI',25,'oap.OAP1_SIDE',1,'oap.OAP2_SIDE',-1"
B="'clear.BODY',struct('Baffle',50,'Detector',50,'TestOptic',90,'PZT',60)"
# the tail retune must have landed
n=0
while [ ! -f oap22d_tail.mat ] && [ $n -lt 240 ]; do sleep 30; n=$((n+1)); done
if [ ! -f oap22d_tail.mat ]; then echo "[ifoseq] ABORT: oap22d_tail.mat never appeared"; exit 1; fi
cp -f oap22d_tail.mat oapifo_tail.mat
cp -f oap22d_tail.mat oapifol_tail.mat
./tg96_batch.sh oapifo  "$D,$B,'stages',{'bench','battery','figs','clearance'}"
./tg96_batch.sh oapifol "$D,$B,'stages',{'bench','loop','figs'}"
echo "[ifoseq] done"
