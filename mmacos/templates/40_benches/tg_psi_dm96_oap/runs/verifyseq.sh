#!/usr/bin/env bash
# THE test that matters for the re-tuned tail: does it READ?
#
# The tune's winner satisfies every new diagnostic -- conc 1.000, wrap 0.63,
# frac 1.000 -- but its DET_TRIM is 46.83, within 1 mm of the 45.96 that the
# OLD objective picked and that reads actuators at gain 0.03.  conc and wrap
# are proxies.  The gain is the truth, and the whole lesson of this arc is that
# a proxy can be satisfied by a broken configuration.
#
#   oapfixb -- the design + the RE-TUNED tail, model 512 / NGRID 193, so it is
#              directly comparable with tailA (tuned/old, gain 0.0338) and
#              tailB (geometric seed, gain 0.9809).
# Sequential (no NOWAIT): takes the shared lock, so it queues behind the gate.
set -u
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$here"
cp -f oapfix_tail.mat oapfixb_tail.mat
D="'bench.optics','oap','bench.POL_IN','source','bench.SRC_AT_FOCUS',true,'bench.D_RC_L2',125,'oap.OAP1_AOI',20,'oap.OAP2_AOI',25,'oap.OAP1_SIDE',1,'oap.OAP2_SIDE',-1"
R="'MODEL',512,'NGRID',193,'grid.N_G',256,'grid.DX_G',0.42,'smoke',true,'stages',{'bench','battery'}"
./tg96_batch.sh oapfixb "$D,$R"
echo "[verifyseq] done"
