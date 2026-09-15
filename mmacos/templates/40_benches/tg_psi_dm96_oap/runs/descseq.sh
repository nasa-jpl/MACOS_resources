#!/usr/bin/env bash
# Item 4's third leg: the DESCENT on the designed reflective bench -- can the
# loop capture the DM's initial figure and walk it down to the set point?  The
# record's reflective rig does NOT (runs/descent_oap, on the 7-deg bench with
# the 25 mm conjugate error): it stalls at 5856 / 23557 / 53150 pm from starts
# of 60 / 150 / 300 nm, never reaching 10 nm from 150 or 300.  Same ladder,
# same seeds, on the corrected bench.
set -u
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$here"
D="'bench.optics','oap','bench.POL_IN','source','bench.SRC_AT_FOCUS',true,'bench.D_RC_L2',125,'oap.OAP1_AOI',20,'oap.OAP2_AOI',25,'oap.OAP1_SIDE',1,'oap.OAP2_SIDE',-1"
cp -f oap22d_tail.mat oapdesc_tail.mat
./tg96_batch.sh oapdesc "$D,'loop.start_rms',[60 150 300],'stages',{'bench','loop','figs'}"
echo "[descseq] done"
