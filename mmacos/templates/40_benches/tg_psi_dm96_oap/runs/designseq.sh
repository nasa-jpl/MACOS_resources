#!/usr/bin/env bash
# Item 2, the design, at full resolution.
#   oap22d -- the chosen reflective bench: OAP1 20 deg / OAP2 25 deg, sides
#             +1/-1, the input polarizer in the source leg, the output optics
#             125 mm ahead of OAP2, the collimator fed at its true focus.
#             Stages bench + figs + clearance: the layout with the mirrors on
#             the beam and the measured part-by-part table.
# The tail retune (tg96_tail for 'oap') follows once the geometry is pinned --
# it is a fminsearch over the four tail parameters and takes far longer.
set -u
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$here"
D="'bench.optics','oap','bench.POL_IN','source','bench.SRC_AT_FOCUS',true,'bench.D_RC_L2',125,'oap.OAP1_AOI',20,'oap.OAP2_AOI',25,'oap.OAP1_SIDE',1,'oap.OAP2_SIDE',-1"
./tg96_batch.sh oap22d "$D,'stages',{'bench','figs','clearance'}"
echo "[designseq] done"
