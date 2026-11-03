#!/usr/bin/env bash
# A/B on the ONE thing I changed in the reading chain: where the input
# polarizer sits.  The designed bench reads with a FLAT gain deficit -- 0.10 to
# 0.21 from 0.7 to 45 cyc/pupil, against the lens rig's 0.988 at every mode.
# Flat in spatial frequency is a scale/contrast signature, not an aberration,
# and the four-step's modulation is set by the polarization state at the
# splitter.  POL_IN 'source' is the only change in that chain (the axis is
# reflected through OAP1 so the state SHOULD be the record's -- reasoned, never
# measured).
#   polA -- the design, POL_IN 'source'   (as measured, the suspect)
#   polB -- the design, POL_IN 'collimated' (the record's placement; it loses
#           ~11 % of rays into the plate at these folds, so it is a DIAGNOSTIC
#           not a candidate design -- but if the gain comes back, the
#           relocation is the cause)
# Model 512 / NGRID 193: Stage C and D discriminate at that resolution.
set -u
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$here"
G="'bench.optics','oap','bench.SRC_AT_FOCUS',true,'bench.D_RC_L2',125,'oap.OAP1_AOI',20,'oap.OAP2_AOI',25,'oap.OAP1_SIDE',1,'oap.OAP2_SIDE',-1"
R="'MODEL',512,'NGRID',193,'grid.N_G',256,'grid.DX_G',0.42,'smoke',true,'stages',{'bench','battery'}"
cp -f oap22d_tail.mat polA_tail.mat
cp -f oap22d_tail.mat polB_tail.mat
./tg96_batch.sh polA "$G,'bench.POL_IN','source',$R"
./tg96_batch.sh polB "$G,'bench.POL_IN','collimated',$R"
echo "[polabseq] done"
