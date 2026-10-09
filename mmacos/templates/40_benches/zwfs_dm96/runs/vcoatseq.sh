#!/usr/bin/env bash
# Does the vector pair's regression come from the COATING at the larger folds,
# or from the fold geometry itself?
#
# On the designed bench (folds 20/25 deg) V reads 633.97 pm against the
# record's 19.6 at 5/9 deg -- while the pinhole IMPROVED 94 -> 0.269 pm.  Bare
# aluminium's diattenuation and retardance grow with incidence angle, so the
# larger folds are the obvious suspect; but that is a hypothesis, and the
# coating is separable from the geometry by simply removing it.
#   oapsens22n -- identical to oapsens22 except coat_oap 'none' (ideal
#                 reflectors: RS = -1, RP = +1, zero retardance).
# If V recovers towards the record's figure, the cost is the COATING at these
# angles -- a real trade line, and one a protected-Al or dielectric stack could
# be specified against.  If V stays at ~600 pm, the coating is exonerated and
# the fold geometry itself is what the vector pair cannot take.
set -u
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$here"
D="'bench.optics','oap','bench.OAP1_AOI',20,'bench.OAP2_AOI',25,'bench.OAP1_SIDE',1,'bench.OAP2_SIDE',-1,'bench.SRC_AT_FOCUS',true,'bench.MASK_TRIM',0,'bench.coat_oap','none'"
./zwfs_batch.sh oapsens22n "$D, 'MODEL',1024, 'NGRID',193, 'readings',{'V'}, 'stages',{'bench','battery'}, 'battery.rows',{'base/single'}, 'battery.calib_surface','base', 'mask.v_arm','engine'"
echo "[vcoatseq] done"
