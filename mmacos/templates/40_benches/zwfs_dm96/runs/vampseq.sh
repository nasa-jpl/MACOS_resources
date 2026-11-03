#!/usr/bin/env bash
# BRIEF_to_gauge_close item 1, the leg vfit22 made necessary.
#
# Measured so far, bare Al, G4: uncalibrated 'ideal' 633.97 pm; 'fit' (the
# per-channel constants kappa+, kappa-, eta fitted on the flat DM's two masked
# images -- what a bench calibration does) 199.45 pm; 'map' (the true
# per-channel pupil maps, the oracle) 0.054 pm PASS.
#
# So constants alone buy 3.2x and stop 17x short of the gate, while the full
# maps clear it by 200x.  The difference between them is the PUPIL-VARYING
# part of the two channel maps.  'amp' is the leg that decides what that costs
# a real bench: the per-channel UNMASKED reference frames -- |qL|, |qR| known,
# the polarization phases NOT -- which every bench already takes for its own
# normalization.  If 'amp' lands near the gate, the redesigned rig needs no new
# calibration hardware at all; if it lands with 'fit', the rig needs a
# polarimetric calibration and that is a line item on the deck.
set -u
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$here"
D="'bench.optics','oap','bench.OAP1_AOI',20,'bench.OAP2_AOI',25,'bench.OAP1_SIDE',1,'bench.OAP2_SIDE',-1,'bench.SRC_AT_FOCUS',true,'bench.MASK_TRIM',0"
C="'MODEL',1024, 'NGRID',193, 'readings',{'V'}, 'stages',{'bench','battery'}, 'battery.rows',{'base/single','base/grid','base/rand'}, 'battery.calib_surface','base', 'mask.v_arm','engine'"
./zwfs_batch.sh vamp22 "$D, 'bench.coat_oap','bareAl', $C, 'mask.v_cal','amp'"
echo "[vampseq] done"
