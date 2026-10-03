#!/usr/bin/env bash
# BRIEF_to_gauge_close item 1, the leg the first two runs made necessary.
#
# Measured in vmap22 / vqw22 / oapsens22 / oapsens22n: the V3 channel PHASE
# difference is 0.0615-0.0620 rad in ALL FOUR configurations, while G4 moves
# 208 -> 319 -> 634 pm.  So on this rig the phase-difference rung the brief
# reads the 634 pm against is NOT the variable; the channel AMPLITUDE
# imbalance is (|qL|/|qR| = 1.0000 none, 1.0313 qwAl, 1.0849 bareAl).
# zwfs_params says as much next to v_arm_damp -- "the OAP rig's term".
#
# That imbalance is very nearly a per-channel CONSTANT, which is what a bench
# calibration removes.  'ideal' knows nothing and 'map' is the oracle; 'fit'
# is the one a real bench performs -- per-channel kappa+, kappa- and eta
# fitted on the flat DM's two masked images.  If 'fit' recovers most of the
# 634 -> 0.054 pm that 'map' recovers, the redesigned rig's vector pair needs
# a routine calibration, not a coating or a fold change, and that is what the
# deck's slide should say.
set -u
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$here"
D="'bench.optics','oap','bench.OAP1_AOI',20,'bench.OAP2_AOI',25,'bench.OAP1_SIDE',1,'bench.OAP2_SIDE',-1,'bench.SRC_AT_FOCUS',true,'bench.MASK_TRIM',0"
C="'MODEL',1024, 'NGRID',193, 'readings',{'V'}, 'stages',{'bench','battery'}, 'battery.rows',{'base/single','base/grid','base/rand'}, 'battery.calib_surface','base', 'mask.v_arm','engine'"
./zwfs_batch.sh vfit22 "$D, 'bench.coat_oap','bareAl', $C, 'mask.v_cal','fit'"
echo "[vfitseq] done"
