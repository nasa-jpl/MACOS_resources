#!/usr/bin/env bash
# V3 record sequence (2026-09-12): the arm's polarization aberration per
# channel.  Engine arm (laser 45 / 90 deg, bare / AR-coated, ideal / oracle
# solver), the engine arm through the on-surface matrix, the synthetic scan
# (differential phase = the diattenuation-type term, differential amplitude
# = the retardance-type term), the oracle at the worst level, the loop.
cd "$(dirname "$0")/.."
B="'stages',{'bench'}, 'readings',{'V'}, 'dm_use',1"
./zwfs_batch.sh v3arm      "$B, 'mask.v_arm','engine'"
./zwfs_batch.sh v3arm_l90  "$B, 'mask.v_arm','engine', 'mask.v_laser_deg',90"
./zwfs_batch.sh v3arm_ar   "$B, 'mask.v_arm','engine', 'mask.v_arm_ar',true"
./zwfs_batch.sh v3arm_map  "$B, 'mask.v_arm','engine', 'mask.v_cal','map'"
R="'stages',{'bench','battery'}, 'readings',{'V'}, 'dm_use',1, 'battery.calib_surface','base', 'battery.rows',{'base/single','base/grid','base/rand'}, 'battery.ladder',[30 60]*1e-6, 'battery.ladder_sites','grid'"
./zwfs_batch.sh v3arm_base "$R, 'mask.v_arm','engine'"
for a in 0.01 0.03 0.1 0.3; do
  ./zwfs_batch.sh v3s_p$a "$R, 'mask.v_arm','synthetic', 'mask.v_arm_dphase',$a"
  ./zwfs_batch.sh v3s_a$a "$R, 'mask.v_arm','synthetic', 'mask.v_arm_damp',$a"
done
./zwfs_batch.sh v3s_p0.3map "$R, 'mask.v_arm','synthetic', 'mask.v_arm_dphase',0.3, 'mask.v_cal','map'"
./zwfs_batch.sh v3loop "'stages',{'bench','loop','figs'}, 'readings',{'V'}, 'loop.readings',{'V'}, 'loop.nph',[1e13 1e15], 'loop.drifts',{'walk'}, 'loop.steps',1e-6, 'mask.v_arm','synthetic', 'mask.v_arm_dphase',0.3"
echo "[$(date '+%F %T')] v3seq done" >> runs/v3seq.log
