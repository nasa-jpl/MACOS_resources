#!/usr/bin/env bash
# V3 record sequence, part 2 (2026-09-12): the engine-arm runs re-done with
# the projected-axes basis (dmg_arm_maps) + the 'amp' solver level (the
# per-channel unmasked reference frames), and the synthetic 0.3 rows under
# 'amp'.  Writes "v3seq2 done" at the end (TO's pseq6 waits for it).
cd "$(dirname "$0")/.."
B="'stages',{'bench'}, 'readings',{'V'}, 'dm_use',1"
./zwfs_batch.sh v3arm      "$B, 'mask.v_arm','engine'"
./zwfs_batch.sh v3arm_l90  "$B, 'mask.v_arm','engine', 'mask.v_laser_deg',90"
./zwfs_batch.sh v3arm_l0   "$B, 'mask.v_arm','engine', 'mask.v_laser_deg',0"
./zwfs_batch.sh v3arm_amp  "$B, 'mask.v_arm','engine', 'mask.v_cal','amp'"
./zwfs_batch.sh v3arm_fit  "$B, 'mask.v_arm','engine', 'mask.v_cal','fit'"
./zwfs_batch.sh v3arm_ar   "$B, 'mask.v_arm','engine', 'mask.v_arm_ar',true"
./zwfs_batch.sh v3arm_map  "$B, 'mask.v_arm','engine', 'mask.v_cal','map'"
R="'stages',{'bench','battery'}, 'readings',{'V'}, 'dm_use',1, 'battery.calib_surface','base', 'battery.rows',{'base/single','base/grid','base/rand'}, 'battery.ladder',[30 60]*1e-6, 'battery.ladder_sites','grid'"
./zwfs_batch.sh v3arm_base  "$R, 'mask.v_arm','engine'"
./zwfs_batch.sh v3s_a0.3amp "$R, 'mask.v_arm','synthetic', 'mask.v_arm_damp',0.3, 'mask.v_cal','amp'"
./zwfs_batch.sh v3s_p0.3amp "$R, 'mask.v_arm','synthetic', 'mask.v_arm_dphase',0.3, 'mask.v_cal','amp'"
echo "[$(date '+%F %T')] v3seq2 done" >> runs/v3seq2.log
