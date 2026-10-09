#!/usr/bin/env bash
# V2 sequence (2026-09-12): metasurface retardance error priced -- G4 bias ladder (bench only),
# on-surface rows (ideal solver vs fit), and the loop with an uncalibrated 0.1 rad error
cd "$(dirname "$0")/.."
B="'stages',{'bench'}, 'readings',{'V'}, 'dm_use',1"
for e in 0.02 0.05 0.10 0.20; do
  ./zwfs_batch.sh v2g_e${e}_a0  "$B, 'mask.v_ret_err',$e, 'mask.v_leak_phase',0"
  ./zwfs_batch.sh v2g_e${e}_a90 "$B, 'mask.v_ret_err',$e, 'mask.v_leak_phase',pi/2"
done
./zwfs_batch.sh v2g_e0.10_a90fit "$B, 'mask.v_ret_err',0.10, 'mask.v_leak_phase',pi/2, 'mask.v_cal','fit'"
R="'stages',{'bench','battery'}, 'readings',{'V'}, 'dm_use',1, 'battery.calib_surface','base', 'battery.rows',{'base/single','base/grid','base/rand'}, 'battery.ladder',[30 60]*1e-6, 'battery.ladder_sites','grid'"
./zwfs_batch.sh v2e10a    "$R, 'mask.v_ret_err',0.10, 'mask.v_leak_phase',pi/2"
./zwfs_batch.sh v2e10afit "$R, 'mask.v_ret_err',0.10, 'mask.v_leak_phase',pi/2, 'mask.v_cal','fit'"
./zwfs_batch.sh v2e20a    "$R, 'mask.v_ret_err',0.20, 'mask.v_leak_phase',pi/2"
./zwfs_batch.sh v2loop "'stages',{'bench','loop','figs'}, 'readings',{'V'}, 'loop.readings',{'V'}, 'loop.nph',[1e13 1e15], 'loop.drifts',{'walk'}, 'loop.steps',1e-6, 'mask.v_ret_err',0.10, 'mask.v_leak_phase',pi/2"
