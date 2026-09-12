#!/usr/bin/env bash
# PDI record sequence 2 (2026-09-12, the P/SRI fiber reference re-pinned to Dube 2024):
# flat-matrix battery + noise (S V P PF); on-surface battery; shutter/pinhole-only frame per state
# (b2 'state'); the 1 lam/D pinhole; the step scheme trade (2% step error, ls vs sh5); the loop rows.
cd "$(dirname "$0")/.."
B="'readings',{'S','V','P','PF'}, 'dm_use',1"
S="'battery.calib_surface','base', 'battery.ladder_sites','grid'"
./zwfs_batch.sh pdi193f     "'stages',{'bench','battery','noise','figs'}, $B"
./zwfs_batch.sh pdi193fbase "'stages',{'bench','battery','figs'}, $B, $S"
./zwfs_batch.sh pdi193state "'stages',{'bench','battery','noise','figs'}, 'readings',{'S','P','PF'}, 'dm_use',1, $S, 'pdi.b2','state'"
./zwfs_batch.sh pdi193d1    "'stages',{'bench','battery','noise','figs'}, 'readings',{'S','P'}, 'dm_use',1, $S, 'pdi.DIA_LAMD',1.0"
./zwfs_batch.sh pdi193se_ls  "'stages',{'bench','battery'}, 'readings',{'P','PF'}, 'dm_use',1, $S, 'pdi.step_err',0.02, 'battery.rows',{'base/single','base/grid','base/rand'}, 'battery.ladder',[30 60]*1e-6"
./zwfs_batch.sh pdi193se_sh5 "'stages',{'bench','battery'}, 'readings',{'P','PF'}, 'dm_use',1, $S, 'pdi.step_err',0.02, 'pdi.scheme','sh5', 'battery.rows',{'base/single','base/grid','base/rand'}, 'battery.ladder',[30 60]*1e-6"
./zwfs_batch.sh ploop193    "'stages',{'bench','loop','figs'}, 'readings',{'L','S','P','PF'}, 'loop.readings',{'P','PF'}"
