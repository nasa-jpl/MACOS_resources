#!/usr/bin/env bash
# Dave's deck questions (2026-09-12): the mask / focal-spot figure at the
# record's own sampling (193/1024 and 385/2048) and the vector reading's
# working-surface ladder past 60 nm.
cd "$(dirname "$0")/.."
./zwfs_batch.sh mask193 "'stages',{'bench','figs'}, 'readings',{'V'}, 'dm_use',1"
./zwfs_batch.sh vlad193 "'stages',{'bench','battery'}, 'readings',{'S','V'}, 'dm_use',1, 'battery.calib_surface','base', 'battery.rows',{'base/single'}, 'battery.ladder',[30 60 120 240 480]*1e-6, 'battery.ladder_sites','grid'"
ZWFS_MEMMAX=20G ./zwfs_batch.sh mask385 "'MODEL',2048, 'NGRID',385, 'param_file','macos_param_2048.txt', 'stages',{'bench','figs'}, 'readings',{'V'}, 'dm_use',1"
echo "[$(date '+%F %T')] v3seq3 done" >> runs/v3seq3.log
