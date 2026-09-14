#!/usr/bin/env bash
# Re-run the model-2048 battery leg of the pinhole trade (deliverable 3).
# pin10_2048 was killed by a SIGTERM from another lane at 14:54 on
# 2026-09-14, sixteen minutes in, and gseq2 moved on to pin10_loop.  This
# re-queues it; pdi_batch.sh's own lock makes it wait for whatever
# DM-gauge MATLAB is running, so it simply falls in behind pin10_loop.
cd "$(dirname "$0")/.."
export ZWFS_MEMMAX=20G
B="'readings',{'P','PF'}, 'dm_use',1"
S="'battery.calib_surface','base', 'battery.ladder_sites','grid'"
R3="'battery.rows',{'base/single','base/grid','base/rand'}"
BIG="'MODEL',2048, 'NGRID',385, 'param_file','macos_param_2048.txt'"
./pdi_batch.sh pin10_2048 "pdi_params, $BIG, 'stages',{'bench','battery','noise'}, $B, $S, $R3, 'pdi.DIA_LAMD',1.0, 'noise.readings',{'P','PF'}, 'noise.nstates',10.^(8:2:14), 'noise.nreal',6, 'battery.ladder',[30 60 120]*1e-6"
echo "gpin done"
