#!/usr/bin/env bash
# Gauge-deck sequence 2 (deliverable 3): the pinhole diameter of record --
# 2.0 lam/D at MODEL 1024 / 193 rays against 1.0 lam/D at 2048 / 385 (where a
# 1.0 lam/D pinhole is sampled as well as 2.0 is at 1024/193).  Rows on the
# 30 nm surface and one loop run each.  Both are recorded; the choice is
# stated with its numbers.
cd "$(dirname "$0")/.."
B="'readings',{'P','PF'}, 'dm_use',1"
S="'battery.calib_surface','base', 'battery.ladder_sites','grid'"
R3="'battery.rows',{'base/single','base/grid','base/rand'}"
L="'loop.readings',{'P','PF'}, 'loop.nph',[1e13 1e15], 'loop.drifts',{'walk'}, 'loop.steps',[1e-6], 'loop.floor',false"
BIG="'MODEL',2048, 'NGRID',385, 'param_file','macos_param_2048.txt'"

./pdi_batch.sh pin20_1024 "pdi_params, 'stages',{'bench','battery','noise'}, $B, $S, $R3, 'pdi.DIA_LAMD',2.0, 'noise.readings',{'P','PF'}, 'noise.nstates',10.^(8:2:14), 'noise.nreal',6, 'battery.ladder',[30 60 120]*1e-6"
./pdi_batch.sh pin20_loop "pdi_params, 'stages',{'bench','loop'}, $B, $L, 'pdi.DIA_LAMD',2.0"
ZWFS_MEMMAX=20G ./pdi_batch.sh pin10_2048 "pdi_params, $BIG, 'stages',{'bench','battery','noise'}, $B, $S, $R3, 'pdi.DIA_LAMD',1.0, 'noise.readings',{'P','PF'}, 'noise.nstates',10.^(8:2:14), 'noise.nreal',6, 'battery.ladder',[30 60 120]*1e-6"
ZWFS_MEMMAX=20G ./pdi_batch.sh pin10_loop "pdi_params, $BIG, 'stages',{'bench','loop'}, $B, $L, 'pdi.DIA_LAMD',1.0"
echo "gseq2 done"
