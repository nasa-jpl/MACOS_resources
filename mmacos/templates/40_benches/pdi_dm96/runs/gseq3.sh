#!/usr/bin/env bash
# Gauge-deck sequence 3 (deliverable 4): the descent -- capturing the DM's
# initial figure -- and the within-measurement DM drift (V4).
#   descent193  the loop started at a 100 nm rms surface (200 nm WFE) with the
#               matrix measured there, gain 0.5, re-calibrated every 10 cycles
#               and never, 1e13 / 1e15 photons per cycle, K 60; readings L S V P PF
#   intra193_0  the walk / thermal loop with the DM STILL during a scan (control)
#   intra193    the same with the whole cycle's drift developing across the scan
cd "$(dirname "$0")/.."
RD="'readings',{'L','S','V','P','PF'}, 'loop.readings',{'L','S','V','P','PF'}, 'dm_use',1"
./pdi_batch.sh descent193 "pdi_params, 'stages',{'bench','loop'}, $RD, 'loop.start_rms',100e-6, 'loop.recal_list',[0 10], 'loop.nph',[1e13 1e15], 'loop.drifts',{}, 'loop.floor',false, 'loop.steps',[], 'loop.K',60"
W="'loop.drifts',{'walk','thermal'}, 'loop.nph',[1e13 1e15], 'loop.steps',[], 'loop.floor',false, 'loop.K',60"
./pdi_batch.sh intra193_0 "pdi_params, 'stages',{'bench','loop'}, $RD, $W, 'loop.intra',0"
./pdi_batch.sh intra193   "pdi_params, 'stages',{'bench','loop'}, $RD, $W, 'loop.intra',1"
echo "gseq3 done"
