#!/usr/bin/env bash
# Gauge-deck sequence 3 (deliverable 4): the descent -- capturing the DM's
# initial figure -- and the within-measurement DM drift (V4).
#
# Order is by what each run settles.  The dev-resolution smoke (sm_knobs)
# showed NOTHING descends from a 100 nm rms surface in 8 cycles, S and P
# actively diverging: at that size the DIFFERENTIAL to the set point is
# ~70 nm rms of surface = ~1.4 waves, so the wrapped phase difference every
# reading returns is wrapped, whatever its absolute range.  The ladder run
# locates where that starts, which is the number the capture slide needs;
# the K = 60 runs then answer the brief's question at 100 nm.
cd "$(dirname "$0")/.."
RD="'readings',{'L','S','V','P','PF'}, 'loop.readings',{'L','S','V','P','PF'}, 'dm_use',1"

# (a) the LADDER moved to runs/gcap.sh (deliverable 9: both ways, the
#     start vector in one invocation).  What is left here is the descent
#     at 100 nm for P in its shutter-frame configuration, the fast-recal
#     probe, and the within-scan drift.

# (b) the brief's descent at 100 nm: every reading, re-calibrated every 10
# cycles and never, two photon levels, K 60.
DESC="'loop.start_rms',100e-6, 'loop.nph',[1e13 1e15], 'loop.drifts',{}, 'loop.floor',false, 'loop.steps',[], 'loop.K',60"
./pdi_batch.sh descent193  "pdi_params, 'stages',{'bench','loop'}, $RD, $DESC, 'loop.recal_list',[0 10]"
./pdi_batch.sh descent193s "pdi_params, 'stages',{'bench','loop'}, 'readings',{'P'}, 'loop.readings',{'P'}, 'dm_use',1, 'pdi.b2','state', $DESC, 'loop.recal_list',[0 10]"
./pdi_batch.sh descent193f "pdi_params, 'stages',{'bench','loop'}, 'readings',{'P'}, 'loop.readings',{'P'}, 'dm_use',1, 'pdi.b2','state', 'loop.start_rms',100e-6, 'loop.nph',[1e15], 'loop.drifts',{}, 'loop.floor',false, 'loop.steps',[], 'loop.K',20, 'loop.recal_list',[2 5]"

# (c) the within-measurement DM / thermal drift (V4), control first
W="'loop.drifts',{'walk','thermal'}, 'loop.nph',[1e13 1e15], 'loop.steps',[], 'loop.floor',false, 'loop.K',60"
./pdi_batch.sh intra193_0 "pdi_params, 'stages',{'bench','loop'}, $RD, $W, 'loop.intra',0"
./pdi_batch.sh intra193   "pdi_params, 'stages',{'bench','loop'}, $RD, $W, 'loop.intra',1"
echo "gseq3 done"
