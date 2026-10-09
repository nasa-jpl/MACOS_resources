#!/usr/bin/env bash
# Dev-resolution smoke of the paths the gauge-deck sequences use, before
# hours of model-1024 compute: the P/SRI bench through noise AND loop, and
# the three new loop knobs (descent, within-scan drift, reference-arm walk)
# on the ZWFS bench.  Model 512, 65 rays, the 48x48 DM.
set -e
cd "$(dirname "$0")/.."
D="'MODEL',512, 'NGRID',65, 'grid.N_G',256, 'grid.DX_G',0.42, 'dm_use',2, 'reg.mode','record'"
# sm_psri_nl (the P/SRI bench through noise AND loop) ran 2026-09-13 09:23 and
# its record is runs/sm_psri_nl.  Re-run it with:
# ./pdi_batch.sh sm_psri_nl "pdi_params, $D, 'pdi.bench','psri', 'readings',{'PF'}, 'noise.readings',{'PF'}, 'loop.readings',{'PF'}, 'stages',{'bench','noise','loop'}, 'noise.nstates',10.^[10 12], 'noise.nreal',2, 'loop.K',6, 'loop.nph',[1e13], 'loop.drifts',{'walk'}, 'loop.steps',[], 'battery.calib_surface','base'"
./pdi_batch.sh sm_knobs "pdi_params, $D, 'readings',{'S','P','PF'}, 'loop.readings',{'S','P','PF'}, 'stages',{'bench','loop'}, 'loop.K',8, 'loop.nph',[1e13], 'loop.drifts',{'walk'}, 'loop.steps',[], 'loop.floor',false, 'loop.intra',1, 'pdi.ref_walk',1e-2, 'loop.start_rms',100e-6, 'loop.recal_list',[0 4]"
echo "gsmoke done"
