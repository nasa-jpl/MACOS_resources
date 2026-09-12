#!/usr/bin/env bash
# PDI sequence 5: the camera drift in its RELATIVE form (a per-pixel offset random-walking 1e-3 of
# the mean photons per lit pixel per frame, per cycle), intra 0 then intra 1 -- the electron form
# (pcam193) was invisible at the campaign's photon levels.  Five readings (I+ floors at 885 pm anyway).
cd "$(dirname "$0")/.."
L="'stages',{'bench','loop','figs'}, 'readings',{'L','S','V','P','PF'}, 'loop.readings',{'L','S','V','P','PF'}, 'loop.drifts',{'cam'}, 'loop.nph',[1e13 1e15], 'loop.steps',[], 'loop.cam_unit','rel', 'loop.cam_walk',1e-3"
./zwfs_batch.sh pcam193r  "$L"
./zwfs_batch.sh pcam193ri "$L, 'loop.cam_intra',1"
