#!/usr/bin/env bash
# PDI sequence 3 (2026-09-12): the CAMERA 1/f drift in the loop (Dave: "add the camera 1/f drift
# knob to the loop stage") -- every reading, camera walk only, two light levels; then the same
# with the whole step developing within each scan (intra 1: what within-scan 1/f costs the
# zero-sum readings).  Runs after pseq2 (the launcher waits for it).
cd "$(dirname "$0")/.."
L="'stages',{'bench','loop','figs'}, 'readings',{'L','I+','S','V','P','PF'}, 'loop.readings',{'L','I+','S','V','P','PF'}, 'loop.drifts',{'cam'}, 'loop.nph',[1e13 1e15], 'loop.steps',[]"
./zwfs_batch.sh pcam193  "$L"
./zwfs_batch.sh pcam193i "$L, 'loop.cam_intra',1"
