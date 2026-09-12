#!/usr/bin/env bash
# PDI sequence 4: the step-scheme trade re-run (2% step error, ls 4-step vs the 5-frame
# Schwider-Hariharan) after the exactness gates were made informational under a step error.
cd "$(dirname "$0")/.."
S="'battery.calib_surface','base', 'battery.ladder_sites','grid', 'battery.rows',{'base/single','base/grid','base/rand'}, 'battery.ladder',[30 60]*1e-6"
./zwfs_batch.sh pdi193se_ls  "'stages',{'bench','battery'}, 'readings',{'P','PF'}, 'dm_use',1, $S, 'pdi.step_err',0.02"
./zwfs_batch.sh pdi193se_sh5 "'stages',{'bench','battery'}, 'readings',{'P','PF'}, 'dm_use',1, $S, 'pdi.step_err',0.02, 'pdi.scheme','sh5'"
