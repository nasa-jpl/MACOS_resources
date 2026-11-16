#!/usr/bin/env bash
# V5 (2026-09-14, plan 11.2): the complex amplitude from the vector sensor at
# the record's resolution: gate G9 (the pair through 5% / 20% pupil
# amplitude dips: the flat's amplitude, the state's clear frame, the pair
# alone), plus the battery rows with the clear-frame reading (mask.v_clear).
set -e
cd "$(dirname "$0")/.."
D="'MODEL',1024, 'NGRID',193, 'readings',{'V'}"
./zwfs_batch.sh an193_clear "zwfs_params, $D, 'stages',{'bench','battery'}, 'battery.rows',{'base/single','base/grid'}, 'battery.calib_surface','base', 'mask.v_dip',[0.05 0.20], 'mask.v_clear',true"
echo "v5seq done"
