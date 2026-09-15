#!/usr/bin/env bash
# V4 (2026-09-14): the vector sensor's ANALYZER leak, priced at the record's
# resolution (model 1024, 193 rays): the MacNeille cube's extinction alone,
# a quarter-wave plate retardance error of lambda/300 and lambda/100, a
# plate azimuth error of 1 deg, and the uniform bound on the ideal plate's
# projected-axis term (1.5e-3 coherent, zero mean over the pupil, which the
# scalar model drops).  Bench stage (G4 on 100 nm pokes, uncalibrated) +
# battery rows on the 30 nm surface with the matrix measured on it (the
# calibrated answer).  an193_ref = the same settings with an ideal analyzer.
set -e
cd "$(dirname "$0")/.."
D="'MODEL',1024, 'NGRID',193, 'readings',{'V'}, 'stages',{'bench','battery'}, 'battery.rows',{'base/single','base/grid'}, 'battery.calib_surface','base'"
./zwfs_batch.sh an193_ref   "zwfs_params, $D"
./zwfs_batch.sh an193_cube  "zwfs_params, $D, 'mask.v_analyzer','engine'"
./zwfs_batch.sh an193_q300  "zwfs_params, $D, 'mask.v_analyzer','engine', 'mask.v_qwp_err',1/300"
./zwfs_batch.sh an193_q100  "zwfs_params, $D, 'mask.v_analyzer','engine', 'mask.v_qwp_err',1/100"
./zwfs_batch.sh an193_az1   "zwfs_params, $D, 'mask.v_analyzer','engine', 'mask.v_qwp_az',1"
./zwfs_batch.sh an193_bound "zwfs_params, $D, 'mask.v_analyzer',struct('lA',4.24e-4,'cA',1.5e-3,'lB',6.34e-4,'cB',-1.5e-3)"
echo "v4seq done"
