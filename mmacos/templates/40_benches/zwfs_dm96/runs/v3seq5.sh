#!/usr/bin/env bash
# Capture range, the photon side (Dave 2026-09-12): N(1 pm) with the matrix
# re-measured on 60 / 120 / 160 nm working surfaces (v193noise is the 30 nm
# point, flat calibration; noise60_c30 = 60 nm surface with the 30 nm matrix).
cd "$(dirname "$0")/.."
N="'stages',{'bench','noise'}, 'readings',{'L','S','V'}, 'noise.readings',{'L','S','V'}, 'dm_use',1, 'noise.nstates',10.^(8:2:14), 'noise.nreal',6"
./zwfs_batch.sh noise193_b30  "$N, 'battery.calib_surface','base', 'battery.base_rms',30e-6"
./zwfs_batch.sh noise193_b60  "$N, 'battery.calib_surface','base', 'battery.base_rms',60e-6"
./zwfs_batch.sh noise193_b120 "$N, 'battery.calib_surface','base', 'battery.base_rms',120e-6"
./zwfs_batch.sh noise193_b160 "$N, 'battery.calib_surface','base', 'battery.base_rms',160e-6"
echo "[$(date '+%F %T')] v3seq5 done" >> runs/v3seq5.log
