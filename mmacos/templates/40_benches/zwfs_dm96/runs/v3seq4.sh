#!/usr/bin/env bash
# Capture range to 10% (Dave 2026-09-12): the working-surface ladder for
# L / I+ / S / V at the record sampling (385 rays), (a) with the matrix
# measured once on the 30 nm surface, (b) re-measured on the working
# surface itself (60 / 90 / 120 / 160 nm).
cd "$(dirname "$0")/.."
C="'NGRID',385, 'dm_use',1, 'readings',{'L','I+','S','V'}, 'stages',{'bench','battery'}, 'battery.calib_surface','base', 'battery.rows',{'base/grid'}, 'battery.ladder_sites','grid'"
./zwfs_batch.sh cap385 "$C, 'battery.ladder',[30 40 50 60 80 100 120 160 240 480]*1e-6"
for b in 60 90 120 160; do
  ./zwfs_batch.sh cap385_b$b "$C, 'battery.base_rms',${b}e-6, 'battery.ladder',[]"
done
echo "[$(date '+%F %T')] v3seq4 done" >> runs/v3seq4.log
