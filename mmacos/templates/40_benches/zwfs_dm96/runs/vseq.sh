#!/usr/bin/env bash
# V1 record sequence (2026-09-11): flat-matrix battery + noise; on-surface battery; loop with S and V
cd "$(dirname "$0")/.."
./zwfs_batch.sh v193flat "'stages',{'bench','battery','noise','figs'}, 'readings',{'L','I+','S','V'}, 'dm_use',1"
./zwfs_batch.sh v193base "'stages',{'bench','battery','figs'}, 'readings',{'L','I+','S','V'}, 'dm_use',1, 'battery.calib_surface','base', 'battery.ladder_sites','grid'"
./zwfs_batch.sh vloop193 "'stages',{'bench','loop','figs'}, 'readings',{'L','I+','S','V'}, 'loop.readings',{'S','V'}"
