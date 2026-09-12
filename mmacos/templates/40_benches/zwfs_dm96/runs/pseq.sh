#!/usr/bin/env bash
# PDI record sequence (2026-09-12): flat-matrix battery + noise (S V P PF); on-surface battery;
# the 1 lam/D pinhole on the surface (+ noise); the loop rows for P and PF (same seeds as S / V)
cd "$(dirname "$0")/.."
./zwfs_batch.sh pdi193 "'stages',{'bench','battery','noise','figs'}, 'readings',{'S','V','P','PF'}, 'dm_use',1"
./zwfs_batch.sh pdi193base "'stages',{'bench','battery','figs'}, 'readings',{'S','V','P','PF'}, 'dm_use',1, 'battery.calib_surface','base', 'battery.ladder_sites','grid'"
./zwfs_batch.sh pdi193d1 "'stages',{'bench','battery','noise','figs'}, 'readings',{'S','P','PF'}, 'dm_use',1, 'pdi.DIA_LAMD',1.0, 'battery.calib_surface','base', 'battery.ladder_sites','grid'"
./zwfs_batch.sh ploop193 "'stages',{'bench','loop','figs'}, 'readings',{'L','S','P','PF'}, 'loop.readings',{'P','PF'}"
