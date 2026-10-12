#!/usr/bin/env bash
# the sensors' camera form (the struct arg was swallowed by zwfs_run's parser as the whole sheet; fields passed as scalars)
cd /home/dcr/dev/MACOS_resources/mmacos/templates/40_benches/zwfs_dm96
C="'noise.cam.pol.name','IMX250MZR class: polarization 2448x2048, 75 fps, well ~10 ke (IMX253 pixel; UNCONFIRMED), read 2.4 e- ASSUMED','noise.cam.pol.npix',2048,'noise.cam.pol.well',1e4,'noise.cam.pol.read',2.4,'noise.cam.pol.rate',75,'noise.cam.pol.fill',0.5,'noise.cam.pol.pol',1"
C="$C,'noise.cam.mono.name','IMX250 mono class: 2448x2048, 75 fps, well ~10 ke (UNCONFIRMED), read 2.4 e- ASSUMED','noise.cam.mono.npix',2048,'noise.cam.mono.well',1e4,'noise.cam.mono.read',2.4,'noise.cam.mono.rate',75,'noise.cam.mono.fill',0.5,'noise.cam.mono.pol',0"
./zwfs_batch.sh camS_oap "'bench.optics','oap','bench.OAP1_AOI',20,'bench.OAP2_AOI',25,'bench.OAP1_SIDE',1,'bench.OAP2_SIDE',-1,'bench.MASK_TRIM',0.632,'mask.v_arm','engine','battery.calib_surface','base','MODEL',1024,'NGRID',193,'readings',{'S','V','P'},'stages',{'bench','noise'},'noise.nstates',1e14,'noise.nreal',12,$C"
echo CAMSENS_DONE
