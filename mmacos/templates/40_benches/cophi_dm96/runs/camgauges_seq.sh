#!/usr/bin/env bash
# the composite deck's "seconds to SNR 1 at 1 pm" row for the other gauges, through their own readings on a named camera (2026-10-12)
B=/home/dcr/dev/MACOS_resources/mmacos/templates/40_benches
POL="struct('name','IMX250MZR class: polarization 2448x2048, 75 fps, well ~10 ke (IMX253 pixel; UNCONFIRMED), read 2.4 e- ASSUMED','npix',2048,'well',1e4,'read',2.4,'rate',75,'fill',0.5,'pol',1)"
MONO="struct('name','IMX250 mono class: 2448x2048, 75 fps, well ~10 ke (UNCONFIRMED), read 2.4 e- ASSUMED','npix',2048,'well',1e4,'read',2.4,'rate',75,'fill',0.5,'pol',0)"
OAP="'bench.optics','oap','bench.POL_IN','source','bench.D_RC_L2',125,'oap.OAP1_AOI',20,'oap.OAP2_AOI',25,'oap.OAP1_SIDE',1,'oap.OAP2_SIDE',-1"
DECK="'stages',{'bench','deck'},'battery.noise',false,'battery.cap_ladder',30e-6,'battery.recap_surf',[],'battery.noise_nreal',12,'battery.noise_cam',$POL"
cd $B/tg_psi_dm96_oap
./tg96_batch.sh camI_lens "'bench.tail_file','redo96_lens_tail.mat',$DECK"
./tg96_batch.sh camI_oap  "$OAP,'bench.tail_file','redo96_oap_tail.mat',$DECK"
cd $B/zwfs_dm96
./zwfs_batch.sh camS_oap "'bench.optics','oap','bench.OAP1_AOI',20,'bench.OAP2_AOI',25,'bench.OAP1_SIDE',1,'bench.OAP2_SIDE',-1,'bench.MASK_TRIM',0.632,'mask.v_arm','engine','battery.calib_surface','base','MODEL',1024,'NGRID',193,'readings',{'S','V','P'},'stages',{'bench','noise'},'noise.nstates',1e14,'noise.nreal',12,'noise.cam',struct('pol',$POL,'mono',$MONO)"
echo CAMGAUGES_DONE
