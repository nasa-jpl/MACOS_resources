#!/usr/bin/env bash
# the converged-estimator re-score (2026-10-11): H then I, both rigs, est_mode 'both', rows only
set -u
B=/home/dcr/dev/MACOS_resources/mmacos/templates/40_benches
EST="'battery.est_mode','both','battery.est_tol',1e-12,'battery.est_maxit',50,'battery.cap_ladder',30e-6,'battery.recap_surf',[]"
OAP="'bench.optics','oap','bench.POL_IN','source','bench.D_RC_L2',125,'oap.OAP1_AOI',20,'oap.OAP2_AOI',25,'oap.OAP1_SIDE',1,'oap.OAP2_SIDE',-1"
cd $B/cophi_dm96
./cophi_batch.sh conv_H_lens "'stages',{'bench','battery'},$EST"
./cophi_batch.sh conv_H_oap  "'bench.optics','oap','stages',{'bench','battery'},$EST"
cd $B/tg_psi_dm96_oap
./tg96_batch.sh conv_I_lensph "'bench.tail_file','redo96_lens_tail.mat','stages',{'bench','deck'},$EST"
./tg96_batch.sh conv_I_oapph  "$OAP,'bench.tail_file','redo96_oap_tail.mat','stages',{'bench','deck'},$EST"
echo CONVSEQ_DONE
