#!/usr/bin/env bash
# the lock-in camera as a READING in every COPHI table (CC 2026-10-12); after the sensors' camera run
cd /home/dcr/dev/MACOS_resources/mmacos/templates/40_benches/cophi_dm96
until grep -q CAMSENS_DONE runs/camsens_seq.log; do sleep 60; done
OAP="'bench.optics','oap','bench.polarizing',false,'cophi.iq',true"
LENS="'bench.polarizing',false,'cophi.iq',true"
./cophi_batch.sh lockin_noise_oap   "$OAP,'stages',{'noise'},'cophi.noise_nreal',12"
./cophi_batch.sh lockin_noise_lens  "$LENS,'stages',{'noise'},'cophi.noise_nreal',12"
./cophi_batch.sh lockin_capture_oap "$OAP,'stages',{'capture'}"
./cophi_batch.sh lockin_battery_oap "$OAP,'stages',{'bench','battery'},'battery.est_mode','both','battery.cap_ladder',30e-6,'battery.recap_surf',[]"
./cophi_batch.sh lockin_loop_oap    "$OAP,'stages',{'loop'},'cophi.loop.cams',{'lockin'},'cophi.loop.fiber',3.5e-3"
./cophi_batch.sh lockin_loop_oap_r  "$OAP,'stages',{'loop'},'cophi.loop.cams',{'lockin_r'},'cophi.loop.rows',{'none'}"
echo LOCKIN_DONE
