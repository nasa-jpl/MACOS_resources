#!/usr/bin/env bash
# the lock-in as a reading, with the MEDIAN annulus fill (a common phase stays a piston); then the sensors' rerun (the record's pupil)
cd /home/dcr/dev/MACOS_resources/mmacos/templates/40_benches/cophi_dm96
OAP="'bench.optics','oap','bench.polarizing',false,'cophi.iq',true,'cophi.annulus_fill','median'"
LENS="'bench.polarizing',false,'cophi.iq',true,'cophi.annulus_fill','median'"
./cophi_batch.sh lockin_noise_lens  "$LENS,'stages',{'noise'},'cophi.noise_nreal',12"
./cophi_batch.sh lockin_capture_oap "$OAP,'stages',{'capture'}"
./cophi_batch.sh lockin_battery_oap "$OAP,'stages',{'bench','battery'},'battery.est_mode','both','battery.cap_ladder',30e-6,'battery.recap_surf',[]"
./cophi_batch.sh lockin_loop_oap    "$OAP,'stages',{'loop'},'cophi.loop.cams',{'lockin'},'cophi.loop.fiber',3.5e-3"
./cophi_batch.sh lockin_loop_oap_r  "$OAP,'stages',{'loop'},'cophi.loop.cams',{'lockin_r'},'cophi.loop.rows',{'none'}"
echo LOCKIN2_DONE
runs/camsens_seq.sh > runs/camsens2_seq.log 2>&1
echo CAMSENS2_DONE >> runs/camsens2_seq.log
