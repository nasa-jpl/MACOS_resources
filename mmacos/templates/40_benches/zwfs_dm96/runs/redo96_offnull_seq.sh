#!/usr/bin/env bash
# The capture-range photon side and the re-measured-matrix rows ON THE BUILT BENCH
# (deck_gauges "Capture range, second half": the sensor rows there were the 385-px
# campaign's cap385_b*/noise193_b*; Dave's rule: the bench as built).  Mirror rig
# (OAPZ as redo96_sensnoise), NGRID 193, S V P; per working surface 60 / 120 / 160 nm:
#   redo96_sensnoise_b<b>  photons for 1 pm with the matrix measured on that surface
#   redo96_senscap_b<b>    1 nm on the grid sites with the matrix re-measured there (gain / floor)
# (the 30 nm points are redo96_sensnoise and redo96_oapsens).  Serial on the gauge lock.
set -u
cd "$(dirname "$0")/.."
export MACOS_HOME=$HOME/dev/macos/macos_f90
LOG=runs/redo96_offnull_seq.log
OAPZ="'bench.optics','oap','bench.OAP1_AOI',20,'bench.OAP2_AOI',25,'bench.OAP1_SIDE',1,'bench.OAP2_SIDE',-1,'bench.MASK_TRIM',0.632,'mask.v_arm','engine','battery.calib_surface','base'"
run() { echo "[$(date '+%F %T')] start $1" >> $LOG; ./zwfs_batch.sh "$1" "$2"; echo "[$(date '+%F %T')] end $1 rc=$?" >> $LOG; }
for b in 60 120 160; do
  run redo96_sensnoise_b$b "$OAPZ,'MODEL',1024,'NGRID',193,'readings',{'S','V','P'},'stages',{'bench','noise'},'battery.base_rms',${b}e-6"
  run redo96_senscap_b$b   "$OAPZ,'MODEL',1024,'NGRID',193,'readings',{'S','V','P'},'stages',{'bench','battery'},'battery.rows',{'base/grid'},'battery.ladder_sites','grid','battery.ladder',[],'battery.base_rms',${b}e-6"
done
echo "[$(date '+%F %T')] ALL DONE" >> $LOG
touch runs/redo96_offnull.done
