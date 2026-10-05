#!/usr/bin/env bash
# The three deck items missing on the redo (built) bench, 2026-10-05 (Dave: "1: yes"):
#   the sensors' photons for 1 pm and their servo hold on the mirror rig (the cycle-3c
#   runs redo96_oapnoise / redo96_oapcap were committed with headers only), and the
#   interferometer's photons for 1 pm on both rigs (battery.noise was off in redo96_*).
# Serial: each batch script waits for any other DM-gauge batch MATLAB.  The TG tags carry
# their own copies of the redo tails (<tag>_tail.mat), else the runner falls back to the
# RECORD's tail (runbook, "One thing to settle before package C").
set -u
cd "$(dirname "$0")/.."
export MACOS_HOME=$HOME/dev/macos/macos_f90
LOG=runs/redo96_ph_seq.log
OAP="'bench.optics','oap','bench.POL_IN','source','bench.D_RC_L2',125,'oap.OAP1_AOI',20,'oap.OAP2_AOI',25,'oap.OAP1_SIDE',1,'oap.OAP2_SIDE',-1"
OAPZ="'bench.optics','oap','bench.OAP1_AOI',20,'bench.OAP2_AOI',25,'bench.OAP1_SIDE',1,'bench.OAP2_SIDE',-1,'bench.MASK_TRIM',0.632,'mask.v_arm','engine','battery.calib_surface','base'"
run() { echo "[$(date '+%F %T')] start $1" >> $LOG; "$2" "$1" "$3"; echo "[$(date '+%F %T')] end $1 rc=$?" >> $LOG; }
( cd ../zwfs_dm96 && run redo96_sensnoise ./zwfs_batch.sh "$OAPZ,'MODEL',1024,'NGRID',193,'readings',{'S','V','P'},'stages',{'bench','noise'}" )
( cd ../zwfs_dm96 && run redo96_sensloop  ./zwfs_batch.sh "$OAPZ,'MODEL',1024,'NGRID',193,'readings',{'S','V','P'},'stages',{'bench','loop','figs'}" )
run redo96_lensph ./tg96_batch.sh "'battery.noise',true,'stages',{'bench','deck'}"
run redo96_oapph  ./tg96_batch.sh "$OAP,'battery.noise',true,'stages',{'bench','deck'}"
echo "[$(date '+%F %T')] ALL DONE" >> $LOG
touch runs/redo96_ph.done
