#!/usr/bin/env bash
# Waits for runs/redo96_ph.done (the four-run sequence), then the lens rig's capture on the
# redo bench since the lit-margin fix (the mirror rig's twin = capt96_oap, 2026-09-30).
set -u
cd "$(dirname "$0")/.."
export MACOS_HOME=$HOME/dev/macos/macos_f90
n=0; until [ -f runs/redo96_ph.done ]; do sleep 60; n=$((n+1)); [ $n -gt 1440 ] && { echo "capt96_lens_after: gave up after 24 h" >> runs/redo96_ph_seq.log; exit 1; }; done
echo "[$(date '+%F %T')] start capt96_lens" >> runs/redo96_ph_seq.log
export TG96_MEMMAX=20G
./tg96_batch.sh capt96_lens "'loop.start_rms',[1e-4 2e-4 3e-5],'loop.recal_every',0,'loop.steps',[],'loop.drifts',{},'loop.floor',false,'loop.nph',1e13,'stages',{'bench','loop','figs'}"
echo "[$(date '+%F %T')] end capt96_lens rc=$?" >> runs/redo96_ph_seq.log
touch runs/capt96_lens.done
