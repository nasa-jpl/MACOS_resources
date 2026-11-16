#!/usr/bin/env bash
# Package A's gate record, the two comparisons that make it readable (TO 2026-09-17).
# Runs AFTER runs/redoseq.sh.
#   1. the SEED tail on the same collimated bench, emitted and put through the
#      pupil stage -- what the tune actually bought, on one bench
#   2. the tuned deck through the pupil stage with the deck's OWN cone
#      (overfill 0): the stage of record re-aims the source cone to 1.06 x the
#      DM aperture, and the emitted bench's cone is 1.21 x, so this says
#      whether the gate numbers depend on that choice
set -u
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$here"
export MACOS_HOME="${MACOS_HOME:-$HOME/dev/macos/macos_f90}"

wait_matlab () {
  local n=0
  while pgrep -x MATLAB >/dev/null 2>&1; do
    sleep 30; n=$((n+1))
    if [ $n -gt 360 ]; then echo "[redoseq2] TIMED OUT waiting for MATLAB"; exit 1; fi
  done
}

echo "[redoseq2] $(date '+%F %T') waiting"
wait_matlab

echo "[redoseq2] $(date '+%F %T') step 1: the seed tail on the collimated bench"
TG96_MEMMAX=14G ./tg96_batch.sh redo_lens_seed "'bench.tail_from_mat',false,'stages',{'bench'}"
echo "[redoseq2] emit exit $?"
./tg96_pupil_batch.sh lens "'tool','sim','deck','$here/runs/redo_lens_seed/redo_lens_seed_test.in','tag','pupilsim_redo_lens_seed'"
echo "[redoseq2] pupilsim seed exit $?"

echo "[redoseq2] $(date '+%F %T') step 2: the tuned deck on its own cone"
./tg96_pupil_batch.sh lens "'tool','sim','deck','$here/runs/redo_lens/redo_lens_test.in','tag','pupilsim_redo_lens_owncone','overfill',0,'dm_ap',0"
echo "[redoseq2] pupilsim own-cone exit $?"

echo "[redoseq2] $(date '+%F %T') chain complete"
