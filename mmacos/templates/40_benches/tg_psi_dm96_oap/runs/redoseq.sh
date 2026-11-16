#!/usr/bin/env bash
# BRIEF_to_tg_redo package A, the rest of it, in one detached chain (TO 2026-09-17).
#   1. the mirror rig's tail: DET_TRIM alone, on the reading objective
#   2. both rigs re-emitted (tg96_run stage 'bench') on the redo's tails
#   3. the pupil stage on each EMITTED deck -- the gate record for package A
# The lens rig's tail (tag lens96) is tuned by the job this chain waits for.
# One MATLAB at a time: every step waits for the previous one's process to go.
set -u
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$here"
export MACOS_HOME="${MACOS_HOME:-$HOME/dev/macos/macos_f90}"
OAP="'bench.optics','oap','bench.POL_IN','source','bench.D_RC_L2',125,'oap.OAP1_AOI',20,'oap.OAP2_AOI',25,'oap.OAP1_SIDE',1,'oap.OAP2_SIDE',-1"

wait_matlab () {                      # bounded: 3 h, then give up loudly
  local n=0
  while pgrep -x MATLAB >/dev/null 2>&1; do
    sleep 30; n=$((n+1))
    if [ $n -gt 360 ]; then echo "[redoseq] TIMED OUT waiting for MATLAB"; exit 1; fi
  done
}

echo "[redoseq] $(date '+%F %T') waiting for the lens tail tune"
wait_matlab

echo "[redoseq] $(date '+%F %T') step 1: the mirror rig's tail (DET_TRIM alone)"
systemd-run --user --scope -p MemoryMax=10G matlab -batch \
  "tg96_tail_batch('tag','oap96',$OAP,'tail.free',{'DET_TRIM'})" > runs/tail_oap96.log 2>&1
echo "[redoseq] step 1 exit $?"
wait_matlab

# the redo's tails, under their own tags, so the record's lens_tail.mat /
# oap_tail.mat stay put until the gate below says the redo is good
cp -f lens96_tail.mat redo_lens_tail.mat 2>/dev/null || echo "[redoseq] NO lens96_tail.mat"
cp -f oap96_tail.mat  redo_oap_tail.mat  2>/dev/null || echo "[redoseq] NO oap96_tail.mat"

echo "[redoseq] $(date '+%F %T') step 2: re-emit both rigs"
TG96_MEMMAX=14G ./tg96_batch.sh redo_lens "'stages',{'clearance','bench','figs'}"
echo "[redoseq] redo_lens exit $?"
TG96_MEMMAX=20G ./tg96_batch.sh redo_oap  "$OAP,'stages',{'clearance','bench','figs'}"
echo "[redoseq] redo_oap exit $?"

echo "[redoseq] $(date '+%F %T') step 3: the pupil stage on the emitted decks"
./tg96_pupil_batch.sh lens "'tool','sim','deck','$here/runs/redo_lens/redo_lens_test.in','tag','pupilsim_redo_lens'"
echo "[redoseq] pupilsim lens exit $?"
./tg96_pupil_batch.sh oap  "'tool','sim','deck','$here/runs/redo_oap/redo_oap_test.in','tag','pupilsim_redo_oap'"
echo "[redoseq] pupilsim oap exit $?"

echo "[redoseq] $(date '+%F %T') chain complete"
