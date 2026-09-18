#!/usr/bin/env bash
# Is the quartet's 6% pitch disagreement the MASK PLATE?  (TO 2026-09-17, package B)
# The through-focus quartet assumes the pupil scales as R2/R1 about the focus.
# A plane-parallel plate between S1 and the focus does not change the ray ANGLE
# but it does displace the convergence, so the station distances R1 and R2 stop
# describing the scaling.  The decided 2 mm mask plate sits exactly there.  This
# emits the same bench with MASK_SUB [] and nothing else changed, and re-runs
# the leg checks: if CHECK 2c collapses, the plate is the cause.
set -u
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$here"
export MACOS_HOME="${MACOS_HOME:-$HOME/dev/macos/macos_f90}"
wait_matlab () { local n=0; while pgrep -x MATLAB >/dev/null 2>&1; do sleep 20; n=$((n+1)); if [ $n -gt 540 ]; then echo "[s2snm] TIMED OUT"; exit 1; fi; done; }
wait_matlab
# THE TAIL MUST TRAVEL WITH THE VARIANT.  tg96_run looks up <tag>_tail.mat and
# then <optics>_tail.mat, so a variant tag with no mat of its own silently picks
# up the RECORD's lens_tail.mat -- which is the tuned tail at 39.9 mm past the
# focus, not the redo's seed station at 10.26.  The first run of this experiment
# did exactly that and moved TWO things at once (the plate and R2, 3.9x).
cp -f redo_lens_tail.mat redo_lens_nomask_tail.mat
echo "[s2snm] $(date '+%F %T') emit (tail copied from redo_lens_tail.mat)"
TG96_MEMMAX=14G ./tg96_batch.sh redo_lens_nomask "'bench.MASK_SUB',[],'stages',{'bench'}"
echo "[s2snm] emit exit $?"
./tg96_pupil_batch.sh lens "'tool','sim','deck','$here/runs/redo_lens_nomask/redo_lens_nomask_test.in','tag','pupilsim_redo_lens_nomask'"
echo "[s2snm] pupilsim exit $?"
wait_matlab
echo "[s2snm] $(date '+%F %T') s2s checks"
systemd-run --user --scope -p MemoryMax=12G matlab -batch \
  "tg96_pupil_s2s('rig','lens','sim','$here/runs/pupilsim_redo_lens_nomask','tag','s2s_nomask','conv',{'+dec'}); exit(0)" \
  > runs/s2s_nomask.log 2>&1
echo "[s2snm] s2s exit $?"
grep -E "CHECK" runs/s2s_nomask/s2s_nomask_report.txt
echo "[s2snm] $(date '+%F %T') done"
