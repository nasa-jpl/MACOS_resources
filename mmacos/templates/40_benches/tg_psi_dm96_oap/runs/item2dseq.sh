#!/usr/bin/env bash
# item2seq's remainder: oapdesc2, matched to the record's descent RUN so that
# only the bench and the tail differ.  Supersedes item2cseq.sh, which had the
# units right but would have run the FULL loop sweep.
#
# The record (runs/descent_oap) ran "3 loop runs of 61 states each" -- ONE run
# per start, at 1.0e+13 photons per cycle, recal never, 35.2 min.  The default
# sweep instead multiplies the descent by every photon level AND adds the step
# and drift runs: 22 loop runs, upwards of three hours, for two starts' worth of
# wanted answer.  oapifol2 (already running) is what supplies the photon and
# drift story; the descent only has to answer whether the loop CAPTURES from
# 100 and 200 nm.
#
#   loop.steps  []      no step-response runs
#   loop.drifts {}      no walk / thermal runs
#   loop.floor  false   no noise-only run  (kinds is then empty)
#   loop.nph    1e13    the record's photon level, so the comparison is fair
#   start_rms   [1e-4 2e-4]   MM -- 100 and 200 nm (see the units commit)
#
# nrun = 0 + 0 + 2 starts = 2 loop runs of 61 states, plus one matrix per start.
#
# CAVEAT for whoever reads the comparison: runs/descent_oap is BS_AOI 7 and
# carries the 25 mm conjugate error, so oapdesc2 differs from it in bench AND
# tail, not in tail alone.  The same trap as the lens ladder (section 4.7).
set -u
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$here"
n=0
while ! grep -q '\] exit' runs/oapifol2.log 2>/dev/null; do
    n=$((n+1)); [ $n -gt 720 ] && { echo "[item2dseq] ABORT: oapifol2 never exited (6 h)"; exit 1; }
    sleep 30
done
echo "[item2dseq] oapifol2 done -> oapdesc2 (descent only, record-matched)"
D="'bench.optics','oap','bench.POL_IN','source','bench.SRC_AT_FOCUS',true,'bench.D_RC_L2',125,'oap.OAP1_AOI',20,'oap.OAP2_AOI',25,'oap.OAP1_SIDE',1,'oap.OAP2_SIDE',-1,'bench.tail_from_mat',false"
L="'loop.start_rms',[1e-4 2e-4],'loop.steps',[],'loop.drifts',{},'loop.floor',false,'loop.nph',1e13,'loop.recal_every',0"
./tg96_batch.sh oapdesc2 "$D,$L,'stages',{'bench','loop','figs'}"
echo "[item2dseq] done"
echo "[item2seq] done" >> runs/item2seq.nohup      # release closechain4
