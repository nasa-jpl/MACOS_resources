#!/usr/bin/env bash
# item2seq's remainder, with oapdesc2's UNITS FIXED.
#
# P.loop.start_rms is in mm -- tg96_run prints it as SR*1e6 nm (line 1202), and
# every neighbouring knob agrees (battery.base_rms 30e-6 for 30 nm, loop.steps
# 1e-6 for 1 nm).  item2seq.sh passed [100 200], i.e. a HUNDRED MILLIMETRES of
# starting surface rms, not 100 nm.  runs/descseq.sh carries the same slip
# ([60 150 300]); it was never run, because oapdesc was deferred, which is why
# nothing caught it.  The record's descent_oap really did run 60/150/300 nm, so
# it was called with 6e-5/1.5e-4/3e-4.
#
# item2seq.sh was stopped rather than edited: bash reads a script incrementally
# and remembers its offset, so editing one that is mid-execution can make it
# resume at the wrong byte.  Killing the sequencer leaves its running child
# (oapifol2) alone -- it is orphaned but alive -- so nothing in flight is lost.
# This script waits for that child, runs the corrected oapdesc2, and then writes
# item2seq's own done-marker so runs/closechain4.sh proceeds as if nothing had
# changed.
set -u
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$here"
n=0
while ! grep -q '\] exit' runs/oapifol2.log 2>/dev/null; do
    n=$((n+1)); [ $n -gt 720 ] && { echo "[item2cseq] ABORT: oapifol2 never exited (6 h)"; exit 1; }
    sleep 30
done
echo "[item2cseq] oapifol2 done -> oapdesc2 (start_rms 100/200 nm, in mm)"
D="'bench.optics','oap','bench.POL_IN','source','bench.SRC_AT_FOCUS',true,'bench.D_RC_L2',125,'oap.OAP1_AOI',20,'oap.OAP2_AOI',25,'oap.OAP1_SIDE',1,'oap.OAP2_SIDE',-1,'bench.tail_from_mat',false"
./tg96_batch.sh oapdesc2 "$D,'loop.start_rms',[1e-4 2e-4],'stages',{'bench','loop','figs'}"
echo "[item2cseq] done"
echo "[item2seq] done" >> runs/item2seq.nohup      # release closechain4
