#!/usr/bin/env bash
# BRIEF_to_gauge_close -- the whole remaining queue in ONE chain.
#
# WHY THIS REPLACES closechain4/5/6.  Those chained on each other's ECHOES
# ("[item2seq] done", "[closechain4] queue drained").  That is fragile in
# exactly the way this session exercised: item2seq was killed and replaced
# TWICE (units, then scope), so its echo was rewritten to be emitted by its
# successor -- and chain4's 8 h bound expired before the 5.5 h loop plus its
# predecessors got there, so chain4 aborted and chains 5 and 6 were left
# waiting on markers that could never arrive.  The queue was dead while a run
# was still going.
#
# This chain waits on RUN ARTIFACTS instead -- the wrapper's own "] exit" line
# in runs/<tag>.log -- which exist regardless of who launched the run or how
# many times the sequencer was replaced.  Bounds are generous and every wait
# says what it is waiting for when it gives up.
set -u
T=/home/dcr/dev/MACOS_resources/mmacos/templates/40_benches/tg_psi_dm96_oap/runs
C=/home/dcr/dev/MACOS_resources/mmacos/templates/30_instruments/bench_ctb

wait_for () {                       # wait_for <log> <hours> <what>
    local log="$1" hrs="$2" what="$3" n=0 cap=$(( $2 * 120 ))
    while ! grep -q '\] exit' "$log" 2>/dev/null; do
        n=$((n+1))
        if [ $n -gt $cap ]; then
            echo "[closefinal] ABORT after ${hrs} h waiting for $what ($log)"; exit 1
        fi
        sleep 30
    done
    echo "[closefinal] $what done: $(grep -h '\] exit' "$log" | tail -1)"
}

wait_for "$T/oapdesc2.log" 4 "item 2's descent"

echo "[closefinal] -> item 3's gate verifies"
"$T/gateseq3.sh"  > "$T/gateseq3b.nohup" 2>&1
echo "[closefinal] -> item 5's AOI ladder"
"$T/aoiseq.sh"    > "$T/aoiseq.nohup"    2>&1
echo "[closefinal] -> item 4's six runs"
"$T/item4seq.sh"  > "$T/item4seq.nohup"  2>&1
echo "[closefinal] -> item 7 step 0: the CTB beam probe"
( cd "$C" && matlab -batch "ctb_beam_probe" ) > "$T/ctb_beam_probe.log" 2>&1
echo "[closefinal] ctb_beam_probe exit $?"

if grep -q 'function stage_wrap_' "$T/../tg96_run.m"; then
    echo "[closefinal] -> item 2's wrap comparison + both rigs' station figures"
    "$T/item2bseq.sh" > "$T/item2bseq.nohup" 2>&1
else
    echo "[closefinal] STOP: tg96_run.m has no wrap stage."
    echo "[closefinal] Apply runs/apply_tg96_pending.py (nothing is in tg96_run.m now), then run runs/item2bseq.sh."
fi
echo "[closefinal] queue drained"
