#!/usr/bin/env bash
# The close-out queue's steps 2 and 3 (BRIEF_to_restart), chained.
#
# Step 1 (runs/apply_tg96_pending.py) is APPLIED and committed -- 7aac1d8 --
# so tg96_run.m now carries the wrap stage, the unsaturated ladder meter, the
# shared camera line, MASK_SUB and the 1800 px stations figure.  Both of the
# sequencers below need it; neither would have been correct before it.
#
#   step 2  item2bseq.sh   wrapoap, wraplens, stnoap, stnlens     ~45 min
#   step 3  item4bseq.sh   thk22_tail (retune), thk22 (gate run)  ~1 h
#
# Each tg96_batch.sh / oap_fold_batch.sh call is synchronous, so a sequencer
# returning IS its jobs finishing -- but the gate between the steps is the
# WRAPPER'S OWN `] exit` marker in each log, never the sequencer's echo, so a
# sequencer that dies without its children is caught here instead of letting
# step 3 start beside step 2's MATLAB.  One model-1024 MATLAB on this box.
set -u
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$here"

need_exit0 () {    # need_exit0 <log> ... : every log carries a zero exit marker
    local bad=0 l m
    for l in "$@"; do
        m="$(grep -h '] exit' "$l" 2>/dev/null | tail -1)"
        if [ -z "$m" ]; then echo "[closefinal2] $l: NO exit marker"; bad=1
        elif ! echo "$m" | grep -q 'exit 0'; then echo "[closefinal2] $l: $m"; bad=1
        else echo "[closefinal2] $l: $m"; fi
    done
    return $bad
}

echo "[closefinal2] $(date '+%F %T') step 2: item2bseq (wrap on both rigs + both station figures)"
./item2bseq.sh
echo "[closefinal2] $(date '+%F %T') item2bseq returned $?"
if need_exit0 wrapoap.log wraplens.log stnoap.log stnlens.log; then
    echo "[closefinal2] step 2 clean"
else
    echo "[closefinal2] STOP: step 2 did not finish clean; step 3 NOT started."
    echo "[closefinal2] queue stopped"
    exit 1
fi

echo "[closefinal2] $(date '+%F %T') step 3: item4bseq (thk22 retune with the advisory gate + gate run)"
./item4bseq.sh
echo "[closefinal2] $(date '+%F %T') item4bseq returned $?"
need_exit0 thk22_tail.log thk22.log || echo "[closefinal2] step 3 had a nonzero exit -- read the log"
echo "[closefinal2] $(date '+%F %T') queue drained"
