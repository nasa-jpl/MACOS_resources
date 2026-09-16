#!/usr/bin/env bash
# Item 4 (queue step 4): RE-TEST the tail tuner's winner gate, now that
# row_gain_ measures by LATTICE DECONVOLUTION instead of a point sample.
#
# The two legs are the ones the brief names and the ones the point-sample
# measure got backwards:
#   gate3_win   objwin3_tail.mat, OAP design  -> must be REFUSED  (battery 0.0338)
#   gate3_lens  lens_tail.mat,    lens rig    -> must be ACCEPTED (battery 0.9968)
# The point-sample run is kept beside these as *.log.pointsample: it accepted
# the first at 0.9804 and refused the second at -0.8285.
#
# Waits for closefinal2 (steps 2 and 3) to drain: ONE model-1024 MATLAB on this
# box.  The wait is bounded and watches the wrapper's own markers, not an echo.
set -u
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$here"
deadline=$(( $(date +%s) + 3*3600 ))            # 3 h: steps 2+3 are ~1h45m
while ! grep -q 'queue drained\|queue stopped' closefinal2.nohup 2>/dev/null; do
    if [ "$(date +%s)" -gt "$deadline" ]; then
        echo "[closefinal3] STOP: closefinal2 did not drain within 3 h; gate NOT run."
        exit 1
    fi
    sleep 60
done
if grep -q 'queue stopped' closefinal2.nohup; then
    echo "[closefinal3] closefinal2 STOPPED (step 2 unclean).  Running the gate anyway:"
    echo "[closefinal3] it is independent of steps 2 and 3 and is the item-4 deliverable."
fi
# belt and braces: no MATLAB may be running when we start
for i in $(seq 1 60); do
    pgrep -x MATLAB >/dev/null || break
    echo "[closefinal3] waiting: a MATLAB is still up"; sleep 30
done
echo "[closefinal3] $(date '+%F %T') item 4: the two-leg gate test"
./gateseq3.sh
echo "[closefinal3] $(date '+%F %T') gateseq3 returned $?"
for l in gate3_win.log gate3_lens.log; do
    echo "[closefinal3] $l: $(grep -h '] exit' "$l" 2>/dev/null | tail -1)"
done
echo "[closefinal3] queue drained"
