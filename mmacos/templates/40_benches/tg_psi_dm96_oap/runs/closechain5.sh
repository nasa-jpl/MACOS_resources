#!/usr/bin/env bash
# BRIEF_to_gauge_close -- the queue after chain4 drains.
#   item4seq        item 4: thicknesses, then substrates, each with its retune
#                   through item 3's gate and its mask-sensor gate run.
#   ctb_beam_probe  item 7's PRECONDITION.  The note's step-2 prediction (50%
#                   Fresnel amplitude conversion at 33 cycles across the beam)
#                   combines a 0.67 mm actuator pitch with a 42.8 mm beam, and
#                   32 * 0.67 = 21.4, not 42.8 -- so one of the two is wrong by
#                   2x, and the documents contradict each other (the README's
#                   R_DM*FILL makes 21.4 a RADIUS; ctb_dm.m uses 21.3 as a
#                   DIAMETER).  Ask the engine before building a measurement
#                   on top of it.
set -u
T=/home/dcr/dev/MACOS_resources/mmacos/templates/40_benches/tg_psi_dm96_oap/runs
C=/home/dcr/dev/MACOS_resources/mmacos/templates/30_instruments/bench_ctb
n=0
while ! grep -q '\[closechain4\] queue drained' "$T/closechain4.nohup" 2>/dev/null; do
    n=$((n+1)); [ $n -gt 1200 ] && { echo "[closechain5] ABORT: chain4 never drained (10 h)"; exit 1; }
    sleep 30
done
echo "[closechain5] chain4 drained -> item 4"
"$T/item4seq.sh" > "$T/item4seq.nohup" 2>&1
echo "[closechain5] item 4 done -> the CTB beam probe (item 7 precondition)"
( cd "$C" && matlab -batch "ctb_beam_probe" ) > "$T/ctb_beam_probe.log" 2>&1
echo "[closechain5] ctb_beam_probe exit $?"
echo "[closechain5] queue drained"
