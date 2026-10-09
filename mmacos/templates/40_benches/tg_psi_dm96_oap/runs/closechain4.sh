#!/usr/bin/env bash
# BRIEF_to_gauge_close -- the queue's tail, after item 2's four runs.
#   gateseq3  item 3 RE-RUN: the first attempt refused both legs with gain NaN
#             because tg96_tail did not have dmg_frame / tg96_place on its
#             path, so the gate was failing CLOSED -- it refused everything.
#             The lens leg (which must ACCEPT) is what caught that; a gate
#             tested only against a tail that should fail cannot see it.
#   aoiseq    item 5: the AOI ladder at the angles each rig is built at.
set -u
T=/home/dcr/dev/MACOS_resources/mmacos/templates/40_benches/tg_psi_dm96_oap/runs
n=0
while ! grep -q '\[item2seq\] done' "$T/item2seq.nohup" 2>/dev/null; do
    n=$((n+1)); [ $n -gt 960 ] && { echo "[closechain4] ABORT: item2seq never finished (8 h)"; exit 1; }
    sleep 30
done
echo "[closechain4] item 2 done -> item 3 gate re-run"
"$T/gateseq3.sh" > "$T/gateseq3b.nohup" 2>&1
echo "[closechain4] item 3 done -> item 5 AOI ladder"
"$T/aoiseq.sh"   > "$T/aoiseq.nohup"    2>&1
echo "[closechain4] queue drained"
