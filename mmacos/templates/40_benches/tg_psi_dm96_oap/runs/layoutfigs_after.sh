#!/usr/bin/env bash
# Regenerate every deck layout figure with the three-leg colour convention
# (dmg_leg_draw, Dave 2026-10-05) once the gauge run queue is done (capt96_lens.done).
set -u
cd "$(dirname "$0")/.."
n=0; until [ -f runs/capt96_lens.done ]; do sleep 60; n=$((n+1)); [ $n -gt 1440 ] && exit 1; done
echo "[$(date '+%F %T')] start layoutfigs" >> runs/redo96_ph_seq.log
./runs/layoutfigs.sh > runs/layoutfigs_after.log 2>&1
echo "[$(date '+%F %T')] end layoutfigs rc=$?" >> runs/redo96_ph_seq.log
touch runs/layoutfigs.done
