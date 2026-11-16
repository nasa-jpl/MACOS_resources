#!/usr/bin/env bash
# The two layout figures, in the deck recipe (deliverable 6).  Small
# model-512 jobs; they do not need the model-1024 lock, but they DO load the
# engine, so run them when the box is not at its memory ceiling.
set -e
cd "$(dirname "$0")/.."
matlab -batch "pdi_layout_fig; exit(0)"   > runs/pdi_layout_fig.log 2>&1
matlab -batch "psri_layout_fig; exit(0)"  > runs/psri_layout_fig.log 2>&1
echo "gfigs done"
