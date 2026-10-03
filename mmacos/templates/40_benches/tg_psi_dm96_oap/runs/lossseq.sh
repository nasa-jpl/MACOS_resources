#!/usr/bin/env bash
# Where does the reflective rig lose rays at a large fold?  The 90-deg fold is
# the one that clears the input polarizer geometrically (the source leg then
# runs PARALLEL to the polarizer's plane), so whether its ray loss is real
# optics or a modelling artifact decides the design.
set -u
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$here"
OAP_ENTRY=oap_loss_probe_batch ./oap_fold_batch.sh loss
OAP_ENTRY=oap_loss_probe_batch ./oap_fold_batch.sh loss_src "'POL_IN','source'"
echo "[lossseq] done"
