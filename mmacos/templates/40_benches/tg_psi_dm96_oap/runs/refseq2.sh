#!/usr/bin/env bash
# Item 2, the design.  Both jobs take the batch wrappers' one-at-a-time lock,
# so this queues behind whatever is already running.
#   fold2 -- the clearance sweep with the input polarizer moved into the
#            DIVERGING source leg (POL_IN 'source').  fold1 showed the 10 mm
#            collimated-leg station cannot be cleared by ANY fold angle.
#   conj  -- is the reflective rig's "fold coma" the FOLD, or the 25 mm
#            zSource conjugate error that the fold amplifies?
set -u
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$here"
./oap_fold_batch.sh fold2 "'A1',6:2:24,'A2',5:5:45,'POL_IN','source'"
OAP_ENTRY=oap_conj_probe_batch ./oap_fold_batch.sh conj
echo "[refseq2] done"
