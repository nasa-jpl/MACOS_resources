#!/usr/bin/env bash
# Descent-stall recovery 2x2 (2026-09-30).  Baseline = runs/samp512 (100 nm: 96.937 pm, rho 0.787;
# 30 nm control: 4.003 pm, rho 0.526).  Same invocation as samp512 plus the fix knobs:
#   fixA  place.lit_margin_mm 1   (control set inside the 48 mm DM aperture by one pitch; the record's lit reaches 49.1)
#   fixB  battery.matrix_reg column (per-column Tikhonov)
#   fixAB both
# tg96_batch.sh runs MATLAB in the foreground and waits for any other DM-gauge batch, so this is serial.
set -u
cd "$(dirname "$0")/.."
BASE="'bench.optics','oap','bench.POL_IN','source','bench.D_RC_L2',125,'oap.OAP1_AOI',20,'oap.OAP2_AOI',25,'oap.OAP1_SIDE',1,'oap.OAP2_SIDE',-1,'NGRID',512,'battery.matrix_step',8,'loop.start_rms',[1e-4 3e-5],'loop.recal_every',0,'loop.steps',[],'loop.drifts',{},'loop.floor',false,'loop.nph',1e13,'stages',{'bench','loop','figs'}"
export TG96_MEMMAX=20G
for cell in "fixA|'place.lit_margin_mm',1" "fixB|'battery.matrix_reg','column'" "fixAB|'place.lit_margin_mm',1,'battery.matrix_reg','column'"; do
    tag="${cell%%|*}"; extra="${cell#*|}"
    echo "[$(date '+%F %T')] start $tag" >> runs/fix2x2_seq.log
    ./tg96_batch.sh "$tag" "$BASE,$extra"
    echo "[$(date '+%F %T')] end $tag rc=$?" >> runs/fix2x2_seq.log
done
echo "[$(date '+%F %T')] ALL DONE" >> runs/fix2x2_seq.log
touch runs/fix2x2.done
