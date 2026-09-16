#!/usr/bin/env bash
# BRIEF_to_gauge_close item 5 (realism item 6) -- the snapshot form's
# polarization AT THE ANGLES EACH RIG IS ACTUALLY BUILT AT.
#
# The record's plate-diattenuation law (0.149*beta^2 of arm rotation) was
# established on a 45-deg ladder; the benches are built at 22.5, and the
# reflective rig adds two metal folds at 20 and 25 deg that the lens ladder
# knows nothing about.  Each run prints three columns, which is the point:
#   gain      the UNCORRECTED four-step scale (what the record quotes)
#   gain_cor  the same frames re-referenced to the arms' MEASURED azimuths --
#             the analyzer sweep, free and exact (analyzer_basis already spans
#             every analyzer angle from three traces per arm)
#   resid     the pupil-VARYING part left over, which is all a matrix
#             calibrated on the bench cannot absorb
# Quoting only `gain` overstates what a calibrated gauge actually suffers.
#
# Model 256 / NGRID 63: geometry and polarization, not a diffraction result.
set -u
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$here"
L=../../90_polarization/tg_psi_dm
OAP="'optics','oap','OAP1_AOI',20,'OAP2_AOI',25,'OAP1_SIDE',1,'OAP2_SIDE',-1,'SRC_AT_FOCUS',true,'POL_IN','source','D_RC_L2',125"

( cd "$L" && matlab -batch "tg_aoi_ladder_batch(22.5, 'tag','lens22')" ) \
    > runs/aoi_lens22.log 2>&1
echo "[aoiseq] lens leg exit $?" | tee -a runs/aoi_lens22.log
( cd "$L" && matlab -batch "tg_aoi_ladder_batch(22.5, $OAP, 'tag','oap22')" ) \
    > runs/aoi_oap22.log 2>&1
echo "[aoiseq] oap leg exit $?" | tee -a runs/aoi_oap22.log
echo "[aoiseq] done"
