#!/usr/bin/env bash
# BRIEF_to_gauge_close item 1 -- the vector pair on the REDESIGNED reflective
# rig: where its 634 pm uncalibrated G4 sits on the zwfs record's channel-phase
# scale, and whether the overcoat moves it.
#
# The v_cal 'ideal' / bareAl leg is NOT re-run: runs/oapsens22 IS that leg at
# these exact settings (its V rows are single 1.0022/4 pm, grid 1.0100/5,
# dense 1.0086/633, ladder 0.9928 at 30 nm / 0.9784 at 40, capture 60 nm).
# Both runs below change ONE thing against it.
#
#   vmap22  -- mask.v_cal 'map': the polarimetrically calibrated solver (the
#              true per-channel pupil maps and constants).  The record says
#              this reproduces the ideal rows at 0.3 rad of channel phase
#              difference; the question is whether it does so here.
#   vqw22   -- bench.coat_oap 'qwAl': the MgF2 overcoat at a QUARTER wave of
#              the bench's own 632.8 nm (114.6 nm physical).  The record's
#              'protectedAl' is 229.3 nm = a HALF wave there.  The engine's
#              measured overcoat rule (macos_f90/CLAUDE.md, "overcoat
#              quarter-wave reversal") is that the polarization trade REVERSES
#              across the quarter-wave condition -- at the true quarter wave
#              the coating's cross-polarization is ~0.05x of bare.  bareAl
#              costs 3.0x here (634 vs 208 pm with no coating), so the
#              prediction is ~1.1x of the coating-free 208, i.e. ~230 pm.
#
# Rows: single / grid / dense, and the break ladder, which always runs.
# Readings V only (S and P do not enter the vector pair's rows and cost time).
# Everything else is oapsensseq.sh verbatim, MASK_TRIM 0 included.
set -u
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$here"
D="'bench.optics','oap','bench.OAP1_AOI',20,'bench.OAP2_AOI',25,'bench.OAP1_SIDE',1,'bench.OAP2_SIDE',-1,'bench.SRC_AT_FOCUS',true,'bench.MASK_TRIM',0"
C="'MODEL',1024, 'NGRID',193, 'readings',{'V'}, 'stages',{'bench','battery'}, 'battery.rows',{'base/single','base/grid','base/rand'}, 'battery.calib_surface','base', 'mask.v_arm','engine'"

./zwfs_batch.sh vmap22 "$D, 'bench.coat_oap','bareAl', $C, 'mask.v_cal','map'"
./zwfs_batch.sh vqw22  "$D, 'bench.coat_oap','qwAl',   $C, 'mask.v_cal','ideal'"
echo "[vcloseseq] done"
