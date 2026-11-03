#!/usr/bin/env bash
# Item 5 of BRIEF_to_reflective: the mask sensors (stepped dimple S, vector
# pair V, pinhole P) on the DESIGNED reflective bench -- OAP1 20 deg / OAP2
# 25 deg, sides +1/-1, the output optics 125 mm ahead of OAP2, and the
# collimator fed at its TRUE focus.  Matched against the 22.5-deg LENS gate
# run, runs/gate22_193, at the same resolution and the same rows.
#
# MASK_TRIM MUST BE 0 HERE.  zwfs_params carries -5.582, which is the LENS
# rig's thin-lens-seed-to-focus correction.  On this bench the seat is at the
# parabola's exact conjugate: oap_focus_probe scans 0.000 lambda F/D of blur at
# MASK_TRIM -0.00 for 20/20, 25/25 and the 20/25 design point (tg_psi_dm96_oap
# runs/zseat, zseat2).  Carrying -5.582 over would seat the mask 5.6 mm off.
#
# The input polarizer does not appear: zwfs_params sets polarizing = false (the
# ZWFS is the test arm alone), so POL_IN is a non-term for these runs.
#
# D_RC_L2 STAYS AT THE ZWFS RECORD'S 55, not the interferometer's 125.  The
# interferometer needed 125 because OAP2's sag envelope (+-50 mm at a 40-deg
# fold) swallowed the ANALYZER 35 mm ahead of its pole -- and the ZWFS has no
# analyzer.  Its element before L2 is the recombination plane, 55 mm ahead,
# against a +-25 mm sag envelope at the design's 25 deg: clear, and measured
# clear (oap_loss_probe runs/loss_a2, 0 rays lost at A2 <= 35 with D_RC_L2 55).
# Moving it to 125 would push L2 from 205 to 275 mm behind the splitter and
# take the ZWFS's TUNED tail (FL_F 42.5325, D_MASK_FL 39.7694, DET_TRIM
# -1.2473) off its station, confounding the front-end measurement this item is
# for with an untuned tail.  The sensors and the interferometer share a
# front-end DESIGN, not a tail.
set -u
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$here"
D="'bench.optics','oap','bench.OAP1_AOI',20,'bench.OAP2_AOI',25,'bench.OAP1_SIDE',1,'bench.OAP2_SIDE',-1,'bench.SRC_AT_FOCUS',true,'bench.MASK_TRIM',0,'bench.coat_oap','bareAl'"
./zwfs_batch.sh oapsens22 "$D, 'MODEL',1024, 'NGRID',193, 'readings',{'S','V','P'}, 'stages',{'bench','battery'}, 'battery.rows',{'base/single','base/grid','base/rand'}, 'battery.calib_surface','base', 'mask.v_arm','engine'"
echo "[oapsensseq] done"
