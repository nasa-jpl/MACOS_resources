#!/usr/bin/env bash
# The reflective tail, re-tuned on the DESIGNED bench (item 2): OAP1 20 deg /
# OAP2 25 deg, sides +1/-1, the polarizer in the source leg, the output optics
# 125 mm ahead of OAP2, the collimator fed at its true focus.  The record's
# oap_tail.mat was fit on the 7-deg bench with the 25 mm conjugate error; both
# are gone, so the tail must be re-fit.  Writes oap22d_tail.mat, which tg96_run
# picks up for the oap22d run (the record's oap_tail.mat is left alone).
set -u
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$here"
D="'bench.optics','oap','bench.POL_IN','source','bench.SRC_AT_FOCUS',true,'bench.D_RC_L2',125,'oap.OAP1_AOI',20,'oap.OAP2_AOI',25,'oap.OAP1_SIDE',1,'oap.OAP2_SIDE',-1"
OAP_ENTRY=tg96_tail_batch ./oap_fold_batch.sh oap22d_tail "$D,'tag','oap22d'"
echo "[tailseq] done"
