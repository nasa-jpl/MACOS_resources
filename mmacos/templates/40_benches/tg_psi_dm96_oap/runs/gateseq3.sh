#!/usr/bin/env bash
# BRIEF_to_gauge_close item 3 -- the TEST of the tail tuner's winner gate.
#
# The gate: after the tune, ONE single-actuator row is read through the ray
# affine in ACTUATOR space, and a winner below 0.95 is REFUSED in favor of the
# geometric seed.  It exists because on the OAP rig every term the cost
# computes (null, poke peak, localization, wrap) preferred a tail that reads a
# single actuator at 0.0338 where the seed reads 0.9809 (REPORT_reflective
# 4.5; runs/tailA vs runs/tailB).
#
# Two verifies, each `tg96_tail('verify_tail',...)`, which runs the gate ALONE
# -- no tune, so this costs one placement + one row per leg instead of 150
# evaluations.  The bench arguments on each line are the ones its tail was
# tuned with; a gate run on a DIFFERENT bench would measure nothing.
#
#   gate3_win  objwin3_tail.mat on the OAP design  -> must be REFUSED
#   gate3_lens lens_tail.mat    on the lens rig    -> must be ACCEPTED (0.9968)
#
# Non-vacuity is the first leg: a gate that accepts everything would pass the
# second leg alone.
set -u
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$here"
OAP="'bench.optics','oap','bench.POL_IN','source','bench.SRC_AT_FOCUS',true,'bench.D_RC_L2',125,'oap.OAP1_AOI',20,'oap.OAP2_AOI',25,'oap.OAP1_SIDE',1,'oap.OAP2_SIDE',-1"

OAP_ENTRY=tg96_tail_batch ./oap_fold_batch.sh gate3_win  "$OAP,'verify_tail','objwin3_tail.mat'"
OAP_ENTRY=tg96_tail_batch ./oap_fold_batch.sh gate3_lens "'bench.optics','lens','verify_tail','lens_tail.mat'"
echo "[gateseq3] done"
