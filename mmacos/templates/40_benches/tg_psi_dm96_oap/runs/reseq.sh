#!/usr/bin/env bash
# Item 4, re-run at the RECORD'S resolution on the seed tail.  tailB shows the
# designed bench reads (0.989 dense-random, 203 pm) but at model 512 / NGRID
# 193, while every record number is model 1024 / NGRID 385.  A cross-
# configuration claim from a resolution-mismatched pair is not a result.
set -u
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$here"
D="'bench.optics','oap','bench.POL_IN','source','bench.SRC_AT_FOCUS',true,'bench.D_RC_L2',125,'oap.OAP1_AOI',20,'oap.OAP2_AOI',25,'oap.OAP1_SIDE',1,'oap.OAP2_SIDE',-1,'bench.tail_from_mat',false"
B="'clear.BODY',struct('Baffle',50,'Detector',50,'TestOptic',90,'PZT',60)"
./tg96_batch.sh oapifo2 "$D,$B,'stages',{'bench','battery','figs','clearance'}"
echo "[reseq] done"
