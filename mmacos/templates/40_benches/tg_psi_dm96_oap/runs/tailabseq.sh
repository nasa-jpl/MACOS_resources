#!/usr/bin/env bash
# The seed-vs-tuned tail A/B.  oapifo's reading BREAKS at the FIRST rung of the
# break ladder -- "30 nm BROKE (wrap: base reads 1.00 of lambda/4)" -- where the
# record's rigs break only at 240 nm.  The base saturates the four-step's
# unambiguous range about 8x too easily, which is the size of the gain deficit.
#
# Leading hypothesis: the RETUNED tail.  tg96_tail's 'oap' objective is
# SHARPNESS (the recovered poke peak), which rewards a sharp poke image without
# enforcing that the detector stays at the DM's PUPIL CONJUGATE.  Its winner
# moved DET_TRIM to +45.96 mm (the lens rig's is -1.25) and the pupil
# magnification changed 1.95x.  A detector off the conjugate carries large
# field curvature into the map, which saturates lambda/4 and gives exactly the
# erratic, sign-flipping, low-correlation rows Stage E shows.
#   tailA -- the design with the TUNED tail   (reproduces oapifo at 512)
#   tailB -- the design with the GEOMETRIC SEED tail (bench.tail_from_mat false)
# If tailB reads and tailA does not, the tail retune is the defect and the fix
# is the objective, not the geometry.
set -u
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$here"
G="'bench.optics','oap','bench.POL_IN','source','bench.SRC_AT_FOCUS',true,'bench.D_RC_L2',125,'oap.OAP1_AOI',20,'oap.OAP2_AOI',25,'oap.OAP1_SIDE',1,'oap.OAP2_SIDE',-1"
R="'MODEL',512,'NGRID',193,'grid.N_G',256,'grid.DX_G',0.42,'smoke',true,'stages',{'bench','battery'}"
cp -f oap22d_tail.mat tailA_tail.mat
./tg96_batch.sh tailA "$G,$R"
./tg96_batch.sh tailB "$G,'bench.tail_from_mat',false,$R"
echo "[tailabseq] done"
