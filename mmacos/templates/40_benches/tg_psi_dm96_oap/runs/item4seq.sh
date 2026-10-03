#!/usr/bin/env bash
# BRIEF_to_gauge_close item 4 -- realism 3, 3b, 4: real glass and a real camera.
#
# Two rounds, each: change the parts -> RETUNE the tail through item 3's gate
# -> run the mask-sensor gate (bench + battery, S / V / P on the lens rig at
# 22.5, the gate22_193 settings).  The retune is not optional bookkeeping: a
# 10 mm splitter and a 2 mm plate under the mask move the focus by
# t*(1-1/n) each, and an un-retuned run would measure the seating error rather
# than the glass.
#
#   THICKNESSES (realism 3, first half)
#     BS_T: the sheet value is PRE-SCALE on the tg96 side (the runner applies
#     s = 96/56 = 1.7143) and PHYSICAL on the zwfs side.  10 mm therefore reads
#     5.8333 for tg96 and 10 for zwfs -- the one asymmetry in these sheets, and
#     the reason it is spelled out here rather than left to the reader.
#     EDGE_MARGIN 4 mm: a real edge on a 103 mm singlet (the record is 2).
#
#   SUBSTRATES (realism 3, second half)
#     PLATE_SUB [1.4585 2]: fused silica at 632.8 nm, 2 mm, under the input
#     polarizer, both arm QWPs, the output plate and the analyzer.
#     MASK_SUB  [1.4585 2]: the mask's own plate, in the CONVERGING beam --
#     the one place a plane-parallel plate is not just path (W040 and a focus
#     shift).  Its faces go ahead of the sandwich's entrance sphere and are
#     inserted INSIDE the existing gap, so the mask does not move and the cost
#     is measurable rather than mixed with a geometry change.
#     MASK_TRIM 'scan' re-finds the mask focus, which is what absorbs the
#     shift; without it the run would blame the glass for a seating error.
#
# THE TWO RIGS HAVE SEPARATE TAILS, and this script does not pretend otherwise.
# tg96_tail retunes the INTERFEROMETER's field-lens tail; the ZWFS carries its
# own (FL_F 42.5325, D_MASK_FL 39.7694, DET_TRIM -1.2473 -- an OLD tg96_tail
# winner from the 7-degree rig, for a bench with a mask SANDWICH the PSI rig
# does not have, which is why its D_MASK_FL is 39.8 against the PSI rig's
# 10.8).  Feeding the PSI retune to the ZWFS would seat a tail tuned on a
# different bench.  What the ZWFS DOES re-find is MASK_TRIM, and that is the
# term the mask plate moves (t*(1-1/n) of focus shift); the field-lens tail
# sets the DETECTOR's pupil conjugate, which plane-parallel glass in the
# collimated legs does not move.  If the ZWFS rows degrade anyway, that names
# the follow-on precisely -- a ZWFS-SIDE tail tune, which does not exist yet --
# rather than hiding it under a borrowed tail.
#
# EACH RETUNE IS CONSUMED.  tg96_tail writes <tag>_tail.mat and tg96_run's
# lookup order is <tag>_tail.mat first, so the tg96_batch line right after each
# retune runs the INTERFEROMETER on the tail that retune just produced -- which
# is what makes the retune a result rather than an exercise of the gate.  The
# gate's own verdict is printed by the retune (TAIL GATE / TAIL GATE REFUSED);
# if it refuses, the run below is on the geometric seed and says so.
#
# The camera (realism 4) needs no run: the parts list and the Stage PLACE line
# print it from P.cam on every run below.
set -u
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$here"
Z=../zwfs_dm96
GATE="'MODEL',1024, 'NGRID',193, 'readings',{'S','V','P'}, 'stages',{'bench','battery'}, 'battery.rows',{'base/single','base/grid','base/rand'}, 'battery.calib_surface','base'"

# ---- round 1: thicknesses -------------------------------------------
THK_TG="'bench.BS_T',5.8333,'bench.EDGE_MARGIN',4.0"
THK_ZW="'bench.BS_T',10.0,'bench.EDGE_MARGIN',4.0"
OAP_ENTRY=tg96_tail_batch ./oap_fold_batch.sh thk22_tail "'bench.optics','lens',$THK_TG,'tag','thk22'"
./tg96_batch.sh thk22 "'bench.optics','lens',$THK_TG,'battery.rows',{'base/single'},'stages',{'bench','battery'}"
( cd "$Z" && ./zwfs_batch.sh thk22 "$THK_ZW, $GATE, 'bench.MASK_TRIM','scan'" )

# ---- round 2: substrates on top of the thicknesses ------------------
SUB="'bench.PLATE_SUB',[1.4585 2.0],'bench.MASK_SUB',[1.4585 2.0]"
OAP_ENTRY=tg96_tail_batch ./oap_fold_batch.sh sub22_tail "'bench.optics','lens',$THK_TG,$SUB,'tag','sub22'"
./tg96_batch.sh sub22 "'bench.optics','lens',$THK_TG,$SUB,'battery.rows',{'base/single'},'stages',{'bench','battery'}"
( cd "$Z" && ./zwfs_batch.sh sub22 "$THK_ZW, $SUB, $GATE, 'bench.MASK_TRIM','scan'" )
echo "[item4seq] done"
