#!/usr/bin/env bash
# Deliverable 9 (BRIEF_to_capture.md): the start-rms ladder, BOTH WAYS.
#
# The question the deck's capture slide asks: how large an initial figure
# can a loop closed through each reading actually capture, and does
# unwrapping the differential move that number.  One runner invocation
# covers the whole ladder -- loop.start_rms takes a VECTOR and the start
# matrix is measured ONCE per starting surface for every class, which is
# what makes the ladder affordable at all.
#
# DEPARTURES, both stated in the report:
#  * 193 rays, not 385.  The box is the limit: the 385-ray ladder is ~4x
#    these states and the queue behind it (the within-scan drift, the
#    reference-arm walk, the pinhole trade) would not run at all.  The
#    wrap is a property of the PHASE, not of the sampling, so the
#    threshold this measures does not depend on the ray count; the
#    pixel-gradient limit the unwrapper trades it for DOES (4 px per
#    actuator at 385, 2.5 at 193), so 193 is the PESSIMISTIC choice for
#    the unwrapped arm.
#  * K 40, not 60.  At gain 0.5 the contraction is 0.5 per cycle, so a
#    descent that has not reached 3 pm by cycle 40 (0.5^40 = 9e-13 of the
#    start) is not going to; the steady-state tail is still 20 cycles.
cd "$(dirname "$0")/.."
RD="'readings',{'L','S','V','P','PF'}, 'loop.readings',{'L','S','V','P','PF'}, 'dm_use',1"
LAD="'loop.start_rms',[30 60 100 150 200 300]*1e-6, 'loop.nph',[1e13 1e15], 'loop.drifts',{}, 'loop.floor',false, 'loop.steps',[], 'loop.K',40, 'loop.recal_list',[0]"

# the two arms of the question, cheapest-informative first
./pdi_batch.sh cap_nouw "pdi_params, 'stages',{'bench','loop'}, $RD, $LAD, 'loop.unwrap',false"
./pdi_batch.sh cap_uw   "pdi_params, 'stages',{'bench','loop'}, $RD, $LAD, 'loop.unwrap',true"

# and what re-calibrating on the surface adds, where the reading alone fails
./pdi_batch.sh cap_uw_recal "pdi_params, 'stages',{'bench','loop'}, $RD, 'loop.start_rms',[100 200 300]*1e-6, 'loop.nph',[1e15], 'loop.drifts',{}, 'loop.floor',false, 'loop.steps',[], 'loop.K',40, 'loop.recal_list',[10], 'loop.unwrap',true"
./pdi_batch.sh cap_nouw_recal "pdi_params, 'stages',{'bench','loop'}, $RD, 'loop.start_rms',[100 200 300]*1e-6, 'loop.nph',[1e15], 'loop.drifts',{}, 'loop.floor',false, 'loop.steps',[], 'loop.K',40, 'loop.recal_list',[10], 'loop.unwrap',false"
echo "gcap done"
