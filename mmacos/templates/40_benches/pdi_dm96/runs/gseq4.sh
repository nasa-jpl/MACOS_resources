#!/usr/bin/env bash
# Gauge-deck sequence 4 (deliverable 5): the reference arm's own drift.  A
# random walk of the P/SRI reference arm's phase relative to the test arm --
# the non-common-path term no common-path reading has -- at three sizes, in
# the loop under the 2 pm walk.  P rides along as the common-path control.
cd "$(dirname "$0")/.."
export ZWFS_MEMMAX=20G

# FIRST, the measurement the deck's recommendation rests on.  cap_uw_recal
# showed that unwrapping AND re-calibrating together take PF's capture from
# 60 nm of surface to 100 (neither alone does it), and cap_state_uw showed
# that P with a shutter frame is PF's twin in capture to four digits without
# recalibration.  The recommended configuration is P-with-shutter, so its
# capture with re-calibration has to be MEASURED, not inferred from the twin.
./pdi_batch.sh cap_state_uw_recal "pdi_params, 'stages',{'bench','loop'}, 'readings',{'P'}, 'loop.readings',{'P'}, 'dm_use',1, 'pdi.b2','state', 'loop.start_rms',[100 150 200]*1e-6, 'loop.nph',[1e13 1e15], 'loop.drifts',{}, 'loop.floor',false, 'loop.steps',[], 'loop.K',40, 'loop.recal_list',[10], 'loop.unwrap',true"

RD="'readings',{'P','PF'}, 'loop.readings',{'P','PF'}, 'dm_use',1"
L="'loop.nph',[1e13 1e15], 'loop.drifts',{'walk'}, 'loop.steps',[], 'loop.floor',true, 'loop.K',60"
for w in 1e-3 1e-2 1e-1; do
    tag="rw193_$(echo $w | tr -d '-.')"
    ./pdi_batch.sh "$tag" "pdi_params, 'stages',{'bench','loop'}, $RD, $L, 'pdi.ref_walk',$w"
done
echo "gseq4 done"
