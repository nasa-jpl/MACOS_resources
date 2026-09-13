#!/usr/bin/env bash
# Gauge-deck sequence 4 (deliverable 5): the reference arm's own drift.  A
# random walk of the P/SRI reference arm's phase relative to the test arm --
# the non-common-path term no common-path reading has -- at three sizes, in
# the loop under the 2 pm walk.  P rides along as the common-path control.
cd "$(dirname "$0")/.."
RD="'readings',{'P','PF'}, 'loop.readings',{'P','PF'}, 'dm_use',1"
L="'loop.nph',[1e13 1e15], 'loop.drifts',{'walk'}, 'loop.steps',[], 'loop.floor',true, 'loop.K',60"
for w in 1e-3 1e-2 1e-1; do
    tag="rw193_$(echo $w | tr -d '-.')"
    ./pdi_batch.sh "$tag" "pdi_params, 'stages',{'bench','loop'}, $RD, $L, 'pdi.ref_walk',$w"
done
echo "gseq4 done"
