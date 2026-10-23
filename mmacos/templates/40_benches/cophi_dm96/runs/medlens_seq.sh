#!/usr/bin/env bash
cd /home/dcr/dev/MACOS_resources/mmacos/templates/40_benches/cophi_dm96
until grep -q CAMSENS2_DONE runs/camsens2_seq.log 2>/dev/null; do sleep 60; done
./cophi_batch.sh noiseX_lens_plain_med "'bench.polarizing',false,'cophi.annulus_fill','median','stages',{'noise'},'cophi.noise_nreal',12"
echo MEDLENS_DONE
