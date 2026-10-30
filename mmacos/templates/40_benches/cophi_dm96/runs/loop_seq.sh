#!/usr/bin/env bash
# the COPHI loop stage, both rigs, the plain splitter (the priced layout); detached (~3.6 h per rig)
cd /home/dcr/dev/MACOS_resources/mmacos/templates/40_benches/cophi_dm96
./cophi_batch.sh loop_lens_plain "'bench.polarizing',false,'stages',{'loop'}"
./cophi_batch.sh loop_oap_plain  "'bench.optics','oap','bench.polarizing',false,'stages',{'loop'}"
echo LOOPSEQ_DONE
