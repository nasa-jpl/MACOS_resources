#!/usr/bin/env bash
# the plain MIRROR loop rows on the FIXED feed (oap.SRC_TRIM 0), the full row set + the corrected fiber budget rows (2026-10-12)
cd /home/dcr/dev/MACOS_resources/mmacos/templates/40_benches/cophi_dm96
./cophi_batch.sh loop_oap_plain "'bench.optics','oap','bench.polarizing',false,'stages',{'loop'},'cophi.loop.fiber',3.5e-3"
echo LOOPFIX_DONE
