#!/usr/bin/env bash
# The reflective front end, item 1: the OAP drawing defect and its fix.
#   lens22g -- the LENS rig re-emitted post-fix at the lens22 settings.  The
#              GATE: its vlayout PNG must be pixel-identical to runs/lens22/
#              lens22_vlayout.png and its clearance table identical to the
#              record's (Bench.station returns vpt for every lens-rig element).
#   oap22   -- the OAP rig on the 22.5 deg bench, post-fix: the first layout
#              with the mirrors on the beam, and the first HONEST clearance
#              table for the reflective rig (pre-fix the tool modelled the
#              source leg as ending 149 mm off OAP1 plus a 150 mm phantom leg).
set -u
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$here"
./tg96_batch.sh lens22g "'stages',{'bench','figs','clearance'}"
./tg96_batch.sh oap22   "'bench.optics','oap','stages',{'bench','figs','clearance'}"
echo "[refseq] done"
