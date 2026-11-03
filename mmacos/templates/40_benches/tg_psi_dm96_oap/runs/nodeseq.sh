#!/usr/bin/env bash
set -u
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$here"
common="'MODEL',512,'NGRID',65,'grid.N_G',256,'grid.DX_G',0.42,'smoke',true"
./tg96_batch.sh node22t "$common,'stages',{'bench','clearance'}"
./tg96_batch.sh nodesolve "$common,'bench.BS_AOI',[],'stages',{'bench'}"
echo "[nodeseq] done"
