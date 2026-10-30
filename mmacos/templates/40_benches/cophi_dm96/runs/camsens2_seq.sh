#!/usr/bin/env bash
cd /home/dcr/dev/MACOS_resources/mmacos/templates/40_benches/cophi_dm96
until grep -q LOCKIN_DONE runs/lockin_seq.log; do sleep 60; done
sed -i 's/CAMSENS_DONE/CAMSENS1_DONE/' runs/camsens_seq.log
runs/camsens_seq.sh > runs/camsens2_seq.log 2>&1
echo CAMSENS2_DONE >> runs/camsens2_seq.log
