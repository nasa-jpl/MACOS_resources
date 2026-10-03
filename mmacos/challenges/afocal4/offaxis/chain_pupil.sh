#!/bin/bash
# Wait for the three decenter arms to finish, THEN run the pupil ladder.
# Sequenced per Dave 2026-09-01: "let the other decenters finish, then run
# the pupil ladder."
cd /home/dcr/dev/MACOS_res_dev/mmacos/challenges/afocal4/offaxis
for p in 943167 943168 943169; do
    while kill -0 "$p" 2>/dev/null; do sleep 30; done
done
echo "[chain] all three decenter arms exited $(date +%H:%M:%S) -- starting pupil ladder"
for N in 4 5 7; do
    OW_N=$N OW_H=0,0.55 OW_EVALS=400 OW_ROUNDS=2 OW_PUPIL=1 OW_TAG=OAP$N \
      MACOS_HOME=/home/dcr/dev/macos/macos_f90 timeout 36000 \
      matlab -batch "run('run_offaxis_wfe.m')" > oap_N$N.log 2>&1 &
done
wait
echo "[chain] pupil ladder complete $(date +%H:%M:%S)"
