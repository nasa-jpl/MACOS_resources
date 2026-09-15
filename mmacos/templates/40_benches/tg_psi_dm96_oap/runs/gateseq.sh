#!/usr/bin/env bash
# Item-2 gates: the builder gained POL_IN and the tail tuner gained the node
# arguments, so re-run the classes that cover twyman_green and the polarizing
# rig.  Waits for the sweep chain's lock.
set -u
root=/home/dcr/dev/MACOS_resources/mmacos
cd "$root"
for t in tBench tTgPol tTgPol2 tPropLayout tDmgLoop; do
    echo "=== $t ==="
    ./run_mmacos_tests.sh "$t" 2>&1 | grep -E "=== class|pass, .* fail|Error|error"
done
echo "[gateseq] done"
