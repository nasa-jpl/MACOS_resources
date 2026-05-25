#!/usr/bin/env bash
#
# Run the GMI regression test series.
#
# Usage:
#   ./run_regression.sh           # run all tests
#   ./run_regression.sh --bootstrap   # (re)generate reference .mat files
#
# Reference state lives in ./reference/*.mat (committed).  Regenerate
# only after intentional behavior changes, then inspect the diff to
# confirm the new numbers are what you wanted.
#
# Exit code 0 on all-pass, non-zero on any failure.

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MATLAB_BIN="$(ls -d /usr/local/MATLAB/R*/bin/matlab 2>/dev/null | tail -1)"
if [ -z "$MATLAB_BIN" ]; then
    echo "ERROR: no MATLAB found under /usr/local/MATLAB/R*/bin/matlab"
    echo "  (GMI regression requires MATLAB to load the mex)"
    exit 1
fi

# The mex file lives one level up; addpath via the entry script.
cd "$HERE"

# NOTE on the exit-code gate.  The current GMI.mexa64 SIGSEGVs in
# MATLAB's mex-cleanup phase at process exit -- AFTER every script
# call has returned and the user's `quit` has fired.  All work
# completes correctly; only MATLAB's teardown hits the bug
# (suspected Fortran-module finalizer in libsmacos.a tripping on
# second unload).  So we can't trust the exit code: a clean run
# still exits non-zero.  Gate is "did the script print its explicit
# completion marker before MATLAB died?"  Drop this gate and trust
# the exit code once the finalizer bug is fixed.

if [ "${1:-}" = "--bootstrap" ]; then
    echo "Bootstrapping reference .mat files (committing CURRENT behavior as ground truth)"
    out="$("$MATLAB_BIN" -batch "bootstrap_reference" 2>&1)" || true
    echo "$out" | tail -8
    if echo "$out" | grep -q '\[bootstrap\] done\.'; then
        echo
        echo "Reference .mat files written to ./reference/"
        echo "Review the diff, commit if intended."
        exit 0
    fi
    echo "FAILED: bootstrap did not reach completion marker"
    exit 1
fi

out="$("$MATLAB_BIN" -batch "regression_main" 2>&1)" || true
echo "$out" | tail -40
if ! echo "$out" | grep -q '=== summary:'; then
    echo "FAILED: suite did not reach summary marker"
    exit 1
fi
# Summary line format: "=== summary: N passed, M failed (of K) ==="
if echo "$out" | grep -qE '=== summary: [0-9]+ passed, 0 failed'; then
    echo
    echo "All GMI regression tests passed."
    exit 0
fi
echo "FAILED: one or more regression tests did not pass"
exit 1
