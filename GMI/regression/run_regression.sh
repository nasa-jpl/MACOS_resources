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

# Both supported GMI builds (gfortran default; ifx via FC=ifx with
# -reentrancy=none in the Makefile) now exit MATLAB cleanly, so we
# trust the exit code directly.  History: a previous marker-based
# gate compensated for the libifcoremt thread-pool SIGSEGV that the
# ifx-built mex hit at MATLAB process exit; resolved by linking the
# single-threaded libifcore variant.  See GMI/Makefile for the
# back-story.

if [ "${1:-}" = "--bootstrap" ]; then
    echo "Bootstrapping reference .mat files (committing CURRENT behavior as ground truth)"
    "$MATLAB_BIN" -batch "bootstrap_reference"
    rc=$?
    if [ "$rc" -eq 0 ]; then
        echo
        echo "Reference .mat files written to ./reference/"
        echo "Review the diff, commit if intended."
        exit 0
    fi
    echo "FAILED: bootstrap exited non-zero ($rc)"
    exit "$rc"
fi

"$MATLAB_BIN" -batch "regression_main"
rc=$?
if [ "$rc" -eq 0 ]; then
    echo
    echo "All GMI regression tests passed."
    exit 0
fi
echo
echo "FAILED: regression suite exited non-zero ($rc)"
exit "$rc"
