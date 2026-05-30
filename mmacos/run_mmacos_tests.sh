#!/usr/bin/env bash
# run_mmacos_tests.sh -- run the mmacos matlab.unittest suite in batch
# mode and exit non-zero on any failure.
#
# Mirror of MACOS_resources/pymacos/run_proper_tests.sh.  See PLAN.md
# §5.4 Phase 3 for the design.
#
# Usage:
#   ./run_mmacos_tests.sh                   # run all tests
#   ./run_mmacos_tests.sh tMacosPkg         # run one class
#   ./run_mmacos_tests.sh -k roundtrip      # run methods matching a tag
#
# Env overrides:
#   MACOS_BUILD_DIR  path to macos cmake build with libsmacos.a
#                    (default: ~/dev/macos/build_release_gfortran)
#   MATLAB_DIR       MATLAB install root (default: latest under
#                    /usr/local/MATLAB)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MACOS_BUILD_DIR="${MACOS_BUILD_DIR:-$HOME/dev/macos/build_release_gfortran}"
MATLAB_DIR="${MATLAB_DIR:-$(ls -1d /usr/local/MATLAB/R[0-9]*[ab] 2>/dev/null | sort -V | tail -1)}"

if [ -z "$MATLAB_DIR" ] || [ ! -d "$MATLAB_DIR" ]; then
    echo "error: MATLAB not found.  Set MATLAB_DIR or install under /usr/local/MATLAB/." >&2
    exit 2
fi

# Rebuild mmacos.mexa64 if any source is newer.
if [ ! -f mmacos.mexa64 ] \
   || [ mmacos_mex.F -nt mmacos.mexa64 ] \
   || [ mmacos_gen.F -nt mmacos.mexa64 ]; then
    echo "(re)building mmacos.mexa64..."
    make FC=gfortran MACOS_BUILD_DIR="$MACOS_BUILD_DIR" >/dev/null
fi

# Pass user filter as a MATLAB-side runtests Tag/Procedure name.
FILTER_ARG=""
if [ $# -ge 1 ]; then
    case "$1" in
        -k) FILTER_ARG=", 'ProcedureName', '*$2*'" ;;
        *)  FILTER_ARG=", 'Name', '$1'" ;;
    esac
fi

SCRIPT="addpath('$SCRIPT_DIR'); suite = matlab.unittest.TestSuite.fromFolder('$SCRIPT_DIR/tests' $FILTER_ARG); if isempty(suite), error('No tests matched filter.'); end; runner = matlab.unittest.TestRunner.withTextOutput; results = runner.run(suite); disp(table(results)); n_failed = sum([results.Failed]); n_passed = sum([results.Passed]); fprintf('=== %d pass, %d fail ===\n', n_passed, n_failed); if n_failed > 0, exit(1); end"

"$MATLAB_DIR/bin/matlab" -batch "$SCRIPT"
