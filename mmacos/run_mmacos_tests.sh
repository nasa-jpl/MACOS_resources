#!/usr/bin/env bash
# run_mmacos_tests.sh -- run the mmacos matlab.unittest suite in batch
# mode and exit non-zero on any failure.
#
# Mirror of MACOS_resources/pymacos/run_proper_tests.sh.  See PLAN.md
# §5.4 Phase 3 for the design.
#
# Usage:
#   ./run_mmacos_tests.sh                   # full suite (split by size)
#   ./run_mmacos_tests.sh fast              # all size=128 EXCEPT masks
#                                           # (~10 s total — dev loop)
#   ./run_mmacos_tests.sh masks             # only the CodeV mask suite
#                                           # (~10 min — heavyweight)
#   ./run_mmacos_tests.sh proper            # only Phase 5 PROPER cmp
#   ./run_mmacos_tests.sh tMacosPkg         # one class by name
#   ./run_mmacos_tests.sh -k roundtrip      # methods matching a substring
#
# Why two batches when no filter is given:
#   macos's Fortran engine corrupts internal state when init() is
#   called with a different model_size in the same process — heap
#   abort surfaces during the next FFT-bearing trace (free()/munmap
#   "invalid size/pointer").  Pymacos runs each phase in its own
#   pytest process for exactly this reason (see
#   pymacos/run_proper_tests.sh).  We mirror that: invoke matlab
#   -batch ONCE per model_size group so each session sees a single
#   init().  Per-group exit codes are aggregated at the bottom; a
#   non-zero result from any group propagates.  This workaround is
#   logged in macos/PLAN.md §0 as a follow-up fix.
#
# Env overrides:
#   MACOS_BUILD_DIR  path to macos cmake build with libsmacos.a
#                    (default: ~/dev/macos/build_release_gfortran)
#   MATLAB_DIR       MATLAB install root (default: latest under
#                    /usr/local/MATLAB)
#   PROPER_DIR       MATLAB PROPER install (default: ~/dev/proper_matlab)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MACOS_BUILD_DIR="${MACOS_BUILD_DIR:-$HOME/dev/macos/build_release_gfortran}"
MATLAB_DIR="${MATLAB_DIR:-$(ls -1d /usr/local/MATLAB/R[0-9]*[ab] 2>/dev/null | sort -V | tail -1)}"
PROPER_DIR="${PROPER_DIR:-$HOME/dev/proper_matlab}"

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

SETUP_PATHS="addpath('$SCRIPT_DIR'); addpath('$SCRIPT_DIR/tests/proper_compare');"
if [ -d "$PROPER_DIR" ]; then
    SETUP_PATHS="$SETUP_PATHS addpath('$PROPER_DIR');"
fi

# Build a single matlab -batch invocation.  $1 = TestSuite expression
# returning a suite; $2 = label printed in the summary line.  Returns
# 0 on all-green, 1 on any failure (Matlab exits with that code; bash
# `set -e` above propagates).
#
# Explicit `exit(0)` at end — without it matlab -batch on R2026a hangs
# at process exit when a mex is loaded (same family as the
# `clear mmacos` bug; see mmacos/CLAUDE.md).
run_batch() {
    local SUITE_EXPR="$1"
    local LABEL="$2"
    local SCRIPT="$SETUP_PATHS suite = $SUITE_EXPR; if isempty(suite), exit(0); end; runner = matlab.unittest.TestRunner.withTextOutput; results = runner.run(suite); disp(table(results)); n_failed = sum([results.Failed]); n_passed = sum([results.Passed]); fprintf('=== $LABEL: %d pass, %d fail ===\n', n_passed, n_failed); if n_failed > 0, exit(1); end; exit(0);"
    echo "--- [$LABEL] ---"
    "$MATLAB_DIR/bin/matlab" -batch "$SCRIPT"
}

# Helper: build a comma-joined TestSuite array expression from one or
# more class-name globs.  Each glob becomes a fromFolder(..., 'Name',
# '<glob>/*') element.
join_suites() {
    local out=""
    for glob in "$@"; do
        local elem="matlab.unittest.TestSuite.fromFolder('$SCRIPT_DIR/tests', 'IncludingSubfolders', true, 'Name', '$glob/*')"
        if [ -z "$out" ]; then
            out="$elem"
        else
            out="$out, $elem"
        fi
    done
    echo "[$out]"
}

# Named groups.  When you add a new test class at a different
# model_size, update the relevant group definition below.
SUITE_FAST=$(join_suites \
    "tMmacosCmd" "tMacosPkg" "tMacosSession" \
    "tCrossSurface" "tPerturbRoundtrip" "tCodeVGrating")
SUITE_MASKS=$(join_suites "tCodeV*Masks*")
SUITE_PROPER=$(join_suites "tProperCompare*")

# Argument handling.
case "${1:-}" in
    "")
        # Full suite: split by model_size group to dodge the
        # init-reinit heap-corruption bug (PLAN.md §0).
        run_batch "[$SUITE_FAST, $SUITE_MASKS]" "model_size=128"
        run_batch "$SUITE_PROPER" "model_size=512"
        echo "=== all groups passed ==="
        ;;
    fast)
        # All size=128 EXCEPT masks — dev iteration loop, ~10 s.
        run_batch "$SUITE_FAST" "fast (size=128, no masks)"
        ;;
    masks)
        # The heavyweight CodeV mask suite, size=128, ~10 min.
        run_batch "$SUITE_MASKS" "masks (size=128)"
        ;;
    proper)
        # Phase 5 PROPER-comparison suite, size=512, ~15 s.
        run_batch "$SUITE_PROPER" "proper (size=512)"
        ;;
    -k)
        # Method-name substring filter.
        FILTER_ARG=", 'ProcedureName', '*$2*'"
        SUITE_EXPR="matlab.unittest.TestSuite.fromFolder('$SCRIPT_DIR/tests', 'IncludingSubfolders', true $FILTER_ARG)"
        run_batch "$SUITE_EXPR" "filter: $2"
        ;;
    *)
        # Single class by name (glob matches all its methods).
        SUITE_EXPR=$(join_suites "$1")
        run_batch "$SUITE_EXPR" "class: $1"
        ;;
esac
