#!/usr/bin/env bash
# run_mmacos_tests.sh -- run the mmacos matlab.unittest suite in batch
# mode and exit non-zero on any failure.
#
# Mirror of MACOS_resources/pymacos/run_proper_tests.sh.  See PLAN.md
# §5.4 Phase 3 for the design.
#
# Usage:
#   ./run_mmacos_tests.sh                   # full suite (split by size)
#   ./run_mmacos_tests.sh quick             # truly-fast smoke subset
#                                           # (~30 s — dev loop)
#   ./run_mmacos_tests.sh fast              # all size=128 EXCEPT masks
#                                           # (minutes — broad, not quick)
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

# License-seat preflight (OPT-IN; default OFF).  In practice a MathWorks
# Home/Individual license runs MULTIPLE concurrent MATLAB sessions under
# the same user, so a `matlab -batch` test run coexists fine with an open
# desktop -- no check is needed by default.  Set MM_SEAT_CHECK=1 to
# re-enable the guard (a strict single-seat setup where a batch run would
# block on a busy seat, surfacing as a silent multi-minute "hang").
if [ "${MM_SEAT_CHECK:-0}" = "1" ]; then
    seat_holder=""
    for p in $(pgrep -f "glnxa64/MATLAB" 2>/dev/null); do
        a=$(ps -o args= -p "$p" 2>/dev/null)
        case "$a" in
            *MathWorksServiceHost*|*ServiceHost*) ;;   # daemon — ignore
            *" -batch "*) ;;                           # a batch run — ignore
            *glnxa64/MATLAB*) seat_holder="$p $seat_holder" ;;
        esac
    done
    if [ -n "$seat_holder" ]; then
        echo "error: an interactive MATLAB session is running (pid ${seat_holder%% })." >&2
        echo "       MM_SEAT_CHECK is on and this looks like a single-seat block." >&2
        echo "       Close the MATLAB desktop and retry, or unset MM_SEAT_CHECK." >&2
        exit 3
    fi
fi

# Rebuild src/mmacos.mexa64 if any source is newer.
if [ ! -f src/mmacos.mexa64 ] \
   || [ src/mmacos_mex.F -nt src/mmacos.mexa64 ] \
   || [ src/mmacos_gen.F -nt src/mmacos.mexa64 ]; then
    echo "(re)building src/mmacos.mexa64..."
    make FC=gfortran MACOS_BUILD_DIR="$MACOS_BUILD_DIR" >/dev/null
fi

# +macos package + the mex live under src/ (see Makefile SRCDIR).
SETUP_PATHS="addpath('$SCRIPT_DIR/src'); addpath('$SCRIPT_DIR/design/runners'); addpath('$SCRIPT_DIR/tests/proper_compare');"
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
    "tCrossSurface" "tPerturbRoundtrip" "tCodeVGrating" \
    "tBandLimitedMask" "tSrsBugFlatZ" "tDwDzZernike" "tDwDx" \
    "tDwDxGroups" "tDesignSystem" "tDesignVary" "tDesignSensitivities" \
    "tDesignOptimize" "tDesignTelescope" "tVeneerXP" "tCoroContrast" \
    "tCompose" "tSysProp" "tSegMirMaker" "tSegmentRx" "tEdgeSensors" \
    "tMet" "tMetView" "tRunMet")
# Truly-fast smoke subset for the dev loop: lightweight, high-signal
# classes only (command dispatch, package/session veneers, pure-math
# mask, perturb roundtrip, first-order props, compose, XP).  EXCLUDES the
# heavy dw/dx sensitivity, design-layer optimisation, and coronagraph-
# contrast classes that make `fast` take minutes.  All size=128, one
# matlab -batch.
SUITE_QUICK=$(join_suites \
    "tMmacosCmd" "tMacosPkg" "tMacosSession" "tBandLimitedMask" \
    "tPerturbRoundtrip" "tSysProp" "tCompose" "tVeneerXP")
# tFreeFormComposite + tCalib + tReadGridFile + tViewRx run at ModelSize=256
SUITE_FREEFORM=$(join_suites "tFreeFormComposite" "tCalib" "tReadGridFile" "tViewRx")
# Note: tBandLimitedMask is pure math (no macos calls), safe in any
# group; lives in "fast" because it's quick.
SUITE_MASKS=$(join_suites "tCodeV*Masks*")
SUITE_PROPER_512=$(join_suites "tProperCompareCassFF" "tProperCompareCassFFAberrations")
SUITE_PROPER_1024=$(join_suites "tProperCompareCoroNFprop" "tProperCompareCoroPhase3" "tProperCompareCoroApodizer" "tProperCompareCoroDMPhase")
# Aggregate for the `proper` shortcut (runs in two batches — the
# initial Cass-FF group at 512 then the Coro group at 1024 — to
# dodge the engine init-reinit heap bug PLAN.md §0 logs).
SUITE_PROPER="$SUITE_PROPER_512"  # back-compat; not directly used

# Argument handling.
case "${1:-}" in
    "")
        # Full suite in ONE matlab -batch.  The historical per-
        # model_size split was a workaround for the init-reinit
        # heap-corruption bug (PLAN.md §0); that bug was fixed in
        # macos commit e2e8bf6 (macos_realloc=.true. on resize +
        # DFOURN grow-on-demand), so all four model_size groups now
        # run cleanly back-to-back in a single matlab process.
        run_batch "[$SUITE_FAST, $SUITE_MASKS, $SUITE_FREEFORM, $SUITE_PROPER_512, $SUITE_PROPER_1024]" \
                  "full suite (all model sizes)"
        ;;
    quick)
        # Truly-fast smoke subset — dev iteration loop, ~30 s.
        run_batch "$SUITE_QUICK" "quick (smoke, size=128)"
        ;;
    fast)
        # All size=128 EXCEPT masks — broad, but minutes (heavy dw/dx +
        # design + coro-contrast classes).  Use `quick` for the dev loop.
        run_batch "$SUITE_FAST" "fast (size=128, no masks)"
        ;;
    masks)
        # The heavyweight CodeV mask suite, size=128, ~10 min.
        run_batch "$SUITE_MASKS" "masks (size=128)"
        ;;
    proper)
        # Phase 5 PROPER-comparison suite — now one batch.
        run_batch "[$SUITE_PROPER_512, $SUITE_PROPER_1024]" "proper (Cass-FF + Coro NF)"
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
