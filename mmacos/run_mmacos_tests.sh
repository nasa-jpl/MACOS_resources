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
#   ./run_mmacos_tests.sh freeform          # the size=256 design-metric
#                                           # group in one process
#                                           # (~10 min — heavyweight)
#   ./run_mmacos_tests.sh proper            # only Phase 5 PROPER cmp
#   ./run_mmacos_tests.sh polfloor          # only Phase 2c contrast floor
#   ./run_mmacos_tests.sh bench1024         # the model-1024 bench gates
#                                           # (tCophiBench, ~22 min, ~12 GB;
#                                           # NOT in the full run)
#   ./run_mmacos_tests.sh tMacosPkg         # one class by name
#   ./run_mmacos_tests.sh -k roundtrip      # methods matching a substring
#   ./run_mmacos_tests.sh -j 8 fast         # the same, sharded over 8 MATLAB
#                                           # sessions (any mode; MM_JOBS=8 too)
#
# Sharding (-j N, 2026-10-27): the suite is enumerated once, partitioned by
# tests/shard_partition.py (one class per unit, never two model sizes in one
# process, longest-first onto the least-loaded shard using tests/timings/*.csv),
# and N `matlab -batch` sessions run the shards at once; each writes
# results_<k>.csv + shard_<k>.log under $MM_SHARD_DIR (default a fresh
# /tmp/mm_shards.<pid>), and the summary names every failure with its log.
# Memory: a shard peaks at ~4 GB (model-512 classes; measured 2026-10-27:
# 3.5-4.1 GB for tRunSensitivities / tRunCompare / tSegmentRx+tRunMet, 1.6-2.6
# GB for the model-128 shards).  Eight shards need ~22 GB and were killed by
# systemd-oomd on the 30 GB box with the editor open, so the launch clamps the
# job count to (MemAvailable - MM_MEM_MARGIN_GB) / MM_SHARD_GB and says so.
# Measured 2026-10-27 (serial fast wall 2645 s): -j 8 unclamped 991 s (2.7x, memory
# at the edge); -j 4 clamped to 3 at once with the thread cap 1541 s (1.7x) --
# the tests themselves ran FASTER than serial once each shard was capped to
# its share of the cores (1976 s of test time vs 2338 s).
# Every run, sharded or not, snapshots `git status` before and after and FAILS
# if a test modified a tracked file (the tRodgers3 lesson, PLAN_CONSOLIDATION
# 1.2 (ttt)); new untracked files are listed as a warning.
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

# -j N (or MM_JOBS=N): shard the batch over N MATLAB sessions.  1 = serial.
JOBS="${MM_JOBS:-1}"
if [ "${1:-}" = "-j" ]; then
    JOBS="$2"; shift 2
fi
case "$JOBS" in ''|*[!0-9]*|0) echo "error: -j needs a positive integer" >&2; exit 2;; esac
# Classes whose tests may be scheduled individually across shards (cheap
# TestClassSetup: init + a fixture path).  A class with a heavy setup stays
# whole so its fixtures are built once.
MM_SHARD_SPLIT="${MM_SHARD_SPLIT:-tDwDx tDwDzZernike}"
MM_SHARD_GB="${MM_SHARD_GB:-4}"            # peak RSS per shard, GB (measured)
MM_MEM_MARGIN_GB="${MM_MEM_MARGIN_GB:-3}"  # headroom kept for the rest of the box

MACOS_BUILD_DIR="${MACOS_BUILD_DIR:-$HOME/dev/macos/build_release_gfortran}"
# MATLAB location differs by OS: /Applications/MATLAB_R20XXx.app on macOS,
# /usr/local/MATLAB/R20XXx on Linux.
if [ -z "$MATLAB_DIR" ]; then
    if [ "$(uname -s)" = "Darwin" ]; then
        MATLAB_DIR="$(ls -1d /Applications/MATLAB_R[0-9]*[ab].app 2>/dev/null | sort -V | tail -1)"
    else
        MATLAB_DIR="$(ls -1d /usr/local/MATLAB/R[0-9]*[ab] 2>/dev/null | sort -V | tail -1)"
    fi
fi
PROPER_DIR="${PROPER_DIR:-$HOME/dev/proper_matlab}"

# The engine loads macos_param.txt from (in order) cwd, exe dir, $MACOS_HOME,
# then a compile-time build dir.  matlab -batch runs from the mmacos root, which
# has no macos_param.txt, and a MISSING param file makes the engine's init abort
# fatally — inside the MEX host that abort is a SIGSEGV, not a clean error.  Point
# MACOS_HOME at the engine's canonical param file so init always resolves it.
# (Tests that need a bespoke param file — e.g. tSegmentRx — copy their own into a
# scratch cwd, which still wins over MACOS_HOME.)
export MACOS_HOME="${MACOS_HOME:-$HOME/dev/macos/macos_f90}"

if [ -z "$MATLAB_DIR" ] || [ ! -d "$MATLAB_DIR" ]; then
    echo "error: MATLAB not found.  Set MATLAB_DIR, install under /usr/local/MATLAB/ (Linux), or /Applications/MATLAB_R*.app (macOS)." >&2
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

# Rebuild the mex if any source is newer.  MEX extension is platform-specific
# (mexa64 Linux, mexmaca64/mexmaci64 macOS) — ask the Makefile so the -nt check
# targets the file that actually gets built.
MEX_FILE="src/mmacos.$(make -s print-mextag 2>/dev/null || echo mexa64)"
if [ ! -f "$MEX_FILE" ] \
   || [ src/mmacos_mex.F -nt "$MEX_FILE" ] \
   || [ src/mmacos_gen.F -nt "$MEX_FILE" ]; then
    echo "(re)building $MEX_FILE..."
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
# Clean-tree guard: a test must not modify a tracked file (it may only write
# under a temp dir).  Snapshot before, compare after; modified tracked files
# fail the batch, new untracked files are listed.
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
tree_snapshot() { git -C "$REPO_DIR" status --porcelain --untracked-files=all 2>/dev/null | sort; }
tree_check() {
    local before="$1" label="$2" after
    after="$(tree_snapshot)"
    local newmod newunt
    newmod="$(comm -13 <(echo "$before") <(echo "$after") | grep -v '^??' || true)"
    newunt="$(comm -13 <(echo "$before") <(echo "$after") | grep '^??' || true)"
    if [ -n "$newunt" ]; then
        echo "warning: [$label] left new untracked files:"; echo "$newunt" | sed 's/^/    /'
    fi
    if [ -n "$newmod" ]; then
        echo "error: [$label] MODIFIED tracked files (a test wrote into the library tree):"
        echo "$newmod" | sed 's/^/    /'
        return 1
    fi
    return 0
}

run_batch() {
    local SUITE_EXPR="$1"
    local LABEL="$2"
    local BEFORE rc=0
    BEFORE="$(tree_snapshot)"
    if [ "$JOBS" -gt 1 ]; then
        run_sharded "$SUITE_EXPR" "$LABEL" || rc=1
    else
        run_serial "$SUITE_EXPR" "$LABEL" || rc=1
    fi
    tree_check "$BEFORE" "$LABEL" || rc=1
    return $rc
}

# N shards: enumerate the suite in one short MATLAB session, partition it,
# launch one `matlab -batch` per shard in the background, wait, summarize.
run_sharded() {
    local SUITE_EXPR="$1"
    local LABEL="$2"
    local OUT="${MM_SHARD_DIR:-/tmp/mm_shards.$$}"
    mkdir -p "$OUT"
    echo "--- [$LABEL] sharded -j $JOBS -> $OUT ---"
    local ENUM="$SETUP_PATHS suite = $SUITE_EXPR; fid = fopen('$OUT/names.txt', 'w'); for k = 1:numel(suite), fprintf(fid, '%s\n', suite(k).Name); end; fclose(fid); exit(0);"
    "$MATLAB_DIR/bin/matlab" -batch "$ENUM" > "$OUT/enumerate.log" 2>&1 || { echo "error: suite enumeration failed ($OUT/enumerate.log)"; return 1; }
    rm -f "$OUT"/shard_*.txt "$OUT"/results_*.csv
    python3 "$SCRIPT_DIR/tests/shard_partition.py" partition --names "$OUT/names.txt" \
        --tests-dir "$SCRIPT_DIR/tests" --timings "$SCRIPT_DIR/tests/timings" \
        --jobs "$JOBS" --out "$OUT" --split "$MM_SHARD_SPLIT" || return 1
    # clamp the parallelism to what memory allows RIGHT NOW (another lane's
    # MATLAB, the editor and the browser all count); shards beyond the clamp
    # run after the first ones finish, in launch order
    local avail_gb fit
    avail_gb=$(awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo 2>/dev/null || echo 0)
    fit=$(( (avail_gb - MM_MEM_MARGIN_GB) / MM_SHARD_GB ))
    [ "$fit" -lt 1 ] && fit=1
    local nshard; nshard=$(ls "$OUT"/shard_*.txt | wc -l)
    local par="$JOBS"; [ "$fit" -lt "$par" ] && par="$fit"
    echo "memory: ${avail_gb} GB available, ${MM_SHARD_GB} GB per shard + ${MM_MEM_MARGIN_GB} GB margin -> $par of $nshard shards at once"
    # each shard gets its share of the cores for MATLAB's own threaded math
    # (the engine mex is single-threaded); N shards at 16 threads each
    # oversubscribe the box and every shard slows
    local ncpu threads
    ncpu=$(nproc 2>/dev/null || echo 4)
    threads=$(( ncpu / par )); [ "$threads" -lt 1 ] && threads=1
    echo "threads: $ncpu cores -> maxNumCompThreads($threads) per shard"
    local pids=() k f t0
    t0=$(date +%s)
    for f in "$OUT"/shard_*.txt; do
        while [ "$(jobs -rp | wc -l)" -ge "$par" ]; do sleep 5; done
        k="${f##*/shard_}"; k="${k%.txt}"
        local SCRIPT="maxNumCompThreads($threads); $SETUP_PATHS suite = $SUITE_EXPR; names = strtrim(string(readlines('$f'))); suite = suite(ismember(string({suite.Name}), names)); if isempty(suite), exit(0); end; runner = matlab.unittest.TestRunner.withTextOutput; results = runner.run(suite); T = table(results); writetable(T(:, {'Name', 'Passed', 'Failed', 'Incomplete', 'Duration'}), '$OUT/results_$k.csv'); n_failed = sum([results.Failed]); fprintf('=== shard $k: %d pass, %d fail ===\n', sum([results.Passed]), n_failed); if n_failed > 0, exit(1); end; exit(0);"
        "$MATLAB_DIR/bin/matlab" -batch "$SCRIPT" > "$OUT/shard_$k.log" 2>&1 &
        pids+=($!)
    done
    local rc=0 p
    for p in "${pids[@]}"; do wait "$p" || rc=1; done
    echo "shards finished in $(( $(date +%s) - t0 )) s wall"
    python3 "$SCRIPT_DIR/tests/shard_partition.py" summarize --out "$OUT" --label "$LABEL" --t0 "$t0" || rc=1
    return $rc
}

run_serial() {
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
    "tDesignOptimize" "tDesignTelescope" "tCheckClippingSource" "tVeneerXP" "tCoroContrast" \
    "tCompose" "tSysProp" "tBeam" "tSegMirMaker" "tSegmentRx" "tEdgeSensors" \
    "tMet" "tMetView" "tRunMet" "tRunSensitivities" "tRunSegmentation" \
    "tRunCompare" "tSpot" "tDrawFrameGuard" "tDwDgridElts" \
    "tApertureFrame" "tAppendRx" "tPropLayout" \
    "tPolarization" "tJonesPupil" "tVecChain" \
    "tPolElement" "tPolRadiometric" "tPolExternal" "tBench" "tOpdRef" \
    "tTgPol" "tTgPol2" "tTgPsiAxes" "tStopReload" "tDmgLoop" "tTmaLongslit" "tSpectrometerSens" "tPupilBlurDemo" "tRxShortCoef" "tRxRefuse" "tRxLineEndings" "tTraceKeepsRx" "tSaveFixedPoint" "tZrnGrRoundTrip" "tGridDropped" "tCalibTargets" \
    "tLinkSave" "tZernikeGridBasis" "tRxBlockComment" \
    "tGratingImmersed" "tGratingAperture" "tRayFailElt" "tGlassDispersion" "tPropMedium" "tRxShortAsph" \
    "tSpectrometerRx" "tGratingOpl" "tAsphCalib" "tTelescopeRx" "tRetraceIdempotent" "tBeamRows" "tTraceRestart" "tFFCut" "tAsphHook" "tBeamDirHook" "tFwdRoot" "tCalibBlindFields" "tCalibStop" "tCalibMatrix" "tSessionModes" "tStopApStop" "tFreeformPole" "tCophi" "tBudgetKernel" "tRunBudget" "tResolutionSweep" "tApiCoverage" "tDefaults" "tSrt" "tSegRay" "tRefRays" "tSint" "tMessages" "tSaveNewFileQuiet" "tRxLoadRulings1018" "tVeneerDefaults" "tZwfsCoat" "tE2eRmsWaves" \
    "tNollMode" "tConvergedEst" "tTmaTelecentric" "tJacobianCheck" "tFingerprint" "tRxHeaderRewrite" "tIodc2010")
# Truly-fast smoke subset for the dev loop: lightweight, high-signal
# classes only (command dispatch, package/session veneers, pure-math
# mask, perturb roundtrip, first-order props, compose, XP).  EXCLUDES the
# heavy dw/dx sensitivity, design-layer optimisation, and coronagraph-
# contrast classes that make `fast` take minutes.  All size=128, one
# matlab -batch.
SUITE_QUICK=$(join_suites \
    "tMmacosCmd" "tMacosPkg" "tMacosSession" "tBandLimitedMask" \
    "tPerturbRoundtrip" "tSysProp" "tCompose" "tVeneerXP" "tSpot")
# tFreeFormComposite + tCalib + tReadGridFile + tViewRx + tStrictKernel
# + tAfocalKernel + tPupilMap + tDesignAfocal + tAfocal4 + tAfocal4Clear
# + tAfocal4Wall + tAfocal4Descent
# + tE2E2Axial + tPolContrast + tRodgers3 + tOffsetImager
# + tDwDsurf + tEltTypeCoverage + tNsFlowOfLight at 256
SUITE_FREEFORM=$(join_suites "tFreeFormComposite" "tCalib" "tReadGridFile" "tViewRx" "tSurfInspect" "tOptFex" "tStrictKernel" "tAfocalKernel" "tPupilMap" "tDesignAfocal" "tAfocal4" "tAfocal4Clear" "tAfocal4Wall" "tAfocal4Descent" "tAfocal4Offaxis" "tE2E2Axial" "tPolContrast" "tRodgers3" "tOffsetImager" "tOiSeedBranch" "tDwDsurf" "tEltTypeCoverage" "tNsFlowOfLight" "tFocalSurface" "tOiDemoStep" "tE2eCoroFold")
# Note: tBandLimitedMask is pure math (no macos calls), safe in any
# group; lives in "fast" because it's quick.
# EP-dome ruling gate (Dave 2026-09-08): sensitivity OPD read at the exit
# pupil, never the tilt-blind FocalPlane.  Model 256 (the e2e6m s3 imager
# fixtures) -- own batch line so it neither drags a 256 deck into the 128
# groups nor rides a size transition.
SUITE_EPDOME=$(join_suites "tEpDomeGate")
SUITE_MASKS=$(join_suites "tCodeV*Masks*")
SUITE_PROPER_512=$(join_suites "tProperCompareCassFF" "tProperCompareCassFFAberrations")
# tPupilAperture (the macos PR #70 ColSource gates) runs every probe at
# model_size 512 — Rx_Mask_Parabolas has nGridpts=512.  Own batch line so
# it neither drags a 512 deck into the 128/256 groups nor rides along with
# the PROPER classes it has nothing to do with.
SUITE_PUPIL_512=$(join_suites "tPupilAperture")
# Phase 2c coronagraph floor -- Rx_Coro declares nGridpts=511 and MUST run
# at model >= 512 (PLAN_POLARIZATION §2c fixture correction).  Its own
# batch so the model-size transition stays contained.
SUITE_POL_512=$(join_suites "tPolContrastCoro")
# CTB coronagraph diffraction chain -- ctb_dcr.in / ctb_s2s_dcr.in declare
# nGridpts=255 and their drivers run at model 512.  Own batch, same reason
# as SUITE_POL_512.  Asset-gated: skips itself when bench_ctb is absent.
SUITE_CTB_512=$(join_suites "tCtbProp" "tCtbDm" "tCfCampaign" "tCtbProbe" "tE2eCoupled" "tE2eSensing" "tE2eCycles")

# reset_xp_method='pupil_find' gates run the zoom fixture at model 512 --
# own batch line, same isolation reason as the other 512 groups.
SUITE_PFM_512=$(join_suites "tPupilFindMethod")
# tEngineMemory deliberately walks model 128 <-> 256 to prove macos.unload
# rebuilds the engine smaller.  Own batch so a size transition it induces
# cannot seed PLAN.md §0 heap corruption into another group.  tEndurance
# (500 trace iterations at 128, ~30 s) rides with it.
SUITE_MEMORY=$(join_suites "tEngineMemory" "tEndurance")
# Model-1024 bench gates: tCophiBench builds both TG96 rigs at 1024 plus a
# 16-min noise leg -- ~22 min and a ~12 GB peak, so it runs ONLY when named
# (`bench1024`), never in the full run or a -j shard set.
SUITE_1024=$(join_suites "tCophiBench")
SUITE_PROPER_1024=$(join_suites "tProperCompareCoroNFprop" "tProperCompareCoroPhase3" "tProperCompareCoroApodizer" "tProperCompareCoroDMPhase")
# Aggregate for the `proper` shortcut (runs in two batches — the
# initial Cass-FF group at 512 then the Coro group at 1024 — to
# dodge the engine init-reinit heap bug PLAN.md §0 logs).
SUITE_PROPER="$SUITE_PROPER_512"  # back-compat; not directly used

# Argument handling.
case "${1:-}" in
    "")
        # Full suite, split into ONE matlab -batch PER model-size group
        # (each group = a fresh MATLAB process).  This is the workaround
        # for the init-reinit heap-corruption bug (PLAN.md §0): that bug
        # was thought fixed in macos e2e8bf6, and the split was briefly
        # collapsed into a single process, but §0 REOPENED (2026-07-16/18)
        # and reconfirmed on the Mac against CCL's partial fix 54270af
        # (2026-07-23) — a single-process run still SIGSEGVs at tViewRx
        # setup from corruption seeded by an earlier size transition.
        # Separate processes contain the corruption to its own group so
        # the rest of the suite still reports.  DO NOT recollapse until
        # PLAN.md §0 REOPENED is closed.
        # Groups are run independently; a failure in one does not abort
        # the others (so a green/red per group is always reported).
        rc=0
        run_batch "$SUITE_FAST"        "full: fast (size 128)"      || rc=1
        run_batch "$SUITE_MASKS"       "full: masks (size 128)"     || rc=1
        run_batch "$SUITE_FREEFORM"    "full: freeform (size 256)"  || rc=1
        run_batch "$SUITE_EPDOME"      "full: ep-dome gate (256)"   || rc=1
        run_batch "$SUITE_PROPER_512"  "full: proper Cass-FF (512)" || rc=1
        run_batch "$SUITE_PUPIL_512"   "full: pupil aperture (512)" || rc=1
        run_batch "$SUITE_POL_512"     "full: pol contrast (512)"   || rc=1
        run_batch "$SUITE_CTB_512"     "full: ctb diffraction (512)" || rc=1
        run_batch "$SUITE_PFM_512"     "full: pupil_find method (512)" || rc=1
        run_batch "$SUITE_PROPER_1024" "full: proper Coro (1024)"   || rc=1
        run_batch "$SUITE_MEMORY"      "full: engine memory (128<->256)" || rc=1
        exit $rc
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
    freeform)
        # The size=256 group in ONE process — the design-metric classes
        # (strict / afocal / pupil kernels, E2E2, CALIB, OptFEX) plus the
        # freeform + viewer ones.  This is the group a design-layer change
        # has to leave green.
        run_batch "$SUITE_FREEFORM" "freeform (size=256)"
        ;;
    proper)
        # Phase 5 PROPER-comparison suite — now one batch.
        run_batch "[$SUITE_PROPER_512, $SUITE_PROPER_1024]" "proper (Cass-FF + Coro NF)"
        ;;
    bench1024)
        # The model-1024 bench gates, own process (~22 min, ~12 GB).
        run_batch "$SUITE_1024" "bench gates (size 1024)"
        ;;
    ctb)
        # CTB coronagraph diffraction chain, size 512.
        run_batch "$SUITE_CTB_512" "ctb diffraction (size 512)"
        ;;
    polfloor)
        # Phase 2c contrast floor: exactness gates at 256, coronagraph
        # chain at 512.  Two batches — different model sizes.
        rc=0
        run_batch "$(join_suites "tPolContrast")" "polfloor (size 256)"      || rc=1
        run_batch "$SUITE_POL_512"                "polfloor coro (size 512)" || rc=1
        exit $rc
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
