#!/usr/bin/env bash
#
# Run the macos-vs-PROPER physical-optics comparison suite.
#
# Usage:
#   ./run_proper_tests.sh              # run tests against the current
#                                      # pymacosf90 .so
#   ./run_proper_tests.sh --build      # rebuild pymacosf90 first
#                                      # (do this after rebuilding the
#                                      #  parent macos libsmacos.a)
#   ./run_proper_tests.sh -v           # pytest -v (one line per test)
#
# Requires:
#   - Intel oneAPI installed (or already activated in the shell)
#   - .venv at ./.venv (run `uv venv --python 3.13 .venv` once if missing)
#   - PyPROPER3 installed in the venv (see tests/proper_compare/README.md)
#

set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$here"

# Parse flags
do_build=0
pytest_args=()
for arg in "$@"; do
    case "$arg" in
        --build)  do_build=1 ;;
        *)        pytest_args+=("$arg") ;;
    esac
done

# 1. Intel oneAPI
if ! command -v ifx >/dev/null 2>&1; then
    if [[ -f /opt/intel/oneapi/setvars.sh ]]; then
        echo "Sourcing Intel oneAPI from /opt/intel/oneapi/setvars.sh..."
        # shellcheck disable=SC1091
        source /opt/intel/oneapi/setvars.sh intel64 >/dev/null
    else
        echo "ERROR: ifx not in PATH and /opt/intel/oneapi/setvars.sh missing."
        echo "  Source your Intel oneAPI setvars.sh before running this script."
        exit 1
    fi
fi

# 2. venv
if [[ ! -d .venv ]]; then
    echo "ERROR: .venv not found at $here/.venv"
    echo "  Run: uv venv --python 3.13 .venv && source .venv/bin/activate &&"
    echo "       uv pip install numpy scipy matplotlib pytest cmake meson ninja"
    echo "  Then install PyPROPER3 per tests/proper_compare/README.md."
    exit 1
fi
# shellcheck disable=SC1091
source .venv/bin/activate

# 3. (optional) rebuild pymacosf90 against the latest libsmacos.a
if [[ $do_build -eq 1 ]]; then
    build_dir="$here/src/cmake/build"
    if [[ ! -d "$build_dir" ]]; then
        echo "ERROR: $build_dir doesn't exist."
        echo "  Run the initial cmake configure first:"
        echo "    cd src/cmake && mkdir build && cd build &&"
        echo "      cmake -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \\"
        echo "            -DCMAKE_Fortran_COMPILER=ifx -S .."
        exit 1
    fi
    echo "Rebuilding pymacosf90..."
    (cd "$build_dir" && make)
fi

# 4. Run each phase in its own pytest process.
#
# Why two invocations: pymacos's init() reallocates the Fortran
# arrays when model_size changes, but some module-level state (most
# visibly the diffraction-grid normalisation) leaks across a 512 <->
# 1024 transition in the same Python process.  Running each phase as
# a fresh process sidesteps this entirely.  Long-term fix is on the
# pymacos side; until then this is the cheap, correct option.
cd tests

set +e   # collect status of each phase rather than aborting at the first
echo
echo "=== Phase 1 (Cass FF) ==="
pytest proper_compare/test_cass_ff.py proper_compare/test_cass_ff_aberrations.py \
       "${pytest_args[@]}"
s1=$?

echo
echo "=== Phase 2 (Coro NF-prop) ==="
pytest proper_compare/test_coro_nfprop.py "${pytest_args[@]}"
s2=$?

# (test_psf.py is a leftover skip; include for completeness but its
# status doesn't gate the overall run.)
echo
echo "=== Auxiliary (test_psf -- mostly skipped) ==="
pytest proper_compare/test_psf.py "${pytest_args[@]}" || true
set -e

echo
echo "Artefacts:"
echo "  $here/tests/proper_compare/results_phase1/   (Cass FF: PNG, .mat, report.md)"
echo "  $here/tests/proper_compare/results_phase2/   (Coro NF-prop: same)"

# Overall status: fail if either phase failed.
if [[ $s1 -ne 0 || $s2 -ne 0 ]]; then
    echo
    echo "FAILURE: phase1=$s1, phase2=$s2"
    exit 1
fi
echo
echo "Both phases passed."
exit 0
