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

# 4. Run the suite
echo
echo "Running proper_compare suite..."
cd tests
pytest proper_compare/ "${pytest_args[@]}"
status=$?

echo
echo "Artefacts: $here/tests/proper_compare/results/"
echo "  - report.md         (cumulative quantitative table)"
echo "  - <test>.png        (3-panel macos / PROPER / diff plot)"
echo "  - <test>.mat        (full arrays + metadata, Matlab-readable)"
echo "  - <test>.macos.txt  (ASCII central crop, sum-normalised)"
echo "  - <test>.proper.txt (ASCII central crop, sum-normalised)"

exit $status
