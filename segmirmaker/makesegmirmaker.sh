#!/bin/bash
# Build SegMirMaker. Requires a pre-built MACOS tree
# (default: $HOME/dev/macos/build_release_giza).
#
# Usage:
#   source ./makesegmirmaker.sh             # Release ifx
#   source ./makesegmirmaker.sh debug       # Debug ifx
#   source ./makesegmirmaker.sh gfortran    # Release gfortran
#   source ./makesegmirmaker.sh debug gfortran

set -e

build_type=Release
fc=ifx
build_tag=release_ifx

for arg in "$@"; do
  case "$arg" in
    debug)    build_type=Debug;   build_tag="${build_tag/release/debug}" ;;
    gfortran) fc=gfortran;        build_tag="${build_tag/ifx/gfortran}" ;;
    ifx)      fc=ifx              ;;
    release)  build_type=Release  ;;
  esac
done

if [ "$fc" = "ifx" ] && [ -f /opt/intel/oneapi/setvars.sh ]; then
  source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
fi

here="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
build_dir="${here}/build_${build_tag}"
mkdir -p "${build_dir}"

cmake -S "${here}" -B "${build_dir}" \
  -DCMAKE_BUILD_TYPE="${build_type}" \
  -DCMAKE_Fortran_COMPILER="${fc}"

cmake --build "${build_dir}" -j

echo
echo "Built: ${build_dir}/SegMirMaker"
