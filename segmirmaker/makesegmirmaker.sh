#!/bin/bash
# Build SegMirMaker. Requires a pre-built MACOS tree; the matching tree is
# selected automatically from the compiler + build type (override with
# MACOS_BUILD_DIR=... in the environment):
#   ifx release -> build_release   ; ifx debug -> build_debug
#   gfortran    -> build_release_gfortran / build_debug_gfortran
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

# On macOS ifx has no arm64 build -- force gfortran (bare symlink from the
# Homebrew gcc), matching the rest of the tree's Mac port.
if [ "$(uname -s)" = "Darwin" ] && [ "$fc" = "ifx" ]; then
  echo "makesegmirmaker: macOS has no ifx; building with gfortran instead."
  fc=gfortran
  build_tag="${build_tag/ifx/gfortran}"
fi

if [ "$fc" = "ifx" ] && [ -f /opt/intel/oneapi/setvars.sh ]; then
  source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
fi

# Select the pre-built MACOS tree that matches this compiler + build type.
if [ -z "${MACOS_BUILD_DIR:-}" ]; then
  bt_lc="$(printf '%s' "$build_type" | tr '[:upper:]' '[:lower:]')"
  if [ "$fc" = "gfortran" ]; then
    macos_tree="build_${bt_lc}_gfortran"
  else
    macos_tree="build_${bt_lc}"
  fi
  MACOS_BUILD_DIR="${HOME}/dev/macos/${macos_tree}"
fi

here="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
build_dir="${here}/build_${build_tag}"
mkdir -p "${build_dir}"

cmake -S "${here}" -B "${build_dir}" \
  -DCMAKE_BUILD_TYPE="${build_type}" \
  -DCMAKE_Fortran_COMPILER="${fc}" \
  -DMACOS_BUILD_DIR="${MACOS_BUILD_DIR}"

cmake --build "${build_dir}" -j

echo
echo "Built: ${build_dir}/SegMirMaker"
