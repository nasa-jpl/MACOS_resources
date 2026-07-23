# ==============================================================================
# mmacos/CMakeLists.txt — Cross-platform CMake build for mmacos MATLAB MEX
#
# mmacos is the MATLAB MEX sibling of pymacos.  It wraps the macos_api_mod
# interface (compiled into libsmacos / smacos.lib) and produces:
#   Linux   : mmacos.mexa64
#   Windows : mmacos.mexw64
#
# Called from the main CMakeLists.txt via add_subdirectory() when BUILD_MMACOS=ON,
# or invoked standalone (pointing at a pre-built macos tree via MACOS_BUILD_DIR).
#
# Key options:
#   MACOS_BUILD_DIR   Pre-built macos CMake tree (standalone mode only)
#   MATLAB_DIR        Override auto-detected MATLAB root
#   BUILD_MMACOS      ON/OFF gate (default ON)
#   RX_INPUT          .in file used by the smoke CTest
#
# Typical standalone usage (Linux):
#   cmake -B build -DMACOS_BUILD_DIR=~/dev/macos/build_release_giza
#   cmake --build build
#   ctest --test-dir build -R smoke
#
# Typical standalone usage (Windows, oneAPI):
#   call setvars.bat --force
#   cmake -B build -G Ninja -DMACOS_BUILD_DIR=C:\path\to\macos\build
#   cmake --build build
# ==============================================================================

# ---------------------------------------------------------------------------
# Standalone vs add_subdirectory guard
# ---------------------------------------------------------------------------
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  if(WIN32)
    set(CMAKE_Fortran_PREPROCESS_SOURCE
      "<CMAKE_Fortran_COMPILER> <DEFINES> <INCLUDES> <FLAGS> /fpp /E <SOURCE> > <PREPROCESSED_SOURCE>")
  endif()
  cmake_minimum_required(VERSION 3.21)
  project(mmacos Fortran C)
endif()

# ---------------------------------------------------------------------------
# Build gate
# ---------------------------------------------------------------------------
option(BUILD_MMACOS "Build the mmacos MATLAB MEX interface" ON)
if(NOT BUILD_MMACOS)
  message(STATUS "mmacos: BUILD_MMACOS=OFF -- skipping")
  return()
endif()

# ==============================================================================
# 1. Validate prerequisites
# ==============================================================================
if(NOT TARGET smacos)
  # Standalone mode — smacos must be pre-built in MACOS_BUILD_DIR
  set(MACOS_BUILD_DIR "" CACHE PATH
      "Path to pre-built macos CMake tree (standalone mode)")
  if(MACOS_BUILD_DIR STREQUAL "")
    set(MACOS_BUILD_DIR "$ENV{HOME}/dev/macos/build_release_giza")
  endif()

  if(WIN32)
    set(_smacos_lib "${MACOS_BUILD_DIR}/lib/smacos.lib")
  else()
    set(_smacos_lib "${MACOS_BUILD_DIR}/lib/libsmacos.a")
  endif()

  if(NOT EXISTS "${_smacos_lib}")
    message(FATAL_ERROR
      "mmacos: MACOS_BUILD_DIR='${MACOS_BUILD_DIR}' does not contain "
      "lib/smacos.lib (Windows) or lib/libsmacos.a (Linux). "
      "Build macos first or set -DMACOS_BUILD_DIR=<path>.")
  endif()

  # Import smacos as a pre-built static library
  add_library(smacos STATIC IMPORTED)
  set_target_properties(smacos PROPERTIES IMPORTED_LOCATION "${_smacos_lib}")

  set(SMACOS_MOD_DIR "${MACOS_BUILD_DIR}/mod_smacos")
  set(_smacos_lib_dir "${MACOS_BUILD_DIR}/lib")
else()
  # Integrated mode — smacos built in same invocation
  get_target_property(_smacos_lib_dir smacos ARCHIVE_OUTPUT_DIRECTORY)
  if(NOT _smacos_lib_dir)
    set(_smacos_lib_dir "${CMAKE_BINARY_DIR}/lib")
  endif()
  set(SMACOS_MOD_DIR "${CMAKE_BINARY_DIR}/mod_smacos")
endif()

# ==============================================================================
# 2. MATLAB auto-detection (mirrors GMI pattern)
# ==============================================================================
set(MATLAB_DIR "" CACHE PATH "MATLAB root directory (auto-detected if empty)")

if(MATLAB_DIR STREQUAL "" OR NOT EXISTS "${MATLAB_DIR}")
  if(WIN32)
    file(GLOB _matlab_candidates
        "C:/Program Files/MATLAB/R[0-9][0-9][0-9][0-9][ab]"
        "C:/Program Files/MATLAB/R[0-9][0-9][0-9][0-9]"
    )
  else()
    file(GLOB _matlab_candidates "/usr/local/MATLAB/R[0-9]*")
  endif()
  if(_matlab_candidates)
    list(SORT _matlab_candidates)
    list(GET _matlab_candidates -1 MATLAB_DIR)
    message(STATUS "mmacos: auto-detected MATLAB at ${MATLAB_DIR}")
  else()
    message(FATAL_ERROR
      "mmacos: No MATLAB installation found. "
      "Set -DMATLAB_DIR=/path/to/MATLAB/Rxxxx.")
  endif()
endif()

if(NOT EXISTS "${MATLAB_DIR}/extern/include/fintrf.h")
  message(FATAL_ERROR
    "mmacos: fintrf.h not found at ${MATLAB_DIR}/extern/include. "
    "Check MATLAB_DIR: ${MATLAB_DIR}")
endif()

message(STATUS "mmacos: MATLAB root: ${MATLAB_DIR}")

# ==============================================================================
# 3. Platform-specific MATLAB paths and MEX suffix
# ==============================================================================
if(WIN32)
  set(MATLAB_ARCH    "win64")
  set(MATLAB_LIB_DIR "${MATLAB_DIR}/extern/lib/win64/microsoft")
  set(MATLAB_BIN_DIR "${MATLAB_DIR}/bin/win64")
  set(MEX_SUFFIX     ".mexw64")
  set(MATLAB_LIBS
      "${MATLAB_LIB_DIR}/libmx.lib"
      "${MATLAB_LIB_DIR}/libmex.lib"
      "${MATLAB_LIB_DIR}/libmat.lib"
  )
elseif(APPLE)
  # macOS / Apple Silicon: MATLAB libs are plain .dylib under bin/maca64
  # (Intel Macs = maci64).  Link with -lmx (NOT the GNU-ld -l:libmx.so form).
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
    set(MATLAB_ARCH "maca64")
  else()
    set(MATLAB_ARCH "maci64")
  endif()
  set(MATLAB_LIB_DIR "${MATLAB_DIR}/bin/${MATLAB_ARCH}")
  set(MEX_SUFFIX     ".mex${MATLAB_ARCH}")
  set(MATLAB_LIBS
      "-L${MATLAB_LIB_DIR}"
      "-lmx"
      "-lmex"
      "-lmat"
  )
else()
  set(MATLAB_ARCH    "glnxa64")
  set(MATLAB_LIB_DIR "${MATLAB_DIR}/bin/glnxa64")
  set(MEX_SUFFIX     ".mexa64")
  set(MATLAB_LIBS
      "-L${MATLAB_LIB_DIR}"
      "-l:libmx.so"
      "-l:libmex.so"
      "-lmat"
  )
endif()

message(STATUS "mmacos: MEX suffix  = ${MEX_SUFFIX}")
message(STATUS "mmacos: MATLAB libs = ${MATLAB_LIB_DIR}")

# ==============================================================================
# 4. Intel runtime paths (Linux ifx only — for RPATH)
# ==============================================================================
if(NOT WIN32 AND CMAKE_Fortran_COMPILER_ID MATCHES "IntelLLVM")
  get_filename_component(_ifx_bin   "${CMAKE_Fortran_COMPILER}" DIRECTORY)
  get_filename_component(_comp_root "${_ifx_bin}" DIRECTORY)
  set(INTEL_LIB_DIR "${_comp_root}/lib")
  get_filename_component(_oneapi_compiler "${_comp_root}" DIRECTORY)
  get_filename_component(_oneapi_root     "${_oneapi_compiler}" DIRECTORY)
  set(MKL_LIB_DIR "${_oneapi_root}/mkl/latest/lib")
  set(MMACOS_RPATHS "${INTEL_LIB_DIR};${MKL_LIB_DIR}")
elseif(NOT WIN32 AND CMAKE_Fortran_COMPILER_ID MATCHES "GNU")
  set(MMACOS_RPATHS "${CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES}")
else()
  set(MMACOS_RPATHS "")
endif()

# ==============================================================================
# 5. Optional support libs (NPSOL, LAPACK, BLAS, FITS)
# ==============================================================================
set(NPSOL_MMACOS_LIBS "")
if(WIN32)
  # slsqplib included: libsmacos references slsqp_ symbols, but the
  # slsqp_optmz.f objects live in their own static archive (CMake target
  # slsqplib), same as the mmacos Makefile's explicit $(SLSQP_LIB) link.
  foreach(_lib npsol.lib lapacklib.lib blaslib.lib fitslib.lib slsqplib.lib)
    if(EXISTS "${_smacos_lib_dir}/${_lib}")
      list(APPEND NPSOL_MMACOS_LIBS "${_smacos_lib_dir}/${_lib}")
    endif()
  endforeach()
else()
  foreach(_lib libnpsol.a liblapacklib.a libblaslib.a libfitslib.a libslsqplib.a)
    if(EXISTS "${_smacos_lib_dir}/${_lib}")
      list(APPEND NPSOL_MMACOS_LIBS "${_smacos_lib_dir}/${_lib}")
    endif()
  endforeach()
endif()

if(NPSOL_MMACOS_LIBS)
  message(STATUS "mmacos: support libs: ${NPSOL_MMACOS_LIBS}")
else()
  message(STATUS "mmacos: no NPSOL/FITS support libs found")
endif()

# ==============================================================================
# 6. mmacos shared library target (same pattern as GMI)
# ==============================================================================
# Write a tiny Fortran stub file -- FKPROCS_ must be first in .text
# so the linker places a single ret at RVA 0x1000, matching the
# layout of the working GMI.mexw64 that MATLAB R2025b accepts.
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/mmacos_stub.F"
"      subroutine fkprocs()\n"
"      return\n"
"      end\n"
)

# Fortran sources moved to src/ in newer mmacos checkouts; fall back to
# the mmacos/ root for older layouts so this works either way.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/mmacos_mex.F")
  set(_mmacos_src_dir "${CMAKE_CURRENT_SOURCE_DIR}/src")
else()
  set(_mmacos_src_dir "${CMAKE_CURRENT_SOURCE_DIR}")
endif()

set(MMACOS_SOURCES
    "${CMAKE_CURRENT_BINARY_DIR}/mmacos_stub.F"
    "${_mmacos_src_dir}/mmacos_mex.F"
    "${_mmacos_src_dir}/mmacos_gen.F"
)

foreach(_src ${MMACOS_SOURCES})
  if(NOT EXISTS "${_src}")
    message(FATAL_ERROR "mmacos source not found: ${_src}")
  endif()
endforeach()

# On macOS a MEX file is a loadable BUNDLE, not a .dylib.  CMake's MODULE
# library type emits `-bundle` automatically (adding it by hand alongside
# SHARED's `-dynamiclib` would conflict).  Linux/Windows keep SHARED.
if(APPLE)
  add_library(mmacos MODULE ${MMACOS_SOURCES})
else()
  add_library(mmacos SHARED ${MMACOS_SOURCES})
endif()

set_target_properties(mmacos PROPERTIES
    OUTPUT_NAME              "mmacos"
    SUFFIX                   "${MEX_SUFFIX}"
    PREFIX                   ""
    Fortran_MODULE_DIRECTORY "${CMAKE_BINARY_DIR}/mod_mmacos"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# ==============================================================================
# 7. Compile definitions
# ==============================================================================
if(WIN32)
  # MX_COMPAT_32 = API 700: symbols are lowercase_700_ matching
  # what libmx.dll/libmex.dll export directly (no C stubs needed)
  target_compile_definitions(mmacos PRIVATE MSWIN MX_COMPAT_32)
else()
  target_compile_definitions(mmacos PRIVATE MX_COMPAT_32 LNXOS)
endif()



# ==============================================================================
# 7b. Windows MATLAB API stubs
# ifx generates UPPERCASE_ symbols; libmx.dll exports UPPERCASE (no trailing _)
# and lowercase_ (with trailing _). We use linker /ALTERNATENAME to alias
# UPPERCASE_ -> UPPERCASE (no stubs needed for most functions).
# mexFunction needs a C gateway since Fortran passes args by reference.
# ==============================================================================
if(WIN32)
  file(WRITE "${CMAKE_BINARY_DIR}/mmacos_win_stubs.c"
"#include <stddef.h>\n"
"#include <string.h>\n"
"#include <mex.h>\n"
"#include <matrix.h>\n"
"extern void MEXFUNCTION_(int *nlhs, mxArray **plhs,\n"
"                         int *nrhs, const mxArray **prhs);\n"
"void mexFunction(int nlhs, mxArray *plhs[],\n"
"                 int nrhs, const mxArray *prhs[]) {\n"
"    MEXFUNCTION_(&nlhs, plhs, &nrhs, prhs); }\n"
"/* mexErrMsgTxt wrapper - libmex exports mexerrmsgtxt_ (lowercase) */\n"
"extern void mexerrmsgtxt_(const char*, int, size_t);\n"
"void MEXERRMSGTXT_(const char *s, int slen, size_t hidden) {\n"
"    mexerrmsgtxt_(s, slen, hidden); }\n"
  )

  add_library(mmacos_stubs STATIC "${CMAKE_BINARY_DIR}/mmacos_win_stubs.c")
  target_include_directories(mmacos_stubs PRIVATE "${MATLAB_DIR}/extern/include")
  target_compile_definitions(mmacos_stubs PRIVATE MSWIN MX_COMPAT_32)
  target_link_libraries(mmacos PRIVATE mmacos_stubs)

  # Use /ALTERNATENAME to alias UPPERCASE_ -> UPPERCASE (libmx exports)
  # This eliminates the need for C wrapper stubs for each MATLAB API function
  target_link_options(mmacos PRIVATE
      /Qoption,link,/EXPORT:mexFunction
      /Qoption,link,/ALTERNATENAME:MXGETM700_=MXGETM700
      /Qoption,link,/ALTERNATENAME:MXGETN700_=MXGETN700
      /Qoption,link,/ALTERNATENAME:MXGETNUMBEROFELEMENTS700_=MXGETNUMBEROFELEMENTS700
      /Qoption,link,/ALTERNATENAME:MXGETSTRING700_=MXGETSTRING700
      /Qoption,link,/ALTERNATENAME:MXISCHAR_=MXISCHAR
      /Qoption,link,/ALTERNATENAME:MXCREATEDOUBLESCALAR_=mxcreatedoublescalar700_
      /Qoption,link,/ALTERNATENAME:MXGETPR_=mxgetpr_
      /Qoption,link,/ALTERNATENAME:MXGETPI_=mxgetpi_
      /Qoption,link,/ALTERNATENAME:MXCOPYPTRTOREAL8700_=MXCOPYPTRTOREAL8700
      /Qoption,link,/ALTERNATENAME:MXCOPYREAL8TOPTR700_=MXCOPYREAL8TOPTR700
      /Qoption,link,/ALTERNATENAME:MXCREATEDOUBLEMATRIX700_=MXCREATEDOUBLEMATRIX700
      /Qoption,link,/ALTERNATENAME:MXCLASSIDFROMCLASSNAME_=mxclassidfromclassname700_
      /Qoption,link,/ALTERNATENAME:MXCREATENUMERICARRAY700_=MXCREATENUMERICARRAY700
  )
endif()

# ==============================================================================
# 8. Compiler flags (mirrors GMI exactly)
# ==============================================================================
if(WIN32)
  if(CMAKE_Fortran_COMPILER_ID MATCHES "IntelLLVM")
    # NOTE: do NOT add /reentrancy:none here. It switches the linked
    # Intel Fortran runtime to libifcorert.dll (single-threaded static),
    # while smacos.lib/fitslib.lib (built by the main CMakeLists.txt)
    # link against libifcoremd.dll (multi-threaded DLL runtime). Mixing
    # both in one binary causes a silent MATLAB exit at MEX load with
    # no crash dump. GMI's working Windows build has no such flag --
    # match that here.
    target_compile_options(mmacos PRIVATE
        /fpp
        /nologo
        /extend-source:132
        /gen-interfaces
        /fp:strict
        /fno-omit-frame-pointer
        /O2
    )
  endif()
else()
  if(CMAKE_Fortran_COMPILER_ID MATCHES "IntelLLVM")
    target_compile_options(mmacos PRIVATE
        -fPIC
        -fpp
        -132
        -gen-interfaces
        -fp-model strict
        -fno-omit-frame-pointer
        -O2
        -xHOST
        -traceback
        -fstack-protector
    )
  elseif(CMAKE_Fortran_COMPILER_ID MATCHES "GNU")
    # Apple Silicon rejects -march=native; use -mcpu=native there.
    if(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
      set(_mmacos_arch_flag -mcpu=native)
    else()
      set(_mmacos_arch_flag -march=native)
    endif()
    target_compile_options(mmacos PRIVATE
        -fPIC
        -cpp
        -ffixed-line-length-132
        -fno-omit-frame-pointer
        -O2
        ${_mmacos_arch_flag}
        -fallow-argument-mismatch
        -std=legacy
        -fstack-protector
    )
  endif()
endif()

# ==============================================================================
# 9. Include directories
# ==============================================================================
target_include_directories(mmacos PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${_mmacos_src_dir}"
    "${SMACOS_MOD_DIR}"
    "${CMAKE_BINARY_DIR}/mod"
    "${MATLAB_DIR}/extern/include"
)

if(TARGET smacos)
  get_target_property(_smacos_includes smacos INCLUDE_DIRECTORIES)
  if(_smacos_includes)
    target_include_directories(mmacos PRIVATE ${_smacos_includes})
  endif()
endif()

# ==============================================================================
# 10. Link libraries
# ==============================================================================
if(WIN32)
  target_link_libraries(mmacos PRIVATE
      smacos
      $<$<TARGET_EXISTS:fitslib>:fitslib>
      ${NPSOL_MMACOS_LIBS}
      ${MATLAB_LIBS}
      gdi32.lib
      user32.lib
  )
elseif(APPLE)
  # macOS mex = a -bundle with dynamic_lookup (MATLAB symbols resolved at
  # load time), NOT a -shared lib with a version-script.  ld64 rejects the
  # GNU-ld flags (--version-script/--no-undefined/-rpath-link/--disable-new-dtags).
  # -lc++ (not -lstdc++) is the Apple C++ runtime.  gfortran is the only
  # compiler here (ifx has no arm64), so no Intel arm.
  target_link_libraries(mmacos PRIVATE
      smacos
      $<$<TARGET_EXISTS:fitslib>:fitslib>
      ${NPSOL_MMACOS_LIBS}
      ${MATLAB_LIBS}
      -lm
      -lc++
  )
  # MODULE library already passes -bundle; just resolve MATLAB symbols at
  # load time (dynamic_lookup) and add an rpath to MATLAB's dylib dir.
  target_link_options(mmacos PRIVATE
      "-Wl,-undefined,dynamic_lookup"
      "-Wl,-rpath,${MATLAB_LIB_DIR}"
  )
else()
  target_link_libraries(mmacos PRIVATE
      smacos
      $<$<TARGET_EXISTS:fitslib>:fitslib>
      ${NPSOL_MMACOS_LIBS}
      ${MATLAB_LIBS}
      -lm
      -lstdc++
  )
  target_link_options(mmacos PRIVATE
      "-Wl,--version-script,${MATLAB_DIR}/extern/lib/glnxa64/fexport.map"
      -Wl,--no-undefined
      "-Wl,-rpath-link,${MATLAB_LIB_DIR}"
      -Wl,--disable-new-dtags
  )
  if(CMAKE_Fortran_COMPILER_ID MATCHES "IntelLLVM")
    target_link_options(mmacos PRIVATE
        -shared-intel
        -reentrancy=none
    )
    target_link_libraries(mmacos PRIVATE -lirc)
  endif()
endif()

# ==============================================================================
# 11. RPATH (Linux only)
# ==============================================================================
if(NOT WIN32 AND MMACOS_RPATHS)
  set_target_properties(mmacos PROPERTIES
      BUILD_RPATH   "${MMACOS_RPATHS}"
      INSTALL_RPATH "${MMACOS_RPATHS}"
  )
endif()

# ==============================================================================
# 12. Build order: ensure smacos is built before mmacos when integrated
# ==============================================================================
if(TARGET smacos AND NOT TARGET smacos-imported)
  add_dependencies(mmacos smacos)
endif()

# ==============================================================================
# 13. CTest targets
# ==============================================================================
enable_testing()

set(RX_INPUT "" CACHE FILEPATH "Rx .in file for smoke test")
if(RX_INPUT STREQUAL "")
  if(WIN32)
    set(RX_INPUT "C:/dev/MACOS_resources/pymacos/tests/Rx/Rx_Cass_FarField.in")
  else()
    set(RX_INPUT "$ENV{HOME}/dev/MACOS_resources/pymacos/tests/Rx/Rx_Cass_FarField.in")
  endif()
endif()

add_test(
  NAME    smoke
  COMMAND "${MATLAB_DIR}/bin/matlab" -batch
          "addpath('${CMAKE_CURRENT_BINARY_DIR}'); test_mmacos('${RX_INPUT}'); exit(0);"
)
set_tests_properties(smoke PROPERTIES LABELS "smoke" TIMEOUT 300)

add_test(
  NAME    unittest
  COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/run_mmacos_tests.sh"
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)
set_tests_properties(unittest PROPERTIES
  LABELS  "unittest"
  TIMEOUT 1800
  DEPENDS smoke
)

# ==============================================================================
# 14. Summary
# ==============================================================================
message(STATUS "------------------------------------------------------------")
message(STATUS "mmacos build configuration")
message(STATUS "  Fortran compiler : ${CMAKE_Fortran_COMPILER} (${CMAKE_Fortran_COMPILER_ID})")
message(STATUS "  MATLAB root      : ${MATLAB_DIR}")
message(STATUS "  MEX suffix       : ${MEX_SUFFIX}")
message(STATUS "  mod_smacos       : ${SMACOS_MOD_DIR}")
message(STATUS "  Support libs     : ${NPSOL_MMACOS_LIBS}")
message(STATUS "  RX smoke input   : ${RX_INPUT}")
message(STATUS "------------------------------------------------------------")
