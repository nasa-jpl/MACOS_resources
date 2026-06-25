# ==============================================================================
# GMI/CMakeLists.txt — Cross-platform CMake build for GMI MATLAB mex interface
#
# Replaces the existing Makefile (Linux-only, ifx/gfortran).
# Supports Linux (mexa64) and Windows (mexw64).
#
# Called from the main CMakeLists.txt via add_subdirectory() when BUILD_GMI=ON:
#
#   if(BUILD_GMI)
#     set(GMI_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../MACOS_resources/GMI"
#         CACHE PATH "Directory containing GMI.F and GMIG.F")
#     add_subdirectory("${GMI_SRC_DIR}" "${CMAKE_BINARY_DIR}/gmi")
#   endif()
#
# Prerequisites:
#   - smacos target must already be defined (BUILD_SMACOS=ON)
#   - MATLAB_ROOT must point to a valid MATLAB installation
#
# Linux output : GMI.mexa64
# Windows output: GMI.mexw64
# ==============================================================================

cmake_minimum_required(VERSION 3.21)

# ==============================================================================
# 1. Validate prerequisites
# ==============================================================================
if(NOT TARGET smacos)
  message(FATAL_ERROR
    "GMI requires the 'smacos' target to be built first.\n"
    "Enable BUILD_SMACOS=ON in the main CMakeLists.txt.")
endif()

# ==============================================================================
# 2. MATLAB root auto-detection
# ==============================================================================
if(WIN32)
  if(NOT MATLAB_ROOT OR NOT EXISTS "${MATLAB_ROOT}")
    file(GLOB _matlab_candidates
        "C:/Program Files/MATLAB/R[0-9][0-9][0-9][0-9][ab]"
        "C:/Program Files/MATLAB/R[0-9][0-9][0-9][0-9]"
    )
    if(_matlab_candidates)
      list(SORT _matlab_candidates)
      list(GET _matlab_candidates -1 MATLAB_ROOT)
      message(STATUS "GMI: auto-detected MATLAB at ${MATLAB_ROOT}")
    else()
      message(FATAL_ERROR
        "No MATLAB installation found under C:/Program Files/MATLAB.\n"
        "Set MATLAB_ROOT on the cmake command line:\n"
        "  -DMATLAB_ROOT=\"C:/Program Files/MATLAB/R2025b\"")
    endif()
  endif()
else()
  if(NOT MATLAB_ROOT OR NOT EXISTS "${MATLAB_ROOT}")
    file(GLOB _matlab_candidates "/usr/local/MATLAB/R[0-9]*")
    if(_matlab_candidates)
      list(SORT _matlab_candidates)
      list(GET _matlab_candidates -1 MATLAB_ROOT)
      message(STATUS "GMI: auto-detected MATLAB at ${MATLAB_ROOT}")
    else()
      message(FATAL_ERROR
        "No MATLAB installation found under /usr/local/MATLAB.\n"
        "Set MATLAB_ROOT on the cmake command line:\n"
        "  -DMATLAB_ROOT=/usr/local/MATLAB/R2025b")
    endif()
  endif()
endif()

message(STATUS "GMI: MATLAB_ROOT = ${MATLAB_ROOT}")

if(NOT EXISTS "${MATLAB_ROOT}/extern/include/mex.h")
  message(FATAL_ERROR
    "MATLAB headers not found at ${MATLAB_ROOT}/extern/include/mex.h.\n"
    "Check MATLAB_ROOT is correct: ${MATLAB_ROOT}")
endif()

# ==============================================================================
# 3. Platform-specific MATLAB paths and MEX suffix
# ==============================================================================
if(WIN32)
  set(MATLAB_ARCH        "win64")
  set(MATLAB_LIB_DIR     "${MATLAB_ROOT}/extern/lib/win64/microsoft")
  set(MATLAB_BIN_DIR     "${MATLAB_ROOT}/bin/win64")
  set(MEX_SUFFIX         ".mexw64")
  set(MATLAB_LIBS
      "${MATLAB_LIB_DIR}/libmx.lib"
      "${MATLAB_LIB_DIR}/libmex.lib"
      "${MATLAB_LIB_DIR}/libmat.lib"
  )
else()
  set(MATLAB_ARCH        "glnxa64")
  set(MATLAB_LIB_DIR     "${MATLAB_ROOT}/bin/glnxa64")
  set(MEX_SUFFIX         ".mexa64")
  set(MATLAB_LIBS
      "-L${MATLAB_LIB_DIR}"
      "-l:libmx.so"
      "-l:libmex.so"
      "-lmat"
  )
endif()

message(STATUS "GMI: MEX suffix  = ${MEX_SUFFIX}")
message(STATUS "GMI: MATLAB libs = ${MATLAB_LIB_DIR}")

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
  set(GMI_RPATHS "${INTEL_LIB_DIR};${MKL_LIB_DIR}")
elseif(NOT WIN32 AND CMAKE_Fortran_COMPILER_ID MATCHES "GNU")
  set(GMI_RPATHS "${CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES}")
else()
  set(GMI_RPATHS "")
endif()

# ==============================================================================
# 5. GMI source files
# Windows: prepend a tiny fkprocs stub so the linker places a single ret
# at RVA 0x1000 (.text start), matching the layout MATLAB R2025b expects.
# ==============================================================================
if(WIN32)
  # Write a tiny Fortran stub -- FKPROCS_ must be first in .text
  # so the linker places ret at RVA 0x1000, matching working MEX layout.
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/gmi_stub.F"
    "      subroutine fkprocs()\n"
    "      return\n"
    "      end\n"
  )
  set(GMI_SOURCES
      "${CMAKE_CURRENT_BINARY_DIR}/gmi_stub.F"
      "${CMAKE_CURRENT_SOURCE_DIR}/GMI.F"
      "${CMAKE_CURRENT_SOURCE_DIR}/GMIG.F"
  )
else()
  set(GMI_SOURCES
      "${CMAKE_CURRENT_SOURCE_DIR}/GMI.F"
      "${CMAKE_CURRENT_SOURCE_DIR}/GMIG.F"
  )
endif()

foreach(_src ${GMI_SOURCES})
  if(NOT EXISTS "${_src}" AND NOT "${_src}" MATCHES "gmi_stub")
    message(FATAL_ERROR "GMI source not found: ${_src}")
  endif()
endforeach()

# ==============================================================================
# 6. NPSOL libs — auto-detect from smacos build tree
# ==============================================================================
get_target_property(_smacos_bin_dir smacos ARCHIVE_OUTPUT_DIRECTORY)
if(NOT _smacos_bin_dir)
  set(_smacos_lib_dir "${CMAKE_BINARY_DIR}/lib")
else()
  set(_smacos_lib_dir "${_smacos_bin_dir}")
endif()

set(NPSOL_GMI_LIBS "")
if(WIN32)
  foreach(_lib npsol.lib lapacklib.lib blaslib.lib)
    if(EXISTS "${_smacos_lib_dir}/${_lib}")
      list(APPEND NPSOL_GMI_LIBS "${_smacos_lib_dir}/${_lib}")
    endif()
  endforeach()
else()
  foreach(_lib libnpsol.a liblapacklib.a libblaslib.a)
    if(EXISTS "${_smacos_lib_dir}/${_lib}")
      list(APPEND NPSOL_GMI_LIBS "${_smacos_lib_dir}/${_lib}")
    endif()
  endforeach()
endif()

if(NPSOL_GMI_LIBS)
  message(STATUS "GMI: NPSOL libs found — linking ${NPSOL_GMI_LIBS}")
else()
  message(STATUS "GMI: NPSOL libs not found — building without NPSOL")
endif()

# ==============================================================================
# 7. GMI target
# ==============================================================================
add_library(GMI SHARED ${GMI_SOURCES})

set_target_properties(GMI PROPERTIES
    OUTPUT_NAME              "GMI"
    SUFFIX                   "${MEX_SUFFIX}"
    PREFIX                   ""
    Fortran_MODULE_DIRECTORY "${CMAKE_BINARY_DIR}/mod_gmi"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# ==============================================================================
# 8. Compile definitions
# ==============================================================================
if(WIN32)
  target_compile_definitions(GMI PRIVATE MSWIN)
else()
  target_compile_definitions(GMI PRIVATE MX_COMPAT_32 LNXOS)
endif()

# ==============================================================================
# 9. Compiler flags + Windows MATLAB API stubs
# ==============================================================================
if(WIN32)
  if(CMAKE_Fortran_COMPILER_ID MATCHES "IntelLLVM")
    # ifx default on Windows generates UPPERCASE_ symbols (same as
    # explicit /names:uppercase /assume:underscore). GMIG.F uses _800
    # API symbols. We bridge via /ALTERNATENAME linker aliases and a
    # small C stubs file that provides the mexFunction gateway.
    target_compile_options(GMI PRIVATE
        /fpp
        /nologo
        /extend-source:132
        /gen-interfaces
        /fp:strict
        /fno-omit-frame-pointer
        /O2
    )
  endif()

  # Windows MATLAB API stubs:
  # GMIG.F generates UPPERCASE_800_ calls (e.g. MXGETM800_).
  # libmx.dll exports MXGETM800 (no trailing _) and mxgetm800_ (lowercase).
  # We provide:
  #   1. mexFunction C gateway (MATLAB requires lowercase C symbol)
  #   2. mexErrMsgTxt wrapper (libmex exports mexerrmsgtxt_ lowercase)
  #   3. /ALTERNATENAME aliases for all _800_ -> _800 symbols
  file(WRITE "${CMAKE_BINARY_DIR}/gmi_win_stubs.c"
"/* Windows MATLAB API gateway + stubs for GMI - generated by CMake */\n\
#include <stddef.h>\n\
#include <string.h>\n\
#include <mex.h>\n\
#include <matrix.h>\n\
\n\
/* mexFunction C gateway: MATLAB requires this exact symbol name.      */\n\
/* ifx generates MEXFUNCTION_ (uppercase+underscore); this C stub      */\n\
/* provides the canonical mexFunction entry point and forwards to it.  */\n\
extern void MEXFUNCTION_(int *nlhs, mxArray **plhs,\n\
                         int *nrhs, const mxArray **prhs);\n\
__declspec(dllexport)\n\
void mexFunction(int nlhs, mxArray *plhs[],\n\
                 int nrhs, const mxArray *prhs[]) {\n\
    MEXFUNCTION_(&nlhs, plhs, &nrhs, prhs);\n\
}\n\
\n\
/* mexfilerequiredapiversion: R2025b version check                     */\n\
__declspec(dllexport)\n\
void mexfilerequiredapiversion(unsigned int *built_by_rel,\n\
                               unsigned int *target_api_ver) {\n\
    if (built_by_rel)   *built_by_rel   = 0x2025bU;\n\
    if (target_api_ver) *target_api_ver = 0x08000000U;\n\
}\n\
\n\
/* mexErrMsgTxt wrapper */\n\
void MEXERRMSGTXT800_(const char *s, int slen) {\n\
    char buf[4096]; int n=slen<4095?slen:4095;\n\
    memcpy(buf,s,n); buf[n]=0; mexErrMsgTxt(buf);\n\
}\n\
"
  )

  add_library(gmi_stubs STATIC "${CMAKE_BINARY_DIR}/gmi_win_stubs.c")
  target_include_directories(gmi_stubs PRIVATE "${MATLAB_ROOT}/extern/include")
  target_compile_definitions(gmi_stubs PRIVATE MSWIN)
  target_link_libraries(GMI PRIVATE gmi_stubs)

  # /ALTERNATENAME: alias UPPERCASE_800_ -> UPPERCASE800 (libmx exports)
  # This resolves GMIG.F's _800_ calls without additional C wrappers.
  target_link_options(GMI PRIVATE
      /Qoption,link,/EXPORT:mexFunction
      /Qoption,link,/EXPORT:mexfilerequiredapiversion
      /Qoption,link,/ALTERNATENAME:MXGETM800_=MXGETM800
      /Qoption,link,/ALTERNATENAME:MXGETN800_=MXGETN800
      /Qoption,link,/ALTERNATENAME:MXGETNUMBEROFELEMENTS800_=MXGETNUMBEROFELEMENTS800
      /Qoption,link,/ALTERNATENAME:MXGETPR800_=MXGETPR800
      /Qoption,link,/ALTERNATENAME:MXCOPYPTRTOREAL8800_=MXCOPYPTRTOREAL8800
      /Qoption,link,/ALTERNATENAME:MXCOPYREAL8TOPTR800_=MXCOPYREAL8TOPTR800
      /Qoption,link,/ALTERNATENAME:MXCREATEDOUBLEMATRIX800_=MXCREATEDOUBLEMATRIX800
      /Qoption,link,/ALTERNATENAME:MXGETSTRING800_=MXGETSTRING800
      /Qoption,link,/ALTERNATENAME:FKPROCS_=fkprocs_
  )

else()
  # Linux ifx
  if(CMAKE_Fortran_COMPILER_ID MATCHES "IntelLLVM")
    target_compile_options(GMI PRIVATE
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
  # Linux gfortran
  elseif(CMAKE_Fortran_COMPILER_ID MATCHES "GNU")
    target_compile_options(GMI PRIVATE
        -fPIC
        -cpp
        -ffixed-line-length-132
        -fno-omit-frame-pointer
        -O2
        -march=native
        -fallow-argument-mismatch
        -std=legacy
        -fstack-protector
    )
  endif()
endif()

# ==============================================================================
# 10. Include directories
# ==============================================================================
target_include_directories(GMI PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_BINARY_DIR}/mod_smacos"
    "${CMAKE_BINARY_DIR}/mod"
    "${MATLAB_ROOT}/extern/include"
)

get_target_property(_smacos_includes smacos INCLUDE_DIRECTORIES)
if(_smacos_includes)
  target_include_directories(GMI PRIVATE ${_smacos_includes})
endif()

# ==============================================================================
# 11. Link libraries
# ==============================================================================
if(WIN32)
  target_link_libraries(GMI PRIVATE
      smacos
      $<$<TARGET_EXISTS:fitslib>:fitslib>
      ${NPSOL_GMI_LIBS}
      ${MATLAB_LIBS}
      gdi32.lib
      user32.lib
  )
else()
  target_link_libraries(GMI PRIVATE
      smacos
      $<$<TARGET_EXISTS:fitslib>:fitslib>
      ${NPSOL_GMI_LIBS}
      ${MATLAB_LIBS}
      -lm
      -lstdc++
  )

  target_link_options(GMI PRIVATE
      "-Wl,--version-script,${MATLAB_ROOT}/extern/lib/glnxa64/fexport.map"
      -Wl,--no-undefined
      "-Wl,-rpath-link,${MATLAB_LIB_DIR}"
      -Wl,--disable-new-dtags
  )

  if(CMAKE_Fortran_COMPILER_ID MATCHES "IntelLLVM")
    target_link_options(GMI PRIVATE
        -shared-intel
        -reentrancy=none
    )
    target_link_libraries(GMI PRIVATE -lirc)
  endif()
endif()

# ==============================================================================
# 12. RPATH (Linux only)
# ==============================================================================
if(NOT WIN32 AND GMI_RPATHS)
  set_target_properties(GMI PROPERTIES
      BUILD_RPATH   "${GMI_RPATHS}"
      INSTALL_RPATH "${GMI_RPATHS}"
  )
endif()

# ==============================================================================
# 13. Summary
# ==============================================================================
message(STATUS "")
message(STATUS "  GMI target configured:")
message(STATUS "    Platform : ${CMAKE_SYSTEM_NAME}")
message(STATUS "    Compiler : ${CMAKE_Fortran_COMPILER_ID}")
message(STATUS "    MATLAB   : ${MATLAB_ROOT}")
message(STATUS "    Output   : GMI${MEX_SUFFIX}")
message(STATUS "")
