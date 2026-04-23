cmake_minimum_required(VERSION 3.20)
project(SegMirMaker Fortran)

# Pre-built MACOS tree supplies libsmacos.a plus its dependency stack and
# the Fortran module files SegMirMaker needs (smacos_mod, surfsub, elt_mod,
# segmir_parent_mod compiled in-tree, param_mod, cfiles_mod).
set(MACOS_BUILD_DIR "$ENV{HOME}/dev/macos/build_release_giza" CACHE PATH
    "Pre-built MACOS build tree (use makeCMdcr.sh to build it)")

if(NOT EXISTS "${MACOS_BUILD_DIR}/lib/libsmacos.a")
  message(FATAL_ERROR
      "libsmacos.a not found in ${MACOS_BUILD_DIR}/lib. "
      "Build MACOS first: cd ../../macos && source ./makeCMdcr.sh")
endif()

find_library(SMACOS_LIB    smacos    PATHS ${MACOS_BUILD_DIR}/lib NO_DEFAULT_PATH REQUIRED)
find_library(FITSLIB       fitslib   PATHS ${MACOS_BUILD_DIR}/lib NO_DEFAULT_PATH REQUIRED)
find_library(NPSOL_LIB     npsol     PATHS ${MACOS_BUILD_DIR}/lib NO_DEFAULT_PATH REQUIRED)
find_library(LAPACK_LIB    lapacklib PATHS ${MACOS_BUILD_DIR}/lib NO_DEFAULT_PATH REQUIRED)
find_library(BLAS_LIB      blaslib   PATHS ${MACOS_BUILD_DIR}/lib NO_DEFAULT_PATH REQUIRED)
find_library(CPGPLOT_LIB   cpgplot   PATHS ${MACOS_BUILD_DIR}/lib NO_DEFAULT_PATH REQUIRED)
find_library(PGPLOT_LIB    pgplot    PATHS ${MACOS_BUILD_DIR}/lib NO_DEFAULT_PATH REQUIRED)
find_library(GIZA_LIB      giza      PATHS ${MACOS_BUILD_DIR}/lib NO_DEFAULT_PATH REQUIRED)
find_library(CAIRO_LIB     cairo     PATHS ${MACOS_BUILD_DIR}/lib NO_DEFAULT_PATH REQUIRED)
find_library(PIXMAN_LIB    pixman    PATHS ${MACOS_BUILD_DIR}/lib NO_DEFAULT_PATH REQUIRED)
find_library(PNG_LIB       png16     PATHS ${MACOS_BUILD_DIR}/lib NO_DEFAULT_PATH REQUIRED)
find_library(Z_LIB         z         PATHS ${MACOS_BUILD_DIR}/lib NO_DEFAULT_PATH REQUIRED)

# readline lives outside the CMake tree (built by makealldcr.sh into the
# macos_f90/readline-8.2 source directory).
set(READLINE_LIB "$ENV{HOME}/dev/macos/macos_f90/readline-8.2/libreadline.a"
    CACHE FILEPATH "Static readline library")
if(NOT EXISTS "${READLINE_LIB}")
  message(FATAL_ERROR "readline static library not found: ${READLINE_LIB}")
endif()

add_executable(SegMirMaker SegMirMaker.f)

# SegMirMaker.f has no CPP directives, but the modules it USEs were built
# with CPP.  Supplying both mod_smacos (for smacos_mod, surfsub, elt_mod,
# param_mod, cfiles_mod) covers everything.
target_include_directories(SegMirMaker PRIVATE
    ${MACOS_BUILD_DIR}/mod_smacos
    ${MACOS_BUILD_DIR}/mod)

# Per-compiler flags match the macos build.
if(CMAKE_Fortran_COMPILER_ID MATCHES "IntelLLVM|Intel")
  target_compile_options(SegMirMaker PRIVATE -extend-source 132 -nowarn)
elseif(CMAKE_Fortran_COMPILER_ID MATCHES "GNU")
  target_compile_options(SegMirMaker PRIVATE
      -std=legacy -fallow-argument-mismatch -ffixed-line-length-132
      -fno-automatic -w)
endif()

target_link_libraries(SegMirMaker PRIVATE
    ${SMACOS_LIB}
    ${FITSLIB}
    ${NPSOL_LIB}
    ${LAPACK_LIB}
    ${BLAS_LIB}
    ${CPGPLOT_LIB}
    ${PGPLOT_LIB}
    ${GIZA_LIB}
    ${CAIRO_LIB}
    ${PIXMAN_LIB}
    ${PNG_LIB}
    ${Z_LIB}
    ${READLINE_LIB}
    X11 ncurses dl pthread m)

# Embed RPATH so the binary runs without sourcing Intel setvars.sh or
# relying on a loaded gfortran module.  Same pattern as macos/CMakeLists.txt.
if(NOT WIN32 AND CMAKE_Fortran_COMPILER_ID MATCHES "IntelLLVM|Intel")
  set(INTEL_LIB_DIR "/opt/intel/oneapi/compiler/latest/lib" CACHE PATH "Intel compiler runtime libs")
  set_target_properties(SegMirMaker PROPERTIES
      BUILD_RPATH   "${INTEL_LIB_DIR};/opt/intel/oneapi/mkl/latest/lib"
      INSTALL_RPATH "${INTEL_LIB_DIR};/opt/intel/oneapi/mkl/latest/lib")
  target_link_options(SegMirMaker PRIVATE -Wl,--disable-new-dtags)
elseif(NOT WIN32 AND CMAKE_Fortran_COMPILER_ID MATCHES "GNU")
  set_target_properties(SegMirMaker PROPERTIES
      BUILD_RPATH   "${CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES}"
      INSTALL_RPATH "${CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES}")
  target_link_options(SegMirMaker PRIVATE -Wl,--disable-new-dtags)
endif()
