cmake_minimum_required(VERSION 3.20)
project(SegMirMaker Fortran)

# Pre-built MACOS tree supplies libsmacos.a plus its dependency stack and
# the Fortran module files SegMirMaker needs (smacos_mod, surfsub, elt_mod,
# segmir_parent_mod compiled in-tree, param_mod, cfiles_mod).
set(MACOS_BUILD_DIR "$ENV{HOME}/dev/macos/build_release" CACHE PATH
    "Pre-built MACOS build tree (use makems.sh to build it)")

if(NOT EXISTS "${MACOS_BUILD_DIR}/lib/libsmacos.a")
  message(FATAL_ERROR
      "libsmacos.a not found in ${MACOS_BUILD_DIR}/lib. "
      "Build MACOS first: cd ../../macos/macos && source ./makems.sh")
endif()

find_library(SMACOS_LIB    smacos    PATHS ${MACOS_BUILD_DIR}/lib NO_DEFAULT_PATH REQUIRED)
find_library(FITSLIB       fitslib   PATHS ${MACOS_BUILD_DIR}/lib NO_DEFAULT_PATH REQUIRED)
# NPSOL was removed from the macos tree (PR #44); SLSQP is the constrained
# back end libsmacos references (design_slsqp_optim_mod -> slsqp_).
find_library(SLSQP_LIB     slsqplib  PATHS ${MACOS_BUILD_DIR}/lib NO_DEFAULT_PATH REQUIRED)
find_library(CPGPLOT_LIB   cpgplot   PATHS ${MACOS_BUILD_DIR}/lib NO_DEFAULT_PATH REQUIRED)
find_library(PGPLOT_LIB    pgplot    PATHS ${MACOS_BUILD_DIR}/lib NO_DEFAULT_PATH REQUIRED)
find_library(GIZA_LIB      giza      PATHS ${MACOS_BUILD_DIR}/lib NO_DEFAULT_PATH REQUIRED)
find_library(CAIRO_LIB     cairo     PATHS ${MACOS_BUILD_DIR}/lib NO_DEFAULT_PATH REQUIRED)
find_library(PIXMAN_LIB    pixman    PATHS ${MACOS_BUILD_DIR}/lib NO_DEFAULT_PATH REQUIRED)
find_library(PNG_LIB       png16     PATHS ${MACOS_BUILD_DIR}/lib NO_DEFAULT_PATH REQUIRED)
find_library(Z_LIB         z         PATHS ${MACOS_BUILD_DIR}/lib NO_DEFAULT_PATH REQUIRED)

# readline lives outside the CMake tree (built by makealldcr.sh into the
# macos_f90/readline-8.2 source directory).  On macOS the engine build uses
# Homebrew's readline instead of the bundled one, so pick that up here too.
if(APPLE)
  execute_process(COMMAND brew --prefix readline
      OUTPUT_VARIABLE _brew_readline OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET)
  set(READLINE_LIB "${_brew_readline}/lib/libreadline.a"
      CACHE FILEPATH "Static readline library")
else()
  set(READLINE_LIB "$ENV{HOME}/dev/macos/macos_f90/readline-8.2/libreadline.a"
      CACHE FILEPATH "Static readline library")
endif()
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
  # -fp-model strict: no FMA contraction — segment geometry reproduces
  # byte-identically across rebuilds (matches the engine's own flags).
  target_compile_options(SegMirMaker PRIVATE -extend-source 132 -nowarn
      -fp-model strict)
elseif(CMAKE_Fortran_COMPILER_ID MATCHES "GNU")
  target_compile_options(SegMirMaker PRIVATE
      -std=legacy -fallow-argument-mismatch -ffixed-line-length-132
      -fno-automatic -w)
endif()

target_link_libraries(SegMirMaker PRIVATE
    ${SMACOS_LIB}
    ${FITSLIB}
    ${SLSQP_LIB}
    ${CPGPLOT_LIB}
    ${PGPLOT_LIB}
    ${GIZA_LIB}
    ${CAIRO_LIB}
    ${PIXMAN_LIB}
    ${PNG_LIB}
    ${Z_LIB}
    ${READLINE_LIB})

# Platform system libraries.  On Linux the giza/pgplot stack needs X11 and
# readline needs ncurses/dl; on macOS -ldl is folded into libSystem and the
# CoreFoundation frameworks the Homebrew libs may reference are picked up
# automatically, so only ncurses + the math/thread libs are named explicitly.
if(APPLE)
  target_link_libraries(SegMirMaker PRIVATE ncurses pthread m)
else()
  target_link_libraries(SegMirMaker PRIVATE X11 ncurses dl pthread m)
endif()

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
  # --disable-new-dtags is a GNU-ld option ld64 rejects; RPATH via the
  # implicit link dirs is already correct on macOS (@rpath entries).
  if(NOT APPLE)
    target_link_options(SegMirMaker PRIVATE -Wl,--disable-new-dtags)
  endif()
endif()
