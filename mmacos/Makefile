##-----------------------------------------------------------------
# Makefile for mmacos (ifx or gfortran + auto-detected MATLAB)
#
# mmacos is the MATLAB sibling of pymacos.  Both wrap the language-
# neutral macos_api_mod module (in §5.1's pymacos refactor).  This
# Makefile compiles macos_api_mod.F from the pymacos source tree as
# part of its own build until the file is promoted to libsmacos.a
# (tracked as a §5 follow-up in PLAN.md).
#
# Usage:
#   make                                     # gfortran (default)
#   make FC=ifx                              # ifx build
#   make MACOS_BUILD_DIR=/path/to/macos/build_release_giza
#   make MATLAB_DIR=/usr/local/MATLAB/R2025b
##-----------------------------------------------------------------

MMACOS_DIR := $(shell pwd)
# macos_api_mod is now compiled into libsmacos.a (lives in macos_f90/
# in the macos repo).  The .mod is taken from $(SMACOS_OBJS), the
# library code is linked via $(SMACOS_LIBS) -- no separate compile step.

# Fortran compiler.  gfortran default (matches GMI) for the same reason:
# the gfortran mex exits MATLAB cleanly; ifx also works but is the
# alternative.  macos_api_mod was ported to be gfortran-clean as part of
# §5.2; both compilers now produce numerically identical results.
FC ?= gfortran

# MATLAB installation — pick the lexicographically latest R20XXx.
MATLAB_BASE := /usr/local/MATLAB
MATLAB_DIR  := $(shell ls -1d $(MATLAB_BASE)/R[0-9]*[ab] 2>/dev/null | sort -V | tail -1)
ifeq ($(MATLAB_DIR),)
  $(error No MATLAB installation found under $(MATLAB_BASE). \
    Install MATLAB or set MATLAB_DIR on the command line.)
endif

# Intel oneAPI paths (used by ifx build only)
INTEL_LIB  := /opt/intel/oneapi/compiler/latest/lib
MKL_LIB    := /opt/intel/oneapi/mkl/latest/lib

# Architecture / MEX extension
OS       := $(shell uname -s)
PLATFORM := $(shell uname -i)
ifeq ($(PLATFORM),x86_64)
    ARCH    := Linux-x86_64
    MEXTAG  := mexa64
endif

# Path to a CMake build tree of macos (build_release_giza by default).
# Falls back to ~/dev/macos/build_release_giza.
MACOS_BUILD_DIR ?= $(HOME)/dev/macos/build_release_giza

ifeq ($(wildcard $(MACOS_BUILD_DIR)/lib/libsmacos.a),)
  $(error MACOS_BUILD_DIR='$(MACOS_BUILD_DIR)' does not contain \
    lib/libsmacos.a.  Build macos first or override MACOS_BUILD_DIR.)
endif

SMACOS_OBJS := $(MACOS_BUILD_DIR)/mod_smacos
SMACOS_LIBS := $(MACOS_BUILD_DIR)/lib/libsmacos.a
NPSOL_LIBS  := $(wildcard $(MACOS_BUILD_DIR)/lib/libnpsol.a) \
               $(wildcard $(MACOS_BUILD_DIR)/lib/liblapacklib.a) \
               $(wildcard $(MACOS_BUILD_DIR)/lib/libblaslib.a)
FITS_LIB    := $(wildcard $(MACOS_BUILD_DIR)/lib/libfitslib.a)

APP      := mmacos
MEX_NAME := $(APP)

# Per-compiler flags.  Same -reentrancy=none rationale as GMI for ifx
# (avoids the parked-thread SIGSEGV at MATLAB process exit).
ifeq ($(FC),ifx)
  FFLAGS := -traceback -fstack-protector \
            -fPIC -fpp -132 -gen-interfaces -fp-model strict \
            -fno-omit-frame-pointer -nologo \
            -O2 -xHOST \
            -DMX_COMPAT_32 \
            -I$(MATLAB_DIR)/extern/include \
            -I$(SMACOS_OBJS) \
            -module $(MMACOS_DIR)
  LDFLAGS := -traceback -fstack-protector \
             -O -shared-intel -shared -reentrancy=none \
             -Wl,--version-script,$(MATLAB_DIR)/extern/lib/glnxa64/fexport.map \
             -Wl,--no-undefined \
             -Wl,-rpath-link,$(MATLAB_DIR)/bin/glnxa64 \
             -Wl,-rpath,$(INTEL_LIB) \
             -Wl,-rpath,$(MKL_LIB) \
             -Wl,--disable-new-dtags
  POST_LIBS := -L$(MATLAB_DIR)/bin/glnxa64 -l:libmx.so -l:libmex.so -lmat \
               -L$(INTEL_LIB) -lirc -lm -lstdc++
else ifeq ($(FC),gfortran)
  FFLAGS := -fPIC -cpp -ffixed-line-length-132 \
            -fno-omit-frame-pointer -O2 -march=native \
            -fallow-argument-mismatch -std=legacy \
            -DMX_COMPAT_32 \
            -I$(MATLAB_DIR)/extern/include \
            -I$(SMACOS_OBJS) \
            -J$(MMACOS_DIR)
  LDFLAGS := -fPIC -O -shared \
             -Wl,--version-script,$(MATLAB_DIR)/extern/lib/glnxa64/fexport.map \
             -Wl,--no-undefined \
             -Wl,-rpath-link,$(MATLAB_DIR)/bin/glnxa64 \
             -Wl,--disable-new-dtags
  POST_LIBS := -L$(MATLAB_DIR)/bin/glnxa64 -l:libmx.so -l:libmex.so -lmat \
               -lgfortran -lm -lstdc++
else
  $(error Unsupported FC='$(FC)' -- use ifx or gfortran)
endif

default: $(MEX_NAME).$(MEXTAG)

MEX_OBJS := mmacos_mex.o mmacos_gen.o

$(MEX_NAME).$(MEXTAG): $(MEX_OBJS) $(SMACOS_LIBS)
	$(FC) $(LDFLAGS) -o $(MEX_NAME).$(MEXTAG) \
	    $(MEX_OBJS) \
	    $(SMACOS_LIBS) $(FITS_LIB) $(NPSOL_LIBS) $(POST_LIBS)

# macos_api_mod's .mod comes from $(SMACOS_OBJS) via -I in FFLAGS.
mmacos_mex.o: mmacos_mex.F
	$(FC) $(FFLAGS) -c mmacos_mex.F

# Auto-generated dispatcher + per-routine helpers.  Regenerate via
# `python3 gen_mex_wrappers.py` whenever macos_api_mod signatures change.
mmacos_gen.o: mmacos_gen.F
	$(FC) $(FFLAGS) -c mmacos_gen.F

clean:
	-rm -f *.o *.mod *.smod $(MEX_NAME).$(MEXTAG)

# Quick smoke target: build then exercise the mex from a MATLAB
# batch session.  Requires Rx_Cass_FarField.in (or set RX=...) on
# the path.
RX ?= $(HOME)/dev/MACOS_resources/pymacos/tests/Rx/Rx_Cass_FarField.in
test: $(MEX_NAME).$(MEXTAG)
	$(MATLAB_DIR)/bin/matlab -batch \
	    "addpath('$(MMACOS_DIR)'); test_mmacos('$(RX)')"

# Full matlab.unittest suite under tests/.  See run_mmacos_tests.sh.
unittest: $(MEX_NAME).$(MEXTAG)
	./run_mmacos_tests.sh
