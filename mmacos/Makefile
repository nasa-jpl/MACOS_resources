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
# Sources (mex + codegen) and the MATLAB +macos package live under src/
# (mirrors pymacos's src/pymacos layout).  The mex is built into src/ too,
# so the MATLAB path only needs src/ on it.
SRCDIR  := src
SRCPATH := $(MMACOS_DIR)/$(SRCDIR)
# macos_api_mod is now compiled into libsmacos.a (lives in macos_f90/
# in the macos repo).  The .mod is taken from $(SMACOS_OBJS), the
# library code is linked via $(SMACOS_LIBS) -- no separate compile step.

# Fortran compiler.  gfortran default (matches GMI) for the same reason:
# the gfortran mex exits MATLAB cleanly; ifx also works but is the
# alternative.  macos_api_mod was ported to be gfortran-clean as part of
# §5.2; both compilers now produce numerically identical results.
# Default to gfortran.  NOTE: a plain `FC ?= gfortran` does NOT work here —
# GNU make ships a built-in default `FC = f77`, and `?=` treats that built-in
# as "already set", so bare `make` would try f77.  Override only when FC comes
# from make's built-in default (an explicit `make FC=ifx` still wins).
ifeq ($(origin FC),default)
  FC := gfortran
endif

# OS detection (used for MATLAB location + arch below)
OS := $(shell uname -s)

# MATLAB installation — pick the lexicographically latest release.
# Linux/Windows: /usr/local/MATLAB/R20XXx.  macOS: /Applications/MATLAB_R20XXx.app.
ifeq ($(OS),Darwin)
  MATLAB_BASE := /Applications
  MATLAB_DIR  := $(shell ls -1d $(MATLAB_BASE)/MATLAB_R[0-9]*[ab].app 2>/dev/null | sort -V | tail -1)
else
  MATLAB_BASE := /usr/local/MATLAB
  MATLAB_DIR  := $(shell ls -1d $(MATLAB_BASE)/R[0-9]*[ab] 2>/dev/null | sort -V | tail -1)
endif
ifeq ($(MATLAB_DIR),)
  $(error No MATLAB installation found under $(MATLAB_BASE). \
    Install MATLAB or set MATLAB_DIR on the command line.)
endif

# Intel oneAPI paths (used by ifx build only)
INTEL_LIB  := /opt/intel/oneapi/compiler/latest/lib
MKL_LIB    := /opt/intel/oneapi/mkl/latest/lib

# Architecture / MEX extension.  `uname -i` is invalid on macOS; use `uname -m`.
MACHINE := $(shell uname -m)
ifeq ($(OS),Darwin)
  ifeq ($(MACHINE),arm64)
    ARCH   := maca64
    MEXTAG := mexmaca64
  else
    ARCH   := maci64
    MEXTAG := mexmaci64
  endif
else
  ARCH   := glnxa64
  MEXTAG := mexa64
endif

# Path to a CMake build tree of macos, matched to the compiler choice:
# gfortran links build_release_gfortran, ifx links build_release.  (The
# old Linux default pointed at the retired build_release_giza tree.)
ifeq ($(OS),Darwin)
  MACOS_BUILD_DIR ?= $(HOME)/dev/macos/build_release_gfortran
else ifeq ($(FC),ifx)
  MACOS_BUILD_DIR ?= $(HOME)/dev/macos/build_release
else
  MACOS_BUILD_DIR ?= $(HOME)/dev/macos/build_release_gfortran
endif

ifeq ($(wildcard $(MACOS_BUILD_DIR)/lib/libsmacos.a),)
  $(error MACOS_BUILD_DIR='$(MACOS_BUILD_DIR)' does not contain \
    lib/libsmacos.a.  Build macos first or override MACOS_BUILD_DIR.)
endif

SMACOS_OBJS := $(MACOS_BUILD_DIR)/mod_smacos
SMACOS_LIBS := $(MACOS_BUILD_DIR)/lib/libsmacos.a
NPSOL_LIBS  := $(wildcard $(MACOS_BUILD_DIR)/lib/libnpsol.a) \
               $(wildcard $(MACOS_BUILD_DIR)/lib/liblapacklib.a) \
               $(wildcard $(MACOS_BUILD_DIR)/lib/libblaslib.a)
# SLSQP -- libsmacos references slsqp_ via design_slsqp_optim_mod but
# the slsqp_optmz.f objects live in their own static archive (CMake
# target slsqplib).  Link it alongside libsmacos here; the wildcard
# tolerates older trees that don't have it yet.
SLSQP_LIB   := $(wildcard $(MACOS_BUILD_DIR)/lib/libslsqplib.a)
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
            -module $(SRCPATH)
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
ifeq ($(OS),Darwin)
  # macOS / Apple Silicon.  MEX = a -bundle with dynamic_lookup (ld64 rejects
  # the GNU-ld version-script/--no-undefined/-rpath-link/--disable-new-dtags).
  # MATLAB libs are plain .dylib (-lmx, not -l:libmx.so); -mcpu=native on arm64
  # (-march=native is rejected); -lc++ not -lstdc++.  Homebrew's libgfortran
  # dir goes on the rpath so the bundle finds it without a module file.
  GFLIB := $(shell dirname $$($(FC) -print-file-name=libgfortran.dylib))
  ifeq ($(MACHINE),arm64)
    ARCHFLAG := -mcpu=native
  else
    ARCHFLAG := -march=native
  endif
  FFLAGS := -fPIC -cpp -ffixed-line-length-132 \
            -fno-omit-frame-pointer -O2 $(ARCHFLAG) \
            -fallow-argument-mismatch -std=legacy \
            -DMX_COMPAT_32 \
            -I$(MATLAB_DIR)/extern/include \
            -I$(SMACOS_OBJS) \
            -J$(SRCPATH)
  LDFLAGS := -fPIC -O -bundle -Wl,-undefined,dynamic_lookup \
             -Wl,-rpath,$(MATLAB_DIR)/bin/$(ARCH) \
             -Wl,-rpath,$(GFLIB)
  POST_LIBS := -L$(MATLAB_DIR)/bin/$(ARCH) -lmx -lmex -lmat \
               -L$(GFLIB) -lgfortran -lm -lc++
else
  # Linux gfortran
  FFLAGS := -fPIC -cpp -ffixed-line-length-132 \
            -fno-omit-frame-pointer -O2 -march=native \
            -fallow-argument-mismatch -std=legacy \
            -DMX_COMPAT_32 \
            -I$(MATLAB_DIR)/extern/include \
            -I$(SMACOS_OBJS) \
            -J$(SRCPATH)
  LDFLAGS := -fPIC -O -shared \
             -Wl,--version-script,$(MATLAB_DIR)/extern/lib/glnxa64/fexport.map \
             -Wl,--no-undefined \
             -Wl,-rpath-link,$(MATLAB_DIR)/bin/glnxa64 \
             -Wl,--disable-new-dtags
  POST_LIBS := -L$(MATLAB_DIR)/bin/glnxa64 -l:libmx.so -l:libmex.so -lmat \
               -lgfortran -lm -lstdc++
endif
else
  $(error Unsupported FC='$(FC)' -- use ifx or gfortran)
endif

default: $(SRCDIR)/$(MEX_NAME).$(MEXTAG)

# Emit just the platform MEX extension (used by run_mmacos_tests.sh to locate
# the built mex without hardcoding mexa64).
.PHONY: print-mextag
print-mextag:
	@echo $(MEXTAG)

MEX_OBJS := $(SRCDIR)/mmacos_mex.o $(SRCDIR)/mmacos_gen.o

$(SRCDIR)/$(MEX_NAME).$(MEXTAG): $(MEX_OBJS) $(SMACOS_LIBS)
	$(FC) $(LDFLAGS) -o $(SRCDIR)/$(MEX_NAME).$(MEXTAG) \
	    $(MEX_OBJS) \
	    $(SMACOS_LIBS) $(SLSQP_LIB) $(FITS_LIB) $(NPSOL_LIBS) $(POST_LIBS)

# macos_api_mod's .mod comes from $(SMACOS_OBJS) via -I in FFLAGS.
$(SRCDIR)/mmacos_mex.o: $(SRCDIR)/mmacos_mex.F
	$(FC) $(FFLAGS) -c $(SRCDIR)/mmacos_mex.F -o $(SRCDIR)/mmacos_mex.o

# Auto-generated dispatcher + per-routine helpers.  Regenerate via
# `python3 src/gen_mex_wrappers.py` whenever macos_api_mod signatures change.
$(SRCDIR)/mmacos_gen.o: $(SRCDIR)/mmacos_gen.F
	$(FC) $(FFLAGS) -c $(SRCDIR)/mmacos_gen.F -o $(SRCDIR)/mmacos_gen.o

clean:
	-rm -f $(SRCDIR)/*.o $(SRCDIR)/*.mod $(SRCDIR)/*.smod \
	    $(SRCDIR)/*__genmod.f90 $(SRCDIR)/$(MEX_NAME).$(MEXTAG)

# Quick smoke target: build then exercise the mex from a MATLAB
# batch session.  Requires Rx_Cass_FarField.in (or set RX=...) on
# the path.
RX ?= $(HOME)/dev/MACOS_resources/pymacos/tests/Rx/Rx_Cass_FarField.in
# Explicit `exit(0)` after the test function — without it matlab -batch
# on R2026a hangs at process exit when a mex is loaded (see
# CLAUDE.md "clear mmacos hangs matlab -batch" note for the family).
test: $(SRCDIR)/$(MEX_NAME).$(MEXTAG)
	$(MATLAB_DIR)/bin/matlab -batch \
	    "addpath('$(MMACOS_DIR)'); addpath('$(SRCPATH)'); test_mmacos('$(RX)'); exit(0);"

# Full matlab.unittest suite under tests/.  See run_mmacos_tests.sh.
unittest: $(SRCDIR)/$(MEX_NAME).$(MEXTAG)
	./run_mmacos_tests.sh
