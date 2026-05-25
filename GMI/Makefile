
##--------------------------------------
# Makefile for GMI (ifx or gfortran + auto-detected MATLAB)
##--------------------------------------

#####################################
# User Configurable Options         #
#####################################

# MACOS source directory (set by makealldcr.sh / makeGMIdcr.sh, or override here)
MACOS_DIR := $(macossrc_dir)
GMI_DIR    := $(shell pwd)

# Fortran compiler.  gfortran is the default — produces a mex that
# exits cleanly under MATLAB.  ifx still works but SIGSEGVs at
# MATLAB process-exit teardown (cosmetic; results are correct).
# Override on the command line: make FC=ifx
FC ?= gfortran

# MATLAB installation — pick the lexicographically latest R20XXx under /usr/local/MATLAB.
# Override on the command line: make MATLAB_DIR=/usr/local/MATLAB/R2023a
MATLAB_BASE := /usr/local/MATLAB
MATLAB_DIR  := $(shell ls -1d $(MATLAB_BASE)/R[0-9]*[ab] 2>/dev/null | sort -V | tail -1)
ifeq ($(MATLAB_DIR),)
  $(error No MATLAB installation found under $(MATLAB_BASE). \
    Install MATLAB or set MATLAB_DIR on the command line.)
endif

# Intel oneAPI paths
INTEL_LIB  := /opt/intel/oneapi/compiler/latest/lib
MKL_LIB    := /opt/intel/oneapi/mkl/latest/lib

# Architecture / MEX extension
OS       := $(shell uname -s)
PLATFORM := $(shell uname -i)
ifeq ($(PLATFORM),x86_64)
    ARCH    := Linux-x86_64
    MEXTAG  := mexa64
endif

# Optional: path to a CMake build tree (build_release_giza, etc.).
# When set, module files and smacos library are taken from there instead
# of the legacy MACOS_OBJS/Linux-x86_64 layout.
# Set via command line: make MACOS_BUILD_DIR=~/dev/macos/build_release_giza
MACOS_BUILD_DIR :=

ifneq ($(MACOS_BUILD_DIR),)
  SMACOS_OBJS := $(MACOS_BUILD_DIR)/mod_smacos
  # libnpsol.a removed by macos rem_npsol PR (Levenberg-Marquardt
  # in nls_optim_dvr replaces NPSOL-based constrained-optim).
  SMACOS_LIBS := $(MACOS_BUILD_DIR)/lib/libsmacos.a
else
  SMACOS_OBJS := $(MACOS_DIR)/SMACOS_OBJS/$(ARCH)
  SMACOS_LIBS := $(SMACOS_OBJS)/smacos_lib.a
endif

######################################################
# No user edits needed below here
######################################################

GMI_DATE := $(shell date '+%Y-%m-%d')

APP      := GMI
MEX_NAME := $(APP)

# Per-compiler flags.  Both produce bit-identical numeric results
# (verified by MACOS_resources/GMI/regression).  gfortran exits
# MATLAB cleanly; ifx SIGSEGVs in MATLAB's mex-cleanup teardown.
#
# -check all removed from FFLAGS/LDFLAGS: it pulls in ifx's
# fsanitize=memory which SIGSEGVs in the mex cleanup path.
# Re-enable when investigating specific runtime issues.

ifeq ($(FC),ifx)
  FFLAGS := -traceback -fstack-protector \
            -fPIC -fpp -132 -gen-interfaces -fp-model strict \
            -fno-omit-frame-pointer -nologo \
            -O2 -xHOST \
            -DMX_COMPAT_32 \
            -I$(MACOS_DIR) -I$(MATLAB_DIR)/extern/include \
            -module $(SMACOS_OBJS)

  # Compiler-only link flags — placed before objects.
  LDFLAGS := -traceback -fstack-protector \
             -O -shared-intel -shared \
             -Wl,--version-script,$(MATLAB_DIR)/extern/lib/glnxa64/fexport.map \
             -Wl,--no-undefined \
             -Wl,-rpath-link,$(MATLAB_DIR)/bin/glnxa64 \
             -Wl,-rpath,$(INTEL_LIB) \
             -Wl,-rpath,$(MKL_LIB) \
             -Wl,--disable-new-dtags

  # Trailing libs — placed after objects/archives so --no-undefined
  # resolves correctly under both bundled and system ld.
  POST_LIBS := -L$(MATLAB_DIR)/bin/glnxa64 -l:libmx.so -l:libmex.so -lmat \
               -L$(INTEL_LIB) -lirc -lm -lstdc++
else ifeq ($(FC),gfortran)
  FFLAGS := -fPIC -cpp -ffixed-line-length-132 \
            -fno-omit-frame-pointer -O2 -march=native \
            -fallow-argument-mismatch -std=legacy \
            -DMX_COMPAT_32 \
            -I$(MACOS_DIR) -I$(MATLAB_DIR)/extern/include \
            -J$(SMACOS_OBJS)

  LDFLAGS := -fPIC -O -shared \
             -Wl,--version-script,$(MATLAB_DIR)/extern/lib/glnxa64/fexport.map \
             -Wl,--no-undefined \
             -Wl,-rpath-link,$(MATLAB_DIR)/bin/glnxa64 \
             -Wl,--disable-new-dtags

  POST_LIBS := -L$(MATLAB_DIR)/bin/glnxa64 -l:libmx.so -l:libmex.so -lmat \
               -lm -lstdc++
else
  $(error Unsupported FC='$(FC)' — use ifx or gfortran)
endif

# Some macos cmake builds emit libfitslib.a as a separate archive;
# link it when present so gfortran's --no-undefined is satisfied.
FITS_LIB := $(wildcard $(MACOS_BUILD_DIR)/lib/libfitslib.a)

default: $(MEX_NAME).$(MEXTAG)

$(MEX_NAME).$(MEXTAG): GMI.o GMIG.o $(SMACOS_LIBS)
	$(FC) $(LDFLAGS) -o $(MEX_NAME).$(MEXTAG) GMI.o GMIG.o $(SMACOS_LIBS) $(FITS_LIB) $(POST_LIBS)

GMI.o: GMI.F
	$(FC) $(FFLAGS) -c GMI.F

GMIG.o: GMIG.F
	$(FC) $(FFLAGS) -c GMIG.F

clean:
	-rm -f *.o *.mod $(MEX_NAME).$(MEXTAG)

# begin fortran depends
GMIG.o: GMIG.F
GMI.o:  GMI.F
# end fortran depends

##--------------------------------------
# Legacy mex-compiler approach (kept for reference — no longer used)
#
# MEX = /usr/local/MATLAB/8.1/bin/mex -f $(GMI_DIR)/mexopts.sh_81
# $(MEX_NAME).$(MEXTAG): $(SRCS) $(SMACOS_LIBS) mexopts.sh
#     $(MEX) -v $(SRCS) -I$(MACOS_DIR) -output $(MEX_NAME)
##--------------------------------------
