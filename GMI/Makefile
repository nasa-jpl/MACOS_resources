
##--------------------------------------
# Makefile for GMI (ifx + auto-detected MATLAB)
##--------------------------------------

#####################################
# User Configurable Options         #
#####################################

# MACOS source directory (set by makealldcr.sh / makeGMIdcr.sh, or override here)
MACOS_DIR := $(macossrc_dir)
GMI_DIR    := $(shell pwd)

# Fortran compiler
FC := ifx

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

SMACOS_OBJS := $(MACOS_DIR)/SMACOS_OBJS/$(ARCH)
SMACOS_LIBS := $(SMACOS_OBJS)/smacos_lib.a

######################################################
# No user edits needed below here
######################################################

GMI_DATE := $(shell date '+%Y-%m-%d')

APP      := GMI
MEX_NAME := $(APP)

# Compiler flags (shared between GMI.F and GMIG.F)
FFLAGS := -check all -traceback -fstack-protector \
          -fPIC -fpp -132 -gen-interfaces -fp-model strict \
          -fno-omit-frame-pointer -nologo \
          -O2 -xHOST \
          -DMX_COMPAT_32 \
          -I$(MACOS_DIR) -I$(MATLAB_DIR)/extern/include \
          -module $(SMACOS_OBJS)

# Linker flags — embed Intel RPATH so MATLAB can load without setvars.sh
LDFLAGS := -check all -traceback -fstack-protector \
           -O -shared-intel -shared \
           -Wl,--version-script,$(MATLAB_DIR)/extern/lib/glnxa64/fexport.map \
           -Wl,--no-undefined \
           -Wl,-rpath-link,$(MATLAB_DIR)/bin/glnxa64 \
           -L$(MATLAB_DIR)/bin/glnxa64 -l:libmx.so -l:libmex.so -lmat \
           -L$(INTEL_LIB) \
           -Wl,-rpath,$(INTEL_LIB) \
           -Wl,-rpath,$(MKL_LIB) \
           -Wl,--disable-new-dtags \
           -lirc -lm -lstdc++

default: $(MEX_NAME).$(MEXTAG)

$(MEX_NAME).$(MEXTAG): GMI.o GMIG.o $(SMACOS_LIBS)
	$(FC) $(LDFLAGS) -o $(MEX_NAME).$(MEXTAG) GMI.o GMIG.o $(SMACOS_LIBS)

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
