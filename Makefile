
##--------------------------------------
# Makefile for GMI, with MACOS v3.34
##--------------------------------------

#####################################
# Typical User Configurable Options #
#####################################


# MACOS_DIR should be the source directory for macos_f90   ---> UPDATE !!!!!!
MACOS_DIR := $(macossrc_dir)
GMI_DIR = $(shell pwd)

#MEX = mex  -largeArrayDims      # Mex Compiler
#MEX = /usr/local/matlab714/bin/mex       # Mex Compiler
MEX = /usr/local/matlab81/bin/mex       # Mex Compiler

# Choose your compiler: lf95 or gfortran:
#FC  = gfortran  # Fortran Compiler
FC  = ifort      # Fortran Compiler

# Choose your Matlab version: 
# (leave blank for system default version)
MATLAB_VERSION := 81

SUFFIX :=
ifeq ($(MATLAB_VERSION),704)
    MEX = /usr/local/matlab704/bin/mex -f $(GMI_DIR)/mexopts.sh_$(MATLAB_VERSION)
    SUFFIX := _$(MATLAB_VERSION)
endif

ifeq ($(MATLAB_VERSION),790)
    MEX = /usr/local/matlab790/bin/mex -f mexopts.sh_$(MATLAB_VERSION)
    SUFFIX := _$(MATLAB_VERSION)
endif

ifeq ($(MATLAB_VERSION),711)  # --> will be default v2010b
    MEX = /usr/local/matlab711/bin/mex -f mexopts.sh_$(MATLAB_VERSION)
    SUFFIX := _$(MATLAB_VERSION)
endif

ifeq ($(MATLAB_VERSION),712)  # --> will be default v2010b
    MEX = /usr/local/matlab712/bin/mex -f mexopts.sh_$(MATLAB_VERSION)
    SUFFIX := _$(MATLAB_VERSION)
endif

ifeq ($(MATLAB_VERSION),714)  
    MEX = /usr/local/matlab-7.14/bin/mex -f $(GMI_DIR)/mexopts.sh_$(MATLAB_VERSION)
    SUFFIX := _$(MATLAB_VERSION)
endif

ifeq ($(MATLAB_VERSION),81) # --> R2013a
    # '-largeArrayDims' flag causes runtime memeory problem in GMIG.F -jzlou 04/2020
    #MEX = /usr/local/matlab-8.1/bin/mex -largeArrayDims -f $(GMI_DIR)/mexopts.sh_$(MATLAB_VERSION) 
    MEX = /usr/local/MATLAB/8.1/bin/mex -f $(GMI_DIR)/mexopts.sh_$(MATLAB_VERSION)
    SUFFIX := _$(MATLAB_VERSION)
endif

# Directory to find SMACOS object files for the specific architecture
export SMACOS_OBJS = $(MACOS_DIR)/SMACOS_OBJS/$(ARCH)
# SMACOS library
export SMACOS_LIBS = $(SMACOS_OBJS)/smacos_lib.a

# For setting/modifying libraries and compile options, see mexopts.sh

######################################################
# Hopefully, there will be no (or little) need for a #
# typical user to edit below here.                   #
######################################################

# Current Revision:
#$export GMI_SVN_REV := $(shell svn info | awk '/Rev:/{print $$4}')

# Current date
export GMI_DATE := $(shell date '+%Y-%m-%d')

OS := $(shell uname -s)
ifeq ($(OS),Darwin)
    PLATFORM := $(shell uname -p)
    ARCH     := $(OS)-$(PLATFORM)
    FC       := gfortran
else
    PLATFORM := $(shell uname -i)
    ifeq ($(PLATFORM),x86_64)
	ARCH   := Linux-x86_64
	MEXTAG := mexa64
    endif
    ifeq ($(PLATFORM),i386)
	ARCH   := Linux-i386
	MEXTAG := mexglx
    endif

    ifeq ($(findstring SUNW,$(PLATFORM)),SUNW)
	ARCH := SUNW-$(shell uname -p)
    endif
endif

#BKBKFMEX = $(MEX) -fortran FC=$(FC)
FMEX = $(MEX)

APP      := GMI
MEX_NAME := $(APP)$(SUFFIX)

default: $(MEX_NAME).$(MEXTAG)

##-------------------------------------------------

MOD_SRCS = $(MACOS_DIR)/param_mod.F $(MACOS_DIR)/elt_mod.F \
           $(MACOS_DIR)/src_mod.F $(MACOS_DIR)/cfiles_mod.F

SCOMP_$(APP)= $(APP).o $(APP)G.o $(SMACOS_OBJS)/smacos_lib.a

SRCS=$(APP).F $(APP)G.F

$(MEX_NAME).$(MEXTAG): $(SRCS) $(SMACOS_LIBS) mexopts.sh
	$(FMEX) -v $(SRCS) -I$(MACOS_DIR) -output $(MEX_NAME)
#	rm -f macos_param.txt
#	cp -p $(MACOS_DIR)/macos_param.txt .
	mv $(MEX_NAME).$(MEXTAG) $(APP).$(MEXTAG)

# Executable
GMImain: $(SCOMP_$(APP)) GMImain.o
	$(FC) -o $@ $(APP).o $(SMACOS_OBJS)/smacos_lib.a GMImain.o
clean:
	-rm -rf *.o *.out *.plot *.trace *.mod core $(MEX_NAME).$(MEXTAG) $(APP).$(MEXTAG)

# begin fortran depends
GMIG.o: GMIG.F param_mod.mod $(MOD_SRCS)
GMI.o: GMI.F $(MOD_SRCS)
GMImain.o: GMImain.F $(MOD_SRCS)
# end fortran depends

