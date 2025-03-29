# GMI
MACOS Generic Matlab Interface

This folder constains source files for compiling GMI on millipede.

The Makefile links object files and a SMACOS library pre-compiled separately
to generate the GMI mex module 'GMI.mexa64'. 

To compile using the Makefile, run 'make' command.

To clean up compiled modules, run 'make clean'.

After GMI compilation, a test can be run by using the 'test_gmi.m' script.

It uses an optiix optical prescription as input.

