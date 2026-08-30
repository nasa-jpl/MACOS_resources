run('/home/dcr/dev/MACOS_res_dev/mmacos/mmacos_setup.m');
here = fileparts(mfilename('fullpath'));  addpath(here);
R = afocal4_packaging('sections', 0:4);
