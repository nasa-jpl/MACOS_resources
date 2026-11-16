function g = cass_farfield()
%CASS_FARFIELD  Geometry parameters for the Cass-FarField PROPER cmp.
%   Mirror of pymacos
%   tests/proper_compare/geometries/cass_farfield.py:CassFarField.
%
%   Returns a struct of physical + numeric parameters shared by the
%   macos_run and proper_run drivers for the Rx_Cass_FarField.in
%   problem.
g.rx_filename         = 'Rx_Cass_FarField.in';
g.macos_detector_srf  = 6;       % FocalPlane element in the Rx
g.macos_size          = 512;     % 2x oversampling vs prescription nGridpts

% Physical params extracted from Rx_Cass_FarField.in
g.wavelength_m        = 1.0e-6;
g.pupil_diameter_m    = 4.0;
g.sec_obs_radius_m    = 0.5;
g.spider_half_width   = 3.125e-3;
g.spider_half_length  = 3.0;

% Macos-reported exit-pupil / focal-plane geometry at model_size=512
g.z_pupil_to_focal_m  = 5.5601;
g.dx_exit_pupil_m     = 3.9032e-3;
g.dx_focal_m          = 2.7823e-6;

% PROPER grid
g.proper_grid_n       = 512;
g.proper_beam_ratio   = 0.5;

% Derived: equivalent thin-lens focal length so PROPER's dx_focal
% matches macos's.  f_eff = dx_focal * D / (lambda * beam_ratio).
g.effective_focal_length_m = ...
    g.dx_focal_m * g.pupil_diameter_m / ...
    (g.wavelength_m * g.proper_beam_ratio);
end
