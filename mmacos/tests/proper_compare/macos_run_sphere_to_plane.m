function [intensity_focal, dx_focal_m, wf_pupil] = macos_run_sphere_to_plane(geom)
%MACOS_RUN_SPHERE_TO_PLANE  Drive mmacos for the sphere-to-plane step.
%   Mirror of pymacos macos_run_sphere_to_plane.
%
%   Returns:
%     intensity_focal  N×N intensity at geom.detector_elt
%     dx_focal_m       focal-plane pitch (m), |macos.dx_at|
%                      (macos can return signed dx at FarField targets;
%                       absolute value is the physical pitch)
%     wf_pupil         struct with the spherical-reference wavefront:
%       .complex_field  N×N complex, macos's WFElt at src_elt
%       .amplitude      N×N real, sqrt(intensity_at_src)
%       .dx_pupil_m     macos pitch at src_elt
%       .dx_focal_m     macos pitch at detector_elt (signed)
arguments
    geom (1,1) struct
end

rx_path = fullfile(getenv('HOME'), 'dev', 'MACOS_resources', ...
    'pymacos', 'tests', 'Rx', geom.rx_filename);

macos.init(geom.macos_size);
macos.load_rx(rx_path);

cfield_at_pupil  = macos.complex_field(geom.src_elt);
intensity_pupil  = macos.intensity(geom.src_elt);
intensity_focal  = macos.intensity(geom.detector_elt);
amplitude_pupil  = sqrt(max(intensity_pupil, 0));

dx_pupil_m = macos.dx_at(geom.src_elt);
dx_focal_m_signed = macos.dx_at(geom.detector_elt);
dx_focal_m = abs(dx_focal_m_signed);

wf_pupil = struct( ...
    'complex_field', cfield_at_pupil, ...
    'amplitude',     amplitude_pupil, ...
    'dx_pupil_m',    dx_pupil_m, ...
    'dx_focal_m',    dx_focal_m_signed);
end
