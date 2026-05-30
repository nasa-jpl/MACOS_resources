function [intensity_at_det, dx_at_det_m, wf_pupil] = macos_run_pupil_to_pupil(geom)
%MACOS_RUN_PUPIL_TO_PUPIL  Drive mmacos through Elt 8 -> 9 -> 10.
%   Mirror of pymacos macos_run_pupil_to_pupil.
%
%   Returns:
%     intensity_at_det   N×N intensity at geom.detector_elt (Elt 10)
%     dx_at_det_m        macos pitch (m) at detector_elt
%     wf_pupil           struct with:
%       .complex_field   cfield at src_elt (Elt 8 spherical reference)
%       .intensity_at_focus  intensity at focus_elt (Elt 9)
%       .dx_pupil_m      pitch at src_elt
%       .dx_focal_m      pitch at focus_elt
%       .dx_at_det_m     pitch at detector_elt
arguments
    geom (1,1) struct
end

rx_path = fullfile(getenv('HOME'), 'dev', 'MACOS_resources', ...
    'pymacos', 'tests', 'Rx', geom.rx_filename);

macos.init(geom.macos_size);
macos.load_rx(rx_path);

cfield_at_pupil    = macos.complex_field(geom.src_elt);
intensity_at_focus = macos.intensity(geom.focus_elt);
intensity_at_det   = macos.intensity(geom.detector_elt);

dx_pupil_m  = macos.dx_at(geom.src_elt);
dx_focal_m  = macos.dx_at(geom.focus_elt);
dx_at_det_m = macos.dx_at(geom.detector_elt);

wf_pupil = struct( ...
    'complex_field',      cfield_at_pupil, ...
    'intensity_at_focus', intensity_at_focus, ...
    'dx_pupil_m',         dx_pupil_m, ...
    'dx_focal_m',         dx_focal_m, ...
    'dx_at_det_m',        dx_at_det_m);
end
