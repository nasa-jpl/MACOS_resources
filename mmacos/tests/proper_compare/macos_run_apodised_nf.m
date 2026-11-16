function [intensity_det, dx_det_m, wf] = macos_run_apodised_nf(geom, mask)
%MACOS_RUN_APODISED_NF  Drive mmacos with an apodiser at src_elt.
%   Mirror of pymacos test_coro_apodizer.py:_macos_apodised.
%
%   Returns:
%     intensity_det  N×N intensity at geom.detector_elt
%     dx_det_m       pitch (m) at detector_elt
%     wf             struct with .complex_field_unapo (so PROPER can
%                    apply the SAME mask itself), .dx_at_src_m,
%                    .dx_at_det_m, .mask
arguments
    geom (1,1) struct
    mask (:,:) double
end

rx_path = fullfile(getenv('HOME'), 'dev', 'MACOS_resources', ...
    'pymacos', 'tests', 'Rx', geom.rx_filename);

macos.init(geom.macos_size);
macos.load_rx(rx_path);

% Populate WFElt at src_elt with the unapodised wavefront.
macos.intensity(geom.src_elt);

% Snapshot the unapodised cfield + dx for PROPER's reference.
cfield_unapo = macos.complex_field(geom.src_elt, 'reset_trace', false);
dx_src_m     = macos.dx_at(geom.src_elt);

% Apodise in place, then propagate WITHOUT resetting trace state
% (reset_trace=true would re-run MODIFY and wipe the mask).
macos.apodize(geom.src_elt, mask);
intensity_det = macos.intensity(geom.detector_elt, 'reset_trace', false);
dx_det_m      = macos.dx_at(geom.detector_elt);

wf = struct( ...
    'complex_field_unapo', cfield_unapo, ...
    'dx_at_src_m',         dx_src_m, ...
    'dx_at_det_m',         dx_det_m, ...
    'mask',                mask);
end
