function [intensity_det, dx_det_m, wf] = macos_run_dm_phase(geom, opd_m)
%MACOS_RUN_DM_PHASE  Apply a complex phase imprint at geom.src_elt, propagate.
%   Mirror of pymacos test_coro_dm_phase.py:_macos_phase.
%
%   Args:
%     geom    struct from geometries.coro_nfprop()
%     opd_m   N×N real OPD map (metres) — the phase imprint to apply
%
%   Returns:
%     intensity_det  N×N intensity at geom.detector_elt
%     dx_det_m       pitch (m) at detector_elt
%     wf             struct with .complex_field_pre, .dx_at_src_m,
%                    .dx_at_det_m, .opd_m (for the PROPER counterpart)
arguments
    geom  (1,1) struct
    opd_m (:,:) double
end

rx_path = fullfile(getenv('HOME'), 'dev', 'MACOS_resources', ...
    'pymacos', 'tests', 'Rx', geom.rx_filename);

macos.init(geom.macos_size);
macos.load_rx(rx_path);

macos.intensity(geom.src_elt);
cfield_pre = macos.complex_field(geom.src_elt, 'reset_trace', false);
dx_src_m   = macos.dx_at(geom.src_elt);

% Build exp(i * 2*pi * OPD / lambda) — pure phase, unit magnitude.
phi = 2 * pi * opd_m / geom.wavelength_m;
mask_complex = complex(cos(phi), sin(phi));

macos.apodize_complex(geom.src_elt, mask_complex);
intensity_det = macos.intensity(geom.detector_elt, 'reset_trace', false);
dx_det_m      = macos.dx_at(geom.detector_elt);

wf = struct( ...
    'complex_field_pre', cfield_pre, ...
    'dx_at_src_m',       dx_src_m, ...
    'dx_at_det_m',       dx_det_m, ...
    'opd_m',             opd_m);
end
