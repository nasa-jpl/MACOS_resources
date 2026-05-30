function [intensity_at_det, dx_m, wf_at_src] = macos_run_coro_nfprop(geom)
%MACOS_RUN_CORO_NFPROP  Drive mmacos for the Coro NF prop step.
%   Mirror of pymacos
%   tests/proper_compare/geometries/coro_nfprop.py:macos_run.
%
%   Returns:
%     intensity_at_det  N×N intensity at geom.detector_elt
%     dx_m              focal-plane pitch (m) from macos.dx_at(detector_elt)
%     wf_at_src         struct with the wavefront at geom.src_elt:
%       .amplitude       N×N real,  sqrt(macos.intensity(src_elt))
%       .complex_field   N×N complex, macos's diffraction-grid WFElt
%                        — the quantity macos's own NF propagator
%                        operates on
%       .opd             OPD map at src_elt (source-ray grid; for
%                        the .mat archive — PROPER uses the cfield
%                        phase instead)
%       .dx_at_src_m     macos pitch (m) at src_elt
%       .dx_at_det_m     macos pitch (m) at detector_elt
arguments
    geom (1,1) struct
end

rx_path = fullfile(getenv('HOME'), 'dev', 'MACOS_resources', ...
    'pymacos', 'tests', 'Rx', geom.rx_filename);

macos.init(geom.macos_size);
macos.load_rx(rx_path);

intensity_at_src = macos.intensity(geom.src_elt);

% Diffraction-grid complex field at src_elt — what macos's own NF
% propagator operates on.  Faithful Phase-2 v2 wavefront hand-off.
cfield_at_src = macos.complex_field(geom.src_elt);

% Source-ray-grid OPD (legacy archive; not used by the PROPER side
% in v2 since the phase content is already in cfield).
macos.trace(geom.src_elt);
opd_at_src = macos.opd();

intensity_at_det = macos.intensity(geom.detector_elt);

dx_at_src_m = macos.dx_at(geom.src_elt);
dx_at_det_m = macos.dx_at(geom.detector_elt);

amplitude_at_src = sqrt(max(intensity_at_src, 0));

wf_at_src = struct( ...
    'amplitude',     amplitude_at_src, ...
    'complex_field', cfield_at_src, ...
    'opd',           opd_at_src, ...
    'dx_at_src_m',   dx_at_src_m, ...
    'dx_at_det_m',   dx_at_det_m);

dx_m = dx_at_det_m;
end
