function [intensity, dx_m, opd_map] = macos_run_cass_ff(geom, opts)
%MACOS_RUN_CASS_FF  Drive mmacos+macos for the Cass-FF geometry.
%   Mirror of pymacos
%   tests/proper_compare/geometries/cass_farfield.py:macos_run.
%
%   Output args:
%       intensity  N×N double from macos.intensity(srf_focal_plane)
%       dx_m       focal-plane pixel pitch (m) — geom.dx_focal_m
%       opd_map    only returned when opts.return_opd is true; OPD map
%                  at the exit pupil
arguments
    geom (1,1) struct
    opts.return_opd      (1,1) logical = false
    opts.exit_pupil_srf  (1,1) double  = 5
    opts.perturbation    cell = {}   % {iElt, [Rx Ry Rz Tx Ty Tz]}
end

rx_path = fullfile(getenv('HOME'), 'dev', 'MACOS_resources', ...
    'pymacos', 'tests', 'Rx', geom.rx_filename);

macos.init(geom.macos_size);
macos.load_rx(rx_path);

if ~isempty(opts.perturbation)
    iElt = opts.perturbation{1};
    prb  = opts.perturbation{2}(:);
    % Use global frame (ifGlobal=1) — Rx_Cass_FarField elements have
    % nECoord=-6 (no local frame).  Translation is in BaseUnits per
    % macos's prb_elt convention (raw mex level — no SI conversion).
    macos.perturb_many(iElt, prb, 1);
end

opd_map = [];
if opts.return_opd
    macos.trace(opts.exit_pupil_srf);
    opd_map = macos.opd();
end

intensity = macos.intensity(geom.macos_detector_srf);
dx_m = geom.dx_focal_m;
end
