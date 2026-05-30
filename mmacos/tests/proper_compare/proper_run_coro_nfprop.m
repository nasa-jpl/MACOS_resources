function [intensity, sampling] = proper_run_coro_nfprop(geom, wf_at_src, opts)
%PROPER_RUN_CORO_NFPROP  Drive MATLAB PROPER for the Phase-2 NF step.
%   Mirror of pymacos
%   tests/proper_compare/geometries/coro_nfprop.py:proper_run.
%
%   Args:
%     geom        struct from geometries.coro_nfprop()
%     wf_at_src   struct from macos_run_coro_nfprop (third return)
%
%   Name-value options:
%     'opd_sign_flip' (default true) — macos OPD sign convention is
%                     opposite to PROPER's prop_add_phase input
%                     (Phase 1 reconciliation; same fix applies here)
arguments
    geom      (1,1) struct
    wf_at_src (1,1) struct
    opts.opd_sign_flip (1,1) logical = true
end

N = geom.macos_size;
% beam_diam = N * dx_at_src so PROPER's pitch = macos's pitch bit-for-bit.
dx_at_src   = wf_at_src.dx_at_src_m;
grid_extent = N * dx_at_src;
% beam_ratio = 1.0 (not 0.5 as in Cass-FF): NF step uses the full
% grid as the beam, not centred-2x-oversampled.
bm = prop_begin(grid_extent, geom.wavelength_m, N, 1.0);

if isfield(wf_at_src, 'complex_field') && ~isempty(wf_at_src.complex_field)
    % Phase 2 v2: use macos's diffraction-grid cfield directly —
    % amplitude AND phase on the matching N×N grid.
    cfield = complex(wf_at_src.complex_field);
    amp        = abs(cfield);
    phase_rad  = angle(cfield);
    opd_metres = phase_rad * geom.wavelength_m / (2 * pi);
    if opts.opd_sign_flip
        opd_metres = -opd_metres;
    end
    bm = prop_multiply(bm, amp);
    bm = prop_add_phase(bm, opd_metres);
else
    % Fallback: amplitude-only.
    bm = prop_multiply(bm, double(wf_at_src.amplitude));
    if isfield(wf_at_src, 'opd') && ~isempty(wf_at_src.opd) ...
            && isequal(size(wf_at_src.opd), size(wf_at_src.amplitude))
        opd = double(wf_at_src.opd);
        if opts.opd_sign_flip, opd = -opd; end
        bm = prop_add_phase(bm, opd);
    end
end

bm = prop_define_entrance(bm);

% Single plane-to-plane Fresnel step.  PROPER selects the propagator
% (angular-spectrum vs convolution Fresnel) from the geometry; for a
% NF step inside the beam this is angular-spectrum.
bm = prop_propagate(bm, geom.propagation_m);

[wfa, sampling] = prop_end(bm);  % default: intensity
intensity = wfa;
end
