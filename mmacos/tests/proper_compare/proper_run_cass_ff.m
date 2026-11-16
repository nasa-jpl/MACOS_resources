function [intensity, sampling] = proper_run_cass_ff(geom, opts)
%PROPER_RUN_CASS_FF  Drive MATLAB PROPER for the Cass-FF geometry.
%   [intensity, sampling] = proper_run_cass_ff(geom)
%   [intensity, sampling] = proper_run_cass_ff(geom, name=value, ...)
%
%   Mirror of pymacos
%   tests/proper_compare/geometries/cass_farfield.py:proper_run.
%
%   Name-value options:
%     'include_obscurations'  (default true)  — analytical path only
%     'macos_opd'             []              — OPD map from macos.opd()
%     'macos_amplitude'       []              — amplitude mask (defaults
%                                               to |macos_opd|>0)
%     'opd_sign_flip'         (default true)  — macos OPD convention is
%                                               opposite to PROPER's
%                                               prop_add_phase input
%     'allow_resample'        (default false) — bilinear-resample on
%                                               shape mismatch
arguments
    geom (1,1) struct
    opts.include_obscurations (1,1) logical = true
    opts.macos_opd       double = []
    opts.macos_amplitude double = []
    opts.opd_sign_flip   (1,1) logical = true
    opts.allow_resample  (1,1) logical = false
end

n          = geom.proper_grid_n;
beam_ratio = geom.proper_beam_ratio;

bm = prop_begin(geom.pupil_diameter_m, geom.wavelength_m, n, beam_ratio);

use_macos_mask = ~isempty(opts.macos_opd) || ~isempty(opts.macos_amplitude);

if use_macos_mask
    if isempty(opts.macos_amplitude)
        amp = double(opts.macos_opd ~= 0);
    else
        amp = opts.macos_amplitude;
    end
    amp_padded = embed_macos_array_in_proper_grid( ...
        amp, geom, opts.allow_resample, 0.0);
    bm = prop_multiply(bm, amp_padded);
else
    bm = prop_circular_aperture(bm, geom.pupil_diameter_m / 2.0);
    if opts.include_obscurations
        bm = prop_circular_obscuration(bm, geom.sec_obs_radius_m);
        bm = prop_rectangular_obscuration(bm, ...
            geom.spider_half_width * 2.0, ...
            geom.spider_half_length * 2.0);
        bm = prop_rectangular_obscuration(bm, ...
            geom.spider_half_length * 2.0, ...
            geom.spider_half_width * 2.0);
    end
end

if ~isempty(opts.macos_opd)
    opd_arr = double(opts.macos_opd);
    if opts.opd_sign_flip
        opd_arr = -opd_arr;
    end
    phase = embed_macos_array_in_proper_grid( ...
        opd_arr, geom, opts.allow_resample, 0.0);
    bm = prop_add_phase(bm, phase);
end

bm = prop_define_entrance(bm);
bm = prop_lens(bm, geom.effective_focal_length_m);
bm = prop_propagate(bm, geom.effective_focal_length_m);
[wfa, sampling] = prop_end(bm);  % default: intensity (modulus squared)

intensity = wfa;
end
