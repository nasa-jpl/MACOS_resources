function [intensity, sampling] = proper_run_pupil_to_pupil(geom, wf_pupil, opts)
%PROPER_RUN_PUPIL_TO_PUPIL  Drive PROPER through Elt 8 -> 9 -> 10.
%   Mirror of pymacos proper_run_pupil_to_pupil.
%
%   Same first-leg recipe as sphere-to-plane, then a SECOND
%   prop_propagate(f) past the focus.
arguments
    geom     (1,1) struct
    wf_pupil (1,1) struct
    opts.opd_sign_flip (1,1) logical = true
end

N = geom.macos_size;
dx_pupil_m  = wf_pupil.dx_pupil_m;
grid_extent = N * dx_pupil_m;
bm = prop_begin(grid_extent, geom.wavelength_m, N, 1.0);

cfield = complex(wf_pupil.complex_field);
bm = prop_multiply(bm, abs(cfield));
opd = angle(cfield) * geom.wavelength_m / (2 * pi);
if opts.opd_sign_flip
    opd = -opd;
end
bm = prop_add_phase(bm, opd);

bm = prop_define_entrance(bm);
bm = prop_lens(bm, geom.focal_length_m);
bm = prop_propagate(bm, geom.focal_length_m);  % to focus (Elt 9)
bm = prop_propagate(bm, geom.focal_length_m);  % to Elt 10
[wfa, sampling] = prop_end(bm);
intensity = wfa;
end
