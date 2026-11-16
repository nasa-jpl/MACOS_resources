function [intensity, sampling] = proper_run_apodised_nf(geom, wf)
%PROPER_RUN_APODISED_NF  Drive PROPER with the same apodiser mask.
%   Combines amp + mask in a single prop_multiply (matches macos's
%   apodize semantics: |cf_apo| = |cf| * mask).
arguments
    geom (1,1) struct
    wf   (1,1) struct
end

cfield = complex(wf.complex_field_unapo);
mask   = double(wf.mask);
dx_src = wf.dx_at_src_m;
N      = geom.macos_size;
grid_extent = N * dx_src;

bm = prop_begin(grid_extent, geom.wavelength_m, N, 1.0);

bm = prop_multiply(bm, abs(cfield) .* mask);
opd = -angle(cfield) * geom.wavelength_m / (2 * pi);
bm = prop_add_phase(bm, opd);

bm = prop_define_entrance(bm);
bm = prop_propagate(bm, geom.propagation_m);
[wfa, sampling] = prop_end(bm);
intensity = wfa;
end
