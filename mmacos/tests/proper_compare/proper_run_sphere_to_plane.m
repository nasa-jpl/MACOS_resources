function [intensity, sampling] = proper_run_sphere_to_plane(geom, wf_pupil, opts)
%PROPER_RUN_SPHERE_TO_PLANE  Drive PROPER for the sphere-to-plane step.
%   Mirror of pymacos proper_run_sphere_to_plane.
%
%   PROPER recipe: ingest macos's complex wavefront at the spherical
%   reference (amplitude + cfield-phase as OPD), then prop_lens(f) +
%   prop_propagate(f) — same FF kernel as Cass-FF Phase 1.
arguments
    geom     (1,1) struct
    wf_pupil (1,1) struct
    opts.opd_sign_flip    (1,1) logical = true
    opts.use_cfield_phase (1,1) logical = true
end

N = geom.macos_size;
dx_pupil_m  = wf_pupil.dx_pupil_m;
grid_extent = N * dx_pupil_m;
bm = prop_begin(grid_extent, geom.wavelength_m, N, 1.0);

cfield = complex(wf_pupil.complex_field);
bm = prop_multiply(bm, abs(cfield));

if opts.use_cfield_phase
    % Phase at a spherical-reference element is residual content
    % beyond the sphere; prop_lens re-applies the convergence.
    opd = angle(cfield) * geom.wavelength_m / (2 * pi);
    if opts.opd_sign_flip
        opd = -opd;
    end
    bm = prop_add_phase(bm, opd);
end

bm = prop_define_entrance(bm);
bm = prop_lens(bm, geom.focal_length_m);
bm = prop_propagate(bm, geom.focal_length_m);
[wfa, sampling] = prop_end(bm);
intensity = wfa;
end
