function g = coro_sphere_to_plane(opts)
%CORO_SPHERE_TO_PLANE  Geometry for the Phase 3b/5 sphere-to-plane step.
%   Mirror of pymacos
%   tests/proper_compare/geometries/coro_nfprop.py:CoroSphereToPlane.
%
%   Default: Elt 8 (1stPropStart, spherical Kr=-774mm) -> Elt 9 (CorMask,
%   plane) of Rx_Coro.in.  PROPER recipe: prop_lens(f) + prop_propagate(f)
%   (same physics as Cass-FF Phase 1).
%
%   Variants (passed as name-value):
%     'rx_filename'    'Rx_Coro_noLyot.in' for Phase 5 baseline (no FPM/Lyot)
%                      'Rx_Coro_FPM.in'    for Phase 5 with FPM + Lyot
%     'src_elt','detector_elt','focal_length_m' for the ExitPupil ->
%                      science FocalPlane step (20 -> 21 at f=0.9514m)
arguments
    opts.rx_filename    (1,:) char   = 'Rx_Coro.in'
    opts.src_elt        (1,1) double = 8        % 1stPropStart (sphere)
    opts.detector_elt   (1,1) double = 9        % CorMask (plane)
    opts.macos_size     (1,1) double = 1024
    opts.wavelength_m   (1,1) double = 8.5e-7
    opts.focal_length_m (1,1) double = 0.774    % |KrElt| at Elt 8
end
g = opts;
end
