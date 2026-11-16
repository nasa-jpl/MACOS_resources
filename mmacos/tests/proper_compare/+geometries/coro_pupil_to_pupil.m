function g = coro_pupil_to_pupil(opts)
%CORO_PUPIL_TO_PUPIL  Geometry for Phase 4a (sphere -> focus -> sphere).
%   Mirror of pymacos
%   tests/proper_compare/geometries/coro_nfprop.py:CoroPupilToPupilThruFocus.
%
%   Default: Elt 8 (1stPropStart, sphere) -> Elt 9 (CorMask, plane) ->
%   Elt 10 (1stPropEnd, sphere on the divergent side).  PROPER recipe:
%   prop_lens(f) + prop_propagate(f) to land at focus, then a SECOND
%   prop_propagate(f) past the focus to Elt 10.  Output sampling
%   rebins back to pupil scale automatically (PROPER's outside-beam
%   Fresnel kernel).
arguments
    opts.rx_filename    (1,:) char   = 'Rx_Coro.in'
    opts.src_elt        (1,1) double = 8       % 1stPropStart (sphere)
    opts.focus_elt      (1,1) double = 9       % CorMask (plane)
    opts.detector_elt   (1,1) double = 10      % 1stPropEnd (sphere)
    opts.macos_size     (1,1) double = 1024
    opts.wavelength_m   (1,1) double = 8.5e-7
    opts.focal_length_m (1,1) double = 0.774
end
g = opts;
end
