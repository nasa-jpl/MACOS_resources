function g = coro_nfprop(opts)
%CORO_NFPROP  Geometry parameters for Coro NF prop comparisons.
%   g = coro_nfprop()                          % Phase 2 default (Elt 2->3)
%   g = coro_nfprop('src_elt', 5, 'detector_elt', 6)            % Phase 3a
%   g = coro_nfprop('rx_filename', 'Rx_Coro_noLyot.in', ...
%                   'src_elt', 13, 'detector_elt', 14)          % Phase 4b
%
%   Mirror of pymacos
%   tests/proper_compare/geometries/coro_nfprop.py:CoroNFprop.
%   Both engines start from the SAME wavefront at src_elt (macos's
%   complex_field) so the comparison isolates the NF kernel.
%
%   macos's interactive INT diagnostic on the default geometry:
%     Wavelength = 850 nm
%     Elt 2 plane: dx1 = 3.3382e-1 mm
%     Elt 3 plane: dx2 = 3.3382e-1 mm  (NF prop preserves pitch)
%   The dx is queried at runtime via macos.dx_at — relying on the
%   displayed 5-sig-fig value would cap agreement at ~1e-5.
arguments
    opts.rx_filename   (1,:) char   = 'Rx_Coro.in'
    opts.src_elt       (1,1) double = 2          % Prop_1_start
    opts.detector_elt  (1,1) double = 3          % Prop_1_end
    opts.macos_size    (1,1) double = 1024       % 2x prescription nGridpts=511
    opts.wavelength_m  (1,1) double = 8.5e-7     % 850 nm
    opts.propagation_m (1,1) double = 0.774      % 774 mm
end
g = opts;
end
