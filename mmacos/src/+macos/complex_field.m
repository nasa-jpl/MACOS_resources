function cf = complex_field(srf, opts)
%MACOS.COMPLEX_FIELD  WFElt at element SRF (N x N complex double).
%   cf = macos.complex_field(SRF) propagates to SRF and returns the
%   complex wavefront on the diffraction grid.  |cf|^2 matches
%   macos.intensity(SRF) to numerical precision.
%
%   Name-value pairs:
%     'reset_trace' true (default) — see macos.intensity note.
arguments
    srf              (1,1) double {mustBeInteger, mustBePositive}
    opts.reset_trace (1,1) logical = true
end
cf = mmacos('complex_field', srf, double(opts.reset_trace));
end
