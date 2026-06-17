function I = intensity(srf, opts)
%MACOS.INTENSITY  |WFElt|^2 at element SRF (the SMACOS INT command).
%   I = macos.intensity(SRF) runs INT at SRF and returns the
%   (mdttl x mdttl) intensity buffer.
%
%   Name-value pairs:
%     'reset_trace' true (default) — issue MODIFY + re-trace before
%                   running INT.  Set false when chaining calls that
%                   share the existing trace state (e.g. after a
%                   macos.apodize that should NOT be wiped).
arguments
    srf              (1,1) double {mustBeInteger, mustBePositive}
    opts.reset_trace (1,1) logical = true
end
I = mmacos('intensity', srf, double(opts.reset_trace));
end
