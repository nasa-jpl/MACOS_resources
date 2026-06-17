function s = trace(srf)
%MACOS.TRACE  Run a full ray trace through element SRF.
%   s = macos.trace(SRF) returns a struct with fields:
%       .nRays   number of successful rays
%       .rmsWFE  RMS wavefront error (WaveUnits)
%   SRF defaults to num_elt() (last element / image plane).
%
%   Required before opd / intensity / complex_field / dx_at queries.
if nargin < 1
    srf = macos.num_elt();
end
validateattributes(srf, {'numeric'}, {'scalar','integer','positive'});
[s.nRays, s.rmsWFE] = mmacos('trace_rays', srf);
end
