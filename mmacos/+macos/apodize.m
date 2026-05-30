function apodize(srf, mask)
%MACOS.APODIZE  Multiply WFElt(:,:,iEltToiWF(srf)) by a real NxN mask.
%   Companion to PROPER's prop_multiply -- pass the same numpy/MATLAB
%   array to both engines and the apodisation is bit-identical.
%
%   Caller must already have propagated to SRF (e.g. via macos.trace
%   or macos.complex_field with default reset_trace=true).  Subsequent
%   intensity/complex_field calls must use 'reset_trace', false to see
%   the apodised wavefront -- default reset_trace=true issues MODIFY
%   and wipes the mask.
%
%   Limitation: modifies only WFElt, NOT the geometric ray channel.
%   For hard-edged apertures in a real chain, use prescription-driven
%   ApType= or Element=Obscuring instead.
arguments
    srf  (1,1) double {mustBeInteger, mustBePositive}
    mask (:,:) double
end
if size(mask,1) ~= size(mask,2)
    error('macos:apodize:notSquare', ...
        'mask must be square (got %d x %d)', size(mask,1), size(mask,2));
end
mmacos('apodize', srf, mask);
end
