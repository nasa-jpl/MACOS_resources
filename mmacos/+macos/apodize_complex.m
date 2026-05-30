function apodize_complex(srf, mask)
%MACOS.APODIZE_COMPLEX  Multiply WFElt(:,:, iEltToiWF(srf)) by a complex mask.
%   Companion to macos.apodize for the complex case — used by the DM
%   phase-imprint tests where the mask is `exp(i * 2*pi * OPD / lambda)`
%   (unit-magnitude, pure phase).
%
%   Same caveat as macos.apodize:
%     - Caller must already have propagated to SRF.
%     - Subsequent intensity/complex_field calls need
%       'reset_trace', false to see the apodised wavefront — default
%       'reset_trace', true wipes the mask via MODIFY.
%
%   The underlying api takes real and imaginary arrays separately
%   (cfield_apodize_complex), so we split here.
arguments
    srf  (1,1) double {mustBeInteger, mustBePositive}
    mask (:,:) double {mustBeNumericOrLogical}   % can be real or complex
end
if size(mask, 1) ~= size(mask, 2)
    error('macos:apodize_complex:notSquare', ...
        'mask must be square (got %d x %d)', size(mask, 1), size(mask, 2));
end
N = size(mask, 1);
mask_re = real(mask);
mask_im = imag(mask);
% Codegen prhs order: MASK_RE, MASK_IM, N, iElt
mmacos('cfield_apodize_complex', mask_re, mask_im, N, srf);
end
