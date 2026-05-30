function s = elt_grating_fnd(srfs)
%MACOS.ELT_GRATING_FND  Find grating-bearing elements.
%   s = macos.elt_grating_fnd()    queries all elements.
%   s = macos.elt_grating_fnd(SRF) queries the listed elements only.
%
%   Returns a struct with fields:
%     .srfs   integer column vector of element ids that carry a grating
%     .types  integer column vector aligned with .srfs:
%               1 = reflective grating, 2 = transmissive grating.
%   Returns an empty struct when no grating is defined anywhere.
arguments
    srfs (:,1) double {mustBeInteger} = []
end
if ~macos.elt_grating_any()
    s = struct('srfs', [], 'types', []);
    return
end
if isempty(srfs)
    srfs = (1:macos.num_elt())';
end
% Cmd 'elt_srf_grating_fnd' has prhs: iElt(N), N
% and returns Grating(N) (one type id per input slot).
n = numel(srfs);
grating = mmacos('elt_srf_grating_fnd', srfs, n);
nz = grating(:) ~= 0;
s.srfs  = srfs(nz);
s.types = grating(nz);
end
