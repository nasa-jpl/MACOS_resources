function s = get_elt_csys(srfs)
%MACOS.GET_ELT_CSYS  Local coordinate-system matrices for elements.
%   s = macos.get_elt_csys(SRFS) returns a struct with fields:
%       .csys      6×6×N double  — TElt frame matrix for each element
%       .csys_lcs  N×1 logical   — true if a local CSYS is defined
%       .csys_upd  N×1 logical   — true if the LCS updates with perturbations
%
%   SRFS is a vector of element ids (length N).  Single-element calls
%   are fine — SRFS can be a scalar.
arguments
    srfs (:,1) double {mustBeInteger, mustBePositive}
end
n = numel(srfs);
[csys, csys_lcs, csys_upd] = mmacos('elt_csys_get', srfs, n);
s.csys      = csys;
s.csys_lcs  = logical(csys_lcs(:));
s.csys_upd  = logical(csys_upd(:));
end
