function perturb_many(srf_vec, prb, is_global)
%MACOS.PERTURB_MANY  Bulk rigid-body PERTURB of multiple elements.
%   macos.perturb_many(SRFS, PRB, IS_GLOBAL) applies the array form
%   (api prb_elt).  Inputs:
%     SRFS       length-N vector of element ids.
%     PRB        6xN matrix; column k = [Rx Ry Rz Tx Ty Tz]' for SRFS(k).
%                Rotations in radians, translations in the Rx's
%                BaseUnits (NOT SI metres -- macos's PERTURB legacy
%                signature).  For SI metres, divide translation by
%                macos.cbm() first, or use macos.perturb in a loop.
%     IS_GLOBAL  length-N logical/numeric; 1=global frame, 0=local.
arguments
    srf_vec   (:,1) double {mustBeInteger}
    prb       (6,:) double
    is_global (:,1) double
end
n = numel(srf_vec);
if size(prb, 2) ~= n
    error('macos:perturb_many:shape', ...
        'PRB must be 6 x %d (got %d x %d)', n, size(prb,1), size(prb,2));
end
if numel(is_global) ~= n
    error('macos:perturb_many:shape', ...
        'IS_GLOBAL must have %d entries (got %d)', n, numel(is_global));
end
mmacos('prb_elt', srf_vec, prb, is_global);
end
