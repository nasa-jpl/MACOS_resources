function prb_grp(iElt, prb, ifGlobal)
%MACOS.PRB_GRP  GPERTURB: rigid-body perturb a group as a single unit.
%   macos.prb_grp(IELT, PRB) applies the 6-DOF rigid-body
%   perturbation vector PRB to the group whose reference element
%   is IELT (the element whose EltGrp array enumerates the members).
%
%   Inputs:
%     IELT     scalar (or 1xK vector) reference element id(s).  Each
%              column of PRB / element of IFGLOBAL is one group.
%     PRB      6xK matrix; column k = [Rx; Ry; Rz; Tx; Ty; Tz]_k.
%              Rotations in radians, translations in the Rx's
%              BaseUnits (NOT SI metres -- this matches macos's raw
%              CPERTURB_GRP_DVR signature, the GPERTURB legacy).
%              For SI metres input, divide the translation rows by
%              macos.cbm() first.
%     IFGLOBAL 1xK logical/numeric.  true (1) -> rotation axes in
%              GLOBAL frame; false (0) -> rotation axes in IELT's
%              local TElt frame.  Pivot is always RptElt(IELT) for
%              all members.
%
%   Used by macos.channels.GroupedRigidBodyChannel.  For per-element
%   rigid-body perturbation, see macos.perturb.
arguments
    iElt     (:,1) double {mustBeInteger, mustBePositive}
    prb      (6,:) double
    ifGlobal (:,1) double = ones(size(iElt))
end
n = numel(iElt);
if size(prb, 2) ~= n
    error('macos:prb_grp:shape', ...
        'PRB must be 6 x %d (got %d x %d)', n, size(prb,1), size(prb,2));
end
if numel(ifGlobal) ~= n
    error('macos:prb_grp:shape', ...
        'ifGlobal must have %d entries (got %d)', n, numel(ifGlobal));
end
mmacos('prb_elt_grp', iElt, prb, ifGlobal, n);
end
