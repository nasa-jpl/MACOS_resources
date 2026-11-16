function del_elt_grp(iElt)
%MACOS.DEL_ELT_GRP  Remove the EltGrp from element(s) IELT.
%   macos.del_elt_grp(IELT) clears the EltGrp(0:N, iElt) array on
%   IELT (or each entry of the column vector IELT).  Companion to
%   macos.set_elt_grp.
arguments
    iElt (:,1) double {mustBeInteger, mustBePositive}
end
n = numel(iElt);
mmacos('elt_grp_del', iElt, n);
end
