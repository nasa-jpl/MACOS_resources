function set_elt_grp(iElt, members)
%MACOS.SET_ELT_GRP  Write EltGrp member list on a reference element.
%   macos.set_elt_grp(IELT, MEMBERS) replaces IELT's EltGrp array with
%   the given member element ids.  IELT itself need not be one of the
%   members (it can be a hidden "group anchor"), but conventionally
%   it is one.  Used by macos.prb_grp to identify which elements move
%   together when GPERTURB is invoked at IELT.
arguments
    iElt    (1,1) double {mustBeInteger, mustBePositive}
    members (:,1) double {mustBeInteger, mustBePositive}
end
n = numel(members);
mmacos('elt_grp_set', iElt, members, n);
end
