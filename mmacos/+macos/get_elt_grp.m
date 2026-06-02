function members = get_elt_grp(iElt)
%MACOS.GET_ELT_GRP  Read the EltGrp member list on a reference element.
%   members = macos.get_elt_grp(IELT) returns a column vector of
%   1-based element ids comprising IELT's EltGrp.  Empty column
%   vector if no group is installed on IELT.
%
%   Companion to macos.set_elt_grp / macos.del_elt_grp / macos.prb_grp.
arguments
    iElt (1,1) double {mustBeInteger, mustBePositive}
end
maxGrpSize = mmacos('elt_grp_max_all');
maxGrpSize = max(double(maxGrpSize), 1);
[jEltGrp, nEltGrp] = mmacos('elt_grp_get', iElt, 1, maxGrpSize);
n = double(nEltGrp);
if numel(n) > 1, n = n(1); end
if n <= 0
    members = zeros(0, 1);
else
    members = double(jEltGrp(1:n));
    members = members(:);
end
end
