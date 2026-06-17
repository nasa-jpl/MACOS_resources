function set_elt_vpt(srf, vpt)
%MACOS.SET_ELT_VPT  Set element SRF's vertex point (3-vector, BaseUnits).
%   Note: this writes VptElt directly without the bookkeeping that
%   macos.perturb performs.  For an optic with a figure error or
%   linked children, prefer macos.perturb so coordinate frames stay
%   consistent.
arguments
    srf (1,1) double {mustBeInteger, mustBePositive}
    vpt (3,1) double
end
mmacos('elt_vpt', [srf], vpt, 1, 1);
end
