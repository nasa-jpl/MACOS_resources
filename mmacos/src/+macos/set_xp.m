function set_xp(vpt, psi, rad)
%MACOS.SET_XP  Set exit-pupil geometry (writes element nElt-1).
%   macos.set_xp(VPT, PSI, RAD) writes the exit-pupil vertex, normal,
%   and reference-sphere radius directly (VptElt/PsiElt/KrElt at nElt-1).
%   VPT and RAD are in BaseUnits, the global frame.
arguments
    vpt (3,1) double
    psi (3,1) double
    rad (1,1) double
end
mmacos('xp_set', vpt, psi, rad);
end
