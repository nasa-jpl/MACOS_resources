function set_elt_csys(srf, xDir, yDir, zDir, opts)
%MACOS.SET_ELT_CSYS  Define the element output local coordinate frame (TElt).
%   macos.set_elt_csys(SRF, XDIR, YDIR, ZDIR) sets the local output coordinate
%   system (the TElt frame reported by TELT-referenced spots and output
%   coordinates) from three 3-vector axes (orthonormalised by the engine).
%   macos.set_elt_csys(..., 'update', true) makes TElt track element
%   perturbations (default false = frame stays fixed).
%
%   This is the element OUTPUT frame (nECoord/TElt), distinct from the surface
%   figure frame set by macos.set_elt_srf_csys.  See also: macos.get_elt_csys,
%   macos.rm_elt_csys.
arguments
    srf         (1,1) double {mustBeInteger, mustBePositive}
    xDir        (3,1) double
    yDir        (3,1) double
    zDir        (3,1) double
    opts.update (1,1) logical = false
end
mmacos('elt_csys_set', srf, xDir, yDir, zDir, double(opts.update), 1);
end
