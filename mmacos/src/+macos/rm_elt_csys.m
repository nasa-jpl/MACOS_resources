function rm_elt_csys(srfs)
%MACOS.RM_ELT_CSYS  Remove the element output local coordinate frame (TElt).
%   macos.rm_elt_csys(SRFS) clears any local output coordinate system on the
%   given elements (resets TElt to identity, nECoord=-6), so their output
%   reverts to the global/beam frame.  See also: macos.set_elt_csys,
%   macos.get_elt_csys.
arguments
    srfs (:,1) double {mustBeInteger, mustBePositive}
end
mmacos('elt_csys_rm', srfs, numel(srfs));
end
