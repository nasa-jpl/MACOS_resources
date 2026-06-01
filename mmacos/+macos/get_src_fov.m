function fov = get_src_fov()
%MACOS.GET_SRC_FOV  Read the source field-of-view state.
%   fov = macos.get_src_fov() returns a struct with fields:
%       .zSrc        scalar  — source distance (BaseUnits)
%       .src_pos     3×1     — ChfRayPos
%       .src_dir     3×1     — ChfRayDir (unit vector along chief ray)
%       .src_finite  logical — true for finite point source, false for
%                              infinite (collimated)
%
%   The (src_pos, src_dir, zSrc) triple is what set_src_fov writes.
%
%   See also: macos.set_src_fov.
[zSrc, SrcPos, SrcDir, IsPtSrc] = mmacos('get_src_fov');
fov.zSrc       = double(zSrc);
fov.src_pos    = SrcPos(:);
fov.src_dir    = SrcDir(:);
fov.src_finite = logical(IsPtSrc);
end
