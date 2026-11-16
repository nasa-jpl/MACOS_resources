function s = get_src_csys()
%MACOS.GET_SRC_CSYS  Source coordinate-frame axes.
%   s = macos.get_src_csys() returns a struct:
%       .xDir  3×1   source frame x-axis (unit vector)
%       .yDir  3×1   source frame y-axis
%       .zDir  3×1   source frame z-axis (chief-ray / propagation direction)
%
%   See also: macos.get_src_fov, macos.get_src_sampling.
[xDir, yDir, zDir] = mmacos('get_src_csys');
s.xDir = xDir(:);
s.yDir = yDir(:);
s.zDir = zDir(:);
end
