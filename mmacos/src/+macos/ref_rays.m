function r = ref_rays(kind)
%MACOS.REF_RAYS  Find the reference rays of the source grid -- the CLI's RRAY / PRAY.
%   r = macos.ref_rays('rray')   the chief ray + the four rays nearest the aperture
%                                edge at +x, +y, -x, -y (5 slots)
%   r = macos.ref_rays('pray')   the chief ray + the eight rays nearest 0.9 of the
%                                aperture at 45-degree steps from +x (9 slots)
%   Each candidate must pass the system.  Fields (slot 1 = the chief ray):
%     r.id      1 x n  the ray's index in the source grid (macos.ray(r.id(k)));
%                      -1 = no ray found (slot 1: the chief ray was lost)
%     r.offset  2 x n  its XY offset from the chief ray in the source grid frame
%                      (BaseUnits; 0 where r.id(k) is -1)
%   These are the rays the CLI's TPR traces.  The trace is marked modified.
%   api ref_rays (BRIEF_to_bindings B.2, 2026-10-17).
%
%   See also: macos.ray, macos.srt.
arguments
    kind (1,:) char {mustBeMember(kind, {'rray', 'pray'})}
end
k = 1 + strcmp(kind, 'pray');
n = 5 + 4*(k - 1);
[id, off] = mmacos('ref_rays', double(k), zeros(9, 1), zeros(2, 9), double(9));
macos.modify();
r = struct('id', reshape(id(1:n), 1, []), 'offset', off(:, 1:n));
end
