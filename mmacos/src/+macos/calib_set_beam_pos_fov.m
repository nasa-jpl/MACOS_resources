function calib_set_beam_pos_fov(pos)
%MACOS.CALIB_SET_BEAM_POS_FOV  Per-field beam position targets for CALIB.
%   macos.calib_set_beam_pos_fov(POS), POS 3 x n (or n x 3): the target
%   position of the beam (see calib_set_beam 'pos') for CALIB field
%   1..n in the order of the deck's OptChfRayDir= / OptChfRayPos= list
%   (n <= 12).  Fields beyond n use the calib_set_beam 'pos' target.
%   macos.calib_set_beam_pos_fov([]) clears the table.
%   Mirrors the Rx keyword OptBeamPosFov= (one row per field, in order).
if isempty(pos)
    mmacos('calib_set_beam_pos_fov', zeros(3, 1), 0);
    return
end
P = double(pos);
if size(P, 1) ~= 3 && size(P, 2) == 3, P = P.'; end
if size(P, 1) ~= 3
    error('macos:calib_set_beam_pos_fov:shape', 'pos must be 3 x n or n x 3');
end
n = size(P, 2);
if n > 12
    error('macos:calib_set_beam_pos_fov:n', 'at most 12 fields (got %d)', n);
end
mmacos('calib_set_beam_pos_fov', P, double(n));
end
