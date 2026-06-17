function stop_obj(x, y, z)
%MACOS.STOP_OBJ  Set the system aperture stop to an object-space point.
%   macos.stop_obj(X, Y, Z) places the system Stop at the global
%   coordinate (X, Y, Z) in BaseUnits.  The chief ray is then aimed
%   to pass through this point.
%
%   Use this for Rx files that don't declare an ApStop= element-id
%   in the header (e.g. FFSegDemoAll.in) when you want SXP / FEX
%   based EP follow-up to work.  Equivalent to the interactive
%   'STOP obj x,y,z' macos command.
%
%   See also: macos.stop, macos.sxp.
arguments
    x (1,1) double
    y (1,1) double
    z (1,1) double
end
mmacos('stop_obj_set', x, y, z);
end
