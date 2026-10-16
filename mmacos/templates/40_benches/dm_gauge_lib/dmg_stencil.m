function stn = dmg_stencil(map_dm, xg, tax, tay, pitch, hw)
%DMG_STENCIL  Kernel stencil: a DM-frame map sampled at actuator-pitch
%   offsets about the poke-A site.  MAP_DM is either the measured
%   response in the DM frame divided by the poke (ZWFS: measured kernel)
%   or the unit influence map (IFO: true kernel).  HW = stencil
%   half-width in actuators (6 across both campaigns).
%   Extracted verbatim from zwfs_s3/tg96_s3 @ 10cf593.
[soff, toff] = meshgrid(-hw:hw, -hw:hw);
stn = interp2(xg, xg.', map_dm, tax + soff*pitch, tay + toff*pitch, ...
              'linear', 0);
end
