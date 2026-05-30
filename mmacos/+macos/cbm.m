function c = cbm()
%MACOS.CBM  Base-units-to-metres conversion factor (CBM).
%   c = macos.cbm() returns the SI-metres value of one BaseUnit in the
%   currently loaded Rx (e.g. 1e-3 for an Rx in mm).  Used internally
%   when callers want to feed/read distances in SI metres while the
%   prescription is in its native units.
c = mmacos('base_unit_to_metres');
end
