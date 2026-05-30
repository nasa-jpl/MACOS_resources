function s = sys_units()
%MACOS.SYS_UNITS  Return a struct of unit identifiers for the loaded Rx.
%   s = macos.sys_units() returns:
%       .base_unit_id  numeric id of the prescription's BaseUnits
%       .wave_unit_id  numeric id of the WaveUnits
%   See macos's BaseUnits / WaveUnits enums for the mapping.
[base_id, wave_id] = mmacos('sys_units');
s.base_unit_id = base_id;
s.wave_unit_id = wave_id;
end
