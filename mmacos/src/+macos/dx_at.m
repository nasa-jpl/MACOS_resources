function dx = dx_at(srf, unit)
%MACOS.DX_AT  Per-element diffraction-grid pitch.
%   dx = macos.dx_at(SRF) returns the pitch at element SRF in SI metres.
%   dx = macos.dx_at(SRF, UNIT) returns it in the requested UNIT:
%       'm'        SI metres (default).
%       'mm'       millimetres.
%       'cm'       centimetres.
%       'um'       micrometres.
%       'native'   the prescription's BaseUnits value (dx_metres / CBM).
arguments
    srf  (1,1) double {mustBeInteger, mustBePositive}
    unit (1,:) char {mustBeMember(unit, ...
        {'m','mm','cm','um','native'})} = 'm'
end
dx_m = mmacos('dx_at', srf);
switch unit
    case 'm',      dx = dx_m;
    case 'mm',     dx = dx_m * 1e3;
    case 'cm',     dx = dx_m * 1e2;
    case 'um',     dx = dx_m * 1e6;
    case 'native'
        c = mmacos('base_unit_to_metres');
        if c == 0.0
            error('macos:dx_at:noCBM', 'CBM unavailable.');
        end
        dx = dx_m / c;
end
end
