function md = modes()
%MACOS.MODES  The five SESSION modes as the engine holds them.
%   md = macos.modes() returns a struct with logical fields .pol (polarization),
%   .vecdif (vector diffraction), .centroid (the FEX/SXP centroid axis), .lneg
%   (negative ray lengths allowed; default TRUE), .cir (colour image rendering;
%   always false in the bindings' engine), and .in_force, a cellstr of the modes
%   NOT at their defaults, named as the CLI's LOAD notice names them
%   ('POLARIZATION', 'VECTOR'/'SCALAR', 'CENTROID', 'NOLNEG', 'CIR').
%   These modes SURVIVE macos.load_rx (Dave 2026-10-11, PLAN_CONSOLIDATION 4
%   item 2); load_rx warns (id mmacos:modesInForce) when any is in force.
%
%   See also: macos.load_rx, macos.polarization.

[p, v, c, l, r] = mmacos('modes_get');
md = struct('pol', p ~= 0, 'vecdif', v ~= 0, 'centroid', c ~= 0, ...
            'lneg', l ~= 0, 'cir', r ~= 0, 'in_force', {{}});
f = {};
if md.pol
    f{end+1} = 'POLARIZATION';
    if md.vecdif, f{end+1} = 'VECTOR'; else, f{end+1} = 'SCALAR'; end
end
if md.centroid, f{end+1} = 'CENTROID'; end
if ~md.lneg,    f{end+1} = 'NOLNEG';   end
if md.cir,      f{end+1} = 'CIR';      end
md.in_force = f;
end
