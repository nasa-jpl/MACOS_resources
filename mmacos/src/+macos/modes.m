function md = modes(opts)
%MACOS.MODES  Query or set the SESSION modes as the engine holds them.
%   md = macos.modes() returns a struct with logical fields .pol (polarization),
%   .vecdif (vector diffraction), .centroid (the FEX/SXP centroid axis), .lneg
%   (negative ray lengths allowed; default TRUE), .cir (colour image rendering;
%   always false in the bindings' engine), .obs (the OBS option in force:
%   'all', 'positive' (default) or 'negative'), and .in_force, a cellstr of the
%   modes NOT at their defaults, named as the CLI's LOAD notice names them
%   ('POLARIZATION', 'VECTOR'/'SCALAR', 'CENTROID', 'NOLNEG', 'CIR',
%   'OBS ALL'/'OBS NEGATIVE').
%
%   md = macos.modes('centroid', TF) / macos.modes('lneg', TF) SETS the two
%   modes that have no setter of their own (api modes_set, Dave 2026-10-28):
%   'centroid' true = CENTROID, false = CHIEFRAY; 'lneg' true = LNEG, false =
%   NOLNEG.  A name left out is left as it is.  Needs macos.init, not a loaded
%   Rx: a mode set before load_rx stays.  Polarization has macos.polarization,
%   the OBS option macos.obs.  Returns the modes after the change.
%
%   These modes SURVIVE macos.load_rx (Dave 2026-10-11, PLAN_CONSOLIDATION 4
%   item 2); load_rx warns (id mmacos:modesInForce) when any is in force.
%
%   Raw equivalents: mmacos('modes_get'), mmacos('modes_set', C, L) with
%   1 = on, 0 = off, -1 = leave; mmacos('obs_opt_get').
%   See also: macos.load_rx, macos.polarization, macos.obs, macos.fex.
arguments
    opts.centroid {mustBeScalarOrEmpty, mustBeNumericOrLogical} = []
    opts.lneg     {mustBeScalarOrEmpty, mustBeNumericOrLogical} = []
end
if ~isempty(opts.centroid) || ~isempty(opts.lneg)
    mmacos('modes_set', code_(opts.centroid), code_(opts.lneg));
end
[p, v, c, l, r] = mmacos('modes_get');
o = mmacos('obs_opt_get');
names = {'all', 'positive', 'negative'};
md = struct('pol', p ~= 0, 'vecdif', v ~= 0, 'centroid', c ~= 0, ...
            'lneg', l ~= 0, 'cir', r ~= 0, 'obs', names{o + 1}, 'in_force', {{}});
f = {};
if md.pol
    f{end+1} = 'POLARIZATION';
    if md.vecdif, f{end+1} = 'VECTOR'; else, f{end+1} = 'SCALAR'; end
end
if md.centroid, f{end+1} = 'CENTROID'; end
if ~md.lneg,    f{end+1} = 'NOLNEG';   end
if md.cir,      f{end+1} = 'CIR';      end
if o == 0,      f{end+1} = 'OBS ALL';      end
if o == 2,      f{end+1} = 'OBS NEGATIVE'; end
md.in_force = f;
end

function c = code_(v)
if isempty(v), c = -1; else, c = double(logical(v)); end
end
