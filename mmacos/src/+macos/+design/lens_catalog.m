function [L, info] = lens_catalog(part, opts)
%MACOS.DESIGN.LENS_CATALOG  A vendor lens from a prescription table, as a macos.design.Lens.
%   L = macos.design.lens_catalog('PAC095AR.14')
%   [L, info] = macos.design.lens_catalog(part, 'file', 'my_parts.txt')
%   parts = macos.design.lens_catalog()          the part names in the table
%
%   The table (default: lens_catalog.txt beside this function; copy it and
%   add your parts) holds one PART line per lens -- radii, centre
%   thicknesses, glasses, diameter, and the vendor's quoted EFL / BFL /
%   design wavelength -- and GLASS lines for any glass the engine's table
%   (macos_f90/macos_glass_list.txt) does not carry, as Sellmeier rows.  The
%   file's own header documents the format and the sign convention.
%
%   L is a Lens with D = the part diameter, its glasses resolved (a table
%   glass becomes a struct row macos.design.glass accepts, so the paraxial
%   solve and the emitted deck's GlassCoef= use the same six numbers), and
%   L.source = the line's provenance.  info carries the vendor's quoted
%   numbers (.efl .bfl .lambda_nm .vendor .source) and the glass rows used
%   (.glass_rows, with each one's published nd).
%
%   See also macos.design.Lens, macos.design.glass, macos.design.Bench.add_optic.
arguments
    part (1,:) char = ''
    opts.file (1,:) char = ''
end
f = opts.file;
if isempty(f), f = fullfile(fileparts(mfilename('fullpath')), 'lens_catalog.txt'); end
assert(isfile(f), 'lens_catalog: table %s not found', f);
txt = splitlines(fileread(f));
parts = struct('part', {}, 'f', {});  G = struct('name', {}, 'B', {}, 'C', {}, 'nd', {}, 'source', {});
for i = 1:numel(txt)
    s = strtrim(txt{i});
    if isempty(s) || s(1) == '#', continue; end
    fld = strtrim(strsplit(s, '|'));
    switch lower(fld{1})
        case 'glass'
            assert(numel(fld) == 5, 'lens_catalog: %s line %d: a glass line has 5 fields', f, i);
            v = sscanf(fld{3}, '%f').';
            assert(numel(v) == 6, 'lens_catalog: %s line %d: six Sellmeier coefficients', f, i);
            G(end+1) = struct('name', fld{2}, 'B', v(1:3), 'C', v(4:6), 'nd', str2double(fld{4}), 'source', fld{5}); %#ok<AGROW>
        case 'part'
            assert(numel(fld) == 11, 'lens_catalog: %s line %d: a part line has 11 fields', f, i);
            parts(end+1) = struct('part', fld{2}, 'f', {fld}); %#ok<AGROW>
        otherwise
            error('lens_catalog: %s line %d: unknown record ''%s''', f, i, fld{1});
    end
end
if isempty(part), L = {parts.part}; info = struct('glass_rows', G, 'file', f); return; end
k = find(strcmpi({parts.part}, part), 1);
assert(~isempty(k), 'lens_catalog: part %s is not in %s (parts: %s)', part, f, strjoin({parts.part}, ', '));
fld = parts(k).f;
R = str2double(strsplit(fld{4}, ';'));
t = str2double(strsplit(fld{5}, ';'));
gn = strtrim(strsplit(fld{6}, ';'));
glass = cell(1, numel(gn));  used = G([]);
for j = 1:numel(gn)
    ig = find(strcmpi({G.name}, gn{j}), 1);
    if isempty(ig)
        glass{j} = gn{j};                                  % resolved from the engine table
    else
        glass{j} = struct('name', G(ig).name, 'B', G(ig).B, 'C', G(ig).C);
        used(end+1) = G(ig); %#ok<AGROW>
    end
end
D = str2double(fld{7});
L = macos.design.Lens('R', R, 't', t, 'glass', glass, 'D', D, 'name', fld{2}, ...
    'source', sprintf('%s %s: %s', fld{3}, fld{2}, fld{11}));
info = struct('part', fld{2}, 'vendor', fld{3}, 'efl', str2double(fld{8}), 'bfl', str2double(fld{9}), ...
    'lambda_nm', str2double(fld{10}), 'source', fld{11}, 'glass_rows', used, 'file', f);
end
