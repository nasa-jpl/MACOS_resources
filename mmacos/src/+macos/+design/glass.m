function [n, S] = glass(name, lambda_nm, opts)
%MACOS.DESIGN.GLASS  Refractive index of a catalog glass, from the ENGINE's own table.
%   n = macos.design.glass('N-BK7', 587.56)   index at 587.56 nm
%   [n, S] = ...  also returns the catalog row:
%     S.name   the catalog spelling ('N-BK7'); 'air' for air
%     S.B      1x3 Sellmeier B coefficients
%     S.C      1x3 Sellmeier C coefficients (um^2)
%     S.file   the table file the row was read from
%   n = macos.design.glass(1.5, ...) returns a NUMERIC index as given (a fixed,
%   non-dispersive medium; S.B and S.C empty).  'air' (any case) is 1.
%   n = macos.design.glass(struct('name','N-F2','B',[..],'C',[..]), ...) uses
%   that Sellmeier row as given: a glass the engine table does not carry
%   (macos.design.lens_catalog supplies these; the emitters write the row as
%   GlassCoef=, so the engine traces it exactly).
%   LAMBDA_NM may be a vector; n then has its shape.
%
%   The table is macos_f90/macos_glass_list.txt, the file the engine reads at
%   start-up (and compiles in as glass_builtin.f90; tools/gen_glass_builtin.py).
%   Each row is NAME B1 B2 B3 C1 C2 C3 and the index is Schott's Sellmeier
%       n^2 - 1 = sum_i B_i L2 / (L2 - C_i),   L2 = lambda_um^2,
%   the formula tracesub.F getIndex evaluates for GlassType= Sellmeier.  A deck
%   that writes these six numbers as GlassCoef= (macos.design.Bench.add_optic,
%   macos.design.lens_emit) traces at the index this function returns, to
%   round-off, at every wavelength: the paraxial model and the engine read ONE
%   table.  (Generalized from challenges/iodc2010/iodc2010_glass, which carried
%   the N-BK7 / N-SF6 rows by hand.)
%
%   Name matching is EXACT, case-insensitive ('n-bk7' finds N-BK7; 'BK' finds
%   nothing).  The engine's own GlassElt= lookup is a case-insensitive PREFIX
%   match that takes the first row in file order, so a deck should always carry
%   GlassCoef= after GlassElt= (the emitters here do; the coefficients written
%   last are the ones traced).
%
%   The file is found at opts.file, else $MACOS_HOME/macos_glass_list.txt, else
%   the macos_f90 directory beside this MACOS_resources checkout.  It is read
%   once per MATLAB session (cached per path).
%
%   See also macos.design.Lens, macos.design.Bench.
arguments
    name
    lambda_nm double {mustBePositive}
    opts.file (1,:) char = ''
end
if isstruct(name)
    assert(isscalar(name) && all(isfield(name, {'name', 'B', 'C'})) && numel(name.B) == 3 && numel(name.C) == 3, ...
        'macos.design.glass: a glass row is struct(''name'', .., ''B'', [1x3], ''C'', [1x3])');
    B = name.B(:).';  C = name.C(:).';
    n = sellmeier_(B, C, lambda_nm);
    S = struct('name', char(name.name), 'B', B, 'C', C, 'file', '(row given)');
    return
end
if isnumeric(name)
    assert(isscalar(name) && name >= 1, 'macos.design.glass: a numeric index must be a scalar >= 1');
    n = name * ones(size(lambda_nm));
    S = struct('name', sprintf('n=%.10g', name), 'B', [], 'C', [], 'file', '');
    return
end
assert(ischar(name) || isstring(name), 'macos.design.glass: NAME must be a glass name or a numeric index');
name = char(name);
if strcmpi(name, 'air')
    n = ones(size(lambda_nm));
    S = struct('name', 'air', 'B', [], 'C', [], 'file', '');
    return
end
T = table_(opts.file);
i = find(strcmpi(T.names, name), 1);
if isempty(i)
    error('macos:design:glass:unknown', ...
        'macos.design.glass: %s is not in %s (%d glasses; names are exact, e.g. N-BK7, SF6, F2, CaF2)', ...
        name, T.file, numel(T.names));
end
B = T.coef(i, 1:3);  C = T.coef(i, 4:6);
n = sellmeier_(B, C, lambda_nm);
S = struct('name', T.names{i}, 'B', B, 'C', C, 'file', T.file);
end

function n = sellmeier_(B, C, lambda_nm)
L2 = (lambda_nm * 1e-3).^2;
n2 = 1;
for k = 1:3, n2 = n2 + B(k) * L2 ./ (L2 - C(k)); end
n = sqrt(n2);
end

% -------------------------------------------------------------------------
function T = table_(file)
persistent cache
if isempty(file), file = default_file_(); end
if isempty(cache), cache = containers.Map('KeyType', 'char', 'ValueType', 'any'); end
if isKey(cache, file), T = cache(file); return; end
assert(isfile(file), 'macos.design.glass: glass table %s not found (set MACOS_HOME to the macos_f90 directory)', file);
fid = fopen(file, 'r');  c = onCleanup(@() fclose(fid));
C = textscan(fid, '%s %f %f %f %f %f %f %*[^\n]');
names = C{1};
coef  = [C{2:7}];
ok = ~cellfun(@isempty, names) & all(isfinite(coef), 2);
T = struct('names', {names(ok)}, 'coef', coef(ok, :), 'file', file);
cache(file) = T;
end

function f = default_file_()
h = getenv('MACOS_HOME');
if ~isempty(h) && isfile(fullfile(h, 'macos_glass_list.txt'))
    f = fullfile(h, 'macos_glass_list.txt');
    return
end
% +design -> +macos -> src -> mmacos -> MACOS_resources -> its parent
here = fileparts(mfilename('fullpath'));
res  = fileparts(fileparts(fileparts(fileparts(here))));
f = fullfile(fileparts(res), 'macos', 'macos_f90', 'macos_glass_list.txt');
end
