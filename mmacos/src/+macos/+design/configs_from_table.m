function cfgs = configs_from_table(T, opts)
%MACOS.DESIGN.CONFIGS_FROM_TABLE  Build a configuration array from a schedule.
%   cfgs = macos.design.configs_from_table(T) turns a zoom / compensation
%   SCHEDULE -- the shape one naturally arrives in, a spreadsheet with one
%   row per configuration -- into the 'configs' struct array the
%   macos.dw_d*_multi supervisors and run_sensitivities take.
%
%   T is a table (or the path to a .csv / .xlsx, which is read with
%   readtable) whose FIRST column is the configuration NAME and whose
%   remaining columns are per-element rigid-body DOFs, named
%
%       <elt>.<DOF>      e.g.  4.Ry    25.Tz
%       <elt>_<DOF>      e.g.  4_Ry    elt25_Tz
%
%   with DOF one of Rx Ry Rz (radians) or Tx Ty Tz (SI metres -- the
%   macos.perturb convention; the engine converts to BaseUnits).  A row
%   whose DOFs are all zero is a legal configuration with an empty setter
%   list: the NOMINAL state, and a useful first row.
%
%   Reading a schedule with '.' in the column names needs
%       readtable(file, 'VariableNamingRule', 'preserve')
%   which this function does for you when handed a filename.
%
%   OPTIONS
%     'frame'   'local' (default) | 'global' -- the frame the DOFs are
%               expressed in, passed straight to macos.perturb.
%     'prefix'  string prepended to every generated name (default '').
%     'tile'    'auto' (default) | 'none' | an Nc x 2 [row col] matrix.
%               'auto' gives each configuration an outer TILE POSITION
%               when the schedule has exactly two DOF columns, by the
%               SAME rule the field set uses: tile column from the rank
%               of the first DOF's value among its sorted unique values,
%               tile row from the second's.  A five-state centre + four
%               corners schedule therefore lands on the corners and
%               centre of a 3x3 grid, and the supervisors lay each
%               configuration's whole field canvas out there -- the zoom
%               grid reads exactly as the field grid does inside one
%               cell.  'none' leaves them untiled (left to right).
%               See macos.config_canvas.
%
%   ONE PERTURB PER (ELEMENT, KIND), NOT PER DOF, AND NEVER MIXED
%   ------------------------------------------------------------
%   Each element's rotations are emitted as ONE perturb and its
%   translations as a SECOND, separate one.  Both parts are deliberate:
%     * combining the three rotations into one call is exact -- macos's
%       Qform is the Rodrigues rotation about the combined axis, so the
%       restore's negated call is its exact inverse;
%     * a perturb carrying a rotation AND a translation together in the
%       LOCAL frame inverts its translation through the ROTATED element
%       frame, leaving O(|theta|*|del|) behind.  Splitting them sidesteps
%       that by construction, which is why this helper never emits a
%       mixed entry.  See private/config_axis.m.
%
%   EXAMPLE -- the zoom_5x5 fixture's FSM schedule
%       t = 1.45444e-4;                 % 0.5 arcmin
%       T = table(["z0";"zUL";"zUR";"zLL";"zLR"], ...
%                 [0; -t; +t; -t; +t], [0; +t; +t; -t; -t], ...
%                 'VariableNames', {'name', '25.Rx', '25.Ry'});
%       cfgs = macos.design.configs_from_table(T);
%       art  = run_sensitivities(rx, 'fov_rad', 2.90888e-4, 'configs', cfgs);
%
%   See also: macos.dw_dx_multi, run_sensitivities.

arguments
    T
    opts.frame  (1,:) char {mustBeMember(opts.frame, {'local','global'})} = 'local'
    opts.prefix (1,:) char = ''
    opts.tile = 'auto'
end

if ischar(T) || isstring(T)
    f = char(T);
    assert(isfile(f), 'macos:configs_from_table:file', ...
        'configs_from_table: %s not found', f);
    T = readtable(f, 'VariableNamingRule', 'preserve');
end
assert(istable(T), 'macos:configs_from_table:type', ...
    'configs_from_table: expected a table or a readtable-able file, got %s', ...
    class(T));
assert(width(T) >= 1, 'macos:configs_from_table:empty', ...
    'configs_from_table: the table has no columns');

vn = string(T.Properties.VariableNames);
names = T.(vn(1));
if isnumeric(names), names = string(names); end
names = string(names);

% ---- parse the DOF column headers ---------------------------------
DOFS = {'Rx','Ry','Rz','Tx','Ty','Tz'};
cols = struct('elt', {}, 'dof', {}, 'idx', {});
for q = 2:numel(vn)
    [e, d] = parse_header(vn(q));
    cols(end+1) = struct('elt', e, 'dof', d, 'idx', q); %#ok<AGROW>
    assert(any(strcmpi(DOFS, d)), 'macos:configs_from_table:dof', ...
        ['configs_from_table: column ''%s'' -- ''%s'' is not a ' ...
         'rigid-body DOF (expected one of %s)'], vn(q), d, ...
        strjoin(DOFS, ' '));
end
assert(~isempty(cols), 'macos:configs_from_table:noDofs', ...
    ['configs_from_table: no DOF columns.  Name them ''<elt>.<DOF>'' ' ...
     '(e.g. 25.Ry) after the leading name column.']);

elts = unique([cols.elt]);

% ---- one configuration per row ------------------------------------
cfgs = struct('name', {}, 'set', {}, 'tile', {});
for r = 1:height(T)
    sl = {};
    for e = elts
        rot = zeros(3,1);  tra = zeros(3,1);
        for q = find([cols.elt] == e)
            v = T{r, cols(q).idx};
            if iscell(v), v = v{1}; end
            v = double(v);
            assert(isscalar(v) && isfinite(v), ...
                'macos:configs_from_table:value', ...
                'configs_from_table: row %d, column ''%s'' is not a finite scalar', ...
                r, vn(cols(q).idx));
            k = find(strcmpi(DOFS, cols(q).dof));
            if k <= 3, rot(k) = v; else, tra(k-3) = v; end
        end
        % rotations and translations go in SEPARATE entries -- see the
        % header: a mixed local-frame perturb does not invert exactly.
        if any(rot ~= 0)
            sl{end+1} = {'perturb', e, 'rotation', rot, ...
                         'translation', zeros(3,1), 'frame', opts.frame}; %#ok<AGROW>
        end
        if any(tra ~= 0)
            sl{end+1} = {'perturb', e, 'rotation', zeros(3,1), ...
                         'translation', tra, 'frame', opts.frame}; %#ok<AGROW>
        end
    end
    c = struct();
    c.name = [opts.prefix char(strtrim(names(r)))];
    c.set  = sl;
    c.tile = [];
    cfgs(end+1) = c; %#ok<AGROW>
end

% ---- outer tile positions -----------------------------------------
tl = assign_tiles(T, cols, height(T), opts.tile);
for r = 1:numel(cfgs)
    if ~isempty(tl), cfgs(r).tile = tl(r, :); end
end
end


% ---------------------------------------------------------------------
function tl = assign_tiles(T, cols, nrow, spec)
%ASSIGN_TILES  Outer [row col] per configuration, or [].
if isnumeric(spec) && ~isempty(spec)
    assert(isequal(size(spec), [nrow 2]), 'macos:configs_from_table:tile', ...
        'configs_from_table: ''tile'' must be %d x 2', nrow);
    tl = double(spec);  return
end
tl = [];
if ~(ischar(spec) || isstring(spec)) || ~strcmpi(char(spec), 'auto'), return; end
% Exactly two DOF columns, or there is no 2-D grid to infer.
if numel(cols) ~= 2, return; end
v1 = zeros(nrow, 1);  v2 = zeros(nrow, 1);
for r = 1:nrow
    a = T{r, cols(1).idx};  if iscell(a), a = a{1}; end
    b = T{r, cols(2).idx};  if iscell(b), b = b{1}; end
    v1(r) = double(a);  v2(r) = double(b);
end
u1 = unique(v1);  u2 = unique(v2);
% rank among sorted unique values -- the same rule make_grid_field_set
% uses for the field set, so the two grids read the same way round
[~, c_idx] = ismember(v1, u1);
[~, r_idx] = ismember(v2, u2);
tl = [r_idx - 1, c_idx - 1];
end


% ---------------------------------------------------------------------
function [e, d] = parse_header(h)
% '<elt>.<DOF>' | '<elt>_<DOF>' | 'elt<elt>_<DOF>' | 'e<elt>.<DOF>'
h = char(h);
tok = regexp(h, '^\s*(?:elt|e)?\s*(\d+)\s*[._]\s*([A-Za-z]{2})\s*$', ...
             'tokens', 'once');
assert(~isempty(tok), 'macos:configs_from_table:header', ...
    ['configs_from_table: cannot parse column header ''%s''.  Use ' ...
     '''<elt>.<DOF>'' (e.g. 25.Ry) or ''<elt>_<DOF>'' (e.g. 25_Ry).'], h);
e = str2double(tok{1});
d = tok{2};
end
