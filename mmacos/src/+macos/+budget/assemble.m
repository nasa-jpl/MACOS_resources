function M = assemble(sens, varargin)
%ASSEMBLE  Load sensitivity Jacobians into one metric-agnostic model.
%   M = macos.budget.assemble(SENS) ingests a sensitivities artifact and
%   concatenates every available channel (dwdx / dwdz / dwdsurf / dwdgrid)
%   into a single Jacobian J with a unified per-column metadata table, the
%   Gram matrix, and the kept-row mask -- the substrate the budget kernels
%   (macos.budget.cov_budget / mc_budget) run on.
%
%   SENS may be:
%     - a path to a <name>_sens.mat (nested ox/oz/og/os, from
%       mmacos/design/runners/run_sensitivities.m), OR
%     - a struct with fields ox/oz/og/os (any may be empty), OR
%     - a single dw_d*_multi output struct (one channel), OR
%     - a flat save_dw_flat .mat path (channel-named matrix + channel_names;
%       metadata recovered by parsing channel_names).
%
%   Options:
%     'metric'   'opd' (default; uses dwdxall) | 'spot' (uses dcdx when
%                present) -- the metric kept generic for the SPOT phase.
%     'trans_per_metre'  true (default): normalize rigid-body translation
%                        columns to per-SI-metre via macos.dwdx_trans_per_metre
%                        so a sigma in metres is unit-consistent.
%
%   M fields:
%     .J      Nw0 x Nz   concatenated Jacobian (may contain NaN rows)
%     .Jk     nw  x Nz   J(keep,:)
%     .G      Nz x Nz    Jk'*Jk  (Gram; the kernel never forms Nw x Nw)
%     .keep   Nw0 x 1    logical finite-row mask
%     .nw     scalar     nnz(keep)
%     .w0     Nw0 x 1    stacked nominal metric
%     .meta   struct of per-column arrays: channel iElt dof kind mode
%             param map name  (length Nz)
%     .cbm    BaseUnit -> metre
%     .base_units, .trans_output, .units(summary), .per_field (for modes)
%
%   See also macos.budget.spec_from_table, macos.budget.suggest_groups.

p = inputParser;
p.addParameter('metric', 'opd', @(s) any(strcmpi(s, {'opd', 'spot'})));
p.addParameter('trans_per_metre', true, @islogical);
p.parse(varargin{:});

S = load_sens_(sens);                 % -> struct with .ox .oz .og .os (any [])

J = []; w0 = []; cbm = 1; base_units = ''; trans_output = '';
per_field = struct('dwdx', {{}}, 'w_nom_2d', {{}});
meta = empty_meta_();
chans = {'ox','dwdx'; 'oz','dwdz'; 'os','dwdsurf'; 'og','dwdgrid'};
for r = 1:size(chans, 1)
    o = S.(chans{r, 1});
    if isempty(o), continue; end
    % (no exist() guard: exist('macos.<fn>','file') is 0 for a package
    % function, which silently skipped this and left an mm deck's
    % translation columns per BaseUnit -- decenter/despace 1/cbm small)
    if strcmp(chans{r, 2}, 'dwdx') && p.Results.trans_per_metre
        o = macos.dwdx_trans_per_metre(o);
        trans_output = 'si';
    end
    [Jc, mc] = channel_(o, chans{r, 2}, p.Results.metric);
    if isempty(Jc), continue; end
    if isempty(J)
        J = Jc; w0 = w0_(o);
    else
        assert(size(Jc, 1) == size(J, 1), 'budget:rows', ...
            'channel %s has %d rows, expected %d (same field set?)', ...
            chans{r, 2}, size(Jc, 1), size(J, 1));
        J = [J, Jc]; %#ok<AGROW>
    end
    meta = cat_meta_(meta, mc);
    if isfield(o, 'cbm') && ~isempty(o.cbm), cbm = o.cbm; end
    if isfield(o, 'base_units') && ~isempty(o.base_units), base_units = o.base_units; end
    if strcmp(chans{r, 2}, 'dwdx') && isfield(o, 'per_field_dwdx')
        per_field.dwdx = o.per_field_dwdx;
        if isfield(o, 'per_field_w_nom_2d'), per_field.w_nom_2d = o.per_field_w_nom_2d; end
    end
end
assert(~isempty(J), 'budget:empty', 'no sensitivity channels found in input');

keep = all(isfinite(J), 2);
Jk = J(keep, :);
M = struct();
M.J = J;  M.Jk = Jk;  M.G = Jk' * Jk;  M.keep = keep;  M.nw = nnz(keep);
M.w0 = w0;  M.meta = meta;  M.cbm = cbm;  M.base_units = base_units;
M.trans_output = trans_output;  M.per_field = per_field;
M.units = struct('cbm', cbm, 'base_units', base_units, ...
    'trans_output', trans_output, 'metric', lower(p.Results.metric));
end

% =========================================================================
function S = load_sens_(sens)
S = struct('ox', [], 'oz', [], 'os', [], 'og', []);
if ischar(sens) || isstring(sens)
    d = load(char(sens));
    sens = d;
end
if isstruct(sens) && (isfield(sens, 'ox') || isfield(sens, 'oz') ...
        || isfield(sens, 'og') || isfield(sens, 'os'))
    for f = {'ox','oz','os','og'}
        if isfield(sens, f{1}), S.(f{1}) = sens.(f{1}); end
    end
    return;
end
% a single channel struct, or a flat save -> route by what it carries
if isstruct(sens) && isfield(sens, 'dwdxall')
    S.ox = sens;                      % canonical alias; classify below by meta
    % if it is really a dwdz/dsurf/dgrid single channel, its own fields
    % (mode/param/map) still drive channel_() detection
    return;
end
if isstruct(sens) && isfield(sens, 'channel_names')
    S.ox = sens; return;              % flat: parse names in channel_()
end
error('budget:input', 'unrecognized sensitivities input');
end

% -------------------------------------------------------------------------
function w0 = w0_(o)
if isfield(o, 'w0_stacked') && ~isempty(o.w0_stacked)
    w0 = o.w0_stacked(:);
elseif isfield(o, 'w_nom_vec')
    w0 = o.w_nom_vec(:);
else
    w0 = [];
end
end

% -------------------------------------------------------------------------
function [J, m] = channel_(o, chdefault, metric)
% Extract the stacked matrix + per-column metadata for one channel struct.
if strcmpi(metric, 'spot')
    if isfield(o, 'dcdx') && ~isempty(o.dcdx), J = o.dcdx;
    else, J = []; m = empty_meta_(); return; end
elseif isfield(o, 'dwdxall') && ~isempty(o.dwdxall)
    J = o.dwdxall;
elseif isfield(o, 'dwdx'), J = o.dwdx;
else, J = []; m = empty_meta_(); return; end
nz = size(J, 2);

names = getcell_(o, 'channel_names', nz);
% detect the real channel from the struct's own metadata fields
has = @(f) isfield(o, f) && ~isempty(o.(f));
m = empty_meta_();
if has('mode')
    ch = 'dwdz';
elseif has('param')
    ch = 'dwdsurf';
elseif has('map_idx')
    ch = 'dwdgrid';
elseif has('dof_idx')
    ch = 'dwdx';
else
    ch = chdefault;                  % flat save: trust the caller's tag
end
m.channel = repmat({ch}, nz, 1);
m.name    = names;
m.iElt    = getnum_(o, 'iElt', nz);
m.dof     = getnum_(o, 'dof_idx', nz);
m.mode    = getnum_(o, 'mode', nz);
m.map     = getnum_(o, 'map_idx', nz);
m.param   = getcell_(o, 'param', nz);
if has('kind')
    m.kind = cellstr(o.kind(:));
else
    m.kind = default_kind_(ch, nz);
end
% recover metadata from channel_names when the struct lacked the fields
m = parse_names_(m, ch);
end

% -------------------------------------------------------------------------
function m = parse_names_(m, ch)
% Fallback parse of 'Elt K Rx' / 'Elt K Z4' / 'Elt K Kr' / 'Source ...'.
dofmap = containers.Map({'Rx','Ry','Rz','Tx','Ty','Tz'}, {0,1,2,3,4,5});
for j = 1:numel(m.name)
    nm = m.name{j};
    if ~isnan(m.iElt(j)) || isempty(nm), continue; end
    tok = regexp(nm, '^Elt\s+(\d+)\s+(.*)$', 'tokens', 'once');
    if ~isempty(tok)
        m.iElt(j) = str2double(tok{1});
        tag = strtrim(tok{2});
        if isKey(dofmap, tag)
            m.dof(j) = dofmap(tag);
        elseif ~isempty(regexp(tag, '^[Zz]\d+$', 'once'))
            m.mode(j) = str2double(tag(2:end));
        elseif any(strcmpi(tag, {'Kr','Kc'}))
            m.param{j} = tag;
        end
    elseif ~isempty(regexp(nm, '^Source', 'once'))
        m.iElt(j) = 0;
    end
end
if strcmp(ch, 'dwdx') && all(cellfun(@isempty, m.kind))
    m.kind = default_kind_(ch, numel(m.name));
end
end

% -------------------------------------------------------------------------
function k = default_kind_(ch, nz)
switch ch
    case 'dwdx',    s = 'RigidBody';
    case 'dwdz',    s = 'MonZern';
    case 'dwdsurf', s = 'Surf';
    case 'dwdgrid', s = 'Grid';
    otherwise,      s = '';
end
k = repmat({s}, nz, 1);
end

function m = empty_meta_()
m = struct('channel', {{}}, 'iElt', [], 'dof', [], 'kind', {{}}, ...
    'mode', [], 'param', {{}}, 'map', [], 'name', {{}});
end

function m = cat_meta_(a, b)
m.channel = [a.channel(:); b.channel(:)];
m.iElt    = [a.iElt(:);    b.iElt(:)];
m.dof     = [a.dof(:);     b.dof(:)];
m.kind    = [a.kind(:);    b.kind(:)];
m.mode    = [a.mode(:);    b.mode(:)];
m.param   = [a.param(:);   b.param(:)];
m.map     = [a.map(:);     b.map(:)];
m.name    = [a.name(:);    b.name(:)];
end

function v = getnum_(o, f, nz)
if isfield(o, f) && ~isempty(o.(f)), v = double(o.(f)(:));
else, v = nan(nz, 1); end
if numel(v) ~= nz, v = nan(nz, 1); end
end

function c = getcell_(o, f, nz)
if isfield(o, f) && ~isempty(o.(f)), c = cellstr(o.(f)(:));
else, c = repmat({''}, nz, 1); end
if numel(c) ~= nz, c = repmat({''}, nz, 1); end
end
