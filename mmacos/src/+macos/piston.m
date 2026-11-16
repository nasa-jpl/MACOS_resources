function out = piston(out, mode, groups)
%MACOS.PISTON  Keep or remove wavefront piston in sensitivity columns.
%   OUT = macos.piston(OUT, MODE) takes a dw_d* result (macos.dw_dx,
%   dw_dx_multi, dw_dz_zernike_multi, dw_dgrid_multi, dw_dsurf_multi, or
%   a run_sensitivities leg) and, for MODE = 'remove', subtracts from
%   every column its mean over each field's pupil (the finite rows), in
%   the per-field cells (per_field_dw*), the stacked matrices (dw*all,
%   rows assigned to fields by indxall + field_table) and the single-field
%   matrices (dwdx, dwdz, dwdg, dwds).  MODE = 'keep' changes nothing.
%   Either way OUT.opd_piston records what the columns now carry:
%   'removed' or 'kept'.  Removing piston from columns already marked
%   'removed' is a no-op, never a second removal.
%
%   M = macos.piston(M, MODE, GROUPS) does the same for a plain matrix:
%   each column loses its mean over the finite rows of each group
%   (GROUPS: one integer per row; omitted = one group).  Runners use it
%   on their engine-side maps so both sides of a comparison follow the
%   same MODE.
%
%   Why it exists (PLAN_CONSOLIDATION 1.2 (www), Dave 2026-10-27): under
%   the chief-ray OPD reference (macos.opd_ref, the default) a poke that
%   moves the chief ray's path adds a constant to the whole map.  An
%   imager's corrector cannot see piston and its image does not depend on
%   it, so imaging analyses remove it ('remove'); an interferometer reads
%   the piston between fields or arms, so its analyses keep it ('keep').
%
%   See also: macos.opd_ref, macos.dw_dx_multi, macos.opd_mask.
arguments
    out
    mode (1,:) char {mustBeMember(mode, {'remove','keep'})}
    groups = []
end

if ~isstruct(out)                                  % plain matrix form
    if strcmp(mode, 'remove'), out = remove_rows_(out, groups); end
    return
end

if strcmp(mode, 'keep')
    if ~isfield(out, 'opd_piston'), out.opd_piston = 'kept'; end
    return                                         % 'removed' stays 'removed'
end
if isfield(out, 'opd_piston') && strcmp(out.opd_piston, 'removed')
    return                                         % already piston-free
end

for f = {'per_field_dwdx', 'per_field_dwdz', 'per_field_dwdg', 'per_field_dwds'}
    if isfield(out, f{1}) && iscell(out.(f{1}))
        out.(f{1}) = cellfun(@(c) remove_rows_(c, []), out.(f{1}), ...
            'UniformOutput', false);
    end
end
g = stacked_groups_(out);
for f = {'dwdxall', 'dwdzall', 'dwdgall', 'dwdsall'}
    if isfield(out, f{1}) && isnumeric(out.(f{1})) && ~isempty(out.(f{1}))
        gf = g;
        if numel(gf) ~= size(out.(f{1}), 1), gf = []; end
        if isempty(gf) && nfields_(out) > 1
            warning('macos:piston:rows', ['macos.piston: the rows of %s ' ...
                'cannot be assigned to their fields -- removing one piston ' ...
                'over all fields'], f{1});
        end
        out.(f{1}) = remove_rows_(out.(f{1}), gf);
    end
end
if nfields_(out) <= 1                              % single-field drivers
    for f = {'dwdx', 'dwdz', 'dwdg', 'dwds'}
        if isfield(out, f{1}) && isnumeric(out.(f{1})) && ~isempty(out.(f{1}))
            out.(f{1}) = remove_rows_(out.(f{1}), []);
        end
    end
end
out.opd_piston = 'removed';
end

function D = remove_rows_(D, g)
if isempty(D), return; end
if isempty(g), g = ones(size(D, 1), 1); end
ok = all(isfinite(D), 2);
for k = unique(g(:)).'
    r = g(:) == k;
    if any(r & ok), D(r, :) = D(r, :) - mean(D(r & ok, :), 1); end
end
end

function n = nfields_(out)
n = 0;
for f = {'per_field_dwdx', 'per_field_dwdz', 'per_field_dwdg', 'per_field_dwds'}
    if isfield(out, f{1}) && iscell(out.(f{1})), n = max(n, numel(out.(f{1}))); end
end
end

function g = stacked_groups_(out)
% field of each stacked row: the multi harvest's canvas tiles (indxall =
% row/col on the tiled canvas, field_table = [dx dy tile_row tile_col])
g = [];
if ~all(isfield(out, {'indxall', 'field_table', 'per_field_w_nom_2d'})), return; end
if ~isstruct(out.indxall) || ~isfield(out.indxall, 'i'), return; end
m = out.per_field_w_nom_2d;
if iscell(m), m = m{1}; end
N = size(m, 1);
tr = floor((out.indxall.i(:) - 1) / N);
tc = floor((out.indxall.j(:) - 1) / N);
g = zeros(numel(tr), 1);
for f = 1:size(out.field_table, 1)
    g(tr == out.field_table(f, 3) & tc == out.field_table(f, 4)) = f;
end
if any(g == 0), g = []; end
end
