function out = dw_dsurf_multi(session, rx_path, opts)
%MACOS.DW_DSURF_MULTI  Multi-field dw/dKr + dw/dKc supervisor.
%   out = macos.dw_dsurf_multi(SESSION, RX_PATH, ...) loads RX_PATH on
%   SESSION, snapshots the nominal source FoV, then loops field points --
%   each iteration absolutely sets ChfRayDir via set_src_fov and runs
%   dw_dsurf at the new field.  The per-field results are tiled into one
%   big OPDall canvas and scattered into one big dwdsall (canonical
%   state-vector form):
%
%       wall = dwdsall * x + w0_stacked
%
%   where wall is the column-major vectorisation of OPDall via m2v, and
%   x is the stacked (Kr, Kc) state of every POWERED optic (Element=
%   Reflector / Refractor with |Kr| << the flat sentinel).
%
%   The powered-surface (radius + conic) companion to
%   macos.dw_dz_zernike_multi, built on the same field-loop + tiling
%   machinery, a different DOF set.
%
%   REQUIRED USER INPUTS:
%     'field_x_rad'   half-FoV in x (direction cosine added to ChfRayDir)
%     'field_y_rad'   half-FoV in y (independent of x)
%
%   FIELD SET (one of):
%     default 5-field (center + 4 corners)
%     'grid' 'NxM'    auto-generate uniform N x M grid (center counted
%                     once when both N and M are odd)
%     'fields' FILE   override: rows of 'name dx_rad dy_rad tile_row tile_col'
%
%   OTHER NAME-VALUE PAIRS (forwarded to dw_dsurf):
%     'params' (cellstr subset of {'Kr','Kc'}), 'delta', 'method',
%     'exit_pupil_elt', 'verbose'.
%
%   OUTPUT STRUCT FIELDS:
%     dwdsall       Nw x Ns canonical state-vector Jacobian
%     dwdxall       alias (== dwdsall) -- canonical-form consumers
%     w0_stacked    Nw x 1 stacked nominal OPDs (m2v of OPDall)
%     indxall       struct with i, j, size -- m2v round-trip metadata
%     OPDall        full tiled OPD canvas (un-vectorised)
%     channel_names Ns x 1 cell array  ('Elt N Kr' / 'Elt N Kc')
%     iElt / param  Ns x 1 per-channel element id + param ('Kr'|'Kc')
%     field_table   Nfields x 4: [dx_rad, dy_rad, tile_row, tile_col]
%     field_names   Nfields x 1 cell array
%     chfraydir_nom 3 x 1 nominal ChfRayDir
%     per_field_dwds      Nfields x 1 cell of single-field dwds blocks
%     per_field_w_nom_2d  Nfields x 1 cell of single-field nominal OPDs
%     rx_path / delta / method / wf_elt / params  -- echoed inputs
%
%   See also: macos.dw_dsurf, macos.dw_dz_zernike_multi, macos.dw_dx_multi.

arguments
    session
    rx_path                     (1,:) char {mustBeNonempty}
    opts.field_x_rad            (1,1) double = NaN
    opts.field_y_rad            (1,1) double = NaN
    opts.fields                 (1,:) char = ''
    opts.grid                   (1,:) char = ''
    opts.params                 cell = {'Kr','Kc'}
    opts.delta                  (1,1) double = 1e-6
    opts.method                 (1,:) char {mustBeMember(opts.method, ...
                                  {'central','forward'})} = 'central'
    opts.exit_pupil_elt         (1,1) double {mustBeInteger} = -1
    opts.verbose                (1,1) logical = false
end

if isnan(opts.field_x_rad) || isnan(opts.field_y_rad)
    error('macos:dw_dsurf_multi:fov', ...
        'field_x_rad and field_y_rad are required');
end

% ---- Field set ----------------------------------------------------
if ~isempty(opts.fields)
    fields = load_field_file(opts.fields);
elseif ~isempty(opts.grid)
    [nx, ny] = parse_grid_spec(opts.grid);
    fields = make_grid_field_set(nx, ny, opts.field_x_rad, opts.field_y_rad);
else
    fields = make_5field_set(opts.field_x_rad, opts.field_y_rad);
end
n_fields = numel(fields);
tile_rows = max(arrayfun(@(s) s.tile_row, fields)) + 1;
tile_cols = max(arrayfun(@(s) s.tile_col, fields)) + 1;
fprintf('[setup] %d field points, tile grid %dx%d\n', ...
    n_fields, tile_rows, tile_cols);
for k = 1:n_fields
    fprintf('  field %-8s: dir-offset=(%+.3e,%+.3e) rad  tile=(%d,%d)\n', ...
        fields(k).name, fields(k).dx, fields(k).dy, ...
        fields(k).tile_row, fields(k).tile_col);
end

% ---- Load + snapshot nominal source -------------------------------
session.load_rx(rx_path);
nom = session.get_src_fov();
fprintf('[setup] nominal ChfRayDir = [%g %g %g]; zSrc = %.3e\n', ...
    nom.src_dir, nom.zSrc);

% ---- Per-field loop -----------------------------------------------
% Each iteration uses set_src_fov ABSOLUTELY (no perturb-undo needed) +
% modify() to flush the trace cache so the new ChfRayDir takes effect,
% and passes reload_rx=false to dw_dsurf so it does NOT call load_rx
% (which would reset ChfRayDir back to the prescription nominal).
per_field_dwds   = cell(n_fields, 1);
per_field_w_nom  = cell(n_fields, 1);
per_field_struct = cell(n_fields, 1);
names = {};  iElt_out = [];  param_out = {};
for k = 1:n_fields
    new_dir = field_to_chfraydir(nom.src_dir, fields(k).dx, fields(k).dy);
    session.set_src_fov('src_pos', nom.src_pos, 'src_dir', new_dir, ...
                        'zSrc', nom.zSrc);
    session.modify();   % flush trace cache so the new dir takes effect
    fprintf('[field %s] ChfRayDir = [%g %g %g]\n', ...
        fields(k).name, new_dir);
    sf = macos.dw_dsurf(session, rx_path, ...
        'params', opts.params, ...
        'delta', opts.delta, ...
        'method', opts.method, ...
        'exit_pupil_elt', opts.exit_pupil_elt, ...
        'verbose', opts.verbose, ...
        'reload_rx', false);    % keep current src_fov state
    per_field_dwds{k} = sf.dwds;
    per_field_w_nom{k} = sf.w_nom_2d;
    per_field_struct{k} = sf;
    if isempty(names)
        names = sf.channel_names;  iElt_out = sf.iElt;  param_out = sf.param;
    end
    col_rms_mean = mean(sqrt(mean(sf.dwds.^2, 1)));
    fprintf('[field %s] dwds shape [%d %d], mean col-RMS %.3e\n', ...
        fields(k).name, size(sf.dwds, 1), size(sf.dwds, 2), col_rms_mean);
end

% Restore source back to nominal.
session.set_src_fov('src_pos', nom.src_pos, 'src_dir', nom.src_dir, ...
                    'zSrc', nom.zSrc);
session.modify();

% ---- Tile OPDall + scatter dwdsall --------------------------------
N = size(per_field_w_nom{1}, 1);
OPDall = zeros(tile_rows * N, tile_cols * N);
for k = 1:n_fields
    r0 = fields(k).tile_row * N;
    c0 = fields(k).tile_col * N;
    OPDall(r0+1:r0+N, c0+1:c0+N) = per_field_w_nom{k};
end

[w0_stacked, indxall] = macos.m2v(OPDall);
Nw = numel(w0_stacked);
Ns = size(per_field_dwds{1}, 2);
fprintf('[stack] OPDall [%d %d]; non-zero pixels = %d\n', ...
    size(OPDall, 1), size(OPDall, 2), Nw);

dwdsall = zeros(Nw, Ns);
indx_i = indxall.i(:);
indx_j = indxall.j(:);
for k = 1:n_fields
    tr = fields(k).tile_row;
    tc = fields(k).tile_col;
    in_tile = (indx_i > tr*N) & (indx_i <= (tr+1)*N) ...
            & (indx_j > tc*N) & (indx_j <= (tc+1)*N);
    i_local = indx_i(in_tile) - tr * N;
    j_local = indx_j(in_tile) - tc * N;
    % Build field-local m2v of this tile so we can map global rows back
    % to the per-field dwds rows.
    [~, field_indx] = macos.m2v(per_field_w_nom{k});
    field_i = field_indx.i(:);
    field_j = field_indx.j(:);
    flat_local  = (j_local - 1) * N + i_local;
    flat_field  = (field_j  - 1) * N + field_i;
    [tf, loc] = ismember(flat_local, flat_field);
    if ~all(tf)
        error('macos:dw_dsurf_multi:scatter', ...
            'field %s: indxall references pixels outside per-field mask', ...
            fields(k).name);
    end
    global_rows = find(in_tile);
    dwdsall(global_rows, :) = per_field_dwds{k}(loc, :);
    fprintf('[stack] field %s: scattered %d rows into dwdsall\n', ...
        fields(k).name, numel(global_rows));
end

fprintf('[stack] dwdsall shape [%d %d]; |dwdsall| max = %.3e\n', ...
    size(dwdsall, 1), size(dwdsall, 2), max(abs(dwdsall(:))));

% Center-tile sanity check: the (0,0) field's rows in dwdsall must
% exactly match its per_field_dwds block (max|diff| = 0 in practice).
ctr_idx = find_center_field_index(fields);
if ~isempty(ctr_idx)
    tr = fields(ctr_idx).tile_row;
    tc = fields(ctr_idx).tile_col;
    in_ctr = (indx_i > tr*N) & (indx_i <= (tr+1)*N) ...
           & (indx_j > tc*N) & (indx_j <= (tc+1)*N);
    dwdsall_ctr = dwdsall(in_ctr, :);
    dwds_C = per_field_dwds{ctr_idx};
    max_diff = max(abs(dwdsall_ctr(:) - dwds_C(:)));
    fprintf('[check] dwdsall@center-tile vs per_field_dwds[center]: ');
    fprintf('max|diff| = %.3e ([%d %d])\n', ...
        max_diff, size(dwdsall_ctr, 1), size(dwdsall_ctr, 2));
    assert(max_diff == 0, ...
        'scatter bug: dwdsall@center-tile differs from per_field_dwds[center]');
else
    fprintf('[check] no (0,0)-offset field -- skipping center-tile check\n');
end

% ---- Pack output struct -------------------------------------------
out = struct();
out.dwdsall              = dwdsall;
out.dwdxall              = dwdsall;     % canonical-form alias
out.w0_stacked           = w0_stacked;
out.indxall              = indxall;
out.OPDall               = OPDall;
out.channel_names        = names;
out.iElt                 = iElt_out;
out.param                = param_out;
out.field_table          = arrayfun( ...
    @(s) [s.dx, s.dy, s.tile_row, s.tile_col], fields, ...
    'UniformOutput', false);
out.field_table          = vertcat(out.field_table{:});
out.field_names          = {fields.name}.';
out.chfraydir_nom        = nom.src_dir(:);
out.per_field_dwds       = per_field_dwds;
out.per_field_w_nom_2d   = per_field_w_nom;
out.rx_path              = rx_path;
out.delta                = opts.delta;
out.method               = opts.method;
out.wf_elt               = per_field_struct{1}.wf_elt;
out.params               = opts.params;
end


% =====================================================================
function fields = make_5field_set(field_x_rad, field_y_rad)
% Default 5-field: center + 4 corners.
fields = struct('name', {}, 'dx', {}, 'dy', {}, ...
                 'tile_row', {}, 'tile_col', {});
fields(end+1) = field_entry('C',  0,            0,            1, 1);
fields(end+1) = field_entry('UL', -field_x_rad, +field_y_rad, 2, 0);
fields(end+1) = field_entry('UR', +field_x_rad, +field_y_rad, 2, 2);
fields(end+1) = field_entry('LL', -field_x_rad, -field_y_rad, 0, 0);
fields(end+1) = field_entry('LR', +field_x_rad, -field_y_rad, 0, 2);
end


function fields = make_grid_field_set(nx, ny, field_x_rad, field_y_rad)
% Uniform NxM grid covering [-field_x_rad..+field_x_rad] x
% [-field_y_rad..+field_y_rad].
if nx > 1, dx_axis = linspace(-field_x_rad, +field_x_rad, nx); else, dx_axis = 0; end
if ny > 1, dy_axis = linspace(-field_y_rad, +field_y_rad, ny); else, dy_axis = 0; end
fields = struct('name', {}, 'dx', {}, 'dy', {}, ...
                 'tile_row', {}, 'tile_col', {});
for ir = 1:numel(dy_axis)
    for ic = 1:numel(dx_axis)
        dy = dy_axis(ir);  dx = dx_axis(ic);
        if (abs(dx) < 1e-30) && (abs(dy) < 1e-30)
            nm = 'C';
        else
            nm = sprintf('F_r%d_c%d', ir-1, ic-1);
        end
        fields(end+1) = field_entry(nm, dx, dy, ir-1, ic-1); %#ok<AGROW>
    end
end
end


function e = field_entry(name, dx, dy, tr, tc)
e.name = name;  e.dx = dx;  e.dy = dy;  e.tile_row = tr;  e.tile_col = tc;
end


function idx = find_center_field_index(fields)
idx = [];
for k = 1:numel(fields)
    if abs(fields(k).dx) < 1e-30 && abs(fields(k).dy) < 1e-30
        idx = k;  return;
    end
end
end


function new_dir = field_to_chfraydir(dir_nom, dx_rad, dy_rad)
% Direction-cosine offset on top of the nominal ChfRayDir, renormalised.
v = dir_nom(:) + [dx_rad; dy_rad; 0];
n = norm(v);
if n == 0
    error('macos:dw_dsurf_multi:zerodir', ...
        'zero-magnitude direction after field offset');
end
new_dir = v / n;
end


function [nx, ny] = parse_grid_spec(spec)
toks = regexp(lower(spec), 'x', 'split');
if numel(toks) ~= 2
    error('macos:dw_dsurf_multi:grid', ...
        '''grid'' must be ''NxM'' (e.g. ''3x3''); got %s', spec);
end
nx = str2double(toks{1});  ny = str2double(toks{2});
if isnan(nx) || isnan(ny) || nx < 1 || ny < 1
    error('macos:dw_dsurf_multi:grid', ...
        '''grid'' must be ''NxM'' with positive integers; got %s', spec);
end
end


function fields = load_field_file(fname)
% Free-form list: lines of 'name dx_rad dy_rad tile_row tile_col'.
fid = fopen(fname, 'r');
if fid < 0
    error('macos:dw_dsurf_multi:fields', ...
        'cannot open fields file: %s', fname);
end
c = onCleanup(@() fclose(fid)); %#ok<NASGU>
fields = struct('name', {}, 'dx', {}, 'dy', {}, ...
                 'tile_row', {}, 'tile_col', {});
while true
    ln = fgetl(fid);
    if ~ischar(ln); break; end
    s = strtrim(ln);
    if isempty(s) || startsWith(s, '#'), continue; end
    toks = regexp(s, '\s+', 'split');
    if numel(toks) < 5
        error('macos:dw_dsurf_multi:fields', ...
            'fields-file row needs 5 columns: %s', s);
    end
    fields(end+1) = field_entry(toks{1}, ...
        str2double(toks{2}), str2double(toks{3}), ...
        str2double(toks{4}), str2double(toks{5})); %#ok<AGROW>
end
end
