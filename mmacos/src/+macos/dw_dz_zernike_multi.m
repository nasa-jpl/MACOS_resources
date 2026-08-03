function out = dw_dz_zernike_multi(session, rx_path, opts)
%MACOS.DW_DZ_ZERNIKE_MULTI  Multi-field dw/dz_Zernike supervisor.
%   out = macos.dw_dz_zernike_multi(SESSION, RX_PATH, ...) loads RX_PATH
%   on SESSION, snapshots the nominal source FoV, then loops field
%   points -- each iteration absolutely sets ChfRayDir via set_src_fov
%   and runs dw_dz_zernike at the new field.  The per-field results
%   are tiled into one big OPDall canvas and scattered into one big
%   dwdxall (canonical state-vector form):
%
%       wall = dwdxall * x + w0_stacked
%
%   where wall is the column-major vectorisation of OPDall via m2v.
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
%   OTHER NAME-VALUE PAIRS (all forwarded to dw_dz_zernike):
%     'kinds', 'elts', 'zmode_start', 'n_zcoef', 'delta', 'method',
%     'exit_pupil_elt', 'verbose'.
%
%   'ngridpts'  (default [] = keep the .in value) ray-grid sampling
%               override, applied once right after the Rx load; persists
%               across the per-field calls.  Clamped by the engine to
%               [3, model-size limit] (warns).
%   'reset_xp'  (default true) re-find the exit pupil (FEX, chief ray) for
%               EACH field before differencing, so the nominal wavefront is
%               referenced to that field's own chief ray and the gross field
%               TILT is removed (off-axis fields otherwise carry a large
%               linear-in-field tilt that swamps the OPD canvas).  A poke's
%               OWN tilt is retained -- the reference is fixed per field, not
%               re-fit after each poke.  Requires a STOP set and > 3 elements.
%               Restore scope: the pre-loop EP is snapshotted/restored via
%               get_xp/set_xp -- vpt/psi/rad (VptElt/PsiElt/KrElt at nElt-1)
%               only.  FEX-written auxiliary fields on the EP element
%               (RptElt, zElt, fElt, eElt, KcElt) are left as re-derived;
%               callers who hand-author those own re-asserting them.
%
%   OUTPUT STRUCT FIELDS:
%     dwdxall       Nw x Nz canonical state-vector Jacobian
%     dwdzall       alias (== dwdxall) -- kind-specific name
%     w0_stacked    Nw x 1 stacked nominal OPDs (m2v of OPDall)
%     indxall       struct with i, j, size -- m2v round-trip metadata
%     OPDall        full tiled OPD canvas (un-vectorised)
%     channel_names Nz x 1 cell array
%     field_table   Nfields x 4: [dx_rad, dy_rad, tile_row, tile_col]
%     field_names   Nfields x 1 cell array
%     chfraydir_nom 3 x 1 nominal ChfRayDir
%     per_field_dwdz  Nfields x 1 cell of single-field dwdz blocks
%     per_field_w_nom_2d  Nfields x 1 cell of single-field nominal OPDs
%     rx_path / delta / method / wf_elt / kinds  -- echoed inputs
%
%   See also: macos.dw_dz_zernike, macos.dwdz_for_current_source.

arguments
    session
    rx_path                     (1,:) char {mustBeNonempty}
    opts.field_x_rad            (1,1) double = NaN
    opts.field_y_rad            (1,1) double = NaN
    opts.fields                 (1,:) char = ''
    opts.grid                   (1,:) char = ''
    opts.kinds                  cell = {'monzern','zern'}
    opts.elts                   (:,1) double = []
    opts.zmode_start            (1,1) double {mustBeInteger, mustBePositive} = 4
    opts.n_zcoef                (1,1) double {mustBeInteger, mustBePositive} = 15
    opts.delta                  (1,1) double = 1e-6
    opts.method                 (1,:) char {mustBeMember(opts.method, ...
                                  {'central','forward'})} = 'central'
    opts.exit_pupil_elt         (1,1) double {mustBeInteger} = -1
    opts.reset_xp               (1,1) logical = true
    opts.verbose                (1,1) logical = false
    opts.ngridpts               double {mustBeScalarOrEmpty} = []
    opts.src_samp               double {mustBeScalarOrEmpty, mustBeInteger} = []
    opts.compute_los            (1,1) logical = false
    opts.spot_elt               double {mustBeScalarOrEmpty, mustBeInteger} = []
end

if isnan(opts.field_x_rad) || isnan(opts.field_y_rad)
    error('macos:dw_dz_zernike_multi:fov', ...
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
apply_ngridpts(session, opts.ngridpts, 'dw_dz_zernike_multi');

% Apply source sampling if specified
if ~isempty(opts.src_samp)
    session.set_src_sampling(opts.src_samp);
    session.modify();  % Flush cache so the new sampling takes effect
end

nom = session.get_src_fov();
fprintf('[setup] nominal ChfRayDir = [%g %g %g]; zSrc = %.3e\n', ...
    nom.src_dir, nom.zSrc);

% Snapshot the prescription's exit-pupil reference (elt nElt-1 geometry)
% so the per-field FEX resets can be undone before returning.
if opts.reset_xp
    xp0 = macos.get_xp();
end

% ---- Per-field loop -----------------------------------------------
% Each iteration uses set_src_fov ABSOLUTELY so no perturb-undo
% round-trip is needed -- each field starts from the nominal source
% state by construction.  modify() after set_src_fov is required so
% the next trace re-derives chief-ray geometry from the new direction
% (without it, the cached trace state would reuse the prior field's
% ChfRayDir and every field would see the nominal OPD).  We also pass
% reload_rx=false to dw_dz_zernike so it does NOT call load_rx (which
% would reset ChfRayDir back to the prescription nominal).
per_field_dwdz   = cell(n_fields, 1);
per_field_w_nom  = cell(n_fields, 1);
per_field_struct = cell(n_fields, 1);
if opts.compute_los
    per_field_dcdx = cell(n_fields, 1);
end
names = {};
iElt_out = [];
for k = 1:n_fields
    new_dir = field_to_chfraydir(nom.src_dir, fields(k).dx, fields(k).dy);
    session.set_src_fov('src_pos', nom.src_pos, 'src_dir', new_dir, ...
                        'zSrc', nom.zSrc);
    session.modify();   % flush trace cache so the new dir takes effect
    fprintf('[field %s] ChfRayDir = [%g %g %g]\n', ...
        fields(k).name, new_dir);
    if opts.reset_xp
        % Re-reference this field's exit pupil to its OWN chief ray (FEX
        % writes the reference into elt nElt-1 = wf_elt) so the nominal
        % wavefront is tilt-removed.  It persists as element geometry across
        % the poke traces, and a poke's own tilt is retained (the reference
        % is NOT re-fit after poking).  The Zern/MonZern/FFZern pokes act on
        % the powered/Zernike optics, not elt nElt-1.  See dw_dgrid_multi.
        macos.fex(1);   % mode 1 = centre on chief ray
    end
    sf = macos.dw_dz_zernike(session, rx_path, ...
        'kinds', opts.kinds, ...
        'elts', opts.elts, ...
        'zmode_start', opts.zmode_start, ...
        'n_zcoef', opts.n_zcoef, ...
        'delta', opts.delta, ...
        'method', opts.method, ...
        'exit_pupil_elt', opts.exit_pupil_elt, ...
        'verbose', opts.verbose, ...
        'reload_rx', false, ...
        'compute_los', opts.compute_los, ...
        'spot_elt', opts.spot_elt);    % keep current src_fov state
    per_field_dwdz{k} = sf.dwdz;
    per_field_w_nom{k} = sf.w_nom_2d;
    per_field_struct{k} = sf;
    if opts.compute_los
        per_field_dcdx{k} = sf.dcdx;
    end
    if isempty(names), names = sf.channel_names; iElt_out = sf.iElt; end
    col_rms_mean = mean(sqrt(mean(sf.dwdz.^2, 1)));
    fprintf('[field %s] dwdz shape [%d %d], mean col-RMS %.3e', ...
        fields(k).name, size(sf.dwdz, 1), size(sf.dwdz, 2), col_rms_mean);
    if opts.compute_los
        los_rms_mean = mean(sqrt(sum(sf.dcdx.^2, 2)));
        fprintf('  mean LOS-RMS %.3e', los_rms_mean);
    end
    fprintf('\n');
end

% Restore source back to nominal.
session.set_src_fov('src_pos', nom.src_pos, 'src_dir', nom.src_dir, ...
                    'zSrc', nom.zSrc);
session.modify();

% Restore the prescription's exit-pupil reference (undo the per-field FEX
% writes to elt nElt-1) so the session is left as loaded.
if opts.reset_xp
    macos.set_xp(xp0.vpt, xp0.psi, xp0.rad);
    session.modify();
end

% ---- Tile OPDall + scatter dwdzall --------------------------------
N = size(per_field_w_nom{1}, 1);
OPDall = zeros(tile_rows * N, tile_cols * N);
for k = 1:n_fields
    r0 = fields(k).tile_row * N;
    c0 = fields(k).tile_col * N;
    OPDall(r0+1:r0+N, c0+1:c0+N) = per_field_w_nom{k};
end

[w0_stacked, indxall] = macos.m2v(OPDall);
Nw = numel(w0_stacked);
Nz = size(per_field_dwdz{1}, 2);
fprintf('[stack] OPDall [%d %d]; non-zero pixels = %d\n', ...
    size(OPDall, 1), size(OPDall, 2), Nw);

dwdzall = zeros(Nw, Nz);
indx_i = indxall.i(:);
indx_j = indxall.j(:);
for k = 1:n_fields
    tr = fields(k).tile_row;
    tc = fields(k).tile_col;
    in_tile = (indx_i > tr*N) & (indx_i <= (tr+1)*N) ...
            & (indx_j > tc*N) & (indx_j <= (tc+1)*N);
    i_local = indx_i(in_tile) - tr * N;
    j_local = indx_j(in_tile) - tc * N;
    % Build field-local m2v of this tile so we can map global rows
    % back to the per-field dwdz rows.
    [~, field_indx] = macos.m2v(per_field_w_nom{k});
    field_i = field_indx.i(:);
    field_j = field_indx.j(:);
    % Match (i_local, j_local) -> per-field row via column-major flat
    flat_local  = (j_local - 1) * N + i_local;
    flat_field  = (field_j  - 1) * N + field_i;
    [tf, loc] = ismember(flat_local, flat_field);
    if ~all(tf)
        error('macos:dw_dz_zernike_multi:scatter', ...
            'field %s: indxall references pixels outside per-field mask', ...
            fields(k).name);
    end
    global_rows = find(in_tile);
    dwdzall(global_rows, :) = per_field_dwdz{k}(loc, :);
    fprintf('[stack] field %s: scattered %d rows into dwdzall\n', ...
        fields(k).name, numel(global_rows));
end

fprintf('[stack] dwdzall shape [%d %d]; |dwdzall| max = %.3e\n', ...
    size(dwdzall, 1), size(dwdzall, 2), max(abs(dwdzall(:))));

% Center-tile sanity check: the (0, 0) field's rows in dwdzall must
% exactly match its per_field_dwdz block.  Catches scatter-logic
% bugs (max|diff| = 0 in practice).
ctr_idx = find_center_field_index(fields);
if ~isempty(ctr_idx)
    tr = fields(ctr_idx).tile_row;
    tc = fields(ctr_idx).tile_col;
    in_ctr = (indx_i > tr*N) & (indx_i <= (tr+1)*N) ...
           & (indx_j > tc*N) & (indx_j <= (tc+1)*N);
    dwdzall_ctr = dwdzall(in_ctr, :);
    dwdz_C = per_field_dwdz{ctr_idx};
    max_diff = max(abs(dwdzall_ctr(:) - dwdz_C(:)));
    fprintf('[check] dwdzall@center-tile vs per_field_dwdz[center]: ');
    fprintf('max|diff| = %.3e ([%d %d])\n', ...
        max_diff, size(dwdzall_ctr, 1), size(dwdzall_ctr, 2));
    assert(max_diff == 0, ...
        'scatter bug: dwdzall@center-tile differs from per_field_dwdz[center]');
else
    fprintf('[check] no (0,0)-offset field -- skipping center-tile check\n');
end

% ---- Pack output struct -------------------------------------------
out = struct();
out.dwdxall              = dwdzall;
out.dwdzall              = dwdzall;
out.w0_stacked           = w0_stacked;
out.indxall              = indxall;
out.OPDall               = OPDall;
out.channel_names        = names;
out.iElt                 = iElt_out;
out.field_table          = arrayfun( ...
    @(s) [s.dx, s.dy, s.tile_row, s.tile_col], fields, ...
    'UniformOutput', false);
out.field_table          = vertcat(out.field_table{:});
out.field_names          = {fields.name}.';
out.chfraydir_nom        = nom.src_dir(:);
out.per_field_dwdz       = per_field_dwdz;
out.per_field_w_nom_2d   = per_field_w_nom;
out.rx_path              = rx_path;
out.delta                = opts.delta;
out.method               = opts.method;
out.wf_elt               = per_field_struct{1}.wf_elt;
out.kinds                = opts.kinds;
out.reset_xp             = opts.reset_xp;

% Add per-field LOS if SPOT was computed
if opts.compute_los
    out.dcdx_per_field = per_field_dcdx;
    if isempty(opts.spot_elt)
        out.spot_elt = session.num_elt();  % Default focal plane
    else
        out.spot_elt = opts.spot_elt;
    end
end
end


% =====================================================================
function fields = make_5field_set(field_x_rad, field_y_rad)
% Default 5-field: center + 4 corners.  Tile-row convention matches
% MATLAB's imagesc(...) origin='lower'-equivalent (we paint OPDall
% directly, so tile_row=2 is "upper" / top of the displayed image
% under axis xy + imagesc).
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
% [-field_y_rad..+field_y_rad].  Center field auto-named "C" when
% it lands at exactly (0, 0); else "F_r{tr}_c{tc}".
if nx > 1
    dx_axis = linspace(-field_x_rad, +field_x_rad, nx);
else
    dx_axis = 0;
end
if ny > 1
    dy_axis = linspace(-field_y_rad, +field_y_rad, ny);
else
    dy_axis = 0;
end
fields = struct('name', {}, 'dx', {}, 'dy', {}, ...
                 'tile_row', {}, 'tile_col', {});
for ir = 1:numel(dy_axis)
    for ic = 1:numel(dx_axis)
        dy = dy_axis(ir);
        dx = dx_axis(ic);
        is_center = (abs(dx) < 1e-30) && (abs(dy) < 1e-30);
        if is_center
            nm = 'C';
        else
            nm = sprintf('F_r%d_c%d', ir-1, ic-1);
        end
        fields(end+1) = field_entry(nm, dx, dy, ir-1, ic-1); %#ok<AGROW>
    end
end
end


function e = field_entry(name, dx, dy, tr, tc)
e.name = name;
e.dx = dx;
e.dy = dy;
e.tile_row = tr;
e.tile_col = tc;
end


function idx = find_center_field_index(fields)
idx = [];
for k = 1:numel(fields)
    if abs(fields(k).dx) < 1e-30 && abs(fields(k).dy) < 1e-30
        idx = k;
        return;
    end
end
end


function new_dir = field_to_chfraydir(dir_nom, dx_rad, dy_rad)
% Direction-cosine offset on top of the nominal ChfRayDir, then
% renormalise.  Per-axis x and y are independent.
v = dir_nom(:) + [dx_rad; dy_rad; 0];
n = norm(v);
if n == 0
    error('macos:dw_dz_zernike_multi:zerodir', ...
        'zero-magnitude direction after field offset');
end
new_dir = v / n;
end


function [nx, ny] = parse_grid_spec(spec)
toks = regexp(lower(spec), 'x', 'split');
if numel(toks) ~= 2
    error('macos:dw_dz_zernike_multi:grid', ...
        '''grid'' must be ''NxM'' (e.g. ''3x3''); got %s', spec);
end
nx = str2double(toks{1});
ny = str2double(toks{2});
if isnan(nx) || isnan(ny) || nx < 1 || ny < 1
    error('macos:dw_dz_zernike_multi:grid', ...
        '''grid'' must be ''NxM'' with positive integers; got %s', spec);
end
end


function fields = load_field_file(fname)
% Free-form list: lines of 'name dx_rad dy_rad tile_row tile_col'.
% '#' starts a comment; blank lines ignored.
fid = fopen(fname, 'r');
if fid < 0
    error('macos:dw_dz_zernike_multi:fields', ...
        'cannot open fields file: %s', fname);
end
c = onCleanup(@() fclose(fid));
fields = struct('name', {}, 'dx', {}, 'dy', {}, ...
                 'tile_row', {}, 'tile_col', {});
while true
    ln = fgetl(fid);
    if ~ischar(ln); break; end
    s = strtrim(ln);
    if isempty(s) || startsWith(s, '#')
        continue;
    end
    toks = regexp(s, '\s+', 'split');
    if numel(toks) < 5
        error('macos:dw_dz_zernike_multi:fields', ...
            'fields-file row needs 5 columns: %s', s);
    end
    fields(end+1) = field_entry(toks{1}, ...
        str2double(toks{2}), str2double(toks{3}), ...
        str2double(toks{4}), str2double(toks{5})); %#ok<AGROW>
end
end
