function out = dw_dx_multi(session, rx_path, opts)
%MACOS.DW_DX_MULTI  Multi-field dw/dx rigid-body supervisor.
%   Mirror of macos.dw_dz_zernike_multi but for the rigid-body
%   Jacobian.  Loads Rx once, builds source / per-element channels
%   once, then loops field points -- each iteration absolutely sets
%   ChfRayDir via set_src_fov + modify(), runs dw_dx with
%   reload_rx=false, and stacks per-field dwdx into the canonical
%   state-vector form  wall = dwdxall * x + w0_stacked.
%
%   REQUIRED:
%     'field_x_rad', 'field_y_rad'
%
%   FIELD SET (one of):
%     default 5-field (C + 4 corners)
%     'grid' 'NxM'
%     'fields' FILE
%
%   FORWARDED TO dw_dx:  dofs, elts, fp_mode, ep_elt, include_source,
%     src_stop_mode, src_stop_pos, src_stop_elt, include_non_optics,
%     stop_elt, stop_obj_pos, rot_output, delta, method,
%     exit_pupil_elt, verbose.
%
%   'delta' can be (1,1) for uniform step or (1,6) for per-DOF steps
%     [Rx Ry Rz Tx Ty Tz]. Rotations in rad. Translation units set by
%     'delta_units' ('si' metres, default | 'base' BaseUnits). Default 1e-8.
%
%   'ngridpts' (default [] = keep the .in value) overrides the ray-grid
%   sampling once, right after the Rx load; it persists across the
%   per-field calls (they run reload_rx=false).  Clamped by the engine
%   to [3, model-size limit] (warns).
%
%   'reset_xp'  (default true) re-find the exit pupil for EACH field
%               before differencing, so the nominal wavefront is
%               referenced to that field's own chief ray and the gross
%               field TILT is removed.  A poke's OWN tilt is retained --
%               the reference is fixed per field, not re-fit after each
%               poke.  With a frozen EP (false) the field tilt is
%               common-mode between w_nom and every poked w, so it
%               cancels in the FD columns; what reset_xp changes is the
%               first-order residual d(frame term)/dx -- negligible at
%               arcminute fields, percent-level on tilt-coupled DOFs at
%               wide fields.  Matches dw_dz_zernike_multi / dw_dsurf_multi
%               / dw_dgrid_multi (family alignment).  Requires a STOP set
%               and > 3 elements.  Set false to keep the prescription's
%               elt nElt-1 reference unchanged (frozen EP).
%               The re-find uses FEX (macos.fex, chief-ray centred);
%               FEX and SXP are merged in the engine, so FEX alone is
%               the well-posed re-reference for all placements.
%               Composes with fp_mode='track': the per-field EP is
%               written into elt nElt-1 BEFORE the FocalPlaneChannel
%               builds its columns, so 'track' saves/restores the
%               post-reset EP pose.
%               Restore scope: the pre-loop EP is snapshotted/restored via
%               get_xp/set_xp -- vpt/psi/rad (VptElt/PsiElt/KrElt at
%               nElt-1) only.  FEX-written auxiliary fields on the EP
%               element (RptElt, zElt, fElt, eElt, KcElt) are left as
%               re-derived; callers who hand-author those own re-asserting
%               them.
%
%   OUTPUT STRUCT FIELDS:
%     dwdxall            Nw x Nz canonical state-vector Jacobian
%     w0_stacked         Nw x 1 stacked nominal OPDs (m2v of OPDall)
%     indxall            i, j, size struct
%     OPDall             full tiled OPD canvas
%     channel_names      Nz x 1 cell
%     field_table        Nfields x 4
%     field_names        Nfields x 1 cell
%     chfraydir_nom      3 x 1
%     per_field_dwdx     Nfields x 1 cell of single-field blocks
%     per_field_w_nom_2d Nfields x 1 cell of single-field nominal OPDs
%     rx_path / delta / method / wf_elt / rot_output / cbm

arguments
    session
    rx_path                  (1,:) char {mustBeNonempty}
    opts.field_x_rad         (1,1) double = NaN
    opts.field_y_rad         (1,1) double = NaN
    opts.fields              (1,:) char = ''
    opts.grid                (1,:) char = ''
    opts.dofs                (:,1) double = (0:5).'
    opts.elts                (:,1) double = []
    opts.fp_mode             (1,:) char {mustBeMember( ...
        opts.fp_mode, {'track','srs','sxp','none'})} = 'track'
    opts.ep_elt              (1,1) double {mustBeInteger} = -1
    opts.include_source      (1,1) logical = false
    opts.src_stop_mode       (1,:) char {mustBeMember( ...
        opts.src_stop_mode, {'obj','elt','none'})} = 'obj'
    opts.src_stop_pos        (1,3) double = [0 0 0]
    opts.src_stop_elt        (1,1) double {mustBeInteger} = 0
    opts.include_non_optics  (1,1) logical = false
    opts.stop_elt            double = []
    opts.stop_obj_pos        double = []
    opts.rot_output          (1,:) char {mustBeMember( ...
        opts.rot_output, {'natural','base-per-rad'})} = 'natural'
    opts.delta               (:,:) double {mustBeDeltaSize} = 1e-8
    opts.delta_units         (1,:) char {mustBeMember(opts.delta_units, ...
                                {'si','base'})} = 'si'
    opts.method              (1,:) char {mustBeMember(opts.method, ...
                                {'central','forward'})} = 'central'
    opts.exit_pupil_elt      (1,1) double {mustBeInteger} = -1
    opts.reset_xp            (1,1) logical = true
    opts.verbose             (1,1) logical = false
    opts.ngridpts            double {mustBeScalarOrEmpty} = []
    opts.src_samp            double {mustBeScalarOrEmpty, mustBeInteger} = []
    opts.compute_los         (1,1) logical = false
    opts.spot_elt            double {mustBeScalarOrEmpty, mustBeInteger} = []
end

if isnan(opts.field_x_rad) || isnan(opts.field_y_rad)
    error('macos:dw_dx_multi:fov', ...
        'field_x_rad and field_y_rad are required');
end

% ---- Field set ----------------------------------------------------
if ~isempty(opts.fields)
    fields = load_field_file(opts.fields);
elseif ~isempty(opts.grid)
    [nx, ny] = parse_grid_spec(opts.grid);
    fields = make_grid_field_set(nx, ny, opts.field_x_rad, ...
                                  opts.field_y_rad);
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

% ---- Load + snapshot nominal --------------------------------------
session.load_rx(rx_path);
apply_ngridpts(session, opts.ngridpts, 'dw_dx_multi');

% Apply source sampling if specified
if ~isempty(opts.src_samp)
    session.set_src_sampling(opts.src_samp);
    session.modify();  % Flush cache so the new sampling takes effect
end

% Apply stop here so it survives across per-field calls (dw_dx with
% reload_rx=false won't touch the stop state).
if ~isempty(opts.stop_elt) && ~isempty(opts.stop_obj_pos)
    error('macos:dw_dx_multi:stop', ...
        'stop_elt and stop_obj_pos are mutually exclusive');
end
if ~isempty(opts.stop_elt)
    session.stop(int32(opts.stop_elt));
elseif ~isempty(opts.stop_obj_pos)
    session.stop_obj(opts.stop_obj_pos(1), opts.stop_obj_pos(2), ...
                      opts.stop_obj_pos(3));
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
per_field_dwdx   = cell(n_fields, 1);
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
    session.modify();
    fprintf('[field %s] ChfRayDir = [%g %g %g]\n', ...
        fields(k).name, new_dir);
    if opts.reset_xp
        % Re-reference this field's exit pupil to its OWN chief ray: FEX
        % writes the reference sphere into elt nElt-1 (= wf_elt), so the
        % nominal (unpoked) wavefront there is tilt-removed.  That
        % reference is element geometry, so it persists across the poke
        % traces below and the rigid-body pokes act on elts 1..nElt-2,
        % never touching elt nElt-1.  Net: the FIELD tilt is removed from
        % the nominal, but a POKE's own tilt is retained.  Writing the EP
        % here -- BEFORE dw_dx builds its channels -- lets a
        % FocalPlaneChannel ('track') save/restore the POST-reset EP pose.
        % FEX and SXP are merged in the engine, so FEX alone is well-posed
        % for all exit-pupil placements.  FEX needs an aperture stop to
        % define the chief ray; if none is set (Rx has no ApStop= and the
        % caller passed no stop_elt / stop_obj_pos) rethrow with an
        % actionable supervisor-level message.  (A stop-state preflight is
        % not viable: get_stop_info reports only element-based stops and
        % errors on an object-space stop, which FEX itself handles fine --
        % so catching FEX's own verdict is the only false-negative-free
        % check.)  The stop state is constant across the loop, so this
        % branch only ever fires on the first field.
        try
            macos.fex(1);   % mode 1 = centre on chief ray
        catch me
            if strcmp(me.identifier, 'macos:fex:noStop')
                error('macos:dw_dx_multi:noStop', ...
                    ['reset_xp=true re-references the exit pupil per ' ...
                     'field via FEX, which needs an aperture stop, but ' ...
                     'none is set.  Add "ApStop= 0 0 1" to the Rx header ' ...
                     '(or 0 0 0 for a stop at the primary), pass ' ...
                     'stop_elt / stop_obj_pos, or set reset_xp=false to ' ...
                     'keep the prescription''s frozen EP.']);
            end
            rethrow(me);
        end
    end
    sf = macos.dw_dx(session, rx_path, ...
        'dofs', opts.dofs, ...
        'elts', opts.elts, ...
        'fp_mode', opts.fp_mode, ...
        'ep_elt', opts.ep_elt, ...
        'include_source', opts.include_source, ...
        'src_stop_mode', opts.src_stop_mode, ...
        'src_stop_pos', opts.src_stop_pos, ...
        'src_stop_elt', opts.src_stop_elt, ...
        'include_non_optics', opts.include_non_optics, ...
        'rot_output', opts.rot_output, ...
        'delta', opts.delta, ...
        'delta_units', opts.delta_units, ...
        'method', opts.method, ...
        'exit_pupil_elt', opts.exit_pupil_elt, ...
        'verbose', opts.verbose, ...
        'reload_rx', false, ...
        'compute_los', opts.compute_los, ...
        'spot_elt', opts.spot_elt);
    per_field_dwdx{k}   = sf.dwdx;
    per_field_w_nom{k}  = sf.w_nom_2d;
    per_field_struct{k} = sf;
    if opts.compute_los
        per_field_dcdx{k} = sf.dcdx;
    end
    if isempty(names), names = sf.channel_names; iElt_out = sf.iElt; end
    col_rms_mean = mean(sqrt(mean(sf.dwdx.^2, 1)));
    fprintf('[field %s] dwdx shape [%d %d], mean col-RMS %.3e', ...
        fields(k).name, size(sf.dwdx, 1), size(sf.dwdx, 2), col_rms_mean);
    if opts.compute_los
        los_rms_mean = mean(sqrt(sum(sf.dcdx.^2, 2)));
        fprintf('  mean LOS-RMS %.3e', los_rms_mean);
    end
    fprintf('\n');
end

session.set_src_fov('src_pos', nom.src_pos, 'src_dir', nom.src_dir, ...
                    'zSrc', nom.zSrc);
session.modify();

% Restore the prescription's exit-pupil reference (undo the per-field FEX
% writes to elt nElt-1) so the session is left as loaded.
if opts.reset_xp
    macos.set_xp(xp0.vpt, xp0.psi, xp0.rad);
    session.modify();
end

% ---- Tile OPDall + scatter dwdxall --------------------------------
N = size(per_field_w_nom{1}, 1);
OPDall = zeros(tile_rows * N, tile_cols * N);
for k = 1:n_fields
    r0 = fields(k).tile_row * N;
    c0 = fields(k).tile_col * N;
    OPDall(r0+1:r0+N, c0+1:c0+N) = per_field_w_nom{k};
end

[w0_stacked, indxall] = macos.m2v(OPDall);
Nw = numel(w0_stacked);
Nz = size(per_field_dwdx{1}, 2);
fprintf('[stack] OPDall [%d %d]; non-zero pixels = %d\n', ...
    size(OPDall, 1), size(OPDall, 2), Nw);

dwdxall = zeros(Nw, Nz);
indx_i = indxall.i(:);
indx_j = indxall.j(:);
for k = 1:n_fields
    tr = fields(k).tile_row;
    tc = fields(k).tile_col;
    in_tile = (indx_i > tr*N) & (indx_i <= (tr+1)*N) ...
            & (indx_j > tc*N) & (indx_j <= (tc+1)*N);
    i_local = indx_i(in_tile) - tr * N;
    j_local = indx_j(in_tile) - tc * N;
    [~, field_indx] = macos.m2v(per_field_w_nom{k});
    field_i = field_indx.i(:);
    field_j = field_indx.j(:);
    flat_local = (j_local - 1) * N + i_local;
    flat_field = (field_j  - 1) * N + field_i;
    [tf, loc] = ismember(flat_local, flat_field);
    if ~all(tf)
        error('macos:dw_dx_multi:scatter', ...
            'field %s: indxall references pixels outside per-field mask', ...
            fields(k).name);
    end
    global_rows = find(in_tile);
    dwdxall(global_rows, :) = per_field_dwdx{k}(loc, :);
    fprintf('[stack] field %s: scattered %d rows into dwdxall\n', ...
        fields(k).name, numel(global_rows));
end

fprintf('[stack] dwdxall shape [%d %d]; |dwdxall| max = %.3e\n', ...
    size(dwdxall, 1), size(dwdxall, 2), max(abs(dwdxall(:))));

% Center-tile sanity check.
ctr_idx = find_center_field_index(fields);
if ~isempty(ctr_idx)
    tr = fields(ctr_idx).tile_row;
    tc = fields(ctr_idx).tile_col;
    in_ctr = (indx_i > tr*N) & (indx_i <= (tr+1)*N) ...
           & (indx_j > tc*N) & (indx_j <= (tc+1)*N);
    max_diff = max(abs(dwdxall(in_ctr, :) - per_field_dwdx{ctr_idx}), ...
                    [], 'all');
    fprintf('[check] dwdxall@center-tile vs per_field_dwdx[center]: ');
    fprintf('max|diff| = %.3e\n', max_diff);
    assert(max_diff == 0, ...
        'scatter bug: dwdxall@center-tile differs from per_field_dwdx[center]');
end

out = struct();
out.dwdxall              = dwdxall;
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
out.per_field_dwdx       = per_field_dwdx;
out.per_field_w_nom_2d   = per_field_w_nom;
out.rx_path              = rx_path;
out.delta                = opts.delta;
out.method               = opts.method;
out.wf_elt               = per_field_struct{1}.wf_elt;
out.rot_output           = opts.rot_output;
out.cbm                  = per_field_struct{1}.cbm;
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
fields = struct('name', {}, 'dx', {}, 'dy', {}, ...
                 'tile_row', {}, 'tile_col', {});
fields(end+1) = field_entry('C',  0,            0,            1, 1);
fields(end+1) = field_entry('UL', -field_x_rad, +field_y_rad, 2, 0);
fields(end+1) = field_entry('UR', +field_x_rad, +field_y_rad, 2, 2);
fields(end+1) = field_entry('LL', -field_x_rad, -field_y_rad, 0, 0);
fields(end+1) = field_entry('LR', +field_x_rad, -field_y_rad, 0, 2);
end


function fields = make_grid_field_set(nx, ny, field_x_rad, field_y_rad)
if nx > 1, dx_axis = linspace(-field_x_rad, +field_x_rad, nx);
else, dx_axis = 0; end
if ny > 1, dy_axis = linspace(-field_y_rad, +field_y_rad, ny);
else, dy_axis = 0; end
fields = struct('name', {}, 'dx', {}, 'dy', {}, ...
                 'tile_row', {}, 'tile_col', {});
for ir = 1:numel(dy_axis)
    for ic = 1:numel(dx_axis)
        dx = dx_axis(ic); dy = dy_axis(ir);
        is_center = (abs(dx) < 1e-30) && (abs(dy) < 1e-30);
        if is_center, nm = 'C';
        else, nm = sprintf('F_r%d_c%d', ir-1, ic-1); end
        fields(end+1) = field_entry(nm, dx, dy, ir-1, ic-1); %#ok<AGROW>
    end
end
end


function e = field_entry(name, dx, dy, tr, tc)
e.name = name; e.dx = dx; e.dy = dy; e.tile_row = tr; e.tile_col = tc;
end


function idx = find_center_field_index(fields)
idx = [];
for k = 1:numel(fields)
    if abs(fields(k).dx) < 1e-30 && abs(fields(k).dy) < 1e-30
        idx = k; return;
    end
end
end


function new_dir = field_to_chfraydir(dir_nom, dx_rad, dy_rad)
v = dir_nom(:) + [dx_rad; dy_rad; 0];
n = norm(v);
if n == 0
    error('macos:dw_dx_multi:zerodir', ...
        'zero-magnitude direction after field offset');
end
new_dir = v / n;
end


function [nx, ny] = parse_grid_spec(spec)
toks = regexp(lower(spec), 'x', 'split');
if numel(toks) ~= 2
    error('macos:dw_dx_multi:grid', ...
        '''grid'' must be ''NxM''; got %s', spec);
end
nx = str2double(toks{1});
ny = str2double(toks{2});
if isnan(nx) || isnan(ny) || nx < 1 || ny < 1
    error('macos:dw_dx_multi:grid', ...
        '''grid'' must be ''NxM'' with positive integers; got %s', spec);
end
end


function fields = load_field_file(fname)
fid = fopen(fname, 'r');
if fid < 0
    error('macos:dw_dx_multi:fields', ...
        'cannot open fields file: %s', fname);
end
c = onCleanup(@() fclose(fid));
fields = struct('name', {}, 'dx', {}, 'dy', {}, ...
                 'tile_row', {}, 'tile_col', {});
while true
    ln = fgetl(fid);
    if ~ischar(ln); break; end
    s = strtrim(ln);
    if isempty(s) || startsWith(s, '#'), continue; end
    toks = regexp(s, '\s+', 'split');
    if numel(toks) < 5
        error('macos:dw_dx_multi:fields', ...
            'fields-file row needs 5 columns: %s', s);
    end
    fields(end+1) = field_entry(toks{1}, ...
        str2double(toks{2}), str2double(toks{3}), ...
        str2double(toks{4}), str2double(toks{5})); %#ok<AGROW>
end
end

function mustBeDeltaSize(d)
    if ~(isequal(size(d), [1 1]) || isequal(size(d), [1 6]))
        error('macos:dw_dx_multi:deltaSize', ...
            'delta must be (1,1) or (1,6)');
    end
end
