function out = dw_dgrid_multi(session, rx_path, opts)
%MACOS.DW_DGRID_MULTI  Multi-field dw/d(grid-data) sensitivity supervisor.
%   out = macos.dw_dgrid_multi(SESSION, RX_PATH, ...) loads RX_PATH on
%   SESSION, snapshots the nominal source FoV, then loops field points --
%   each iteration absolutely sets ChfRayDir via set_src_fov and runs
%   dw_dgrid at the new field.  The per-field results are tiled into one
%   big OPDall canvas and scattered into one big dwdgall (canonical
%   state-vector form):
%
%       wall = dwdgall * x + w0_stacked
%
%   where wall is the column-major vectorisation of OPDall via m2v, and
%   x is the stacked influence-map amplitude of every grid poke on every
%   grid-bearing surface.
%
%   The grid-data (GMI pgrid) companion to macos.dw_dz_zernike_multi and
%   macos.dw_dsurf_multi, built on the same field-loop + tiling machinery,
%   a different DOF set.
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
%   OTHER NAME-VALUE PAIRS (forwarded to dw_dgrid):
%     'influence'  [N x N x K] maps for every grid element, OR a per-segment
%                  basis (a macos.segment_grid_basis struct, or a cell per grid
%                  element).  Default: a low-order Zernike-on-grid basis.
%     'zmodes'     Noll/ANSI modes for the default basis.  Default [4 5 6 7 8 11].
%     'elts'       Vector of element IDs to include.  Default [] (auto-detect all
%                  grid-bearing elements from the loaded prescription).
%     'delta', 'method', 'exit_pupil_elt', 'verbose'.
%
%   'ngridpts'  (default [] = keep the .in value) ray-grid sampling
%               override, applied once right after the Rx load; persists
%               across the per-field calls.  Clamped by the engine to
%               [3, model-size limit] (warns).
%   'reset_xp'  (default true) re-find the exit pupil (FEX, chief ray) for
%               EACH field before differencing, so the nominal wavefront is
%               referenced to that field's own chief ray and the gross field
%               TILT is removed (without it, an off-axis field carries a
%               large linear-in-field tilt that swamps the OPD canvas).  A
%               poke's OWN tilt is retained -- the reference is fixed per
%               field, not re-fit after each poke.  Requires a STOP set and
%               > 3 elements.  Set false to keep the prescription's
%               elt nElt-1 reference unchanged.
%               Restore scope: the pre-loop EP is snapshotted and restored
%               via get_xp/set_xp, i.e. vpt/psi/rad (VptElt/PsiElt/KrElt at
%               nElt-1) only.  FEX-written auxiliary fields on the EP
%               element (RptElt, zElt, fElt, eElt, KcElt) are left as
%               re-derived, not rolled back -- callers who hand-author
%               those on the EP element own re-asserting them afterward.
%   'reset_xp_method'  DEPRECATED.  FEX and SXP are merged in the engine
%               (FEX radius = chief-ray distance to iElt+1 = the FP), so
%               FEX is the only path.  'sxp' is accepted as an alias with
%               a one-time warning; do not rely on it.
%
%   OUTPUT STRUCT FIELDS:
%     dwdgall       Nw x Ng canonical state-vector Jacobian
%     dwdxall       alias (== dwdgall) -- canonical-form consumers
%     w0_stacked    Nw x 1 stacked nominal OPDs (m2v of OPDall)
%     indxall       struct with i, j, size -- m2v round-trip metadata
%     OPDall        full tiled OPD canvas (un-vectorised)
%     channel_names Ng x 1 cell array
%     iElt / map_idx  Ng x 1 per-channel element id + influence-map index
%     field_table   Nfields x 4: [dx_rad, dy_rad, tile_row, tile_col]
%     field_names   Nfields x 1 cell array
%     chfraydir_nom 3 x 1 nominal ChfRayDir
%     per_field_dwdg      Nfields x 1 cell of single-field dwdg blocks
%     per_field_w_nom_2d  Nfields x 1 cell of single-field nominal OPDs
%     rx_path / delta / method / wf_elt / zmodes  -- echoed inputs
%
%   See also: macos.dw_dgrid, macos.dw_dz_zernike_multi, macos.dw_dsurf_multi.

arguments
    session
    rx_path                     (1,:) char {mustBeNonempty}
    opts.field_x_rad            (1,1) double = NaN
    opts.field_y_rad            (1,1) double = NaN
    opts.fields                 (1,:) char = ''
    opts.grid                   (1,:) char = ''
    opts.influence              = []   % [NxNxK] | per-segment struct | cell
    opts.zmodes                 (1,:) double = [4 5 6 7 8 11]
    opts.elts                   (:,1) double = []
    opts.delta                  (1,1) double = 1e-6
    opts.method                 (1,:) char {mustBeMember(opts.method, ...
                                  {'central','forward'})} = 'central'
    opts.exit_pupil_elt         (1,1) double {mustBeInteger} = -1
    opts.reset_xp               (1,1) logical = true
    opts.verbose                (1,1) logical = false
    opts.reload_rx              (1,1) logical = true
    opts.reset_xp_method        (1,:) char {mustBeMember(opts.reset_xp_method, ...
                                  {'fex','sxp'})} = 'fex'
    opts.ngridpts               double {mustBeScalarOrEmpty} = []
    opts.src_samp               double {mustBeScalarOrEmpty, mustBeInteger} = []
    opts.compute_los            (1,1) logical = false
    opts.spot_elt               double {mustBeScalarOrEmpty, mustBeInteger} = []
end

if isnan(opts.field_x_rad) || isnan(opts.field_y_rad)
    error('macos:dw_dgrid_multi:fov', ...
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
% reload_rx=false lets the caller pre-load the Rx AND install state that
% must persist across the field loop -- e.g. a nominal grid figure added
% with macos.elt_grid_add (a fresh load_rx here would wipe it).  Each
% field below runs dw_dgrid with reload_rx=false, so a single load is all
% that happens regardless.
if opts.reload_rx
    session.load_rx(rx_path);
end
apply_ngridpts(session, opts.ngridpts, 'dw_dgrid_multi');

% Apply source sampling if specified
if ~isempty(opts.src_samp)
    session.set_src_sampling(opts.src_samp);
    session.modify();  % Flush cache so the new sampling takes effect
end

nom = session.get_src_fov();
fprintf('[setup] nominal ChfRayDir = [%g %g %g]; zSrc = %.3e\n', ...
    nom.src_dir, nom.zSrc);

% reset_xp_method is DEPRECATED: FEX and SXP are merged in the engine, so
% FEX is the only path now.  'sxp' is accepted as an alias (SegDemo3-era
% scripts pass it) but warned once per session.
if strcmp(opts.reset_xp_method, 'sxp')
    warn_reset_xp_method_deprecated_();
end

% Snapshot the prescription's exit-pupil reference (elt nElt-1 geometry)
% so the per-field FEX resets can be undone before returning.  Track
% whether FEX actually moves the EP (writes only into a Return/Reference
% nElt-1; no-pupil decks are silent no-ops) + guard a powered-optic
% clobber -- see private/reset_xp_guard.
reset_ep_moved = false;
if opts.reset_xp
    xp0 = macos.get_xp();
    ep_is_powered = reset_xp_guard('is_powered', session);
end

% Resolve the influence basis ONCE so every field shares identical
% columns (same channel order, same map amplitudes).  When the caller
% supplies 'influence' it is used verbatim; otherwise a default
% Zernike-on-grid basis is built at the first grid element's sampling.
% Building here (after load_rx) -- rather than letting each per-field
% dw_dgrid rebuild it -- guarantees consistency and avoids redundant
% basis evaluations across the field loop.
infl = opts.influence;
if isempty(infl)
    g = macos.find_grid_elts();
    if ~isempty(opts.elts)
        g = intersect(g, opts.elts);
    end
    if isempty(g)
        error('macos:dw_dgrid_multi:nogrid', ...
            'no grid-bearing elements in the loaded prescription');
    end
    nsz  = double(mmacos('elt_srf_grid_size', g(1), 1));
    infl = macos.zernike_grid_basis(nsz, opts.zmodes);
end

% ---- Per-field loop -----------------------------------------------
% Each iteration uses set_src_fov ABSOLUTELY (no perturb-undo needed) +
% modify() to flush the trace cache so the new ChfRayDir takes effect,
% and passes reload_rx=false to dw_dgrid so it does NOT call load_rx
% (which would reset ChfRayDir back to the prescription nominal).
per_field_dwdg   = cell(n_fields, 1);
per_field_w_nom  = cell(n_fields, 1);
per_field_struct = cell(n_fields, 1);
if opts.compute_los
    per_field_dcdx = cell(n_fields, 1);
end
names = {};  iElt_out = [];  map_idx_out = [];
for k = 1:n_fields
    new_dir = field_to_chfraydir(nom.src_dir, fields(k).dx, fields(k).dy);
    session.set_src_fov('src_pos', nom.src_pos, 'src_dir', new_dir, ...
                        'zSrc', nom.zSrc);
    session.modify();   % flush trace cache so the new dir takes effect
    fprintf('[field %s] ChfRayDir = [%g %g %g]\n', ...
        fields(k).name, new_dir);
    if opts.reset_xp
        % Re-reference this field's exit pupil to its OWN chief ray: FEX
        % writes the reference sphere into elt nElt-1 (= wf_elt), so the
        % nominal (unpoked) wavefront there is tilt-removed.  That reference
        % is element geometry, so it persists across the poke traces below
        % -- and the grid pokes live on elts 1..nElt-2, never touching elt
        % nElt-1.  Net: the FIELD tilt is removed from the nominal, but a
        % POKE's own tilt is retained in the sensitivity (the reference is
        % NOT re-fit after poking).
        % FEX and SXP are merged in the engine: post-rework FEX sets the
        % EP reference radius to the chief-ray distance to iElt+1 (= the
        % FP), identical to what SXP produced -- so FEX is the single
        % well-posed path for all EP placements (including the near-EP
        % SegDemo3* layouts that once needed SXP).  reset_xp_method is
        % retained only as a deprecated alias (warned once above).
        % shared guard: raises the supervisor-level no-stop error and
        % absorbs the no-pupil-element FAIL -- see private/reset_xp_guard.
        reset_xp_guard('fex', session);
        reset_ep_moved = reset_xp_guard('check', session, xp0, ...
            reset_ep_moved, ep_is_powered);
    end
    sf = macos.dw_dgrid(session, rx_path, ...
        'influence', infl, ...
        'elts', opts.elts, ...
        'delta', opts.delta, ...
        'method', opts.method, ...
        'exit_pupil_elt', opts.exit_pupil_elt, ...
        'verbose', opts.verbose, ...
        'reload_rx', false, ...
        'compute_los', opts.compute_los, ...
        'spot_elt', opts.spot_elt);    % keep current src_fov state
    per_field_dwdg{k} = sf.dwdg;
    per_field_w_nom{k} = sf.w_nom_2d;
    per_field_struct{k} = sf;
    if opts.compute_los
        per_field_dcdx{k} = sf.dcdx;
    end
    if isempty(names)
        names = sf.channel_names;  iElt_out = sf.iElt;  map_idx_out = sf.map_idx;
    end
    col_rms_mean = mean(sqrt(mean(sf.dwdg.^2, 1)));
    fprintf('[field %s] dwdg shape [%d %d], mean col-RMS %.3e', ...
        fields(k).name, size(sf.dwdg, 1), size(sf.dwdg, 2), col_rms_mean);
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
reset_xp_stamp = reset_xp_guard('finalize', opts.reset_xp, ...
    reset_ep_moved, session.num_elt() - 1);

% ---- Tile OPDall + scatter dwdgall --------------------------------
N = size(per_field_w_nom{1}, 1);
OPDall = zeros(tile_rows * N, tile_cols * N);
for k = 1:n_fields
    r0 = fields(k).tile_row * N;
    c0 = fields(k).tile_col * N;
    OPDall(r0+1:r0+N, c0+1:c0+N) = per_field_w_nom{k};
end

[w0_stacked, indxall] = macos.m2v(OPDall);
Nw = numel(w0_stacked);
Ng = size(per_field_dwdg{1}, 2);
fprintf('[stack] OPDall [%d %d]; non-zero pixels = %d\n', ...
    size(OPDall, 1), size(OPDall, 2), Nw);

dwdgall = zeros(Nw, Ng);
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
    % to the per-field dwdg rows.
    [~, field_indx] = macos.m2v(per_field_w_nom{k});
    field_i = field_indx.i(:);
    field_j = field_indx.j(:);
    flat_local  = (j_local - 1) * N + i_local;
    flat_field  = (field_j  - 1) * N + field_i;
    [tf, loc] = ismember(flat_local, flat_field);
    if ~all(tf)
        error('macos:dw_dgrid_multi:scatter', ...
            'field %s: indxall references pixels outside per-field mask', ...
            fields(k).name);
    end
    global_rows = find(in_tile);
    dwdgall(global_rows, :) = per_field_dwdg{k}(loc, :);
    fprintf('[stack] field %s: scattered %d rows into dwdgall\n', ...
        fields(k).name, numel(global_rows));
end

fprintf('[stack] dwdgall shape [%d %d]; |dwdgall| max = %.3e\n', ...
    size(dwdgall, 1), size(dwdgall, 2), max(abs(dwdgall(:))));

% Center-tile sanity check: the (0,0) field's rows in dwdgall must
% exactly match its per_field_dwdg block (max|diff| = 0 in practice).
ctr_idx = find_center_field_index(fields);
if ~isempty(ctr_idx)
    tr = fields(ctr_idx).tile_row;
    tc = fields(ctr_idx).tile_col;
    in_ctr = (indx_i > tr*N) & (indx_i <= (tr+1)*N) ...
           & (indx_j > tc*N) & (indx_j <= (tc+1)*N);
    dwdgall_ctr = dwdgall(in_ctr, :);
    dwdg_C = per_field_dwdg{ctr_idx};
    max_diff = max(abs(dwdgall_ctr(:) - dwdg_C(:)));
    fprintf('[check] dwdgall@center-tile vs per_field_dwdg[center]: ');
    fprintf('max|diff| = %.3e ([%d %d])\n', ...
        max_diff, size(dwdgall_ctr, 1), size(dwdgall_ctr, 2));
    assert(max_diff == 0, ...
        'scatter bug: dwdgall@center-tile differs from per_field_dwdg[center]');
else
    fprintf('[check] no (0,0)-offset field -- skipping center-tile check\n');
end

% ---- Pack output struct -------------------------------------------
out = struct();
out.dwdgall              = dwdgall;
out.dwdxall              = dwdgall;     % canonical-form alias
out.w0_stacked           = w0_stacked;
out.indxall              = indxall;
out.OPDall               = OPDall;
out.channel_names        = names;
out.iElt                 = iElt_out;
out.map_idx              = map_idx_out;
out.field_table          = arrayfun( ...
    @(s) [s.dx, s.dy, s.tile_row, s.tile_col], fields, ...
    'UniformOutput', false);
out.field_table          = vertcat(out.field_table{:});
out.field_names          = {fields.name}.';
out.chfraydir_nom        = nom.src_dir(:);
out.per_field_dwdg       = per_field_dwdg;
out.per_field_w_nom_2d   = per_field_w_nom;
out.rx_path              = rx_path;
out.delta                = opts.delta;
out.method               = opts.method;
out.wf_elt               = per_field_struct{1}.wf_elt;
out.zmodes               = opts.zmodes;
out.reset_xp             = reset_xp_stamp;   % true | false | 'no-effect'

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
    error('macos:dw_dgrid_multi:zerodir', ...
        'zero-magnitude direction after field offset');
end
new_dir = v / n;
end


function [nx, ny] = parse_grid_spec(spec)
toks = regexp(lower(spec), 'x', 'split');
if numel(toks) ~= 2
    error('macos:dw_dgrid_multi:grid', ...
        '''grid'' must be ''NxM'' (e.g. ''3x3''); got %s', spec);
end
nx = str2double(toks{1});  ny = str2double(toks{2});
if isnan(nx) || isnan(ny) || nx < 1 || ny < 1
    error('macos:dw_dgrid_multi:grid', ...
        '''grid'' must be ''NxM'' with positive integers; got %s', spec);
end
end


function fields = load_field_file(fname)
% Free-form list: lines of 'name dx_rad dy_rad tile_row tile_col'.
fid = fopen(fname, 'r');
if fid < 0
    error('macos:dw_dgrid_multi:fields', ...
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
        error('macos:dw_dgrid_multi:fields', ...
            'fields-file row needs 5 columns: %s', s);
    end
    fields(end+1) = field_entry(toks{1}, ...
        str2double(toks{2}), str2double(toks{3}), ...
        str2double(toks{4}), str2double(toks{5})); %#ok<AGROW>
end
end


function warn_reset_xp_method_deprecated_()
% One-time-per-session deprecation notice for reset_xp_method='sxp'.
persistent warned
if isempty(warned)
    warning('macos:dw_dgrid_multi:resetXpMethodDeprecated', ...
        ['reset_xp_method is deprecated: FEX and SXP are merged in the ' ...
         'engine, so FEX is used for the per-field exit-pupil reset ' ...
         'regardless of this option.  ''sxp'' is accepted as an alias.']);
    warned = true;
end
end
