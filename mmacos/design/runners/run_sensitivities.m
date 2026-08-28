function art = run_sensitivities(rx_in, opts)
%RUN_SENSITIVITIES  Sensitivity stage runner: .in -> dwdx/dwdz/dwdgrid.
%
%   art = run_sensitivities(RX_IN, 'fov_rad', F, ...) is the
%   sensitivities stage of the design pipeline
%       design -> segmentation -> sensitivities -> MET -> compare -> simulate
%   (see design/runners/README.md).  It harvests the three wavefront
%   Jacobian channels on RX_IN over a multi-point field set, each in
%   the canonical state-vector form  wall = J*x + w0_stacked
%   (the macos.dw_d*_multi supervisors -- the TESTED configuration of
%   the sensitivities/ examples; Dave 2026-07-19: reuse runners, don't
%   rewrite per case):
%
%     dwdx     rigid-body 6-DOF per element, LOCAL/TElt triads
%              (macos.dw_dx_multi, fp_mode='track')
%     dwdz     segment-LOCAL MonZernike figure modes
%              (macos.dw_dz_zernike_multi, kinds={'monzern'})
%     dwdgrid  per-segment grid-poke channel: RX_IN is grid-augmented
%              in each segment's CLOCKED Mon frame
%              (macos.design.grid_augment_rx -- REPLACES any stale
%              parent-frame grid lines, the e5-corpus trap), pokes are
%              a per-segment Gram-Schmidt Zernike influence basis
%              (macos.segment_grid_basis + macos.dw_dgrid_multi)
%
%   Artifacts (in OUT_DIR, default beside RX_IN):
%     <name>_sens.mat          ox / oz / og supervisor outputs + config
%     <name>_grid.in           the grid-augmented Rx (+ flat grid file)
%     <name>_sens_report.txt   sizes, conditioning (all-column AND
%                              segment-only -- non-segment lMon optics
%                              inflate the raw number), per-segment
%                              column norms
%     figures                  nominal-OPD field canvas, SV spectra,
%                              per-channel overviews, and the e5-style
%                              PER-ELEMENT pages (center + multi field)
%
%   REQUIRED:
%     'fov_rad'    half-field (rad): field set = center + 4 corners
%                  (pass 'fields' or 'grid' through to the supervisors
%                  for other sets)
%   OPTIONS:
%     'configs'    1xNc struct array of CONFIGURATIONS (default [] =
%                  today's single-block call, byte-identical).  Each
%                  entry is .name + .set, a cell of setter invocations
%                  {fname, elt, args...} applied to every channel's
%                  harvest; the Jacobian is then evaluated per
%                  (configuration, field) and the blocks stack as extra
%                  ROWS.  Pass-through to the dw_d*_multi supervisors --
%                  they own validation, the modify()-after-setters rule,
%                  and the snapshot / restore / ASSERT cycle.  Build one
%                  with macos.design.configs_from_table.  See
%                  design/PLAN_CONFIGURATIONS.md.
%     'resume_dir' checkpoint directory for a long multi-configuration
%                  harvest (default "" = off).  With more than one
%                  configuration, each channel's per-configuration block
%                  is saved there as it completes and reloaded instead of
%                  recomputed on a restart, then the blocks are stitched
%                  into exactly the array a single all-configurations
%                  call produces -- the stitch goes through
%                  macos.config_canvas, the same function the
%                  supervisors use, so the tile layout and the
%                  configuration-major row order come out the same way.
%                  The directory is pruned when the run completes.  A 5x5
%                  harvest at model 512 is a multi-hour run; without this
%                  a kill at block 24 costs the whole thing.
%     'stop_elt'   set the aperture stop at this ELEMENT ([] = off).  The
%                  alternative to injecting a header ApStop= position via
%                  'stop' -- the zoom fixture's pupil IS an element (its
%                  steering mirror), and the deck carries no ApStop=.
%     'channels'   subset of {'dwdx','dwdz','dwdgrid'} (default all)
%     'dofs'       dwdx DOF subset (default (0:5)': Rx..Tz)
%     'elts'       element subset for the dwdx / dwdz / dwdgrid channel
%                  builders ([] = every eligible optic).  A cheap way to
%                  scope a harvest to the optics you are actually
%                  budgeting.
%     'groups'     containers.Map name -> column vector of element ids:
%                  RIGID-BODY groups, perturbed as one unit (GPERTURB).
%                  Their 6 columns per group are APPENDED AFTER the
%                  per-element block of the dwdx channel.  Default [].
%     'groups_auto' parse EltGrp= declarations out of RX_IN and add them
%                  as groups (positive-N explicit lists, deduped by
%                  member tuple).  Default false.  Merged with 'groups';
%                  an explicit entry wins a name collision.
%     'group_coords'    'global' (default) | 'local' (ref-elt TElt frame)
%     'group_fp_mode'   'auto' (default) | 'none' | 'sxp' | 'srs' --
%                  post-perturbation follow-up for a group that contains
%                  the focal plane
%     'group_stop_mode' 'obj' (default) | 'elt' | 'none'
%     'group_stop_pos'  1x3 object-space stop coords.  Default [0 0 0].
%                  SCOPE: groups are RIGID-BODY groups and reach the
%                  'dwdx' channel ONLY.  dwdz / dwdsurf / dwdgrid are
%                  figure/surface channel kinds with no group analogue
%                  in the engine; grouping them is deferred.
%     'zmodes_fig' dwdz MonZernike modes (default 4:11)
%     'zmodes_grid' dwdgrid basis modes (default 4:9)
%     'ng'         grid size for the augmentation (default 256)
%     'span_frac'  grid span fraction of the parent Aperture (1.0 = the dxGrid convention)
%     'pm_ref_elt' segment_grid_basis footprint target (default 1)
%     'reset_xp_method'  'fex' (default) | 'sxp' (deprecated alias of fex
%                  -- FEX and SXP are merged in the engine) |
%                  'pupil_find': place the cone-convergence best-fit
%                  exit-pupil sphere (design/src/pupil_find).
%                  Needs a stop: 'stop_elt', or the deck's own ApStop=
%                  (object-space header form included -- the segmented-
%                  primary idiom); 'pupil_find_opts' forwards
%                  finder name-values.  Fit metrics land in the report.
%                  'pf_scope' 'field' (default; flipped from 'config'
%                  2026-08-27, the w_nom audit): one 3x3 MINI-CONE fit
%                  per (configuration, field) block, half-width
%                  'pf_probe_rad' (NaN = 0.15x the field half-width),
%                  centered on that field -- the sphere axis parallels
%                  the combo's own chief, the field tilt is absorbed
%                  per block, and each block subtracts its OWN w_nom;
%                  nominals and Jacobian match fex (4e-11 mm / 1e-6).
%                  'config': ONE frozen field-set-wide sphere per
%                  configuration -- a frozen-reference / pupil-wander
%                  DIAGNOSTIC: the per-field tilt stays in w_nom
%                  (0.64 mm RMS at +-1' on the zoom fixture) and leaks
%                  3-5% into the rigid-body dwdx columns (dw_dx_multi's
%                  help has the full story and the conditioning caveat).
%                  Historical: 'sxp' was accepted as an alias
%                  (warned once); retained so near-EP legacy decks that
%                  pass it keep running.
%     'delta_x','delta_z','delta_g'  FD steps ([] = supervisor default;
%                  native units -- mind BaseUnits m vs mm)
%     'ngridpts'   ray-grid override ([] = keep the .in value); a
%                  coarse grid is a legitimate memory/runtime lever
%     'model_size' engine model (default 512, >= ng)
%     'out_dir','name','visible','verbose'  as in run_met
%     'per_element' per-element page field modes (default
%                  ["center" "multi"]; [] = skip the pages).  The pages
%                  are numerous, so they land in the <name>_pages/
%                  subfolder; the top level keeps only the overview
%                  figures (Dave 2026-07-19)
%
%   art: ox/oz/og + artifact paths + conditioning table.
%
%   See also: macos.dw_dx_multi, macos.dw_dz_zernike_multi,
%             macos.dw_dgrid_multi, macos.design.grid_augment_rx,
%             macos.segment_grid_basis, run_met.

arguments
    rx_in (1,1) string
    opts.fov_rad (1,1) double {mustBePositive}
    opts.configs = []           % configuration axis (see above)
    opts.resume_dir (1,1) string = ""   % per-configuration checkpoints
    opts.stop_elt double = []   % aperture stop AT an element
    opts.channels string = ["dwdx" "dwdz" "dwdgrid"]  % + "dwdsurf" opt-in
    opts.dofs (:,1) double = (0:5).'
    opts.elts (:,1) double = []   % element subset ([] = all eligible)
    opts.groups = []            % containers.Map name -> member ids
                                % (rigid-body groups; dwdx channel ONLY)
    opts.groups_auto (1,1) logical = false   % parse EltGrp= from RX_IN
    opts.group_coords (1,:) char {mustBeMember(opts.group_coords, ...
        {'global','local'})} = 'global'
    opts.group_fp_mode (1,:) char {mustBeMember(opts.group_fp_mode, ...
        {'auto','none','sxp','srs'})} = 'auto'
    opts.group_stop_mode (1,:) char {mustBeMember(opts.group_stop_mode, ...
        {'obj','elt','none'})} = 'obj'
    opts.group_stop_pos (1,3) double = [0 0 0]
    opts.zmodes_fig (1,:) double = 4:11
    opts.zmodes_grid (1,:) double = 4:9
    opts.zkinds cell = {'monzern'}   % dwdz kinds: subset of
                                     % {'monzern','ffzern','zern'};
                                     % monzern = the SEGMENT-LOCAL basis
    opts.surf_params cell = {'Kr','Kc'}  % dwdsurf channel parameters
    opts.surf_remove_ptt (1,1) logical = false  % dwdsurf: project piston +
                                     % tip + tilt out of each Kr/Kc response
                                     % (aligned out during assembly)
    opts.grid_basis (1,:) char {mustBeMember(opts.grid_basis, ...
        {'multi','single'})} = 'multi'   % per-segment bespoke basis
                                % (segment_grid_basis; the general case)
                                % vs ONE reference-segment basis shared
                                % by every segment
                                % (gs_zernike_segment_basis; cheaper,
                                % exact for congruent segments)
    opts.ref_seg (1,1) double {mustBeInteger, mustBePositive} = 1
    opts.influence = []         % explicit influence basis for dwdgrid
                                % ([NxNxK] maps or a per-segment struct,
                                % e.g. DM actuators): used VERBATIM on
                                % the Rx's EXISTING grids -- no
                                % augmentation, no basis build; the
                                % caller owns frame/span consistency
    opts.ng (1,1) double {mustBeInteger, mustBePositive} = 256
    opts.span_frac (1,1) double {mustBePositive} = 1.0
    opts.pm_ref_elt (1,1) double {mustBeInteger, mustBePositive} = 1
    opts.reset_xp_method (1,:) char {mustBeMember(opts.reset_xp_method, ...
        {'fex','sxp','pupil_find'})} = 'fex'
    opts.pupil_find_opts cell = {}
    opts.pf_scope (1,:) char {mustBeMember(opts.pf_scope, ...
        {'config','field'})} = 'field'
    opts.pf_probe_rad (1,1) double = NaN
    opts.delta_x double = []
    opts.delta_z double = []
    opts.delta_g double = []
    opts.stop double = []       % 3-vector StopPos POSITION to inject as
                                % a header ApStop= when RX_IN has none
                                % ([0 0 0] = stop at the PM).  The
                                % exit-pupil machinery (fp 'track', FEX/
                                % SXP resets) needs a stop; SMM-derived
                                % fixtures (e5 corpus) ship without one.
    opts.ngridpts double = []
    opts.model_size (1,1) double = 512
    opts.out_dir (1,1) string = ""
    opts.name (1,1) string = ""
    opts.visible (1,1) logical = false
    opts.verbose (1,1) logical = true
    opts.per_element string = ["center" "multi"]
end
assert(isfile(rx_in), 'run_sensitivities: %s not found', rx_in);
[rx_dir, stem] = fileparts(char(rx_in));
if strlength(opts.out_dir) == 0, opts.out_dir = string(rx_dir); end
if strlength(opts.name) == 0, opts.name = string(stem); end
od   = char(opts.out_dir);
name = char(opts.name);
if ~isfolder(od), mkdir(od); end
% The engine resolves GridFile= names relative to the cwd at load time
% (both RX_IN's own grid refs and the augmented Rx's), so run the whole
% harvest from OUT_DIR; grid files must sit beside RX_IN/OUT_DIR.
rx_in = string(fullpath_(char(rx_in)));
oldd = cd(od);  restore_cwd = onCleanup(@() cd(oldd));

log_ = fopen(fullfile(od, [name '_sens_report.txt']), 'w');
closer = onCleanup(@() fclose(log_));
if opts.verbose
    say = @(varargin) fprintf(1, varargin{:}) + fprintf(log_, varargin{:});
else
    say = @(varargin) fprintf(log_, varargin{:});
end
say('==== run_sensitivities: %s ====\n', char(rx_in));
say('field set: center + 4 corners at +-%.4g rad\n', opts.fov_rad);

% preflight: the exit-pupil machinery needs an aperture stop
txt = fileread(char(rx_in));
if isempty(regexp(txt, '^\s*ApStop=', 'once', 'lineanchors')) ...
        && isempty(opts.stop_elt)
    assert(~isempty(opts.stop), ['run_sensitivities: %s declares no ' ...
        'ApStop (exit-pupil machinery needs one). Add "ApStop= 0 0 0" ' ...
        'to the Rx header, or pass ''stop'', [0 0 0] to inject that ' ...
        'position, or ''stop_elt'', N to set the stop AT an element ' ...
        '(the zoom fixture''s pupil IS an element).'], char(rx_in));
    rx_stop = fullfile(od, [name '_stop.in']);
    ap = sprintf('           ApStop=%s', sprintf('  %.6E', opts.stop));
    L0 = splitlines(string(txt));
    % ApStop must land in the HEADER key chain: after Obscratn= /
    % Aperture= if present, else before nElt= (the parser leaves the
    % header at nElt/Element -- a line after that is silently ignored)
    ie = find(startsWith(strtrim(L0), 'Obscratn='), 1);
    if isempty(ie), ie = find(startsWith(strtrim(L0), 'Aperture='), 1); end
    if ~isempty(ie)
        L0 = [L0(1:ie); string(ap); L0(ie+1:end)];
    else
        ie = find(startsWith(strtrim(L0), 'nElt=') | ...
                  startsWith(strtrim(L0), 'Element='), 1);
        L0 = [L0(1:ie-1); string(ap); L0(ie:end)];
    end
    fid = fopen(rx_stop, 'w'); fprintf(fid, '%s\n', L0); fclose(fid);
    say('NOTE: no ApStop in the Rx -- injected StopPos %s (-> %s)\n', ...
        mat2str(opts.stop(:).'), rx_stop);
    rx_in = string(rx_stop);
    txt = fileread(char(rx_in));
end

% segment census from the prescription text (dwdz/dwdgrid need them)
nseg = numel(regexp(txt, 'Element=\s*Segment', 'match'));
segs = 1:nseg;
say('segments: %d\n\n', nseg);

m = macos.Session(opts.model_size);
FOV = opts.fov_rad;
sup = {'field_x_rad', FOV, 'field_y_rad', FOV, 'ngridpts', opts.ngridpts, ...
       'reset_xp_method', opts.reset_xp_method, ...
       'pupil_find_opts', opts.pupil_find_opts, ...
       'pf_scope', opts.pf_scope, 'pf_probe_rad', opts.pf_probe_rad};
if strcmp(opts.reset_xp_method, 'pupil_find')
    % the finder lives in design/src, which is not on the default path
    addpath(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'src'));
    % No pfNeedsStopElt refusal here: the general preflight above already
    % guarantees a stop in SOME form (deck ApStop= header, injected
    % 'stop', or 'stop_elt'), and pupil_find now honors a deck-declared
    % stop -- the object-space / segmented-primary idiom (Luis
    % 2026-08-26).  The guard's own check remains for the truly stopless
    % configured state.
end
if ~isempty(opts.stop_elt)
    sup = [sup, {'stop_elt', opts.stop_elt}];
end
% the configuration axis is passed PER CALL (run_channel_ substitutes a
% single configuration when checkpointing), never baked into sup
CF = opts.configs;
RD = opts.resume_dir;
% Checkpoint key: the resume files must be method-aware.  A checkpoint
% written under fex and resumed under pupil_find is served VERBATIM (the
% resume key used to be channel+config only), silently making the two
% methods' outputs identical -- exactly the trap Luis hit.  'fex'/'sxp'
% keep the historical bare filenames; 'pupil_find' forks its own.
XK = '';
if strcmp(opts.reset_xp_method, 'pupil_find')
    XK = '_pf';
    if strcmp(opts.pf_scope, 'field'), XK = '_pff'; end
end
ncfg = numel(opts.configs);
if ncfg > 0
    say('configurations: %d\n', ncfg);
    for c = 1:ncfg
        d = cfg_describe_(opts.configs(c));
        say('  %-10s %s\n', opts.configs(c).name, d{1});
        for q = 2:numel(d), say('  %-10s %s\n', '', d{q}); end
    end
end

ox = [];  oz = [];  og = [];  os = [];  rxg = '';
%% dwdx
if any(opts.channels == "dwdx")
    say('[dwdx] rigid-body 6-DOF per element, LOCAL triads...\n');
    a = {};  if ~isempty(opts.delta_x), a = {'delta', opts.delta_x}; end
    % groups reach the dwdx channel ONLY -- they are RIGID-BODY groups;
    % the figure/surface channels have no group analogue in the engine
    g = {'groups', opts.groups, 'groups_auto', opts.groups_auto, ...
         'group_coords', opts.group_coords, ...
         'group_fp_mode', opts.group_fp_mode, ...
         'group_stop_mode', opts.group_stop_mode, ...
         'group_stop_pos', opts.group_stop_pos};
    ox = run_channel_(@(cf) macos.dw_dx_multi(m, char(rx_in), sup{:}, ...
        'configs', cf, 'dofs', opts.dofs, 'elts', opts.elts, g{:}, a{:}), ...
        'dwdx', CF, RD, say, XK);
    ngrp = 0;   % isfield guard: a checkpoint written before the
                % supervisor carried 'kind' resumes without it
    if isfield(ox, 'kind'), ngrp = nnz(strcmp(ox.kind, 'Group')); end
    if ngrp > 0
        say('    + %d group channel(s) appended after the per-element block\n', ...
            ngrp);
    end
    say('    dwdxall %d x %d over %d fields\n\n', size(ox.dwdxall, 1), ...
        size(ox.dwdxall, 2), size(ox.field_table, 1));
    say_pf_(say, ox, 'dwdx');
end

%% dwdz (segment-LOCAL MonZernike)
if any(opts.channels == "dwdz") && nseg > 0
    say('[dwdz] segment-LOCAL MonZernike modes %s...\n', mat2str(opts.zmodes_fig));
    a = {};  if ~isempty(opts.delta_z), a = {'delta', opts.delta_z}; end
    oz = run_channel_(@(cf) macos.dw_dz_zernike_multi(m, char(rx_in), sup{:}, ...
        'configs', cf, 'kinds', opts.zkinds, 'elts', opts.elts, ...
        'zmode_start', opts.zmodes_fig(1), ...
        'n_zcoef', opts.zmodes_fig(end), a{:}), 'dwdz', CF, RD, say, XK);
    say('    dwdzall %d x %d\n\n', size(oz.dwdxall, 1), size(oz.dwdxall, 2));
    say_pf_(say, oz, 'dwdz');
end

%% dwdsurf (per-element Kr/Kc -- opt-in)
if any(opts.channels == "dwdsurf")
    say('[dwdsurf] per-element %s...\n', strjoin(opts.surf_params, '/'));
    os = run_channel_(@(cf) macos.dw_dsurf_multi(m, char(rx_in), sup{:}, ...
        'configs', cf, 'params', opts.surf_params, ...
        'remove_ptt', opts.surf_remove_ptt), 'dwdsurf', CF, RD, say, XK);
    say('    dwdsall %d x %d\n\n', size(os.dwdxall, 1), size(os.dwdxall, 2));
    say_pf_(say, os, 'dwdsurf');
end

%% dwdgrid (per-segment G-S basis on the grid-augmented Rx, or a
%% caller-supplied influence basis on the Rx's existing grids)
if any(opts.channels == "dwdgrid") && (nseg > 0 || ~isempty(opts.influence))
    if ~isempty(opts.influence)
        say('[dwdgrid] caller-supplied influence basis on the Rx grids...\n');
        rxg = char(rx_in);
        sgb = opts.influence;
    else
        say('[dwdgrid] grid-augmenting in the clocked Mon frames...\n');
        rxg = fullfile(od, [name '_grid.in']);
        ga = macos.design.grid_augment_rx(rx_in, rxg, ...
            'ng', opts.ng, 'span_frac', opts.span_frac);
        if any(ga.replaced)
            say('    NOTE: stale grid lines replaced in %d segment blocks\n', ...
                nnz(ga.replaced));
        end
        nge = macos.load_rx(rxg);  tg = macos.trace(nge);
        say('    %s_grid.in: %d elts, %d/%d rays, gdx %.4g\n', name, nge, ...
            nnz(logical(macos.get_ray_info(tg.nRays).ok_pass)), tg.nRays, ga.gdx(1));
        if strcmp(opts.grid_basis, 'multi')
            sgb = macos.segment_grid_basis(m, rxg, 'pm_ref_elt', opts.pm_ref_elt, ...
                'modes', opts.zmodes_grid, 'orthogonalize', true);
        else
            seg_elts = find_seg_elts_(txt);
            sgb = macos.gs_zernike_segment_basis(m, rxg, ...
                'pm_ref_elt', opts.pm_ref_elt, 'ref_seg', opts.ref_seg, ...
                'seg_elts', seg_elts, 'modes', opts.zmodes_grid);
        end
    end
    a = {};  if ~isempty(opts.delta_g), a = {'delta', opts.delta_g}; end
    og = run_channel_(@(cf) macos.dw_dgrid_multi(m, rxg, sup{:}, ...
        'configs', cf, 'influence', sgb, 'elts', opts.elts, ...
        a{:}), 'dwdgrid', CF, RD, say, XK);
    % persist the influence basis WITH the harvest: the basis is part
    % of the Jacobian's definition, and rebuilding it in a later
    % session is not bit-stable (the last G-S mode can come out
    % rotated -- caught by run_compare 2026-07-19); downstream
    % consumers (run_compare pokes, dmdgrid) use og.sgb verbatim
    og.sgb = sgb;
    say('    dwdgall %d x %d (influence basis saved in og.sgb)\n\n', ...
        size(og.dwdgall, 1), size(og.dwdgall, 2));
    say_pf_(say, og, 'dwdgrid');
end

%% conditioning report
say('[conditioning] finite rows only:\n');
say('    %-8s %12s %6s %10s %10s %10s\n', 'channel', 'size', 'rank', ...
    'sv_max', 'sv_min+', 'cond+');
tab = {};  SV = {};  SVc = {};
J = {'dwdx', ox; 'dwdz', oz; 'dwdgrid', og; 'dwdsurf', os};
for q = 1:size(J, 1)
    o = J{q, 2};
    if isempty(o), continue; end
    A = jmat_(o);  A = A(all(isfinite(A), 2), :);
    s = svd(full(A), 'econ');  SV{end+1} = s; %#ok<AGROW>
    tol = max(size(A)) * eps(max(s));
    sp = s(s > tol);
    tab(end+1, :) = {J{q,1}, size(A), nnz(s > tol), s(1), sp(end), s(1)/sp(end)}; %#ok<AGROW>
    say('    %-8s %5dx%-6d %6d %10.3e %10.3e %10.3e\n', J{q,1}, ...
        size(A,1), size(A,2), nnz(s > tol), s(1), sp(end), s(1)/sp(end));
    % per-configuration conditioning.  The STACKED number above is the
    % design-relevant one -- it is the conditioning of the estimation
    % problem you actually have -- but a single configuration that is
    % far worse than the stack is worth seeing.
    if ncfg > 0 && isfield(o, 'indxall') && isfield(o.indxall, 'config')
        A0 = jmat_(o);
        for c = 1:ncfg
            rows = (o.indxall.config == c);
            Ac = A0(rows, :);  Ac = Ac(all(isfinite(Ac), 2), :);
            if isempty(Ac), continue; end
            sc = svd(full(Ac), 'econ');
            SVc{end+1} = struct('ch', J{q,1}, 'cfg', ...
                opts.configs(c).name, 's', sc); %#ok<AGROW>
            tolc = max(size(Ac)) * eps(max(sc));
            spc = sc(sc > tolc);
            say('    %-8s cfg %-8s %5dx%-6d %6d %10.3e %10.3e %10.3e\n', ...
                J{q,1}, opts.configs(c).name, size(Ac,1), size(Ac,2), ...
                nnz(sc > tolc), sc(1), spc(end), sc(1)/spc(end));
        end
    end
    % segment-only restriction: non-segment lMon optics legitimately
    % appear in dwdz but inflate cond+ (e5pie: 9.2e3 -> 5.4)
    if nseg > 0 && ~strcmp(J{q,1}, 'dwdgrid')
        cn = o.channel_names;
        isseg = false(size(cn));
        for s2 = segs
            isseg = isseg | startsWith(cn, sprintf('Elt %d ', s2));
        end
        if any(isseg) && ~all(isseg)
            As = jmat_(o);  As = As(all(isfinite(As), 2), isseg);
            ss = svd(full(As), 'econ');
            say('    %-8s segment-only: %dx%d  cond+ %.3e\n', J{q,1}, ...
                size(As, 1), size(As, 2), ss(1)/ss(end));
        end
    end
end
if ~isempty(ox) && nseg > 0
    say('\n    dwdx per-segment column norms (rms native-unit per unit DOF):\n');
    say('      seg     rx        ry        rz        tx        ty        tz\n');
    cn = ox.channel_names;
    for s2 = segs
        ic = find(startsWith(cn, sprintf('Elt %d ', s2)));
        if numel(ic) == 6
            nv = sqrt(mean(ox.dwdxall(:, ic).^2, 1, 'omitnan'));
            say('      %3d  %s\n', s2, sprintf('%9.2e ', nv));
        end
    end
end

%% figures (the tested sensitivities-example presentation)
vv = 'off';  if opts.visible, vv = 'on'; end %#ok<NASGU>
ref = firstnonempty_(ox, oz, og, os);
if ~isempty(ref)
    if ncfg > 0
        ct = sprintf('%s -- nominal OPD, %d configurations x %d fields', ...
            name, ncfg, size(ref.field_table, 1));
    else
        ct = sprintf('%s -- nominal OPD, %d fields', name, ...
            size(ref.field_table, 1));
    end
    plot_opd_canvas(ref, ct, od, [name '_opdall.png']);
end
if ~isempty(SV)
    f = figure('Visible', 'off', 'Position', [0 0 760 520]);
    mk = {'-o', '-s', '-^'};
    for q = 1:numel(SV)
        semilogy(SV{q}/SV{q}(1), mk{q}, 'MarkerSize', 3); hold on
    end
    grid on; legend(tab(:, 1), 'Location', 'southwest');
    xlabel('singular value index'); ylabel('\sigma_i / \sigma_1');
    title(sprintf('%s: Jacobian spectra (%d segs)', name, nseg));
    print(f, fullfile(od, [name '_svspec.png']), '-dpng', '-r120'); close(f);
end
if ~isempty(SVc)
    chs = unique(cellfun(@(v) string(v.ch), SVc), 'stable');
    f = figure('Visible', 'off', ...
        'Position', [0 0 max(760, 420*numel(chs)) 500]);
    for q = 1:numel(chs)
        subplot(1, numel(chs), q); hold on
        % stacked FIRST and thick-but-pale, so the per-configuration
        % curves -- which is what this figure exists to show -- are not
        % buried under it.  They separate only in the small-sigma tail,
        % which is where the rank difference lives.
        is = find(strcmp(tab(:,1), char(chs(q))), 1);
        if ~isempty(is)
            semilogy(SV{is}/SV{is}(1), '-', 'LineWidth', 4, ...
                'Color', [0.78 0.78 0.78], 'DisplayName', 'stacked');
        end
        sel = find(cellfun(@(v) string(v.ch) == chs(q), SVc));
        for r = sel(:).'
            semilogy(SVc{r}.s / SVc{r}.s(1), '-', 'LineWidth', 0.9, ...
                'DisplayName', SVc{r}.cfg);
        end
        set(gca, 'YScale', 'log'); grid on
        ylim([1e-18 2]);   % below this is round-off, not structure
        legend('Location', 'southwest', 'FontSize', 7);
        xlabel('singular value index'); ylabel('\sigma_i / \sigma_1');
        title(char(chs(q)), 'Interpreter', 'none');
    end
    sgtitle(sprintf('%s: per-configuration vs stacked spectra', name), ...
        'Interpreter', 'none');
    print(f, fullfile(od, [name '_svspec_configs.png']), '-dpng', '-r120');
    close(f);
    say('per-configuration spectra: %s_svspec_configs.png\n', name);
end
pages = {'dwdx', ox; 'dwdz', oz; 'dwdgrid', og; 'dwdsurf', os};
pdir = fullfile(od, [name '_pages']);      % the pages are numerous --
if ~isempty(opts.per_element) && ~isfolder(pdir), mkdir(pdir); end  % own folder
for q = 1:size(pages, 1)
    o = pages{q, 2};
    if isempty(o), continue; end
    plot_dw_channels(o, sprintf('%s %s -- each channel', name, pages{q,1}), ...
        od, [name '_' pages{q,1} '_channels.png']);
    for pm = opts.per_element(:).'
        plot_dw_per_element(o, char(pm), pdir, [name '_' pages{q,1}]);
    end
end
say('\nfigures: %s_opdall/svspec/<ch>_channels + per-element pages in %s_pages/\n', ...
    name, name);

%% save
matp = fullfile(od, [name '_sens.mat']);
cfg = opts;
save(matp, 'ox', 'oz', 'og', 'os', 'cfg', '-v7.3');
say('\nDone: %s + %s_sens_report.txt\n', matp, name);

% prune the checkpoints -- the run completed, the combined .mat above is
% the artifact and the per-block files are just a crash cushion
if strlength(RD) > 0 && isfolder(char(RD))
    rmdir(char(RD), 's');
    say('checkpoints pruned: %s\n', char(RD));
end

art = struct('ox', ox, 'oz', oz, 'og', og, 'os', os, 'mat', string(matp), ...
    'grid_in', string(rxg), 'report', ...
    string(fullfile(od, [name '_sens_report.txt'])), 'nseg', nseg, ...
    'conditioning', {tab}, 'nconfig', ncfg);
end

% ---------------------------------------------------------------------
function A = jmat_(o)
if isfield(o, 'dwdgall'), A = o.dwdgall; else, A = o.dwdxall; end
end

function o = firstnonempty_(varargin)
o = [];
for k = 1:nargin
    if ~isempty(varargin{k}), o = varargin{k}; return; end
end
end

function p = fullpath_(p)
%FULLPATH_  Absolute form of P.  The absoluteness test must be
%   platform-correct: a bare startsWith(p,'/') reads a Windows
%   'C:\...' as RELATIVE and prepends pwd, producing 'cwd\C:\...'
%   (Luis, 2026-08-24 -- run_dwdx_5zoom_5fov crashed on Windows while
%   Linux, where every absolute path starts with '/', never saw it).
%   Absolute here = leading / or \ (POSIX / drive-rooted), X:\ or X:/
%   (Windows drive), or \\server (UNC, via the leading-\ case).
if isempty(regexp(p, '^([/\\]|[A-Za-z]:[/\\])', 'once'))
    p = fullfile(pwd, p);
end
end

function o = run_channel_(fn, tag, cfgs, resume_dir, say, key)
%RUN_CHANNEL_  One channel's harvest, optionally checkpointed per config.
%   Without a resume_dir (or with fewer than two configurations) this is
%   exactly fn(cfgs) -- ONE supervisor call carrying the whole
%   configuration axis, which is the normal path.  With both, each
%   configuration is harvested and saved on its own so a killed run
%   resumes at the block it died on, and the blocks are stitched back
%   into the same arrays the single call produces.
if isempty(cfgs) || numel(cfgs) < 2 || strlength(resume_dir) == 0
    o = fn(cfgs);
    return
end
rd = char(resume_dir);
if ~isfolder(rd), mkdir(rd); end
outs = cell(1, numel(cfgs));
for c = 1:numel(cfgs)
    ck = fullfile(rd, sprintf('%s%s_%s.mat', tag, key, cfgs(c).name));
    if isfile(ck)
        S = load(ck);  outs{c} = S.o;
        say('    [resume] %s / %s <- %s\n', tag, cfgs(c).name, ck);
    else
        o = fn(cfgs(c));
        save(ck, 'o', '-v7.3');
        outs{c} = o;
        say('    [checkpoint] %s / %s -> %s\n', tag, cfgs(c).name, ck);
    end
end
o = stitch_configs_(outs, cfgs);
end


function o = stitch_configs_(outs, cfgs)
%STITCH_CONFIGS_  Reassemble per-configuration blocks into one harvest.
%   Identical to the single all-configurations call by construction: the
%   assembly goes through macos.config_canvas, the SAME function the
%   supervisors use, so the outer tile layout and the
%   configuration-major row order come out the same way.  Each
%   checkpointed block is a single-configuration harvest, whose OPDall
%   is just that configuration's own field canvas.
o = outs{1};
canv = cellfun(@(u) u.OPDall, outs, 'UniformOutput', false);
tile_rc = [];
if isfield(cfgs, 'tile') && ~any(arrayfun(@(c) isempty(c.tile), cfgs))
    tile_rc = vertcat(cfgs.tile);
end
[o.OPDall, ix] = macos.config_canvas(canv, tile_rc);
o.indxall = ix;
o.w0_stacked = macos.m2v(o.OPDall, ix);
for f = intersect(fieldnames(o), ...
        {'dwdxall','dwdzall','dwdsall','dwdgall'}).'
    C = cellfun(@(u) u.(f{1}), outs, 'UniformOutput', false);
    o.(f{1}) = vertcat(C{:});
end
for f = fieldnames(o).'
    if startsWith(f{1}, 'per_field_')
        C = cellfun(@(u) u.(f{1}), outs, 'UniformOutput', false);
        o.(f{1}) = vertcat(C{:});          % Nc x Nf
    end
end
C = cellfun(@(u) u.config_table, outs, 'UniformOutput', false);
o.config_table = vertcat(C{:});
% pupil_find placement metrics are per-configuration too: each
% checkpointed block carries exactly its own configuration's entry, and
% dropping the merge under-reports every config after the first (caught
% by tPupilFindMethod/test_resume_checkpoints_are_method_aware).
if isfield(o, 'pupil_find')
    C = cellfun(@(u) u.pupil_find, outs, 'UniformOutput', false);
    for c = 1:numel(C)          % each block ran as its own single-config
        for q = 1:numel(C{c})   % call, so its entries all say config 1:
            if isfield(C{c}, 'config'), C{c}(q).config = c; end
        end                     % renumber to the stitched axis
    end
    o.pupil_find = [C{:}];
end
o.config_names = {o.config_table.name}.';
end


function d = cfg_describe_(cfg)
% One printable line per setter in a configuration (the report names
% what each configuration DID).  Accepts either the caller's raw cell
% list or the supervisors' normalised records.
sl = cfg.set;
if isempty(sl), d = {'(no setters -- nominal state)'}; return; end
d = cell(1, numel(sl));
for k = 1:numel(sl)
    e = sl{k};
    if isstruct(e)
        switch e.fn
            case 'perturb'
                d{k} = sprintf('perturb elt %d rot %s trans %s (%s)', ...
                    e.elt, mat2str(e.rotation.', 4), ...
                    mat2str(e.translation.', 4), e.frame);
            case {'set_elt_vpt','set_elt_psi','set_elt_rpt'}
                d{k} = sprintf('%s elt %d = %s', e.fn, e.elt, ...
                    mat2str(e.value.', 6));
            otherwise
                d{k} = sprintf('%s elt %d', e.fn, e.elt);
        end
    elseif iscell(e) && numel(e) >= 2
        rest = '';
        for q = 3:numel(e)
            if ischar(e{q}) || isstring(e{q})
                rest = [rest ' ' char(e{q})]; %#ok<AGROW>
            else
                rest = [rest ' ' mat2str(reshape(double(e{q}),1,[]), 4)]; %#ok<AGROW>
            end
        end
        d{k} = sprintf('%s elt %d%s', char(e{1}), double(e{2}), rest);
    else
        d{k} = '(unrecognised setter entry)';
    end
end
end

function e = find_seg_elts_(txt)
% element indices of the Element= Segment blocks, by block order
blk = regexp(txt, 'Element=\s*(\w+)', 'tokens');
types = string(cellfun(@(c) c{1}, blk, 'UniformOutput', false));
e = find(types == "Segment");
end

function say_pf_(say, o, tag)
%SAY_PF_  One report line per pupil_find placement (per config, or per
%   (config, field) block under pf_scope='field').
if ~isfield(o, 'pupil_find'), return; end
for i = 1:numel(o.pupil_find)
    m = o.pupil_find(i);
    if isfield(m, 'field') && m.field > 0
        lbl = sprintf('cfg %d fld %d', m.config, m.field);
    elseif isfield(m, 'config')
        lbl = sprintf('cfg %d', m.config);
    else
        lbl = sprintf('cfg %d', i);   % pre-scope checkpoints
    end
    say(['    [%s] pupil_find %s: sphere vtx-FEX %.3g, dep_rms %.3g, ' ...
         'conv R %.4g, rad %.6g\n'], tag, lbl, m.vtx_minus_fex, m.dep_rms, ...
        m.conv_radius, m.rad);
end
end
