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
%     'channels'   subset of {'dwdx','dwdz','dwdgrid'} (default all)
%     'dofs'       dwdx DOF subset (default (0:5)': Rx..Tz)
%     'zmodes_fig' dwdz MonZernike modes (default 4:11)
%     'zmodes_grid' dwdgrid basis modes (default 4:9)
%     'ng'         grid size for the augmentation (default 256)
%     'span_frac'  grid span fraction of the parent Aperture (1.0 = the dxGrid convention)
%     'pm_ref_elt' segment_grid_basis footprint target (default 1)
%     'reset_xp_method'  'fex' (default) | 'sxp' for the dwdz/dwdgrid
%                  per-field exit-pupil reset (near-EP layouts: 'sxp')
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
    opts.channels string = ["dwdx" "dwdz" "dwdgrid"]  % + "dwdsurf" opt-in
    opts.dofs (:,1) double = (0:5).'
    opts.zmodes_fig (1,:) double = 4:11
    opts.zmodes_grid (1,:) double = 4:9
    opts.zkinds cell = {'monzern'}   % dwdz kinds: subset of
                                     % {'monzern','ffzern','zern'};
                                     % monzern = the SEGMENT-LOCAL basis
    opts.surf_params cell = {'Kr','Kc'}  % dwdsurf channel parameters
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
        {'fex','sxp'})} = 'fex'
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
if isempty(regexp(txt, '^\s*ApStop=', 'once', 'lineanchors'))
    assert(~isempty(opts.stop), ['run_sensitivities: %s declares no ' ...
        'ApStop (exit-pupil machinery needs one). Add "ApStop= 0 0 0" ' ...
        'to the Rx header or pass ''stop'', [0 0 0].'], char(rx_in));
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
sup = {'field_x_rad', FOV, 'field_y_rad', FOV, 'ngridpts', opts.ngridpts};

ox = [];  oz = [];  og = [];  os = [];  rxg = '';
%% dwdx
if any(opts.channels == "dwdx")
    say('[dwdx] rigid-body 6-DOF per element, LOCAL triads...\n');
    a = {};  if ~isempty(opts.delta_x), a = {'delta', opts.delta_x}; end
    ox = macos.dw_dx_multi(m, char(rx_in), sup{:}, 'dofs', opts.dofs, a{:});
    say('    dwdxall %d x %d over %d fields\n\n', size(ox.dwdxall, 1), ...
        size(ox.dwdxall, 2), size(ox.field_table, 1));
end

%% dwdz (segment-LOCAL MonZernike)
if any(opts.channels == "dwdz") && nseg > 0
    say('[dwdz] segment-LOCAL MonZernike modes %s...\n', mat2str(opts.zmodes_fig));
    a = {};  if ~isempty(opts.delta_z), a = {'delta', opts.delta_z}; end
    oz = macos.dw_dz_zernike_multi(m, char(rx_in), sup{:}, ...
        'kinds', opts.zkinds, 'zmode_start', opts.zmodes_fig(1), ...
        'n_zcoef', opts.zmodes_fig(end), a{:});
    say('    dwdzall %d x %d\n\n', size(oz.dwdxall, 1), size(oz.dwdxall, 2));
end

%% dwdsurf (per-element Kr/Kc -- opt-in)
if any(opts.channels == "dwdsurf")
    say('[dwdsurf] per-element %s...\n', strjoin(opts.surf_params, '/'));
    os = macos.dw_dsurf_multi(m, char(rx_in), sup{:}, ...
        'params', opts.surf_params);
    say('    dwdsall %d x %d\n\n', size(os.dwdxall, 1), size(os.dwdxall, 2));
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
    og = macos.dw_dgrid_multi(m, rxg, sup{:}, 'influence', sgb, ...
        'reset_xp_method', opts.reset_xp_method, a{:});
    % persist the influence basis WITH the harvest: the basis is part
    % of the Jacobian's definition, and rebuilding it in a later
    % session is not bit-stable (the last G-S mode can come out
    % rotated -- caught by run_compare 2026-07-19); downstream
    % consumers (run_compare pokes, dmdgrid) use og.sgb verbatim
    og.sgb = sgb;
    say('    dwdgall %d x %d (influence basis saved in og.sgb)\n\n', ...
        size(og.dwdgall, 1), size(og.dwdgall, 2));
end

%% conditioning report
say('[conditioning] finite rows only:\n');
say('    %-8s %12s %6s %10s %10s %10s\n', 'channel', 'size', 'rank', ...
    'sv_max', 'sv_min+', 'cond+');
tab = {};  SV = {};
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
    plot_opd_canvas(ref, sprintf('%s -- nominal OPD, %d fields', name, ...
        size(ref.field_table, 1)), od, [name '_opdall.png']);
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

art = struct('ox', ox, 'oz', oz, 'og', og, 'os', os, 'mat', string(matp), ...
    'grid_in', string(rxg), 'report', ...
    string(fullfile(od, [name '_sens_report.txt'])), 'nseg', nseg, ...
    'conditioning', {tab});
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
if ~startsWith(p, '/'), p = fullfile(pwd, p); end
end

function e = find_seg_elts_(txt)
% element indices of the Element= Segment blocks, by block order
blk = regexp(txt, 'Element=\s*(\w+)', 'tokens');
types = string(cellfun(@(c) c{1}, blk, 'UniformOutput', false));
e = find(types == "Segment");
end
