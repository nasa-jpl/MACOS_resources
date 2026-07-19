function art = run_segmentation(parent_in, opts)
%RUN_SEGMENTATION  Segmentation stage runner: parent .in -> segmented .in.
%
%   art = run_segmentation(PARENT_IN, 'rings', R, 'grid', G, ...) is
%   the segmentation stage of the design pipeline
%       design -> segmentation -> sensitivities -> MET -> compare -> simulate
%   (see design/runners/README.md).  It segments element ELT of the
%   parent prescription (SegMirMaker via macos.design.segment_rx),
%   verifies the result against the monolithic parent by ENGINE TRUTH,
%   and emits the handoff artifact:
%
%     <name>.in            the segmented Rx WITH physical polygonal
%                          apertures declared (emit_apertures: pie =
%                          center hexagon + chorded sectors, hex =
%                          exact corners); gap rays clip -- physically
%                          honest
%     <name>Hx.m           the SegMirMaker edge-sensor sidecar (dedx)
%     <name>_seg_report.txt  parity + clip census + frames table
%     <name>_footprints.png  traced per-segment footprints (poke-diff
%                          engine truth) + tiling boundary + emitted
%                          aperture polygons
%     <name>_views.png     view_std 4-panel of the segmented system
%
%   Verification ladder (each says PASS/numbers in the report):
%     [0] monolithic parent baseline trace
%     [1] bare segmentation trace parity (the parent figure rides on
%         every segment; obscurations carry to the center segment)
%     [2] engine-truth footprints (macos.design.seg_footprints)
%     [3] aperture variant: pass counts + first-fail element census
%     [5] standalone artifact reload == in-session counts
%
%   REQUIRED:
%     'rings'   tiling rings (Pie: 1; Hex: 1 or 2)
%     'grid'    'Pie' | 'Hex'
%   OPTIONS:
%     'elt'          parent element to segment (default 1)
%     'gap'          inter-segment gap, parent BaseUnits (default 50 mm
%                    equivalent: 50 for mm prescriptions, 0.05 for m)
%     'dofs'         SegMirMaker DOF count (default 6)
%     'meas_config'  SegMirMaker measurement config (default 1)
%     'grid_npts'    source-grid override for the splice ([] = keep)
%     'emit_apertures' declare physical apertures (default true)
%     'ap_pad'       aperture clearance beyond the physical edge (0)
%     'fp_poke'      footprint poke, BaseUnits ([] = auto: 1e-6 for mm
%                    prescriptions, 1e-7 for m)
%     'views'        emit view_std figure (default true)
%     'model_size'   engine model (default 512)
%     'out_dir','name','visible','verbose'  as in run_met
%
%   art: seg (bare) / segA (apertures) structs, labels, pass counts,
%   artifact paths.
%
%   See also: macos.design.segment_rx, macos.design.seg_footprints,
%             macos.design.seg_boundary, run_sensitivities, run_met.

arguments
    parent_in (1,1) string
    opts.rings (1,1) double {mustBeInteger, mustBePositive}
    opts.grid (1,:) char {mustBeMember(opts.grid, {'Pie','Hex'})}
    opts.elt (1,1) double {mustBeInteger, mustBePositive} = 1
    opts.gap double = []
    opts.dofs (1,1) double = 6
    opts.meas_config (1,1) double = 1
    opts.grid_npts double = []
    opts.emit_apertures (1,1) logical = true
    opts.ap_pad (1,1) double = 0
    opts.fp_poke double = []
    opts.views (1,1) logical = true
    opts.model_size (1,1) double = 512
    opts.out_dir (1,1) string = ""
    opts.name (1,1) string = ""
    opts.visible (1,1) logical = false
    opts.verbose (1,1) logical = true
end
assert(isfile(parent_in), 'run_segmentation: %s not found', parent_in);
[pdir, pstem] = fileparts(char(parent_in));
if strlength(opts.out_dir) == 0, opts.out_dir = string(pdir); end
if strlength(opts.name) == 0
    opts.name = string(sprintf('%s_%s%d', pstem, lower(opts.grid), opts.rings));
end
od = char(opts.out_dir);  name = char(opts.name);
if ~isfolder(od), mkdir(od); end

% unit-aware defaults from the parent's BaseUnits
ptxt = fileread(char(parent_in));
bu = regexp(ptxt, 'BaseUnits=\s*(\w+)', 'tokens', 'once');
is_m = ~isempty(bu) && strcmpi(bu{1}, 'm');
if isempty(opts.gap),     opts.gap     = tern_(is_m, 0.05, 50);   end
if isempty(opts.fp_poke), opts.fp_poke = tern_(is_m, 1e-7, 1e-6); end

log_ = fopen(fullfile(od, [name '_seg_report.txt']), 'w');
closer = onCleanup(@() fclose(log_));
if opts.verbose
    say = @(varargin) fprintf(1, varargin{:}) + fprintf(log_, varargin{:});
else
    say = @(varargin) fprintf(log_, varargin{:});
end
say('==== run_segmentation: %s (elt %d, %s rings=%d, gap %g) ====\n', ...
    char(parent_in), opts.elt, opts.grid, opts.rings, opts.gap);

%% [0] monolithic parent baseline
macos.init(opts.model_size);
macos.load_rx(char(parent_in));
t0 = macos.trace();
r0 = macos.get_ray_info(t0.nRays);
p0 = nnz(logical(r0.ok_pass) & logical(r0.ok_trace));
say('[0] parent: %d src rays, %d pass, rmsWFE %.4g (WaveUnits)\n\n', ...
    t0.nRays, p0, t0.rmsWFE);

%% [1] bare segmentation + trace parity
sargs = {'elt', opts.elt, 'rings', opts.rings, 'grid', opts.grid, ...
    'gap', opts.gap, 'dofs', opts.dofs, 'meas_config', opts.meas_config};
if ~isempty(opts.grid_npts), sargs = [sargs, {'grid_npts', opts.grid_npts}]; end
seg = macos.design.segment_rx(char(parent_in), sargs{:});
say('[1] %d segments, width %.4g flat-to-flat, %d elements\n', ...
    seg.nseg, seg.width, seg.n_elt);
if isfield(seg, 'carried_obs')
    say('    parent obscurations carried onto the center segment: %d\n', ...
        seg.carried_obs);
end
for s = 1:seg.nseg
    fr = seg.frames(s);
    say('      %-8s center [%+10.4g %+10.4g %+10.4g]  lMon %.4g\n', ...
        fr.name, fr.rpt(1), fr.rpt(2), fr.rpt(3), fr.lmon);
end
macos.load_rx(seg.in);
t1 = macos.trace();
r1 = macos.get_ray_info(t1.nRays);
p1 = nnz(logical(r1.ok_pass) & logical(r1.ok_trace));
w1 = macos.opd();
say('    bare segmented: %d src rays, %d pass, rmsWFE %.4g (parent %.4g, %+.2f%%)\n', ...
    t1.nRays, p1, t1.rmsWFE, t0.rmsWFE, 100*(t1.rmsWFE - t0.rmsWFE)/t0.rmsWFE);

%% [2] engine-truth footprints + boundary
labels = macos.design.seg_footprints(seg, w1, 'poke', opts.fp_poke);
B = macos.design.seg_boundary(seg);

%% [3] physical apertures
segA = seg;  pA = p1;  tA = t1;
if opts.emit_apertures
    segA = macos.design.segment_rx(char(parent_in), sargs{:}, ...
        'emit_apertures', true, 'ap_pad', opts.ap_pad);
    macos.load_rx(segA.in);
    tA = macos.trace();
    rA = macos.get_ray_info(tA.nRays);
    pA = nnz(logical(rA.ok_pass) & logical(rA.ok_trace));
    rs = macos.get_ray_status(tA.nRays);
    fe = rs.fail_elt(rs.status ~= 0);
    say('[3] physical apertures (pad %g): %d pass (bare %d) -- %d gap/rim rays clip\n', ...
        opts.ap_pad, pA, p1, p1 - pA);
    say('    first-fail elements %s; rmsWFE %.4g (%+.2f%% vs parent)\n', ...
        mat2str(unique(fe(fe > 0)).'), tA.rmsWFE, ...
        100*(tA.rmsWFE - t0.rmsWFE)/t0.rmsWFE);
end

fpng = fullfile(od, [name '_footprints.png']);
apk = {};
if opts.emit_apertures && isfield(segA, 'apertures')
    apk = {'apertures', segA.apertures};
end
f = macos.design.seg_footprint_view(labels, B, seg, ...
    'units', tern_(is_m, 'm', 'mm'), apk{:}, 'alpha', 0.55, ...
    'title', sprintf('%s: traced footprints + emitted apertures (%d segs)', ...
                     name, seg.nseg), 'save', fpng);
close(f);
say('    footprint/aperture figure: %s\n', fpng);

%% [4] views
vpng = '';
if opts.views
    vpng = fullfile(od, [name '_views.png']);
    try
        fv = macos.view_std('args', {'show','beam'}, 'visible', opts.visible, ...
            'title', sprintf('%s: segmented system (%d segments)', name, seg.nseg), ...
            'save', vpng);
        close(fv);
        say('[4] standard views: %s\n', vpng);
    catch ME
        vpng = '';
        say('[4] view_std skipped (%s)\n', ME.message);
    end
end

%% [5] artifact + standalone reload verification
art_in = fullfile(od, [name '.in']);
copyfile(segA.in, art_in);
hx_out = fullfile(od, [name 'Hx.m']);
copyfile(segA.hx, hx_out);
macos.init(opts.model_size);
nv = macos.load_rx(art_in);
sv = macos.trace(nv);
rv = macos.get_ray_info(sv.nRays);
np = nnz(logical(rv.ok_pass) & logical(rv.ok_trace));
ok = (np == pA);
say('[5] artifact %s: standalone reload %d elts, %d/%d rays pass -> %s\n', ...
    art_in, nv, np, sv.nRays, tern_(ok, 'VERIFIED', '** MISMATCH **'));

art = struct('in', string(art_in), 'hx', string(hx_out), ...
    'report', string(fullfile(od, [name '_seg_report.txt'])), ...
    'footprints_png', string(fpng), 'views_png', string(vpng), ...
    'seg', seg, 'segA', segA, 'labels', labels, ...
    'pass_parent', p0, 'pass_bare', p1, 'pass_ap', pA, 'src', t1.nRays, ...
    'rms_parent', t0.rmsWFE, 'rms_bare', t1.rmsWFE, 'rms_ap', tA.rmsWFE, ...
    'verified', ok);
end

% ---------------------------------------------------------------------
function s = tern_(c, a, b), if c, s = a; else, s = b; end, end
