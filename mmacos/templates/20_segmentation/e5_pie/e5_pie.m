%E5_PIE  Pie segmentation with physical polygonal apertures.
%
% Manual worked example (companion to ../e5_seg, the hex case).  Run
% after mmacos_setup.  Every step writes a figure beside this script;
% numeric findings collect in findings.txt.
%
%   Step 1  segment the e5 monolithic primary into a 1-ring PIE
%           (macos.design.segment_rx: 7 segments = center + 6 wedges).
%   Step 2  measure each segment's TRUE ray footprint from the trace
%           (poke one segment, diff the OPD) and overlay the tiling
%           boundary reconstruction (macos.design.seg_boundary).  The
%           central cell of the (X,L,R) hex-coordinate tiling is a
%           HEXAGON, not a disc -- the footprint proves it.
%   Step 3  re-emit the prescription with each segment's physical
%           boundary declared as a polygonal aperture
%           (segment_rx emit_apertures=true -> macos.design.
%           seg_apertures: center hexagon + ring-1 wedges as TRUE
%           convex chorded polygons -- side edges parallel to the
%           sector rays, so inter-segment gaps are uniform-width
%           slots that do not converge at the center, and no
%           obscuration is needed.  Deeper rings, whose arc inner
%           edge is non-convex, would emit convex sector + inner
%           obscuration per the engine convention).
%   Step 4  trace parity: rays the source tiling places in the
%           inter-segment GAPS are clipped by the apertures --
%           physically honest (they have no glass under them); the
%           wavefront is unchanged where rays survive.
%   Step 5  perturbation honesty: decenter one wedge 100 mm -- rays
%           walking off the physical segment clip only when the
%           aperture is declared.
%   Step 6  launcher placement on the Rx-DECLARED edges: with the
%           polygons in the prescription, macos.design.seg_boundary
%           auto-switches to its 'rxpoly' source, so add_met places
%           launchers on the aperture polygons themselves (the general
%           case, covering imported segmented prescriptions), and
%           met_view renders the MET scene.
%
% See also: assess history in README.md (this example grew out of the
% 2026-07-16 poly-aperture assessment).

PAD  = 0;       % aperture clearance beyond the physical edge, mm.
                % 0 = the physical segment (gap rays clip); gap/2 would
                % put the aperture at the tiling midline (trace-neutral).
POKE = 0.100;   % wedge decenter for step 5, metres
NF   = 6;       % fiducials on the M2 rim (step 6)
R_FID = 590;    % fiducial ring radius, mm (~25 inside M2's 615 rim)
R_EXTRA = 100;  % aft/M3 launcher ring radius, mm (that truss's size)

here = fileparts(mfilename('fullpath'));
res_root = fullfile(getenv('HOME'), 'dev', 'MACOS_resources');
tin = fullfile(res_root, 'segmirmaker', 'test_in');
log_ = fopen(fullfile(here, 'findings.txt'), 'w');
say = @(varargin) fprintf(1, varargin{:}) + fprintf(log_, varargin{:});

%% ---- Step 1: pie segmentation ------------------------------------------
seg = macos.design.segment_rx(fullfile(tin, 'e5mono.in'), 'elt', 1, ...
    'rings', 1, 'grid', 'Pie', 'gap', 50, 'dofs', 6, 'meas_config', 1);
say('step 1: pie system, %d segments, width %.1f mm, gap %g mm\n', ...
    seg.nseg, seg.width, seg.gap);

old = cd(seg.run.workdir); restore = onCleanup(@() cd(old));
macos.init(512);
macos.load_rx(seg.in);
t0 = macos.trace();
pass_ = @(t) nnz(subsref(macos.get_ray_info(t.nRays), ...
    substruct('.', 'ok_pass')));
p0 = pass_(t0);
w0 = macos.opd();
say('        baseline: %d src rays, %d pass, rmsWFE %.6g mm\n', ...
    t0.nRays, p0, t0.rmsWFE);

%% ---- Step 2: true footprints vs the tiling boundary --------------------
% Poke each segment in turn (local piston) and mark the OPD pixels that
% move -- macos.design.seg_footprints, the shared engine-truth
% measurement; figure via macos.design.seg_footprint_view (both hoisted
% from this example so e2e s3 and future runners consume the same code).
labels = macos.design.seg_footprints(seg, w0, 'poke', 1e-6);
B = macos.design.seg_boundary(seg);               % tiling reconstruction
[f, xmm, ymm, toimg] = macos.design.seg_footprint_view(labels, B, seg, ...
    'units', 'mm', ...
    'title', 'Step 2 -- traced segment footprints + tiling boundary');
text(0, 0, 'hexagonal center cell', 'Horiz', 'center', 'FontWeight', 'bold');
print(f, fullfile(here, 'e5pie_step2_footprints.png'), '-dpng', '-r120');
close(f);
say('step 2: footprint map + boundary overlay -> e5pie_step2_footprints.png\n');

%% ---- Step 3: emit the physical apertures --------------------------------
segA = macos.design.segment_rx(fullfile(tin, 'e5mono.in'), 'elt', 1, ...
    'rings', 1, 'grid', 'Pie', 'gap', 50, 'dofs', 6, 'meas_config', 1, ...
    'emit_apertures', true, 'ap_pad', PAD);
copyfile(segA.in, fullfile(here, 'e5pie_polyap.in'));
ap = segA.apertures;
say('step 3: apertures emitted (pad %g mm): center %s with %d vertices, wedges %d each -> e5pie_polyap.in\n', ...
    PAD, 'hexagon', size(ap.poly{1}, 2), size(ap.poly{2}, 2));
f = macos.design.seg_footprint_view(labels, B, seg, 'units', 'mm', ...
    'apertures', ap, 'boundary', false, 'alpha', 0.35, ...
    'title', ['Step 3 -- emitted aperture polygons ', ...
              '(black; red = obscurations, none for 1 ring)'], ...
    'save', fullfile(here, 'e5pie_step3_apertures.png'));
close(f);

%% ---- Step 4: trace parity (gap rays clip) --------------------------------
cd(segA.run.workdir);                             % its own GridFile dir
macos.load_rx(segA.in);
t1 = macos.trace();
p1 = pass_(t1);
w1 = macos.opd();
rs = macos.get_ray_status(t1.nRays);
fe = rs.fail_elt(rs.status ~= 0);
say('step 4: with apertures: %d pass (baseline %d), rmsWFE %.6g mm (%+.2f%%)\n', ...
    p1, p0, t1.rmsWFE, 100*(t1.rmsWFE - t0.rmsWFE)/t0.rmsWFE);
say('        %d rays clipped; first-fail element(s): %s\n', ...
    p0 - p1, mat2str(unique(fe(fe > 0)).'));
dm = (w0 ~= 0 & isfinite(w0)) & ~(w1 ~= 0 & isfinite(w1));
f = figure('Visible','off', 'Position', [0 0 720 660]);
imagesc(xmm, ymm, toimg(double(dm)), 'AlphaData', 0.25 + 0.75*toimg(double(dm)));
axis xy image; hold on
colormap([0.94 0.94 0.94; 0.85 0.2 0.2]); clim([0 1]);
overlay_(B, seg.nseg, 'k-', 0.8);
title(sprintf('Step 4 -- %d clipped rays: the inter-segment gaps + rim margin', nnz(dm)));
xlabel('pupil x, mm'); ylabel('pupil y, mm');
print(f, fullfile(here, 'e5pie_step4_clipped.png'), '-dpng', '-r120');
close(f);

%% ---- Step 5: perturbation honesty ---------------------------------------
% decenter wedge 2 by POKE: rays walking off the physical segment must
% not survive.  Without the aperture some do (they re-intersect the
% mathematical surface where no glass exists); with it, the loss lands
% at the DECLARED edge.
pass5 = zeros(1, 2);
for variant = 1:2
    if variant == 1
        cd(seg.run.workdir);  macos.load_rx(seg.in);
    else
        cd(segA.run.workdir); macos.load_rx(segA.in);
    end
    macos.perturb(seg.seg_elts(2), 'translation', [POKE; 0; 0], 'frame','local');
    macos.modify();
    tp = macos.trace();
    pass5(variant) = pass_(tp);
end
say('step 5: %g mm wedge decenter: %d rays pass with apertures, %d without --\n', ...
    POKE*1e3, pass5(2), pass5(1));
say('        the aperture-less trace keeps %d rays that have no glass under them\n', ...
    pass5(1) - pass5(2));
wp = macos.opd();
f = figure('Visible','off', 'Position', [0 0 720 660]);
dmp = (w1 ~= 0 & isfinite(w1)) & ~(wp ~= 0 & isfinite(wp));
imagesc(xmm, ymm, toimg(double(dmp)), 'AlphaData', 0.25 + 0.75*toimg(double(dmp)));
axis xy image; hold on
colormap([0.94 0.94 0.94; 0.85 0.2 0.2]); clim([0 1]);
overlay_(B, seg.nseg, 'k-', 0.8);
title(sprintf('Step 5 -- %g mm decenter of wedge 2: %d rays clip at the aperture', ...
      POKE*1e3, p1 - pass5(2)));
xlabel('pupil x, mm'); ylabel('pupil y, mm');
print(f, fullfile(here, 'e5pie_step5_poke.png'), '-dpng', '-r120');
close(f);

%% ---- Step 6: launchers on the Rx-declared edges + MET scene -------------
Brx = macos.design.seg_boundary(segA, 5);
say('step 6: seg_boundary source = %s (auto-detected from the Rx)\n', Brx.kind);
am = macos.design.add_met(segA.in, segA, 'hub', segA.nseg+1, ...
    'r_fid', R_FID, 'nf', NF, 'extra_sources', segA.n_elt-2, ...
    'r_extra', R_EXTRA, 'edge_off', 5);
macos.load_rx(am.in); macos.trace();
fv = macos.design.met_view(segA, am, 'visible', false, 'edge_off', 5, ...
    'title', 'e5 PIE: launchers on the Rx-declared aperture polygons', ...
    'save', fullfile(here, 'e5pie_step6_met.png'));
close(fv);
say('        %d MET beams; launchers sampled from the aperture polygons -> e5pie_step6_met.png\n', ...
    am.n_beams);
fclose(log_);
fprintf('findings.txt + 5 figures written to %s\n', here);

% ---------------------------------------------------------------------------
function overlay_(B, nseg, style, lw)
for s = 1:nseg
    plot(B.poly{s}(1,:), B.poly{s}(2,:), style, 'LineWidth', lw);
end
end
