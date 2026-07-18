% s2_instrument.m  (mmacos/design/examples/e2e/ -- stage 2 of 6)
% =====================================================================
%  E2E STAGE 2 -- IMAGING INSTRUMENT: a 3-mirror bench relay widens the
%  corrected field, solved JOINTLY with the telescope.
% =====================================================================
%  Consumes s1_telescope.mat.  The stage-1 telescope is perfect at its
%  design field point and hands over a PURE field-differential residual
%  across the patch; this stage adds the optics that correct it:
%    [1] rebuild the s1 chain (solved conics carried via 'conic',
%        freeform figures re-applied) and append the THREE-mirror relay
%        (M4 weak corrector at an intermediate conjugate / M5
%        collimator / M6 camera), zigzagged across the bench by fold
%        tilts (Bauer path).  Field correction = freeform at staggered
%        intermediate conjugates; NO active control in this part of the
%        telescope (Dave).  Probed and rejected: a per-field patch
%        corrector at a focus (global-Zernike rank collapse) and a
%        4th weak mirror near the relayed pupil (acts common-mode);
%    [2] baseline: the +-2' ladder of telescope+bare relay;
%    [3] M5/M7 conic+ROC solve at the bias point (FP aligned first --
%        the s1 lesson: mm-scale defocus poisons the solve; the weak
%        correctors are HELD -- with ROC+conic free they turn into
%        K~-2000 pathologies);
%    [4] field-zone lMon over the +-2' set (M2/M3 zones GROW with the
%        wider field -- Dave: M2/M3 may get larger as instrument optics
%        are added) + JOINT freeform field solve (CALIB on the powered
%        set, SVD engine on the weak correctors + polish + M1 common
%        mode over the full field set);
%    [5] clearance + M1 hole re-check;
%    [6] standard views + field maps at +-2';
%    [7] deliverables (s2_instrument.in/.mat) + standalone reload;
%    [8] design report + stage-2 addendum -> s2_report.txt.
%
%  Run AFTER s1_telescope.m:
%    >> run('.../design/examples/e2e/s2_instrument.m')
% =====================================================================
addpath(fullfile(getenv('HOME'),'dev/MACOS_resources/mmacos/src'));
addpath(fullfile(getenv('HOME'),'dev/MACOS_resources/mmacos/design/src'));
exdir = fileparts(mfilename('fullpath'));  if isempty(exdir), exdir = pwd; end
P = e2e_params();
D = P.D_m;  LAM = P.lambda_m;
s1 = load(fullfile(exdir, 's1_telescope.mat'));
imodes = P.inst.modes;  if isempty(imodes), imodes = P.modes; end

fprintf('====================================================================\n');
fprintf(' E2E stage 2: imaging instrument | 3-mirror relay | field +-%g'' -> +-%g''\n', ...
        P.fov_arcmin, P.inst.fov_arcmin);
fprintf('====================================================================\n');

%% -- [1] rebuild telescope + append the relay -------------------------
e1  = s1.spec.elt;
i3  = s1.pm(end);                                   % M3 in the s1 chain
ifp = find(strcmp({e1.kind}, 'FocalPlane'), 1, 'last');
b   = norm(e1(ifp).Vpt(:) - e1(i3).Vpt(:));         % M3 -> telescope focus
K1  = arrayfun(@(k) e1(k).Kc, s1.pm);
zf  = P.fold_frac*D;
r_fold = P.fold_margin * (zf - s1.lay.int_focus_z) / ...
         (2*P.primary_fnum*P.secondary_mag);
is_offner = strcmpi(string(P.inst.type), "offner");
if is_offner
    % Concentric ring-field 1:1 Offner: concave R (twice) + convex R/2
    % at the stop, chief-path legs/tilts from the closed geometry.
    % M4 = a FLAT routing fold + corrector station inside the object
    % leg (flat => zero tilt astigmatism at any routing angle).
    [olegs, otilts, og] = offner_layout(P.inst.offner_R, P.inst.offner_h, ...
                                        'fno', P.system_fnum);
    Rrel = [1e6, P.inst.offner_R, P.inst.offner_R/2, P.inst.offner_R];
    rlegs = [olegs(1) - P.inst.dpast_m, olegs(2), olegs(3)];
    rtilts = [P.inst.tilt_deg(1), otilts];
else
    Rrel = P.inst.R_m;
    if isempty(Rrel)
        % zigzag relay: collimator f5 = its distance from the telescope
        % focus (exact whatever the telescope conjugates); camera f6 =
        % f5 (unit magnification); M4 stays weak.
        f5 = P.inst.dpast_m + P.inst.legs_m(1);
        Rrel = [20, 2*f5, 2*f5];
    end
    rlegs = P.inst.legs_m;
    rtilts = P.inst.tilt_deg;
end
t = macos.design.Telescope('family','TMA', ...
        'aperture_diameter_m', D, 'wavelength_m', LAM, ...
        'model_size', P.model_size, 'grid_npts', P.grid_npts);
t.add_mirror('M1','radius_m',s1.R(1),'spacing_after_m',s1.tsp(1),'conic',K1(1));
t.add_mirror('M2','radius_m',s1.R(2),'spacing_after_m',s1.tsp(2),'convex',true, ...
             'conic',K1(2));
t.add_mirror('M3','radius_m',s1.R(3),'spacing_after_m', b + P.inst.dpast_m, ...
             'conic',K1(3));
t.add_mirror('M4','radius_m',Rrel(1),'spacing_after_m',rlegs(1), ...
             'tilt_deg',rtilts(1),'conic',0);
t.add_mirror('M5','radius_m',Rrel(2),'spacing_after_m',rlegs(2), ...
             'tilt_deg',rtilts(2),'conic',0);
if is_offner
    t.add_mirror('M6','radius_m',Rrel(3),'spacing_after_m',rlegs(3), ...
                 'tilt_deg',rtilts(3),'convex',true,'conic',0);
    t.add_mirror('M7','radius_m',Rrel(4),'spacing_after','derive', ...
                 'tilt_deg',rtilts(4),'conic',0);
else
    t.add_mirror('M6','radius_m',Rrel(3),'spacing_after','derive', ...
                 'tilt_deg',rtilts(3),'conic',0);
end
t.add_focal_plane('FP','ap_r',P.fp_body_r);
t.set_field_bias(s1.bias);
t.add_fold('FM','after','M2','dist_m', s1.tsp(1)+zf, 'to',[1 0 0], ...
           'ap_r', r_fold);
t.set_hole('M1', s1.r_hole);
t.build();
% re-apply the s1 telescope freeform figures (same modes/lmon contract)
for k = s1.pm
    ff = e1(k).freeform;
    if isstruct(ff) && ~isempty(ff) && isfield(ff,'modes')
        t.set_freeform(k, ff.modes, ff.coef, 'type', ff.type, 'lmon', ff.lmon);
    end
end
t.build('', 'init', false);
t.center_focal_plane();
e = t.spec.elt;  pm = powered_(t);
fprintf('\n[1] chain (telescope + bench relay; M3->M4 = %.3f m):\n', ...
        b + P.inst.dpast_m);
for k = 1:numel(e)
    fprintf('    %-4s %-10s Vpt=[%8.3f %8.3f %8.3f]\n', e(k).name, e(k).kind, e(k).Vpt);
end

%% -- [2] baseline: the +-2' ladder with the bare relay ----------------
h2 = P.inst.fov_arcmin;
F2 = [ (h2/2)*[1 0; 0 1; 0 -1; 1 1; 1 -1] ;
        h2   *[1 0; 0 1; 0 -1; 1 1; 1 -1] ] * pi/180/60;
w2 = [1, 1 + (F2(:,1).' > 0)];
% DENSE field grid for the SVD stages + final scoring (Dave: more field
% points smooth the OPD between solve samples).  CALIB is capped at 12
% FOVs; the SVD engine traces its own fields with no cap.
F2s = macos.design.field_grid(h2, P.inst.nfield_svd, 'units','arcmin');
d0 = wfe_field_diag(t, F2, 'quiet', true);
fprintf('\n[2] baseline over +-%g'': worst %.3f raw / %.3f -tilt waves\n', ...
        h2, max(d0.rms_raw), max(d0.rms_tilt));

%% -- [3] M5/M6 conic+ROC solve at the bias point ----------------------
% The weak correctors (M4/M6) are HELD OUT: with ROC+conic free a
% corrector turns into a strongly-powered K~-2000 surface (conic
% abused as high-order sag) and drags the first order off spec.  Their
% correction budget is the freeform in [4].
if is_offner
    % pm = [M1 M2 M3 M4 M5 M6 M7]; the convex stop mirror M6 is
    % pupil-conjugate (M1-degenerate) and stays OUT of the solves.
    relay = pm(4:7);  cams = [];  weak = pm(4);
    strong = pm([2 3 5 7]);  polishset = pm([2 3 4 5 7]);
else
    relay = pm(4:6);  cams = pm([5 6]);  weak = pm(4);
    strong = pm([2 3 5 6]);  polishset = pm(2:6);
end
t.align_focal_plane('grid', 3, 'span_arcmin', h2/2);
if isempty(cams)
    % Offner: the spheres ARE the design -- no ROC/conic solve (a solve
    % here would un-Offner the concentricity).  Measure only.
    t.align_focal_plane('grid', 3, 'span_arcmin', h2/2);
    macos.trace(numel(t.spec.elt));  wfe_pt = rms_waves(macos.opd(), LAM);
    r3 = struct('wfe_before', wfe_pt*LAM, 'wfe_after', wfe_pt*LAM);
    fprintf('\n[3] Offner relay held (concentric spheres by design): bias point %.4f waves\n', ...
            wfe_pt);
else
t.optimize('fields_arcmin', [], 'elts', cams, ...
           'dofs', [0 0 0 0 0 0 1 1], 'max_iters', P.max_iters);
t.align_focal_plane('grid', 3, 'span_arcmin', h2/2);
r3 = t.optimize('fields_arcmin', [], 'elts', cams, ...
                'dofs', [0 0 0 0 0 0 1 1], 'max_iters', P.max_iters);
macos.trace(numel(t.spec.elt));  wfe_pt = rms_waves(macos.opd(), LAM);
fprintf('\n[3] M5/M6 conic+ROC @ bias point (FP aligned, correctors held): %.4f -> %.4f waves\n', ...
        max(r3.wfe_before)/LAM, wfe_pt);
end
fprintf('    relay R = %s m, K = %s\n', ...
        mat2str(arrayfun(@(k) abs(e_now(t,k).Kr), relay), 4), ...
        mat2str(arrayfun(@(k) e_now(t,k).Kc, relay), 4));

%% -- [4] joint freeform field solve over +-2' -------------------------
lz = field_zone_lmon(t, pm, F2);
fprintf('\n[4] field-zone lMon (m): %s\n', mat2str(lz, 3));
fprintf('    (s1 zones were %s -- M2/M3 grow with the wider field)\n', ...
        mat2str(s1.lz, 3));
% [4a] joint CALIB FIELD solve on the powered set M2/M3/M5/M6 (M1 out:
% pupil-degenerate with the relayed stop; the weak correctors get the
% SVD engine next -- CALIB's FD-LM goes singular on their sparser
% support).
[~, si] = ismember(strong, pm);
r4 = t.optimize_freeform(strong, 'modes', imodes, 'type', P.ztype, ...
                         'fields', F2, 'weights', w2, 'lmon', lz(si), ...
                         'max_iters', P.inst.max_iters_ff);
fprintf('    [4a] joint field solve (elts %s): worst %.4f -> %.4f waves\n', ...
        mat2str(strong), max(r4.wfe_before)/LAM, max(r4.wfe_after)/LAM);
% [4b] the weak correctors at their intermediate conjugates (SVD
% engine: truncated SVD + damping + line search on the true
% multi-field merit -- rank-safe where CALIB blows up).
r4w = zern_jacobian_solve(t, weak, 'modes', imodes, 'type', P.ztype, ...
        'lmon', lz(4), 'fields', F2s, 'iters', 2);
fprintf('    [4b] M4 corrector (SVD rank %d): worst %.4f -> %.4f -tilt waves\n', ...
        r4w.rank, max(r4w.wfe(1,:))/LAM, max(last_wfe_(r4w))/LAM);
% [4c] SVD polish of the full instrument+telescope set (M2..M7).
[~, ci] = ismember(polishset, pm);
r4c = zern_jacobian_solve(t, polishset, 'modes', imodes, 'type', P.ztype, ...
        'lmon', lz(ci), 'fields', F2s, 'iters', 2);
fprintf('    [4c] SVD polish (elts %s): worst %.4f -> %.4f -tilt waves\n', ...
        mat2str(polishset), max(r4c.wfe(1,:))/LAM, max(last_wfe_(r4c))/LAM);
% [4d] M1 common mode over the FULL field set (the single-field CALIB
% null wandered -- tilt is a gauge at one field -- and traded
% worst-field 0.72 -> 3.6; the SVD engine projects the gauge out).
r4b = zern_jacobian_solve(t, pm(1), 'modes', P.modes, 'type', P.ztype, ...
        'lmon', lz(1), 'fields', F2s, 'iters', 2);
d4 = wfe_field_diag(t, F2s, 'quiet', true);   % scored on the DENSE grid
wfe_ff = max(d4.rms_raw);  wfe_ft = max(d4.rms_tilt);
fprintf(['    [4d] M1 common mode (SVD): worst %.4f -> %.4f -tilt; ', ...
         'final +-%g'''': %.4f raw / %.4f -tilt -> %s\n'], ...
        max(r4b.wfe(1,:))/LAM, max(last_wfe_(r4b))/LAM, h2, wfe_ff, wfe_ft, ...
        ternary(wfe_ft < P.dl_waves, 'DIFFRACTION-LIMITED', 'residual'));
cmax = arrayfun(@(k) coef_max_(t, k), pm);
fprintf('    coefficient sanity: max|coef| = %s m\n', mat2str(cmax, 2));
if any(cmax > 1e-2)
    warning('e2e:s2:coef', ['metre/cm-scale Zernike coefficients -- the ', ...
        'canceling-pair pathology; revisit lMon / staging.']);
end

%% -- [4e] distortion: M4 as the reflective field corrector ------------
% The raw-vs-tilt gap is relay DISTORTION: per-field chief-ray landing
% error after the best AFFINE map (magnification / rotation /
% anamorphism are plate-scale calibration, not distortion).  Detector
% tilt cannot correct it (Dave) -- but M4, near the focus, bends each
% field's chief individually: its per-field-tilt channel (useless for
% blur, which is why its blur rank collapsed) is exactly the
% distortion knob.  Linear solve: poke each M4 mode, build the
% chief-displacement Jacobian (affine part projects out inside the
% metric), damped LSQ, verify the blur is untouched.
[dx0, ~] = distortion_(t, F2s);
DIST_TOL = 1e-4;                              % m at the detector; below
if dx0 < DIST_TOL                             % this, correction is not
    dx1 = dx0;  d4e = d4;  a_pick = 0;        % worth ANY blur trade
    fprintf(['\n[4e] distortion %.1f um already below the %.0f um bar -- ', ...
             'M4 corrector stands down\n'], dx0*1e6, DIST_TOL*1e6);
else
i4 = pm(4);  ff4 = t.spec.elt(i4).freeform;
c0 = ff4.coef(:).';  ds = 2e-7;
[~, r0] = distortion_(t, F2s);
J = zeros(numel(r0), numel(c0));
for j = 1:numel(c0)
    cj = c0;  cj(j) = cj(j) + ds;
    t.set_freeform(i4, ff4.modes, cj, 'type', ff4.type, 'lmon', ff4.lmon);
    t.build('', 'init', false);
    [~, rj] = distortion_(t, F2s);
    J(:,j) = (rj(:) - r0(:)) / ds;
end
lamd = 1e-3 * norm(J);
dc = -(J.'*J + lamd^2*eye(numel(c0))) \ (J.'*r0(:));
% BLUR-GUARDED step: an unguarded full step fixed the mapping 4x but
% cost 6 waves of blur (large tilt-channel excursions carry
% within-patch structure).  Scan the step scale and take the best
% distortion whose blur cost stays within BLUR_TOL of the pre-solve
% state.
BLUR_TOL = 0.02;                              % waves, allowed blur cost
alphas = [1 0.5 0.25 0.1 0.05 0.02];
dx1 = dx0;  d4e = d4;  a_pick = 0;
for a = alphas
    t.set_freeform(i4, ff4.modes, c0 + a*dc.', 'type', ff4.type, 'lmon', ff4.lmon);
    t.build('', 'init', false);
    da = wfe_field_diag(t, F2s, 'quiet', true);
    if max(da.rms_tilt) <= wfe_ft + BLUR_TOL
        [dxa, ~] = distortion_(t, F2s);
        if dxa < dx1, dx1 = dxa;  d4e = da;  a_pick = a; end
        break                                  % largest guarded step wins
    end
end
t.set_freeform(i4, ff4.modes, c0 + a_pick*dc.', 'type', ff4.type, 'lmon', ff4.lmon);
t.build('', 'init', false);
fprintf(['\n[4e] M4 distortion solve (blur-guarded, alpha %.2f): rms %.1f -> %.1f um ', ...
         '(%.3f -> %.3f arcsec); blur worst %.4f -> %.4f -tilt waves\n'], ...
        a_pick, dx0*1e6, dx1*1e6, dx0/(P.system_fnum*D)*206265, ...
        dx1/(P.system_fnum*D)*206265, wfe_ft, max(d4e.rms_tilt));
end
wfe_ff = max(d4e.rms_raw);  wfe_ft = max(d4e.rms_tilt);

%% -- [5] clearance + M1 hole re-check ---------------------------------
r_hole = m1_hole_radius_(t, P.hole_margin);
if isfinite(r_hole), t.set_hole('M1', r_hole); end
fa = t.align_focal_plane('grid', 5, 'span_arcmin', min(0.5, h2/2));
rep = t.check_clipping('noload', true, 'quiet', true);
bad = {rep(~[rep.ok]).name};
fprintf('\n[5] M1 hole r = %.3f m; true FP: tilt %.3f deg, sag %+.1f..%+.1f um\n', ...
        r_hole, fa.tilt_deg, min(fa.sag_m)*1e6, max(fa.sag_m)*1e6);
fprintf('    clearance: %d/%d bodies clear%s\n', sum([rep.ok]), numel(rep), ...
        ternary(isempty(bad), '', [' (' strjoin(bad,',') ')']));

%% -- [6] views --------------------------------------------------------
try
    fv = macos.view_std('args', {'show','beam'}, 'visible', false, ...
            'title', sprintf('e2e s2: +3-mirror imaging relay, field +-%g''', h2), ...
            'save', fullfile(exdir, 's2_views.png'));
    close(fv);
    fprintf('\n[6] standard views: s2_views.png\n');
catch ME, fprintf('\n[6] view_std skipped (%s)\n', ME.message); end
try
    Fmap = macos.design.field_grid(h2, 7, 'units','arcmin');
    dmap = wfe_field_diag(t, Fmap, 'quiet', true);
    scan = struct('fields', Fmap*180*60/pi, 'wfe', dmap.rms_raw(:));
    f1 = t.view_field_map(scan, 'kind', 'contour');
    saveas(f1, fullfile(exdir, 's2_wfe_field.png'));  close(f1);
    fg = figure('Visible','off');
    contourf(fa.map.thx_arcmin, fa.map.thy_arcmin, fa.map.sag_m*1e6, ...
             15, 'LineColor','none');
    axis equal tight; colormap(parula); cb = colorbar;
    cb.Label.String = 'focus sag from fitted FP  [\mum]';
    xlabel('\theta_x  [arcmin]'); ylabel('\theta_y  [arcmin]');
    title(sprintf('field curvature (FP tilt %.3f\\circ)', fa.tilt_deg));
    saveas(fg, fullfile(exdir, 's2_fpmap.png'));  close(fg);
    fprintf('    field maps: s2_wfe_field.png + s2_fpmap.png\n');
catch ME, fprintf('    field maps skipped (%s)\n', ME.message); end

%% -- [7] deliverables + standalone verification -----------------------
t.add_pupil(numel(t.spec.elt));
rxfile  = fullfile(exdir, 's2_instrument.in');
matfile = fullfile(exdir, 's2_instrument.mat');
t.save(rxfile);  t.save_spec(matfile);
fprintf('\n[7] saved: %s\n           + %s\n', rxfile, matfile);
macos.init(P.model_size);
nv = macos.load_rx(rxfile);  sv = macos.trace(nv);
rv = macos.get_ray_info(sv.nRays);
np = nnz(logical(rv.ok_pass) & logical(rv.ok_trace));
fprintf('    standalone reload: %d elts, %d/%d rays pass -> %s\n', ...
        nv, np, sv.nRays, ternary(np > 0.9*sv.nRays, 'VERIFIED', '** BROKEN **'));
t.build('', 'init', false);

%% -- [8] the design report (+ stage-2 addendum) -----------------------
fprintf('\n[8] design report:\n');
rpt = design_report(t, 'rings_arcmin', [0.5 1 1.5 h2], 'align', fa);
add = { ...
 ' -- stage-2 addendum: the imaging instrument --'
 sprintf('   relay: M4 corrector %.2f m past the telescope focus / M5 collimator / M6 corrector / M7 camera', ...
         P.inst.dpast_m)
 sprintf('   relay R = %s m, K = %s (M5/M7 solved at the bias point)', ...
         mat2str(arrayfun(@(k) abs(e_now(t,k).Kr), relay), 4), ...
         mat2str(arrayfun(@(k) e_now(t,k).Kc, relay), 4))
 sprintf('   joint solve: CALIB M2/M3/M5/M6 (11 pts) + SVD M4 + polish M2..M6 + M1 common mode (%dx%d grid)', ...
         P.inst.nfield_svd, P.inst.nfield_svd)
 sprintf('   field-zone lMon (m): %s  (s1: %s)', mat2str(lz,3), mat2str(s1.lz,3))
 sprintf('   baseline +-%g'': %.3f -tilt -> after joint solve %.4f raw / %.4f -tilt waves (%s)', ...
         h2, max(d0.rms_tilt), wfe_ff, wfe_ft, ...
         ternary(wfe_ft < P.dl_waves,'DL','residual'))
 sprintf('   bias point %.4f -tilt waves | max|coef| %s m', bias_pt_wfe_(d4e, F2s), mat2str(cmax,2))
 sprintf('   distortion (M4 reflective field-corrector solve): %.1f -> %.1f um rms at the detector', ...
         dx0*1e6, dx1*1e6)
 '======================================================='};
addtxt = sprintf('%s\n', add{:});
fprintf('%s', addtxt);
fid = fopen(fullfile(exdir, 's2_report.txt'), 'w');
fprintf(fid, '%s%s', rpt.text, addtxt);  fclose(fid);
fprintf('    report: s2_report.txt\n');

save(matfile, 'P', 'r3', 'r4', 'r4b', 'lz', 'fa', 'rpt', 'pm', 'relay', ...
     'r4w', 'r4c', 'r_hole', 'd0', 'd4', 'd4e', 'dx0', 'dx1', '-append');
fprintf('\nStage 2 complete.  Next: s3_segmentation.m.\n');

% ---- helpers --------------------------------------------------------
function ek = e_now(t, k), ek = t.spec.elt(k); end

function c = coef_max_(t, k)
%COEF_MAX_  Max |freeform coef| of elt k; 0 when it carries no figure
%   (e.g. the Offner convex, held out of every solve).
    ff = t.spec.elt(k).freeform;
    if isstruct(ff) && ~isempty(ff) && isfield(ff, 'coef') && ~isempty(ff.coef)
        c = max(abs(ff.coef));
    else
        c = 0;
    end
end

function [rms_m, res] = distortion_(t, F)
%DISTORTION_  Chief-ray mapping error over field set F: landing points
%   in the detector plane minus their best AFFINE map of field angle
%   (residual = true distortion).  Returns the rms (m) and the 2xN
%   residual matrix.
    e = t.spec.elt;
    ifp = find(strcmp({e.kind}, 'FocalPlane'), 1, 'last');
    ps = e(ifp).psi(:);  vpt = e(ifp).Vpt(:);
    [~, i0] = min(abs(ps));  u = zeros(3,1);  u(i0) = 1;
    u1 = u - ps*(ps.'*u);  u1 = u1/norm(u1);  u2 = cross(ps, u1);
    n = size(F,1);  Pp = zeros(2, n);
    for i = 1:n
        t.trace_at_field(F(i,:));
        sc = macos.trace(ifp);
        b = macos.get_ray_info(sc.nRays);
        q = b.pos(:,1) - vpt;              % ray 1 = the chief
        Pp(:,i) = [u1.'*q; u2.'*q];
    end
    t.trace_at_field([]);
    A = [F, ones(n,1)];
    C = (A \ Pp.').';
    res = Pp - C*A.';
    rms_m = sqrt(mean(sum(res.^2, 1)));
end

function w = last_wfe_(r)
%LAST_WFE_  Last APPLIED wfe row of a zern_jacobian_solve result (a
%   pass that found no improving step leaves later rows zero).
    i = find(any(r.wfe, 2), 1, 'last');
    w = r.wfe(i, :);
end

function w = bias_pt_wfe_(d, F)
%BIAS_PT_WFE_  Tilt-removed WFE (waves) at the field point nearest the
%   bias center in a wfe_field_diag result over field set F.
    [~, i0] = min(vecnorm(F.'));
    w = d.rms_tilt(i0);
end

function pm = powered_(t)
    e = t.spec.elt;
    pm = find(arrayfun(@(x) strcmp(x.kind,'Reflector') && abs(x.Kr) < 1e21, e));
end

function r = m1_hole_radius_(t, margin)
%M1_HOLE_RADIUS_  Largest crossing radius of any post-M2 beam leg
%   through the M1 plane (see s1_telescope.m).
    macos.ray_hist('on');
    s = macos.trace();
    hh = macos.ray_hist(s.nRays);
    macos.ray_hist('off');
    p0 = t.spec.elt(1).Vpt(:);  ps = t.spec.elt(1).psi(:);
    r = NaN;  hit = [];
    for leg = 3:size(hh.P,3)-1
        A = squeeze(hh.P(:,:,leg));  B = squeeze(hh.P(:,:,leg+1));
        ok = hh.ok(:,leg) & hh.ok(:,leg+1);
        for i = find(ok(:)).'
            d  = B(:,i) - A(:,i);
            dn = dot(ps, d);
            if abs(dn) < 1e-12, continue; end
            sfr = dot(ps, p0 - A(:,i)) / dn;
            if sfr > 0 && sfr < 1
                q = A(:,i) + sfr*d;
                hit(end+1) = norm(q - p0 - ps*dot(ps, q - p0));  %#ok<AGROW>
            end
        end
    end
    if ~isempty(hit), r = margin * max(hit); end
end

function w = rms_waves(W, lam)
    v = W(isfinite(W) & W ~= 0 & abs(W) < 1e30);
    if isempty(v), w = NaN; else, w = std(v)/lam; end
end
function s = ternary(c, a, b), if c, s = a; else, s = b; end, end
