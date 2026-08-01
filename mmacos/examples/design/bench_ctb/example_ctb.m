% example_ctb.m
% ===================================================================
%  MACOS DESIGN LAYER -- CORONAGRAPH TESTBED (CTB) WORKED EXAMPLE
% ===================================================================
%  Build an all-reflective coronagraph testbed relay SEQUENTIALLY
%  with the macos.design.Bench add-optic utilities, optimize it in
%  physically-staged steps, study its POLARIZATION behaviour, and add a
%  pupil-imaging-lens design step.
%
%  The bench is laid out PLANAR (all folds in one plane -- the compact,
%  DST2R-style packaging; folds alternate side for layout only).
%
%  POLARIZATION is reported as a DIAGNOSTIC (Al-coated mirrors ->
%  jones_pupil/pol_maps at the FPA).  Fold polarization compensation is
%  non-trivial and NOT attempted here: measured, an in-plane reversed fold
%  does NOT cancel retardance (mean identical, variation worse), and a
%  naive 90-deg crossed-plane (3-D) fold cuts diattenuation ~90x but not
%  the contrast-relevant retardance VARIATION and costs ~10x WFE.  Proper
%  compensation pairs equal-AOI folds with exactly-crossed planes AND
%  balances the induced astigmatism -- a layout+optimization follow-up.
%
%  TOPOLOGY (light order) -- a 2-DM coronagraph relay alternating PUPIL
%  and FOCUS planes:
%    source -> OAP1 -> DM1(pupil) -> DM2(pupil) -> OAP2 -> [focus23]
%      -> OAP3 -> apodizer -> OAP4 -> FPM -> OAP5 -> Lyot -> OAP6
%      -> field_stop -> OAP7 -> backend -> OAP8 -> FPA
%  8 off-axis parabolas (Kc=-1); DM1/DM2 flat fold DMs (probed on the
%  collimated pupil); apod/FPM/Lyot/field_stop/backend passive Reference
%  markers (conjugate targets + real coronagraph mask sites).
%
%  OAP geometry (arbitrary fold): add_oap takes the parent focal length
%  'f' directly; conjugate r = f/cos^2(AOI) = 2f/(1-cos theta).  DST2R
%  seeds: f = [2500 1524 1143 1350 675 635 635 762] mm, AOI ~ 5 deg.
%
%  Source model: point source = section of a sphere; Aperture = the
%  NUMERICAL APERTURE, sized to fill the LIMITING aperture (the DM stop).
%
%  Staged optimization: each conjugate solved with a GEOMETRIC cost on
%  its own plane, in light order, freezing upstream optics (a single WFE
%  solve at the FPA can hide mis-collimation as defocus).  RMS WFE and
%  polarization are reported as final figures of merit.
%
%  Run:   >> run('.../example_ctb.m')     (Requires MACOS_HOME.)
% ===================================================================

addpath('~/dev/MACOS_resources/mmacos/src');
exdir = fileparts(mfilename('fullpath'));
if isempty(exdir), exdir = pwd; end
assert(~isempty(getenv('MACOS_HOME')), ...
    'MACOS_HOME must be set (engine needs macos_param.txt).');
macos.init(256);

% ===================================================================
%  PARAMETER BLOCK  (all lengths mm; edit and re-run)
% ===================================================================
P.F_OAP = [2500 1524 1143 1350 675 635 635 762];  % DST2R parent focal lengths
P.AOI   = 5;                                       % OAP/DM angle of incidence, deg
P.R_OAP = 75;                                       % OAP clear-aperture radius, mm
P.R_DM  = 22.5;                                     % DM aperture radius, mm (stop)
P.FILL  = 0.95;                                     % source NA fills FILL of DM
P.WAVLEN= 500e-6;                                   % 500 nm in mm
P.r     = P.F_OAP / cosd(P.AOI)^2;                  % conjugate distances (mm)
% free (collimated-space) leg lengths -- set generously, no overlap:
P.L_DM1=600; P.L_DM2=500; P.L_O2=700; P.L_APOD=350; P.L_O4=500;
P.L_LYOT=350; P.L_O6=450; P.L_BACK=300; P.L_O8=450;
P.AP    = P.R_DM*P.FILL / P.r(1);                   % source numerical aperture
% aluminium coating for the polarization study (index n - i*kappa):
P.AL_N = 1.2;  P.AL_K = 7.0;  P.AL_T = 1e-4;        % ~100 nm Al

% pupil beam radii down the relay (demagnify by focus/collimate f-ratios):
w_DM = P.AP*P.r(1);
fprintf('pupil beam radii (mm): DM %.2f  apod %.2f  Lyot %.2f  backend %.2f\n', ...
    w_DM, w_DM*P.F_OAP(3)/P.F_OAP(2), ...
    w_DM*P.F_OAP(3)/P.F_OAP(2)*P.F_OAP(5)/P.F_OAP(4), ...
    w_DM*P.F_OAP(3)/P.F_OAP(2)*P.F_OAP(5)/P.F_OAP(4)*P.F_OAP(7)/P.F_OAP(6));
assert(w_DM < P.R_DM, 'beam (%.2f) overfills DM aperture (%.2f)', w_DM, P.R_DM);

% ===================================================================
%  BUILD + OPTIMIZE THE (PLANAR) CTB
% ===================================================================
%  The bench is laid out PLANAR (all folds in one plane -- the compact,
%  DST2R-style packaging).  A crossed-plane 3-D variant was studied for
%  polarization compensation but is NOT a net win (it cuts diattenuation
%  ~90x yet does not beat planar on the contrast-relevant retardance
%  VARIATION, and costs ~10x WFE) -- see the polarization diagnostic note
%  below.  So we ship the planar bench and REPORT polarization on it.
out = run_variant('planar', P, exdir);

% ===================================================================
%  POLARIZATION DIAGNOSTIC (planar bench, Al-coated mirrors)
% ===================================================================
%  Per macos.pol_maps: the pupil-MEAN retardance after folds is largely a
%  geometric FRAME rotation (not an aberration); the VARIATION across the
%  pupil is what sets a coronagraph contrast floor.  Report both.
fprintf('\n=== POLARIZATION at FPA (Al-coated mirrors) ===\n');
fprintf('  retardance : mean %.5f rad   var_rms %.3e rad\n', ...
    out.pol.ret_mean, out.pol.ret_var);
fprintf('  diattenuat.: mean %.5f       var_rms %.3e\n', ...
    out.pol.diat_mean, out.pol.diat_var);
fprintf([ ...
'  NOTE: fold polarization compensation is non-trivial -- in-plane fold\n' ...
'  alternation does NOT cancel retardance, and naive crossed-plane folding\n' ...
'  cuts diattenuation but not retardance-variation and costs WFE.  Proper\n' ...
'  compensation pairs equal-AOI folds with exactly-crossed planes AND\n' ...
'  balances the induced astigmatism (flagged follow-up).\n']);

% ===================================================================
%  PUPIL-IMAGING-LENS (PIL) DESIGN STEP
% ===================================================================
%  An additional design step (emits its own Rx per lens): insert a
%  pupil-imaging lens near the star focus to switch the camera from
%  SOURCE imaging to PUPIL imaging.  Per D. Marx's DST2R layouts, two
%  PILs give a size/camera-position trade -- a longer lens makes a bigger
%  pupil image and pushes the camera back.  We build both and report the
%  pupil-image size and camera shift, emitting ctb_pil150.in / ctb_pil75.in.
fprintf('\n=== PIL DESIGN STEP (pupil-imaging lens) ===\n');
PILS = [150 75];                            % lens focal lengths (mm)
pil = [];
for fpil = PILS
    p = build_pil(fpil, P, exdir);
    if isempty(pil), pil = p; else, pil(end+1) = p; end   %#ok<AGROW>
end
out.pil = pil;
% report the camera-position trade (relative to the first, per Marx)
cam0 = out.pil(1).cam_shift_mm;
for q = 1:numel(out.pil)
    pl = out.pil(q);
    fprintf('  PIL f=%3gmm: pupil image diam %.2f mm, camera %+.1f mm vs f=%gmm  (%s)\n', ...
        pl.f, pl.pupil_diam_mm, pl.cam_shift_mm - cam0, PILS(1), pl.rx);
end
fprintf(['  (size ratio %.2fx and camera move ~%.0f mm mirror Marx''s ' ...
         '150mm/75mm 1000/500-px, +106mm trade)\n'], ...
    out.pil(1).pupil_diam_mm/out.pil(2).pupil_diam_mm, ...
    abs(out.pil(2).cam_shift_mm - out.pil(1).cam_shift_mm));

save(fullfile(exdir,'ctb.mat'), 'out');
fprintf('\nDONE.  Optimized source-imaging Rx: %s\n', out.rx_opt);
fprintf('       PIL pupil-imaging Rx: %s\n', strjoin({out.pil.rx}, ', '));

% ===================================================================
%  BUILD + OPTIMIZE + DIAGNOSE ONE VARIANT
% ===================================================================
function R = run_variant(mode, P, exdir)
    [b, ix] = build_ctb(mode, P);
    b.print_chain();
    rx_seed = fullfile(exdir, sprintf('ctb_%s_seed.in', mode));
    b.emit(rx_seed);

    % ---- sketch: the planar fold layout renders true in the XY plane --
    fsk = b.sketch('title', sprintf('ctb (%s) -- OAP legs pinned by r=f/cos^2(AOI)', mode));
    set(fsk, 'Position', [100 100 1600 1000]);
    print(fsk, fullfile(exdir, sprintf('ctb_%s_params.png', mode)), '-dpng', '-r150');
    close(fsk);

    % ---- verify: element count, trace success, no vignetting, chief ---
    macos.load_rx(rx_seed);
    nE = macos.num_elt();
    assert(nE == numel(b.E), 'engine %d elts, builder %d', nE, numel(b.E));
    okc = zeros(1,nE);  dchief = zeros(1,nE);
    for k = 1:nE
        sk = macos.trace(k);  info = macos.get_ray_info(sk.nRays);
        okc(k) = nnz(info.ok_trace(:) & info.ok_pass(:));
        dchief(k) = norm(info.pos(:,1) - b.E(k).rpt);
    end
    fprintf('  ok/elt [%s]  chief max %.3g mm\n', num2str(okc), max(dchief));
    assert(all(okc > 0), 'trace FAILED (0 ok rays at some element)');
    assert(max(dchief) < 1e-6, 'chief-ray model disagrees with engine');
    assert(okc(ix.DM1)-okc(ix.FPA) <= 0.02*okc(ix.DM1), 'seed over-vignetted');
    macos.stop(ix.DM1);  macos.save_rx(rx_seed);

    % ---- staged optimization (light order, freeze upstream) -----------
    R.elt = ix;  R.rx_seed = rx_seed;
    rxc = rx_seed;
    % A: collimate at DM1 (OAP1, Kr-only sphere -- cheaper to fab)
    [R.A.Kr,R.A.Kc] = optimize_conic(rxc, ix.OAP1, @()collimation_cost(ix.DM1), 'kr');
    rxc = save_stage(exdir,mode,'A');
    % B: pupil relay DM->apodizer (OAP2 focus, OAP3 collimate)
    optimize_conic(rxc, ix.OAP2, @()spot_cost(ix.Focus23));       rxc = save_stage(exdir,mode,'B2');
    optimize_conic(rxc, ix.OAP3, @()collimation_cost(ix.Apodizer)); rxc = save_stage(exdir,mode,'B3');
    % C: focus at FPM (OAP4)
    optimize_conic(rxc, ix.OAP4, @()spot_cost(ix.FPM));          rxc = save_stage(exdir,mode,'C');
    % D: collimate at Lyot (OAP5)
    optimize_conic(rxc, ix.OAP5, @()collimation_cost(ix.Lyot));  rxc = save_stage(exdir,mode,'D');
    % E: focus at field stop (OAP6)
    optimize_conic(rxc, ix.OAP6, @()spot_cost(ix.FieldStop));    rxc = save_stage(exdir,mode,'E');
    % F: collimate at backend (OAP7)
    optimize_conic(rxc, ix.OAP7, @()collimation_cost(ix.Backend)); rxc = save_stage(exdir,mode,'F');
    % G: focus at FPA (OAP8)
    optimize_conic(rxc, ix.OAP8, @()spot_cost(ix.FPA));
    rx_opt = fullfile(exdir, sprintf('ctb_%s_opt.in', mode));
    macos.save_rx(rx_opt);  R.rx_opt = rx_opt;

    % ---- figures of merit ---------------------------------------------
    macos.load_rx(rx_opt);  macos.modify();
    sF = macos.trace(macos.num_elt());
    R.rmsWFE_waves = sF.rmsWFE / P.WAVLEN;
    z = macos.pupil_zone_map(ix.DM1, ix.FPA, 'ngrid',5, 'quiet',true);
    R.zone = z;
    fprintf('  RMS WFE %.4g waves | DM1->FPA zone spot med %.4g um worst %.4g um\n', ...
        R.rmsWFE_waves, 1e3*z.med_spot, 1e3*z.max_spot);

    % ---- POLARIZATION: Al-coat mirrors, Jones pupil at FPA ------------
    R.pol = pol_report(rx_opt, ix, P);
    fprintf('  pol @ FPA: ret mean %.5f (var %.2e), diatten mean %.5f (var %.2e)\n', ...
        R.pol.ret_mean, R.pol.ret_var, R.pol.diat_mean, R.pol.diat_var);

    % ---- render (beam through optics, fold plane) ---------------------
    macos.load_rx(rx_opt);  macos.modify();  macos.trace(macos.num_elt());
    f1 = macos.view_rx('show','beam','bundle','rings','nrings',3,'nspokes',12,'bodies','solid');
    set(f1, 'Color','w', 'Position',[100 100 1500 1000]);  axis equal; grid on;
    title(sprintf('ctb (%s) -- beam through the coronagraph relay', mode));
    print(f1, fullfile(exdir, sprintf('ctb_%s_view_rx.png', mode)), '-dpng', '-r150');
    close(f1);
    f2 = macos.view_std('title', sprintf('ctb (%s)', mode));
    print(f2, fullfile(exdir, sprintf('ctb_%s_view_std.png', mode)), '-dpng', '-r150');
    close(f2);
end

% ===================================================================
%  BUILD THE BENCH IN A GIVEN FOLD MODE
% ===================================================================
function [b, ix] = build_ctb(mode, P, tail)
%BUILD_CTB  Lay out the CTB relay (planar folds, compact DST2R-style).
%   MODE is retained for artifact naming ('planar'); every fold turns in
%   the global XY plane, alternating side so the near-retro bounces
%   zig-zag down the bench without doubling back through an upstream
%   optic.  (A crossed-plane 3-D variant was studied for polarization but
%   dropped -- see the header.)
%   TAIL: 'source' (default) terminates with OAP8 -> FPA (star imaging);
%   'none' stops after OAP8, leaving the bench ready for the PIL step to
%   append a pupil-imaging lens (build_pil).
    if nargin < 3, tail = 'source'; end
    b = macos.design.Bench('ctb', 'dir',[1;0;0], 'aperture',P.AP, ...
                           'ngridpts',63, 'wavelen',P.WAVLEN);
    S.k = 0;                              % fold counter (persists via nested)
    d0 = [1;0;0];

    function o = nextdir(d)
        % turn the chief by theta = 180 - 2*AOI about +z, alternating sign
        S.k = S.k + 1;  th = 180 - 2*P.AOI;
        o = rot_axis(d, [0;0;1], (1-2*mod(S.k,2))*th);
    end

    O1 = b.add_oap(P.r(1), nextdir(d0), 'mode','collimate','f',P.F_OAP(1),'name','OAP1','aprad',P.R_OAP); d0=b.dir;
    ix.OAP1=O1.i;
    ix.DM1 = b.add_mirror(P.L_DM1,'out',nextdir(d0),'name','DM1','aprad',P.R_DM); d0=b.dir;
    ix.DM2 = b.add_mirror(P.L_DM2,'out',nextdir(d0),'name','DM2','aprad',P.R_DM); d0=b.dir;
    O2 = b.add_oap(P.L_O2, nextdir(d0),'mode','focus','f',P.F_OAP(2),'name','OAP2','aprad',P.R_OAP); d0=b.dir;
    ix.OAP2=O2.i;
    ix.Focus23 = b.add_reference(P.r(2), 'Focus23');
    O3 = b.add_oap(P.r(3), nextdir(d0),'mode','collimate','f',P.F_OAP(3),'name','OAP3','aprad',P.R_OAP); d0=b.dir;
    ix.OAP3=O3.i;
    ix.Apodizer = b.add_reference(P.L_APOD, 'Apodizer');
    O4 = b.add_oap(P.L_O4, nextdir(d0),'mode','focus','f',P.F_OAP(4),'name','OAP4','aprad',P.R_OAP); d0=b.dir;
    ix.OAP4=O4.i;
    ix.FPM = b.add_reference(P.r(4), 'FPM');
    O5 = b.add_oap(P.r(5), nextdir(d0),'mode','collimate','f',P.F_OAP(5),'name','OAP5','aprad',P.R_OAP); d0=b.dir;
    ix.OAP5=O5.i;
    ix.Lyot = b.add_reference(P.L_LYOT, 'Lyot');
    O6 = b.add_oap(P.L_O6, nextdir(d0),'mode','focus','f',P.F_OAP(6),'name','OAP6','aprad',P.R_OAP); d0=b.dir;
    ix.OAP6=O6.i;
    ix.FieldStop = b.add_reference(P.r(6), 'FieldStop');
    O7 = b.add_oap(P.r(7), nextdir(d0),'mode','collimate','f',P.F_OAP(7),'name','OAP7','aprad',P.R_OAP); d0=b.dir;
    ix.OAP7=O7.i;
    ix.Backend = b.add_reference(P.L_BACK, 'Backend');
    % --- TAIL: source-imaging (OAP8 focuses the star to the FPA) --------
    %  The PIL design step (build_pil) reuses everything up to Backend and
    %  swaps in a pupil-imaging lens instead of this OAP8->FPA tail.
    O8 = b.add_oap(P.L_O8, nextdir(d0),'mode','focus','f',P.F_OAP(8),'name','OAP8','aprad',P.R_OAP); d0=b.dir;
    ix.OAP8=O8.i;
    if strcmp(tail,'source')
        ix.FPA = b.add_detector(P.r(8), 'FPA');   % star image (source imaging)
    end
    % 'none': leave the chief at OAP8's output; build_pil appends the PIL.
end

% ===================================================================
%  PIL DESIGN STEP: append a pupil-imaging lens to the CTB front end
% ===================================================================
function pl = build_pil(fpil, P, exdir)
%BUILD_PIL  Build a pupil-imaging configuration with an FPIL-mm lens.
%   Reuses the CTB front end through OAP8 (tail 'none'), then places a
%   thin lens that IMAGES THE EXIT PUPIL onto the camera.  OAP8 focuses
%   the star r8 ahead; the exit pupil sits at OAP8, i.e. a distance
%   (r8 + d_focus_lens) before the lens.  With the lens 'focus' mode the
%   camera lands at the pupil-image conjugate s_i = 1/(1/f - 1/s_o).  A
%   longer f -> larger |magnification| and a farther camera (Marx's trade).
    [b, ix] = build_ctb('planar', P, 'none');
    D_FL = 60;                       % OAP8-focus -> lens standoff (mm)
    s_o  = P.r(8) + D_FL;            % exit pupil (at OAP8) -> lens distance
    L = b.add_lens(D_FL, fpil, 40, 'mode','focus', 'n',1.5, 'name',sprintf('PIL%g',fpil));
    s_i  = 1/(1/fpil - 1/s_o);       % lens -> pupil image (thin-lens)
    mag  = s_i/s_o;                  % pupil-image magnification (signed)
    ix.PIL = L.i_pow;
    ix.PupilCam = b.add_detector(s_i - L.thickness, 'PupilCam');

    rx = fullfile(exdir, sprintf('ctb_pil%g.in', fpil));
    b.emit(rx);
    macos.load_rx(rx);  macos.stop(ix.DM1);  macos.save_rx(rx);

    % verify trace + measure the pupil image: rays from the DM pupil should
    % now form an extended PUPIL image (not a point) on the camera; its
    % diameter = the beam spread at PupilCam.
    macos.load_rx(rx);  macos.modify();
    s = macos.trace(ix.PupilCam);  info = macos.get_ray_info(s.nRays);
    ok = info.ok_trace(:) & info.ok_pass(:);
    dch = info.dir(:,1)/norm(info.dir(:,1));
    d = info.pos(:,ok) - info.pos(:,1);  dt = d - dch*(dch.'*d);
    pdiam = 2*max(sqrt(sum(dt.^2,1)));

    pl = struct('f',fpil, 'rx',rx, 'cam_shift_mm',s_i, ...
                'pupil_diam_mm',pdiam, 'mag',mag, 'nrays_ok',nnz(ok));
end

% ===================================================================
%  POLARIZATION REPORT (Al mirrors + Jones pupil at the FPA)
% ===================================================================
function pol = pol_report(rx, ix, P)
    macos.load_rx(rx);
    mir = [ix.OAP1 ix.DM1 ix.DM2 ix.OAP2 ix.OAP3 ix.OAP4 ix.OAP5 ix.OAP6 ix.OAP7 ix.OAP8];
    for e = mir
        macos.coating(e, 'index',P.AL_N, 'extinc',P.AL_K, 'thickness',P.AL_T);
    end
    macos.modify();
    jp = macos.jones_pupil(ix.FPA);
    pm = macos.pol_maps(jp);
    pol = struct('ret_mean',pm.mean.ret, 'ret_var',pm.var_rms.ret, ...
                 'diat_mean',pm.mean.D,  'diat_var',pm.var_rms.D);
end

% ===================================================================
%  LOCAL FUNCTIONS
% ===================================================================
function d = rot_axis(d0, ax, ang_deg)
%ROT_AXIS  Rodrigues rotation of d0 about unit axis AX by ANG_DEG.
    d0=d0(:); ax=ax(:)/norm(ax(:)); a=deg2rad(ang_deg);
    d = d0*cos(a) + cross(ax,d0)*sin(a) + ax*(ax.'*d0)*(1-cos(a));
    d = d/norm(d);
end
function rx = save_stage(exdir, mode, tag)
    rx = fullfile(exdir, sprintf('ctb_%s_stage%s.in', mode, tag));
    macos.save_rx(rx);
end

function [Kr, Kc] = optimize_conic(rx, pow_elt, costfn, dof)
%OPTIMIZE_CONIC  fminsearch over the conic DOFs of POW_ELT; RX reloaded
%   each eval; engine left at the optimum.  DOF 'kr' (sphere/fixed-conic,
%   cheaper) or 'kr_kc' (default, full conic).
    if nargin < 4, dof = 'kr_kc'; end
    macos.load_rx(rx);
    Kr0 = macos.get_elt_kr(pow_elt);  Kc0 = macos.get_elt_kc(pow_elt);
    o = optimset('Display','off','TolX',1e-6,'TolFun',1e-16,'MaxFunEvals',500);
    switch dof
        case 'kr'
            x = fminsearch(@(x) eval_conic(rx,pow_elt,[x(1) Kc0],costfn), Kr0, o);
            Kr = x(1);  Kc = Kc0;
        case 'kr_kc'
            x = fminsearch(@(x) eval_conic(rx,pow_elt,x,costfn), [Kr0 Kc0], o);
            Kr = x(1);  Kc = x(2);
        otherwise, error('optimize_conic: dof must be ''kr'' or ''kr_kc''.');
    end
    macos.load_rx(rx);
    macos.set_elt_kr(pow_elt, Kr);  macos.set_elt_kc(pow_elt, Kc);
end

function c = eval_conic(rx, pow_elt, x, costfn)
    macos.load_rx(rx);
    macos.set_elt_kr(pow_elt, x(1));  macos.set_elt_kc(pow_elt, x(2));
    c = costfn();
end

function c = collimation_cost(pupil_elt)
%COLLIMATION_COST  Mean-squared ray angle vs the chief at PUPIL_ELT (rad^2).
    s = macos.trace(pupil_elt);
    if s.nRays < 10, c = 1e6; return; end
    info = macos.get_ray_info(s.nRays);
    ok = info.ok_trace(:) & info.ok_pass(:);
    D  = info.dir(:, ok);  D = D ./ vecnorm(D);
    dch = info.dir(:,1) / norm(info.dir(:,1));
    ct = max(min(dch.'*D, 1), -1);
    c  = mean(acos(ct).^2);
end

function r = spot_cost(foc_elt)
%SPOT_COST  RMS TRANSVERSE ray spread on the plane at FOC_ELT (mm).
    s = macos.trace(foc_elt);
    if s.nRays < 10, r = 1e6; return; end
    info = macos.get_ray_info(s.nRays);
    ok = info.ok_trace(:) & info.ok_pass(:);
    P  = info.pos(:, ok);
    pch = info.pos(:,1);  dch = info.dir(:,1)/norm(info.dir(:,1));
    d  = P - pch;  dt = d - dch*(dch.'*d);
    r  = sqrt(mean(sum(dt.^2, 1)));
end
