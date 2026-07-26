function run_pol_validation(polvalDir)
%RUN_POL_VALIDATION  Regenerate the polarization validation report evidence.
%   run_pol_validation(POLVALDIR) re-runs every polarization validation case
%   that this box can measure, writes the figures to POLVALDIR/media/*.png
%   and the measured numbers to POLVALDIR/generated/numbers.json, then
%   returns.  render_polval.py substitutes those numbers into the report
%   prose (polval/*.md.in -> polval/*.md), so NO number in the report is
%   ever hand-copied.
%
%   POLVALDIR defaults to <macos repo>/docs/macos-manual/polval, resolved
%   from this file's own location (mmacos and macos are siblings under
%   ~/dev by convention): tools/pol_validation_report -> ../../.. = the
%   MACOS_resources parent -> ../macos.  Pass an explicit path to override.
%
%   Everything here runs at ONE model size (128) in ONE MATLAB session --
%   see mmacos/CLAUDE.md on macos_init_all() heap corruption across
%   model_size transitions.  Do not add a case at a different size without
%   splitting the driver into per-size batch invocations.
%
%   Gates that this driver CANNOT measure (other language, other engine
%   build, or a historical pre-fix binary) are not silently omitted: they
%   live in external.json beside this file, carry the command that produces
%   them and the date they were captured, and the report labels them as
%   externally sourced.  See README.md.
%
%   Usage (normally via make_polval.sh):
%       matlab -batch "run_pol_validation('/path/to/polval')"

    if nargin < 1 || isempty(polvalDir)
        here = fileparts(mfilename('fullpath'));                % .../tools/pol_validation_report
        mmacosRoot = fileparts(fileparts(here));                % .../mmacos
        devRoot = fileparts(fileparts(mmacosRoot));             % .../dev
        polvalDir = fullfile(devRoot, 'macos', 'docs', 'macos-manual', 'polval');
    end
    mediaDir = fullfile(polvalDir, 'media');
    genDir   = fullfile(polvalDir, 'generated');
    if ~exist(mediaDir, 'dir'), mkdir(mediaDir); end
    if ~exist(genDir,   'dir'), mkdir(genDir);   end

    MODEL = 128;
    V = struct();                       % token -> entry (see addval)

    fprintf('polval: model size %d, output %s\n', MODEL, polvalDir);
    macos.init(MODEL);

    V = phase1_gates(V, MODEL);
    V = phase2_gates(V, MODEL, mediaDir);
    V = phase3a_gates(V, MODEL, mediaDir);

    assert_gates(V);
    write_numbers(V, genDir, MODEL);
    fprintf('polval: wrote %s\n', fullfile(genDir, 'numbers.json'));
end

% =====================================================================
%  Gate thresholds -- the report must not be able to document a broken
%  gate.  These mirror the assertions in tPolarization / tJonesPupil /
%  tVecChain: if a number regresses past the value its CI test allows,
%  regeneration FAILS instead of quietly publishing the degraded result
%  next to prose that still calls it round-off.  This is a guard on the
%  report, not a substitute for the test suite -- run that too.
% =====================================================================
function assert_gates(V)
    lim = {
    % token             op     limit    (mirrors)
      'G11_BITWISE',    '==',  1
      'G11_STATUS_DIFF','<=',  0
      'G12_COAT_ROUNDTRIP','<', 1e-12
      'G21_MAXD',       '<',   1e-12
      'G21_MAXRET',     '<',   1e-12
      'G21_LEAK',       '<',   1e-12
      'G21_TNONUNIF',   '<',   1e-12
      'G22_DMAG',       '<',   1e-12
      'G22_DPHASE',     '<',   1e-12
      'G22_DRESID',     '<',   1e-12
      'G23_AZ_RESID',   '<',   1e-10
      'G24_DINV',       '<',   1e-12
      'G24_RATIO',      '>',   10
      'G31_BITWISE',    '==',  1
      'G32_WORST',      '<',   1e-13
      'G33_ELEG1',      '<',   1e-14
      'G33_ELEG2',      '<',   1e-14
      'G34_THRU_RESID', '<',   1e-14
      'G35_TOT_RESID',  '<',   1e-12
      };
    bad = {};
    for i = 1:size(lim, 1)
        tokv = lim{i,1};  op = lim{i,2};  L = lim{i,3};
        if ~isfield(V, tokv)
            bad{end+1} = sprintf('%s: not measured', tokv); %#ok<AGROW>
            continue
        end
        x = V.(tokv).value;
        switch op
            case '==', ok = (x == L);
            case '<',  ok = (x <  L);
            case '<=', ok = (x <= L);
            case '>',  ok = (x >  L);
        end
        if ~ok
            bad{end+1} = sprintf('%s = %g  (needs %s %g)', tokv, x, op, L); %#ok<AGROW>
        end
    end
    if ~isempty(bad)
        error('polval:gate', ['validation gate(s) FAILED -- report not ' ...
            'regenerated:\n  %s\n'], strjoin(bad, sprintf('\n  ')));
    end
    fprintf('polval: %d gate thresholds pass\n', size(lim, 1));
end

% =====================================================================
%  Phase 1 -- exposure gates (state, round-trip, geometry invariance)
% =====================================================================
function V = phase1_gates(V, ~)
    fprintf('polval: Phase 1 gates\n');
    rx = polval_rx('Rx_Cass_FarField.in');
    DET = 6;  FOLD = 3;

    % ---- G1.1 geometry invariance: pol ON must not perturb the geometry.
    % Phase 2 assembles a Jones matrix by pairing rays ACROSS two traces
    % with different input states; that pairing is only meaningful if the
    % geometry is bit-identical between them.
    macos.load_rx(rx);
    macos.polarization('off');
    t0 = macos.trace(DET);
    W0 = macos.opd();  s0 = macos.get_ray_status(t0.nRays);
    macos.polarization('on', 'Ex', [1 0], 'Ey', [0 0]);
    t1 = macos.trace(DET);
    W1 = macos.opd();  s1 = macos.get_ray_status(t1.nRays);
    s0 = double(s0.status);  s1 = double(s1.status);
    dW = max(abs(W1(:) - W0(:)));
    V = addval(V, 'G11_OPD_DIFF', dW, '%.3e', 'waves', ...
        'geometry invariance under ifPol', 'tPolarization/test_pol_on_off_roundtrip (state); this driver (OPD)');
    V = addval(V, 'G11_STATUS_DIFF', max(abs(s1(:) - s0(:))), '%d', 'rays', ...
        'ray-status invariance under ifPol', 'this driver');
    V = addval(V, 'G11_BITWISE', double(isequal(W0, W1)), '%d', 'bool', ...
        'OPD bitwise equality under ifPol', 'this driver');

    % ---- G1.2 coat_get o coat_set == identity (physical thickness in,
    % waves-normalized in storage, inverted on the way out).
    nAl = 1.45; kAl = 7.54; thk = [2.0e-4; 1.1e-4];
    idx = [nAl; 1.38];  ext = [kAl; 0.0];
    macos.coating(FOLD, 'index', idx, 'extinc', ext, 'thickness', thk);
    c = macos.coating(FOLD);
    err = max([abs(c.index(:) - idx); abs(c.extinc(:) - ext); ...
               abs(c.thickness(:) - thk) ./ thk]);
    V = addval(V, 'G12_COAT_ROUNDTRIP', err, '%.3e', 'relative', ...
        'coat_get o coat_set identity', 'tPolarization/test_coat_roundtrip_identity');
    V = addval(V, 'G12_NLAYER', numel(thk), '%d', 'layers', ...
        'round-trip stack depth', 'this driver');
end

% =====================================================================
%  Phase 2a/2b -- Jones pupil + polarization-aberration maps
% =====================================================================
function V = phase2_gates(V, ~, mediaDir)
    rx = polval_rx('Rx_Cass_FarField.in');
    DET = 6;  PRIM = 2;  SEC = 3;
    nAl = 1.45; kAl = 7.54; thkAl = 2.0e-4;

    % ---- G2.1 unitarity gate -----------------------------------------
    % Stock Cass mirrors carry the perfect-conductor idiom (IndRef=1,
    % Extinc=1e22) => RP=RS=-1: the Jones pupil must be unitary times a
    % scalar at every unvignetted point.  One check that catches basis,
    % normalization and sign errors together.
    fprintf('polval: G2.1 unitarity\n');
    macos.load_rx(rx);
    jp = macos.jones_pupil(DET);
    pm = macos.pol_maps(jp);
    m  = jp.mask;
    V = addval(V, 'G21_NPTS',  nnz(m), '%d', 'points', ...
        'unvignetted pupil points', 'tJonesPupil/test_unitarity_gate');
    V = addval(V, 'G21_MAXD',  max(pm.D(m)), '%.2e', '', ...
        'max diattenuation, perfect conductors', 'tJonesPupil/test_unitarity_gate');
    V = addval(V, 'G21_MAXRET', max(pm.ret(m)), '%.2e', 'rad', ...
        'max retardance, perfect conductors', 'tJonesPupil/test_unitarity_gate');
    % Transmission uniformity needs care at this level.  T here spans ~30
    % distinct doubles; the RELATIVE spread is a few times eps.  mean() over
    % ~1e4 terms carries its own summation error of the same order -- in
    % fact LARGER than the spread it is being used to measure -- so
    % std/mean (what the CI gate asserts, and a perfectly valid upper
    % bound) is dominated by that floor rather than by any physics.  Report
    % both, referenced to the MEDIAN (an exactly-selected element, no
    % accumulation), and publish the floor itself so the gap is evidenced
    % rather than asserted.
    Tm = pm.T(m);  Tref = median(Tm);
    V = addval(V, 'G21_TNONUNIF', std(Tm)/mean(Tm), '%.2e', '', ...
        'transmission non-uniformity, std/mean (the CI-gate statistic)', ...
        'tJonesPupil/test_unitarity_gate');
    V = addval(V, 'G21_TSPREAD', (max(Tm)-min(Tm))/Tref, '%.2e', '', ...
        'transmission peak-to-valley about the median', 'this driver');
    V = addval(V, 'G21_TRMS', sqrt(mean((Tm/Tref - 1).^2)), '%.2e', '', ...
        'transmission RMS non-uniformity about the median', 'this driver');
    V = addval(V, 'G21_TMEANFLOOR', abs(mean(Tm)/Tref - 1), '%.2e', '', ...
        'summation-error floor of mean() over the pupil', 'this driver');
    V = addval(V, 'G21_TNUNIQ', numel(unique(Tm)), '%d', 'values', ...
        'distinct doubles in the transmission map', 'this driver');
    V = addval(V, 'G21_LEAK', jp.leak, '%.2e', '', ...
        'longitudinal leak max |E.k|/|E|', 'tJonesPupil/test_unitarity_gate');
    fig_unitarity(pm, fullfile(mediaDir, 'polval_unitarity.png'));

    % ---- G2.2 Fresnel-analytic fold ----------------------------------
    % Bench-emitted 45 deg flat fold + optically thick Al.  The measured
    % per-ray RS/RP ratio is convention-free (frame factors cancel), so it
    % compares to the closed form with nothing to fit.
    fprintf('polval: G2.2 Fresnel fold\n');
    b = macos.design.Bench('foldrig', 'aperture', 0.06, 'ngridpts', 41);
    fold = b.add_fold(50, [0;0;1]);
    b.add_detector(60);
    tmp = [tempname '_dir'];  mkdir(tmp);
    foldRx = fullfile(tmp, 'foldrig.in');
    b.emit(foldRx);
    macos.load_rx(foldRx);
    macos.coating(fold, 'index', nAl, 'extinc', kAl, 'thickness', thkAl);
    macos.polarization('on', 'Ex', [1/sqrt(2) 0], 'Ey', [1/sqrt(2) 0]);
    macos.trace(fold);
    rf = macos.ray_field(fold);
    mm = rf.status == 0;

    [ratio_meas, RSa, RPa, aoi] = fold_fresnel(rf, mm, nAl, kAl);
    dmag = abs(abs(ratio_meas) - abs(RSa./RPa));
    dph  = abs(angle(ratio_meas ./ (RSa./RPa)));
    V = addval(V, 'G22_NRAYS', nnz(mm), '%d', 'rays', ...
        'unvignetted rays on the fold', 'tJonesPupil/test_fold_fresnel_analytic');
    V = addval(V, 'G22_AOI_MIN', min(aoi), '%.2f', 'deg', ...
        'min AOI over the footprint', 'tJonesPupil/test_fold_fresnel_analytic');
    V = addval(V, 'G22_AOI_MAX', max(aoi), '%.2f', 'deg', ...
        'max AOI over the footprint', 'tJonesPupil/test_fold_fresnel_analytic');
    V = addval(V, 'G22_DMAG', max(dmag), '%.2e', '', ...
        'max |RS/RP| residual vs Fresnel', 'tJonesPupil/test_fold_fresnel_analytic');
    V = addval(V, 'G22_DPHASE', max(dph), '%.2e', 'rad', ...
        'max arg(RS/RP) residual vs Fresnel', 'tJonesPupil/test_fold_fresnel_analytic');

    jpf = macos.jones_pupil(fold);
    pmf = macos.pol_maps(jpf);
    Da  = analytic_D(jpf, rf, nAl, kAl);
    V = addval(V, 'G22_DRESID', max(abs(pmf.D(jpf.mask) - Da)), '%.2e', '', ...
        'max per-ray diattenuation residual vs Fresnel', ...
        'tJonesPupil/test_fold_fresnel_analytic');
    V = addval(V, 'G22_DMEAN', mean(Da), '%.4f', '', ...
        'mean diattenuation of the Al fold at 45 deg', 'this driver');
    fig_fresnel(aoi, ratio_meas, RSa, RPa, dmag, dph, ...
        fullfile(mediaDir, 'polval_fresnel_fold.png'));

    % ---- G2.3 2-theta rotational symmetry ----------------------------
    fprintf('polval: G2.3 2-theta symmetry\n');
    macos.load_rx(rx);
    macos.coating(PRIM, 'index', nAl, 'extinc', kAl, 'thickness', thkAl);
    macos.coating(SEC,  'index', nAl, 'extinc', kAl, 'thickness', thkAl);
    pm2 = macos.pol_maps(macos.jones_pupil(DET));
    [resid, D3max, ringD, innerD, TH, R, rmax] = two_theta(pm2);
    V = addval(V, 'G23_AZ_RESID', max(abs(resid)), '%.2e', 'rad', ...
        'azimuth-lock residual of the diattenuation axis', ...
        'tJonesPupil/test_2theta_symmetry');
    V = addval(V, 'G23_CIRC', D3max, '%.2e', '', ...
        'max circular diattenuation component', 'tJonesPupil/test_2theta_symmetry');
    V = addval(V, 'G23_RING_RATIO', ringD/innerD, '%.2f', 'x', ...
        'D(outer ring) / D(inner ring)', 'tJonesPupil/test_2theta_symmetry');
    V = addval(V, 'G23_DMEAN', pm2.mean.D, '%.3e', '', ...
        'pupil-mean diattenuation, Al Cass', 'this driver');
    V = addval(V, 'G23_DVAR', pm2.var_rms.D, '%.3e', '', ...
        'pupil-VARIATION (RMS) diattenuation, Al Cass', 'this driver');
    fig_2theta(pm2, TH, R, rmax, resid, fullfile(mediaDir, 'polval_2theta.png'));

    % ---- G2.4 basis artifact: double-pole vs local-sp -----------------
    fprintf('polval: G2.4 basis artifact\n');
    macos.load_rx(rx);
    macos.coating(SEC, 'index', nAl, 'extinc', kAl, 'thickness', thkAl);
    pdp = macos.pol_maps(macos.jones_pupil(DET, 'basis', 'double-pole'));
    psp = macos.pol_maps(macos.jones_pupil(DET, 'basis', 'local-sp'));
    mb  = pdp.mask & psp.mask;
    V = addval(V, 'G24_DINV', max(abs(pdp.D(mb) - psp.D(mb))), '%.2e', '', ...
        'D basis-invariance residual', 'tJonesPupil/test_basis_invariance_and_sp_artifact');
    V = addval(V, 'G24_RET_DP', pdp.var_rms.ret, '%.3e', 'rad', ...
        'retardance variation, double-pole basis', ...
        'tJonesPupil/test_basis_invariance_and_sp_artifact');
    V = addval(V, 'G24_RET_SP', psp.var_rms.ret, '%.3e', 'rad', ...
        'retardance variation, local-sp basis', ...
        'tJonesPupil/test_basis_invariance_and_sp_artifact');
    V = addval(V, 'G24_RATIO', psp.var_rms.ret/pdp.var_rms.ret, '%.1f', 'x', ...
        'local-sp retardance artifact inflation', ...
        'tJonesPupil/test_basis_invariance_and_sp_artifact');
    fig_basis(pdp, psp, fullfile(mediaDir, 'polval_basis.png'));
end

% =====================================================================
%  Phase 3a Tranche 1 -- vector propagation across the chain
% =====================================================================
function V = phase3a_gates(V, ~, mediaDir)
    rxChain = polval_rx('Rx_VecChain.in');
    rxFF    = polval_rx('Rx_Cass_FarField.in');
    LEG = [2 4];  FFDET = 6;

    fprintf('polval: G3.x vector chain\n');
    states = {'scalar', 'polsc', 'vec_x', 'vec_45', 'vec_circ'};
    I = struct();
    for k = 1:numel(states)
        for L = 1:2
            I.(states{k}){L} = vecchain_case(rxChain, states{k}, LEG(L));
        end
    end

    % ---- G3.1 polarized-scalar reduces EXACTLY to the scalar path -----
    % The fix that made this true also fixed vignetting at the seed:
    % RayE carries no aperture clipping.
    bit = [isequal(I.polsc{1}, I.scalar{1}), isequal(I.polsc{2}, I.scalar{2})];
    V = addval(V, 'G31_BITWISE', double(all(bit)), '%d', 'bool', ...
        'pol-ON/vector-OFF bitwise equal to pol-OFF at both legs', ...
        'tVecChain/test_polarized_scalar_is_bit_identical');

    % ---- G3.2 vector == scalar at every leg, every input state --------
    st = {'vec_x', 'vec_45', 'vec_circ'};
    tok = {'X', 'L45', 'CIRC'};
    worst = 0;
    for k = 1:3
        for L = 1:2
            a = I.(st{k}){L};  s = I.scalar{L};
            r = norm(a(:)/sum(a(:)) - s(:)/sum(s(:))) / norm(s(:)/sum(s(:)));
            worst = max(worst, r);
            V = addval(V, sprintf('G32_%s_LEG%d', tok{k}, L), r, '%.2e', '', ...
                sprintf('vector(%s) vs scalar, leg %d', st{k}, L), ...
                'tVecChain/test_vector_equals_scalar_every_state');
        end
    end
    V = addval(V, 'G32_WORST', worst, '%.2e', '', ...
        'worst vector-vs-scalar residual over all states and legs', ...
        'tVecChain/test_vector_equals_scalar_every_state');

    % ---- G3.3 energy conservation per leg -----------------------------
    for L = 1:2
        a = I.vec_x{L};  s = I.scalar{L};
        V = addval(V, sprintf('G33_ELEG%d', L), ...
            abs(sum(a(:)) - sum(s(:)))/sum(s(:)), '%.2e', 'relative', ...
            sprintf('total-power residual, vector vs scalar, leg %d', L), ...
            'tVecChain/test_energy_conserved_per_leg');
    end

    % ---- G3.4 mask throughput identical on the vector path ------------
    ts = sum(I.scalar{2}(:)) / sum(I.scalar{1}(:));
    V = addval(V, 'G34_THRU_SCALAR', ts, '%.6f', '', ...
        'leg2/leg1 throughput, scalar (MidStop obscuration)', ...
        'tVecChain/test_mask_throughput_identical_on_vector_path');
    worstT = 0;
    for k = 1:3
        tv = sum(I.(st{k}){2}(:)) / sum(I.(st{k}){1}(:));
        worstT = max(worstT, abs(tv - ts)/ts);
    end
    V = addval(V, 'G34_THRU_RESID', worstT, '%.2e', 'relative', ...
        'worst mask-throughput mismatch on the vector path', ...
        'tVecChain/test_mask_throughput_identical_on_vector_path');
    fig_vecchain(I, LEG, fullfile(mediaDir, 'polval_vecchain_legs.png'));

    % ---- G3.5 far-field normalization (PFFPROP -> FFPROP x3) ----------
    fprintf('polval: G3.5 far-field A/B\n');
    Isf = vecchain_case(rxFF, 'scalar', FFDET);
    Ivf = vecchain_case(rxFF, 'vec_x',  FFDET);
    V = addval(V, 'G35_SCALAR_TOT', sum(Isf(:)), '%.6e', '', ...
        'scalar far-field total power', ...
        'tVecChain/test_far_field_vector_matches_scalar_normalization');
    V = addval(V, 'G35_VECTOR_TOT', sum(Ivf(:)), '%.6e', '', ...
        'vector far-field total power (post-fix)', ...
        'tVecChain/test_far_field_vector_matches_scalar_normalization');
    V = addval(V, 'G35_TOT_RESID', abs(sum(Ivf(:))-sum(Isf(:)))/sum(Isf(:)), ...
        '%.2e', 'relative', 'vector-vs-scalar total-power residual', ...
        'tVecChain/test_far_field_vector_matches_scalar_normalization');
    V = addval(V, 'G35_MAP_DIFF', ...
        norm(Ivf(:)-Isf(:))/norm(Isf(:)), '%.2e', '', ...
        'vector-vs-scalar far-field map difference (off-normal train)', ...
        'tVecChain/test_far_field_vector_matches_scalar_normalization');

    % |Ez|/|Ex| at the exit pupil -- the SUSPECTED source of G35_MAP_DIFF.
    % Measured here so the report can show the order of magnitude while
    % stating plainly that the attribution is NOT verified: there is no
    % plane-selectable complex-field getter, so the per-plane contribution
    % to the propagated intensity cannot be isolated.  Gate the probe on
    % status==0 -- obscured rays carry RayE=0 and read as a spurious zero.
    macos.load_rx(rxFF);
    macos.polarization('on', 'Ex', [1 0], 'Ey', [0 0]);
    macos.trace(5);
    rfe = macos.ray_field(5);
    ok = rfe.status == 0;
    ez = abs(rfe.Ez(ok)) ./ abs(rfe.Ex(ok));
    V = addval(V, 'G35_EZ_RATIO', median(ez), '%.2e', '', ...
        'median |Ez|/|Ex| at the exit pupil (UNVERIFIED attribution)', ...
        'this driver');
    V = addval(V, 'G35_EZ_MAX', max(ez), '%.2e', '', ...
        'max |Ez|/|Ex| at the exit pupil (UNVERIFIED attribution)', ...
        'this driver');
    V = addval(V, 'G35_EZ_NRAYS', nnz(ok), '%d', 'rays', ...
        'unvignetted rays in the |Ez|/|Ex| probe', 'this driver');
    fig_farfield(Isf, Ivf, fullfile(mediaDir, 'polval_farfield_ab.png'));
end

% =====================================================================
%  case runners
% =====================================================================
function I = vecchain_case(rx, mode, elt)
    macos.load_rx(rx);
    switch mode
        case 'scalar',   macos.polarization('off');
        case 'polsc'
            macos.polarization('on', 'Ex', [1 0], 'Ey', [0 0]);
            macos.vector_diffraction(false);
        case 'vec_x'
            macos.polarization('on', 'Ex', [1 0], 'Ey', [0 0]);
            macos.vector_diffraction(true);
        case 'vec_45'
            macos.polarization('on', 'Ex', [1 0], 'Ey', [1 0]);
            macos.vector_diffraction(true);
        case 'vec_circ'
            macos.polarization('on', 'Ex', [1 0], 'Ey', [0 1]);
            macos.vector_diffraction(true);
        otherwise, error('polval:mode', 'unknown mode %s', mode);
    end
    I = macos.intensity(elt);
end

function [ratio_meas, RSa, RPa, aoi] = fold_fresnel(rf, m, nAl, kAl)
% Per-ray measured RS/RP (convention-free ratio form) and the closed form.
% Frames follow the engine exactly: s = ki x n, p_i = s x ki, p_r = s x ko;
% the launch frame is the ssrcray point-source pair yray = unit(k x xGrid).
    kox=rf.kx(m); koy=rf.ky(m); koz=rf.kz(m);
    nx=rf.nx(m);  ny=rf.ny(m);  nz=rf.nz(m);
    kd = kox.*nx + koy.*ny + koz.*nz;
    kix = kox-2*kd.*nx;  kiy = koy-2*kd.*ny;  kiz = koz-2*kd.*nz;
    sx=kiy.*nz-kiz.*ny; sy=kiz.*nx-kix.*nz; sz=kix.*ny-kiy.*nx;
    sm=sqrt(sx.^2+sy.^2+sz.^2); sx=sx./sm; sy=sy./sm; sz=sz./sm;
    pix=sy.*kiz-sz.*kiy; piy=sz.*kix-sx.*kiz; piz=sx.*kiy-sy.*kix;
    prx=sy.*koz-sz.*koy; pry=sz.*kox-sx.*koz; prz=sx.*koy-sy.*kox;
    xg = [0;1;0];
    yrx=kiy*xg(3)-kiz*xg(2); yry=kiz*xg(1)-kix*xg(3); yrz=kix*xg(2)-kiy*xg(1);
    ym=sqrt(yrx.^2+yry.^2+yrz.^2); yrx=yrx./ym; yry=yry./ym; yrz=yrz./ym;
    xrx=yry.*kiz-yrz.*kiy; xry=yrz.*kix-yrx.*kiz; xrz=yrx.*kiy-yry.*kix;
    einx=(xrx+yrx)/sqrt(2); einy=(xry+yry)/sqrt(2); einz=(xrz+yrz)/sqrt(2);
    Es = rf.Ex(m).*sx  + rf.Ey(m).*sy  + rf.Ez(m).*sz;
    Ep = rf.Ex(m).*prx + rf.Ey(m).*pry + rf.Ez(m).*prz;
    qs = einx.*sx  + einy.*sy  + einz.*sz;
    qp = einx.*pix + einy.*piy + einz.*piz;
    ratio_meas = (Es./Ep).*(qp./qs);
    N1 = 1.0;  N2 = complex(nAl, -kAl);
    cthi = abs(kix.*nx + kiy.*ny + kiz.*nz);
    ctht = sqrt(1 - (N1/N2)^2*(1 - cthi.^2));
    RPa = (N1*ctht - N2*cthi)./(N2*cthi + N1*ctht);
    RSa = (N1*cthi - N2*ctht)./(N1*cthi + N2*ctht);
    aoi = acosd(cthi);
end

function Da = analytic_D(jp, rf, nAl, kAl)
    ko=[jp.kx(jp.mask), jp.ky(jp.mask), jp.kz(jp.mask)];
    n =[rf.nx(jp.mask), rf.ny(jp.mask), rf.nz(jp.mask)];
    ki = ko - 2*sum(ko.*n,2).*n;
    c  = abs(sum(ki.*n,2));
    N1 = 1.0;  N2 = complex(nAl, -kAl);
    ct = sqrt(1 - (N1/N2)^2*(1 - c.^2));
    RP = (N1*ct - N2*c)./(N2*c + N1*ct);
    RS = (N1*c - N2*ct)./(N1*c + N2*ct);
    Da = abs(abs(RS).^2 - abs(RP).^2)./(abs(RS).^2 + abs(RP).^2);
end

function [resid, D3max, ringD, innerD, TH, R, rmax] = two_theta(pm)
    m = pm.mask;
    [ii, jj] = find(m);
    N = size(m,1);
    [JJ, II] = meshgrid(1:N, 1:N);
    R  = sqrt((II-mean(ii)).^2 + (JJ-mean(jj)).^2);
    TH = atan2(JJ-mean(jj), II-mean(ii));
    rmax = max(R(m));
    ring  = m & (R > 0.60*rmax) & (R < 0.75*rmax);
    inner = m & (R > 0.25*rmax) & (R < 0.40*rmax);
    D1 = pm.Dvec(:,:,1); D2 = pm.Dvec(:,:,2); D3 = pm.Dvec(:,:,3);
    ang = 0.5*atan2(D2(ring), D1(ring));
    off = 0.5*angle(mean(exp(2i*(ang - TH(ring)))));
    resid = mod(ang - TH(ring) - off + pi/2, pi) - pi/2;
    D3max = max(abs(D3(m)));
    ringD = mean(pm.D(ring));  innerD = mean(pm.D(inner));
end

% =====================================================================
%  figures  (autoscaled panels, NaN off-mask, non-obscuring legends)
% =====================================================================
function fig_unitarity(pm, out)
    f = newfig([1100 420]);
    tiledlayout(f, 1, 3, 'Padding', 'compact', 'TileSpacing', 'compact');
    panel_map(pm.D,   pm.mask, 'diattenuation D');
    panel_map(pm.ret, pm.mask, 'retardance \delta  [rad]');
    nexttile;
    % Referenced to the MEDIAN, not the mean: mean() over ~1e4 points
    % accumulates more round-off than the map's entire spread, which would
    % paint a spurious uniform offset across the whole pupil.
    Tn = pm.T ./ median(pm.T(pm.mask)) - 1;
    panel_here(Tn, pm.mask, 'transmission T/median(T) - 1');
    sgtitle(f, ['Unitarity gate: perfect-conductor Cassegrain ' ...
                '(RP=RS=-1) -- every map is round-off'], ...
            'FontWeight', 'bold');
    savefig_(f, out);
end

function fig_fresnel(aoi, ratio, RSa, RPa, dmag, dph, out)
    f = newfig([1100 420]);
    tiledlayout(f, 1, 3, 'Padding', 'compact', 'TileSpacing', 'compact');
    ana = RSa ./ RPa;
    nexttile; hold on; grid on;
    plot(aoi, abs(ratio), '.', 'MarkerSize', 6, 'DisplayName', 'measured');
    plot(aoi, abs(ana), 'o', 'MarkerSize', 3, 'LineWidth', 0.6, ...
         'DisplayName', 'Fresnel closed form');
    xlabel('AOI  [deg]'); ylabel('|R_s / R_p|');
    title('magnitude'); legend('Location', 'best'); legend boxoff;
    nexttile; hold on; grid on;
    plot(aoi, angle(ratio), '.', 'MarkerSize', 6, 'DisplayName', 'measured');
    plot(aoi, angle(ana), 'o', 'MarkerSize', 3, 'LineWidth', 0.6, ...
         'DisplayName', 'Fresnel closed form');
    xlabel('AOI  [deg]'); ylabel('arg(R_s / R_p)  [rad]');
    title('phase'); legend('Location', 'best'); legend boxoff;
    nexttile; hold on; grid on;
    semilogy(aoi, max(dmag, realmin), '.', 'MarkerSize', 6, ...
             'DisplayName', 'magnitude');
    semilogy(aoi, max(dph, realmin), '.', 'MarkerSize', 6, ...
             'DisplayName', 'phase  [rad]');
    set(gca, 'YScale', 'log');
    xlabel('AOI  [deg]'); ylabel('|measured - analytic|');
    title('residual'); legend('Location', 'best'); legend boxoff;
    sgtitle(f, ['Fresnel-analytic gate: 45 deg flat fold, optically thick Al ' ...
                '-- convention-free R_s/R_p ratio'], 'FontWeight', 'bold');
    savefig_(f, out);
end

function fig_2theta(pm, TH, R, rmax, resid, out)
    f = newfig([1100 420]);
    tiledlayout(f, 1, 3, 'Padding', 'compact', 'TileSpacing', 'compact');
    panel_map(pm.D, pm.mask, 'diattenuation D (grows with AOI off-axis)');
    % orientation quiver: the diattenuation axis is a DIRECTOR (mod pi),
    % so draw it as a headless double-ended segment.
    nexttile; hold on;
    m = pm.mask;
    N = size(m,1);
    step = max(1, round(N/22));
    [JJ, II] = meshgrid(1:N, 1:N);
    sel = false(N); sel(1:step:end, 1:step:end) = true;
    sel = sel & m & (R < 0.95*rmax);
    ang = 0.5*atan2(pm.Dvec(:,:,2), pm.Dvec(:,:,1));
    L = 0.42*step;
    u = L*cos(ang(sel));  v = L*sin(ang(sel));
    quiver(JJ(sel), II(sel),  u,  v, 0, 'k', 'ShowArrowHead', 'off');
    quiver(JJ(sel), II(sel), -u, -v, 0, 'k', 'ShowArrowHead', 'off');
    axis image ij; xlim([1 N]); ylim([1 N]); box on;
    set(gca, 'XTick', [], 'YTick', []);
    title('diattenuation axis (director, mod \pi)');
    nexttile;
    histogram(resid, 40);
    xlabel('azimuth-lock residual  [rad]'); ylabel('rays'); grid on;
    title('0.5\cdotatan2(D_2,D_1) - \theta_{pupil}');
    sgtitle(f, ['2\theta rotational-symmetry invariant: Al on both Cassegrain ' ...
                'mirrors, on-axis'], 'FontWeight', 'bold');
    savefig_(f, out);
end

function fig_basis(pdp, psp, out)
    f = newfig([1100 420]);
    tiledlayout(f, 1, 3, 'Padding', 'compact', 'TileSpacing', 'compact');
    mb = pdp.mask & psp.mask;
    panel_map(pdp.ret, mb, 'retardance, double-pole basis  [rad]');
    panel_map(psp.ret, mb, 'retardance, local-s/p basis  [rad]');
    nexttile;
    panel_here(pdp.D - psp.D, mb, 'D(double-pole) - D(local-s/p)');
    sgtitle(f, ['Basis artifact: the s/p coordinate singularity imprints ' ...
                'retardance the double-pole basis does not; D is invariant'], ...
            'FontWeight', 'bold');
    savefig_(f, out);
end

function fig_vecchain(I, LEG, out)
    f = newfig([1100 700]);
    tiledlayout(f, 2, 3, 'Padding', 'compact', 'TileSpacing', 'compact');
    for L = 1:2
        s = I.scalar{L};  v = I.vec_circ{L};
        sn = s/sum(s(:));  vn = v/sum(v(:));
        nexttile; panel_here(sn, true(size(sn)), ...
            sprintf('scalar, leg %d (elt %d)', L, LEG(L)));
        nexttile; panel_here(vn, true(size(vn)), ...
            sprintf('vector (circular), leg %d', L));
        nexttile; panel_here(abs(vn-sn), true(size(sn)), ...
            sprintf('|difference|, leg %d', L));
    end
    sgtitle(f, ['Chain closure: two bracketed near-field legs through a ' ...
                'central obscuration -- vector reproduces scalar at round-off'], ...
            'FontWeight', 'bold');
    savefig_(f, out);
end

function fig_farfield(Is, Iv, out)
    f = newfig([1100 420]);
    tiledlayout(f, 1, 3, 'Padding', 'compact', 'TileSpacing', 'compact');
    lg = @(A) log10(max(A/max(A(:)), 1e-12));
    nexttile; panel_here(lg(Is), true(size(Is)), 'scalar  log_{10}(I/I_{max})');
    nexttile; panel_here(lg(Iv), true(size(Iv)), 'vector  log_{10}(I/I_{max})');
    % log scale here too: the difference is concentrated near the core, and
    % on a linear ruler the whole field outside a couple of pixels reads as
    % a flat null -- which is precisely the structure worth showing.
    D = log10(max(abs(Iv-Is)/max(Is(:)), 1e-12));   % normalized to I_max
    nexttile; panel_here(D, true(size(Is)), ...
        'log_{10}(|vector - scalar| / I_{max})');
    sgtitle(f, ['Far-field leg after PFFPROP -> FFPROP\times3: identical total ' ...
                'power; the residual map is an off-normal train'], ...
            'FontWeight', 'bold');
    savefig_(f, out);
end

function f = newfig(sz)
    f = figure('Visible', 'off', 'Color', 'w', ...
               'Position', [100 100 sz(1) sz(2)]);
end

function panel_map(A, mask, ttl)
    nexttile; panel_here(A, mask, ttl);
end

function panel_here(A, mask, ttl)
    B = A; B(~mask) = NaN;
    h = imagesc(B); set(h, 'AlphaData', ~isnan(B));
    axis image ij off; colormap(gca, parula); colorbar; title(ttl);
end

function savefig_(f, out)
    exportgraphics(f, out, 'Resolution', 150);
    close(f);
    fprintf('polval: wrote %s\n', out);
end

% =====================================================================
%  bookkeeping
% =====================================================================
function V = addval(V, name, value, fmt, units, gate, test)
    e.value = value;
    if isnumeric(value) && isscalar(value)
        e.text = sprintf(fmt, value);
    else
        e.text = char(string(value));
    end
    e.units  = units;
    e.gate   = gate;
    e.test   = test;
    e.source = 'driver';
    V.(name) = e;
end

function write_numbers(V, genDir, model)
    p = struct();
    p.generated     = char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss'));
    [p.engine_sha, p.engine_branch, p.engine_dirty] = ...
        gitinfo(fullfile(getenv('HOME'), 'dev', 'macos'));
    [p.resources_sha, p.resources_branch, p.resources_dirty] = ...
        gitinfo(fullfile(getenv('HOME'), 'dev', 'MACOS_resources'));
    p.matlab     = version('-release');
    p.model_size = model;
    [~, h] = system('hostname'); p.host = strtrim(h);
    out = struct('provenance', p, 'values', V);
    fid = fopen(fullfile(genDir, 'numbers.json'), 'w');
    fprintf(fid, '%s\n', jsonencode(out, 'PrettyPrint', true));
    fclose(fid);
end

function [sha, br, dirty] = gitinfo(repo)
    [a, s] = system(sprintf('git -C %s rev-parse --short HEAD 2>/dev/null', repo));
    [b, r] = system(sprintf('git -C %s rev-parse --abbrev-ref HEAD 2>/dev/null', repo));
    [c, d] = system(sprintf('git -C %s status --porcelain --untracked-files=no 2>/dev/null', repo));
    sha = 'unknown'; br = 'unknown'; dirty = false;
    if a == 0, sha = strtrim(s); end
    if b == 0, br  = strtrim(r); end
    if c == 0, dirty = ~isempty(strtrim(d)); end
end

function p = polval_rx(name)
%POLVAL_RX  Resolve an Rx fixture by name (mirrors tests/private/rx_fixture_path).
%   The tests' resolver lives in a private/ folder and is not reachable
%   from tools/, so the same two-root search is repeated here.
    roots = { fullfile(getenv('HOME'), 'dev', 'MACOS_resources', 'pymacos', 'tests', 'Rx'), ...
              fullfile(getenv('HOME'), 'dev', 'MACOS_resources', 'mmacos', 'tests', 'Rx') };
    for i = 1:numel(roots)
        p = fullfile(roots{i}, name);
        if exist(p, 'file'), return; end
    end
    error('polval:rx', 'Rx fixture not found: %s', name);
end
