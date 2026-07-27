function run_pol_validation(polvalDir, model)
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
%   ONE model size per MATLAB session -- see mmacos/CLAUDE.md on
%   macos_init_all() heap corruption across model_size transitions.  The
%   driver is therefore invoked once per size and each run writes a PART
%   file, generated/parts/numbers_<model>.json; merge_numbers.py combines
%   the parts into generated/numbers.json.  make_polval.sh does all of it.
%
%       model 128   Phase 1 exposure, Phase 2a/2b Jones pupil, Phase 3a
%                   Tranche 1, the r_p sign fix
%       model 256   Phase 2c exactness gates (Rx_VecChain, Rx_Cass_FarField
%                   -- both declare nGridpts=256)
%       model 512   Phase 2c on the coronagraph chain (Rx_Coro declares
%                   nGridpts=511, so it MUST run at >= 512)
%
%   A new case at a new size needs a new branch in the switch below, its
%   own gate-limit block in gate_limits(), and a line in make_polval.sh.
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
    if nargin < 2 || isempty(model), model = 128; end
    mediaDir = fullfile(polvalDir, 'media');
    genDir   = fullfile(polvalDir, 'generated');
    partDir  = fullfile(genDir, 'parts');
    if ~exist(mediaDir, 'dir'), mkdir(mediaDir); end
    if ~exist(genDir,   'dir'), mkdir(genDir);   end
    if ~exist(partDir,  'dir'), mkdir(partDir);  end

    MODEL = model;
    V = struct();                       % token -> entry (see addval)

    fprintf('polval: model size %d, output %s\n', MODEL, polvalDir);

    % Capture git provenance BEFORE writing anything.  The driver's own
    % outputs (media/*.png, generated/numbers.json) live in the macos repo,
    % so sampling git state at the END always reports that tree as dirty --
    % the stamp would describe the tree the run CREATED rather than the one
    % it measured, which is the opposite of the point.
    prov0 = capture_provenance(MODEL);

    macos.init(MODEL);

    switch MODEL
        case 128
            V = phase1_gates(V, MODEL);
            V = phase2_gates(V, MODEL, mediaDir);
            V = phase3a_gates(V, MODEL, mediaDir);
            V = spsign_gates(V, MODEL, mediaDir);
        case 256
            V = phase2c_exact_gates(V, MODEL, mediaDir);
        case 512
            V = phase2c_coro_gates(V, MODEL, mediaDir);
        otherwise
            error('polval:model', 'no gate group defined for model size %d', MODEL);
    end

    assert_gates(V, MODEL);
    part = fullfile(partDir, sprintf('numbers_%d.json', MODEL));
    write_numbers(V, part, prov0);
    fprintf('polval: wrote %s\n', part);
end

% =====================================================================
%  Gate thresholds -- the report must not be able to document a broken
%  gate.  These mirror the assertions in tPolarization / tJonesPupil /
%  tVecChain: if a number regresses past the value its CI test allows,
%  regeneration FAILS instead of quietly publishing the degraded result
%  next to prose that still calls it round-off.  This is a guard on the
%  report, not a substitute for the test suite -- run that too.
% =====================================================================
function lim = gate_limits(model)
%GATE_LIMITS  Per-model-size gate thresholds.  A token listed here
%   MUST have been measured by that size's gate group -- a missing
%   measurement fails the run exactly like a regressed one.
    switch model
      case 128
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
        'G25_OTHER',      '<',   1e-10
        'G25_CIRC',       '<',   1e-10
        'G25_RHO4',       '<',   1e-2
        'G25_PAIR_RESID', '<',   1e-6
        'G25_RADSYM',     '<',   1e-6
        'G25_DZERO',      '<',   1e-3
        'G31_BITWISE',    '==',  1
        'G32_WORST',      '<',   1e-13
        'G33_ELEG1',      '<',   1e-14
        'G33_ELEG2',      '<',   1e-14
        'G34_THRU_RESID', '<',   1e-14
        'G35_TOT_RESID',  '<',   1e-12
        'G36_PLANESUM',   '<',   1e-14
        'G36_RESID',      '<',   1e-3
        'G36_DECORR',     '<',   1e-4
        'G41_RESID_MAX',  '<',   1e-11
        'G41_RET',        '<',   1e-14
        'G42_BOUNDRATIO', '<',   1.05
        'G42_PYPX1',      '<',   1e-3
        'G42_SLOPE',      '>',   1.7
        };
      case 256
        lim = {
        % token                op     limit
          'C21_VC_CROSS',      '<=',  0
          'C21_VC_SCALAR',     '<',   1e-13
          'C21_VC_CURVE',      '<',   1e-12
          'C22_PARSEVAL',      '<',   1e-18
          'C22_CLOSURE',       '<',   1e-14
          'C23_ANALYZER',      '<',   1e-12
          'C23_CIRC_ORTHO',    '>',   1e5
          'C24_CARRIED',       '<',   1.001
          'C24_CARRIED_LO',    '>',   0.999
          'C25_SWEEP_MONO',    '==',  1
          };
      case 512
        lim = {
        % token                op     limit
          'C31_PARSEVAL',      '<',   1e-15
          'C31_CLOSURE',       '<',   1e-14
          'C32_DOP',           '>',   0.99999
          'C33_CARRIED',       '<',   0.95
          'C33_CARRIED_COAT',  '<',   0.7
          };
      otherwise
        error('polval:gate', 'no gate table for model %d', model);
    end
end

function assert_gates(V, model)
    lim = gate_limits(model);
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
    fprintf('polval: %d gate thresholds pass (model %d)\n', size(lim, 1), model);
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
    % Al at 632.8 nm.  200 nm of it -- but macos.coating takes thickness in
    % ELEMENT BaseUnits, and the two fixtures here differ: Rx_Cass_FarField
    % is BaseUnits=m, the Bench-emitted fold rig is BaseUnits=mm.  Same
    % constant in both would silently mean 200 um on one of them (which is
    % still optically thick, so the gates would pass -- it just would not
    % be the thickness the text claims, and it made the mmacos and pymacos
    % coefficients differ in the 8th digit).  Mirrors tJonesPupil.
    nAl = 1.45; kAl = 7.54;
    thkAl = 2.0e-7;        % Rx_Cass_FarField (BaseUnits = m)
    thkAlBench = 2.0e-4;   % Bench fold rig   (BaseUnits = mm)

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
    macos.coating(fold, 'index', nAl, 'extinc', kAl, 'thickness', thkAlBench);
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

    % ---- G2.5 low-order expansion: the two-mirror literature form -----
    % Phase 2b.  Standard polarization-aberration theory for an on-axis
    % rotationally symmetric two-mirror system predicts diattenuation and
    % retardance growing as rho^2 with a 2*theta azimuth, i.e. in the
    % Pauli representation pure ASTIGMATISM -- astig0 in s1, astig45 in
    % s2, equal magnitude, no circular part, no defocus.  pm2 above is
    % exactly that system, so reuse it.
    fprintf('polval: G2.5 low-order expansion\n');
    pz = macos.pol_zernike(pm2);
    mo = pz.modes;
    iA0 = find(mo==6); iA45 = find(mo==4); i2A0 = find(mo==14);
    a0 = abs(pz.D(iA0,1));  a45 = abs(pz.D(iA45,2));
    r0 = abs(pz.ret(iA0,1));
    keep = true(numel(mo),1); keep([iA0 iA45 i2A0 find(mo==12)]) = false;
    V = addval(V, 'G25_D_ASTIG0', pz.D(iA0,1), '%.4e', '', ...
        'diattenuation astig0 coefficient (Pauli s1)', ...
        'tJonesPupil/test_pol_zernike_two_mirror_form');
    V = addval(V, 'G25_D_ASTIG45', pz.D(iA45,2), '%.4e', '', ...
        'diattenuation astig45 coefficient (Pauli s2)', ...
        'tJonesPupil/test_pol_zernike_two_mirror_form');
    V = addval(V, 'G25_PAIR_RESID', abs(a0-a45)/a0, '%.2e', 'relative', ...
        'astig0/astig45 magnitude mismatch (pupil discretization)', ...
        'tJonesPupil/test_pol_zernike_two_mirror_form');
    V = addval(V, 'G25_RET_ASTIG0', pz.ret(iA0,1), '%.4e', 'rad', ...
        'retardance astig0 coefficient (Pauli s1)', ...
        'tJonesPupil/test_pol_zernike_two_mirror_form');
    V = addval(V, 'G25_OTHER', max(max(abs(pz.D(keep,1:2))))/a0, '%.2e', '', ...
        'largest non-astigmatic linear coefficient, relative', ...
        'tJonesPupil/test_pol_zernike_two_mirror_form');
    V = addval(V, 'G25_CIRC', max(abs(pz.D(:,3)))/a0, '%.2e', '', ...
        'largest circular (s3) coefficient, relative', ...
        'tJonesPupil/test_pol_zernike_two_mirror_form');
    V = addval(V, 'G25_RHO4', abs(pz.D(i2A0,1))/a0, '%.2e', '', ...
        'rho^4 astigmatism companion, relative to the primary term', ...
        'tJonesPupil/test_pol_zernike_two_mirror_form');
    % radial law: |D| expands to piston + defocus only, and its on-axis
    % extrapolation vanishes -- the fit is never told to arrange that.
    cm = pz.Dmag;
    kk = true(numel(mo),1); kk([find(mo==1) find(mo==5) find(mo==13)]) = false;
    V = addval(V, 'G25_RADSYM', max(abs(cm(kk)))/abs(cm(mo==1)), '%.2e', '', ...
        'largest non-rotationally-symmetric term in |D|, relative', ...
        'tJonesPupil/test_pol_zernike_two_mirror_form');
    D0 = zern_sum_(mo, cm, 0, 0);  D1 = zern_sum_(mo, cm, 1, 0);
    V = addval(V, 'G25_DZERO', abs(D0)/abs(D1), '%.2e', 'relative', ...
        'extrapolated on-axis diattenuation (physics requires 0)', ...
        'tJonesPupil/test_pol_zernike_two_mirror_form');
    % Report the UNEXPLAINED fraction: "explained = 1.000000" tells the
    % reader nothing and reads as an exact claim it is not.
    V = addval(V, 'G25_FIT_MISS', 1 - min(pz.frac.D(1:2)), '%.2e', '', ...
        'fraction of the linear Dvec maps NOT captured by modes 1-15', ...
        'this driver');
    V = addval(V, 'G25_RESID_REL', max(pz.resid_rms.D(1:2))/abs(pz.D(iA0,1)), ...
        '%.2e', '', 'fit residual RMS relative to the astigmatic term', ...
        'this driver');
    V = addval(V, 'G25_COND', pz.cond, '%.3f', '', ...
        'condition number of the fit over this annular pupil', 'this driver');
    fig_zernike(pz, pm2, fullfile(mediaDir, 'polval_zernike.png'));

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
        'median |Ez|/|Ex| at the exit pupil', ...
        'this driver');
    V = addval(V, 'G35_EZ_MAX', max(ez), '%.2e', '', ...
        'max |Ez|/|Ex| at the exit pupil', ...
        'this driver');
    V = addval(V, 'G35_EZ_NRAYS', nnz(ok), '%d', 'rays', ...
        'unvignetted rays in the |Ez|/|Ex| probe', 'this driver');
    fig_farfield(Isf, Ivf, fullfile(mediaDir, 'polval_farfield_ab.png'));

    % ---- G3.6 the attribution, DECOMPOSED (was unverifiable) ----------
    % Tranche 1 could only guess why the vector run differs from the
    % scalar one on an off-normal train.  With the plane-selectable
    % complex-field getter the per-plane contributions are measurable, and
    % the guess turns out to be HALF right: two mechanisms, not one.
    fprintf('polval: G3.6 attribution decomposition\n');
    macos.load_rx(rxFF);
    macos.polarization('on', 'Ex', [1 0], 'Ey', [0 0]);
    macos.vector_diffraction(true);
    Ivd = macos.intensity(FFDET);
    Ipx = abs(macos.complex_field(FFDET, 'plane', 1)).^2;
    Ipy = abs(macos.complex_field(FFDET, 'plane', 2)).^2;
    Ipz = abs(macos.complex_field(FFDET, 'plane', 3)).^2;
    fpow = sum(Ipx(:)) / sum(Isf(:));
    rel = @(A,B) norm(A(:)-B(:))/norm(B(:));
    raw = rel(Ivd, Isf);
    resid = rel(fpow*Isf + Ipy + Ipz, Ivd);
    cc = corrcoef(Ipx(:), Isf(:));
    V = addval(V, 'G36_F', fpow, '%.6f', '', 'in-plane (Ex) power fraction', ...
        'tVecChain/test_vector_scalar_difference_decomposition');
    V = addval(V, 'G36_OUTFRAC', 1-fpow, '%.4e', '', ...
        'out-of-plane (Ey+Ez) power fraction', ...
        'tVecChain/test_vector_scalar_difference_decomposition');
    V = addval(V, 'G36_RAW', raw, '%.4e', '', ...
        'vector-vs-scalar map difference being explained', ...
        'tVecChain/test_vector_scalar_difference_decomposition');
    V = addval(V, 'G36_RESID', resid, '%.4e', '', ...
        'residual of the two-term decomposition f*Is + Iy + Iz', ...
        'tVecChain/test_vector_scalar_difference_decomposition');
    % Report 1-corr: "corr = 1.000000" is uninformative and reads as an
    % exact claim (same trap as the fit-fraction number in G2.5).
    V = addval(V, 'G36_DECORR', 1-cc(1,2), '%.2e', '', ...
        '1 - corr(Ex, scalar map): mechanism 1 is a near-pure rescale', ...
        'tVecChain/test_vector_scalar_difference_decomposition');
    V = addval(V, 'G36_PLANESUM', rel(Ipx+Ipy+Ipz, Ivd), '%.2e', '', ...
        'component planes summed vs intensity()', ...
        'tVecChain/test_component_planes_sum_to_intensity');
end

% =====================================================================
%  case runners
% =====================================================================
% =====================================================================
%  Section 4 -- the reflected-p-hat / Fresnel r_p sign correction
%  (macos cb29ea5).  Everything else in this report is measured on
%  EVEN-mirror trains, where the pre-fix defect cancelled exactly.  These
%  gates read the field after exactly ONE reflection, which is the only
%  place it was ever visible.
% =====================================================================
function V = spsign_gates(V, ~, mediaDir)
    rx = polval_rx('Rx_Cass_FarField.in');
    STOP = 1;  PRIM = 2;  SEC = 3;     % Obscuring stop, primary, secondary

    fprintf('polval: G4.x odd-mirror s/p sign\n');
    macos.load_rx(rx);
    macos.polarization('on', 'Ex', [1 0], 'Ey', [0 0]);
    macos.trace(STOP);  r0 = macos.ray_field(STOP);
    macos.trace(PRIM);  r1 = macos.ray_field(PRIM);
    macos.trace(SEC);   r2 = macos.ray_field(SEC);
    ok  = (r0.status == 0) & (r1.status == 0);
    ok2 = r2.status == 0;

    % Geometry from the RAY DIRECTIONS only: AOI from the stop-to-mirror
    % deflection (pi - 2*AOI for a mirror), azimuth from the outgoing
    % transverse direction.  No pixel-grid-to-pupil mapping is assumed.
    kdot = r0.kx.*r1.kx + r0.ky.*r1.ky + r0.kz.*r1.kz;
    aoi  = (pi - acos(min(max(kdot, -1), 1))) / 2;
    phi  = atan2(r1.ky, r1.kx);
    rEy  = r1.Ey ./ r1.Ex;
    rEz  = r1.Ez ./ r1.Ex;

    % ---- G4.1 the whole single-reflection Jones, closed form ----------
    % Perfect conductor (r_s = -1, r_p = +1 in the engine's ray-following
    % p-hat basis) plus geometry gives, EXACTLY in the AOI:
    %   Ey/Ex = -sin(2 phi) sin^2(a) / den,  Ez/Ex = -sin(2a) cos(phi)/den
    %   den   = 1 - 2 sin^2(a) cos^2(phi)
    % written from Born & Wolf, NOT transcribed from the engine -- the
    % circularity that let the old fold gate pass either way.
    den   = 1 - 2*sin(aoi).^2 .* cos(phi).^2;
    predY = -sin(aoi).^2 .* sin(2*phi) ./ den;
    predZ =  sin(2*aoi)  .* cos(phi)   ./ den;
    sy = ok & abs(sin(2*phi)) > 0.2 & aoi > deg2rad(1);   % away from own zeros
    sz = ok & abs(cos(phi))   > 0.2 & aoi > deg2rad(1);
    ry = abs((rEy(sy) - predY(sy)) ./ predY(sy));
    rz = abs((rEz(sz) - predZ(sz)) ./ predZ(sz));

    V = addval(V, 'G41_NRAYS', nnz(sy), '%d', 'rays', ...
        'rays entering the closed-form comparison', ...
        'tPolarization/test_odd_mirror_crosspol_pec_analytic');
    V = addval(V, 'G41_AOIMAX', rad2deg(max(aoi(ok))), '%.2f', 'deg', ...
        'largest angle of incidence on the primary', 'this driver');
    V = addval(V, 'G41_RESID_MED', median(ry), '%.2e', '', ...
        'median relative residual, cross-pol vs closed form', ...
        'tPolarization/test_odd_mirror_crosspol_pec_analytic');
    V = addval(V, 'G41_RESID_MAX', max(ry), '%.2e', '', ...
        'max relative residual, cross-pol vs closed form', ...
        'tPolarization/test_odd_mirror_crosspol_pec_analytic');
    V = addval(V, 'G41_RESIDZ_MAX', max(rz), '%.2e', '', ...
        'max relative residual, longitudinal component vs closed form', ...
        'tPolarization/test_odd_mirror_crosspol_pec_analytic');
    V = addval(V, 'G41_RET', max(abs(imag(rEy(ok)))), '%.2e', '', ...
        'retardance introduced by a perfect conductor', ...
        'tPolarization/test_odd_mirror_crosspol_pec_analytic');

    % ---- G4.2 the fixture-free half: bound, magnitude, rho^2 law ------
    ratio = abs(rEy);
    V = addval(V, 'G42_BOUNDRATIO', max(ratio(ok))/max(sin(aoi(ok)).^2), ...
        '%.3f', '', 'max cross-pol as a fraction of the O(sin^2 AOI) bound', ...
        'tPolarization/test_odd_mirror_crosspol_rho2_law');
    V = addval(V, 'G42_PYPX1', mean(ratio(ok).^2), '%.3e', '', ...
        'cross-polarized power fraction after ONE mirror', ...
        'tPolarization/test_odd_mirror_crosspol_rho2_law');
    V = addval(V, 'G42_PYPX2', ...
        sum(abs(r2.Ey(ok2)).^2)/sum(abs(r2.Ex(ok2)).^2), '%.4e', '', ...
        'cross-polarized power fraction after TWO mirrors (unchanged by the fix)', ...
        'tools/pol_sp_sign_probe');

    N = size(r1.Ex, 1);  c = (N + 1)/2;
    [jj, ii] = meshgrid(1:N, 1:N);
    rho = hypot(ii - c, jj - c);  rmax = max(rho(ok));
    frac = [0.25 0.5 0.75 1.0];  med = nan(size(frac));
    for t = 1:numel(frac)
        sel = ok & abs(rho - frac(t)*rmax) < 2;
        med(t) = median(ratio(sel));
    end
    for t = 1:numel(frac)
        V = addval(V, sprintf('G42_R%d', t), med(t), '%.3e', '', ...
            sprintf('median |Ey/Ex| at rho = %.2f of the pupil edge', frac(t)), ...
            'tPolarization/test_odd_mirror_crosspol_rho2_law');
    end
    sl = polyfit(log(frac(:)), log(med(:)), 1);
    V = addval(V, 'G42_SLOPE', sl(1), '%.3f', '', ...
        'log-log slope of cross-pol vs pupil radius (physics requires 2)', ...
        'tPolarization/test_odd_mirror_crosspol_rho2_law');

    fig_spsign(ratio, predY, aoi, ok, rho/rmax, frac, med, ...
        fullfile(mediaDir, 'polval_spsign.png'));
end

function fig_spsign(ratio, predY, aoi, ok, rn, frac, med, out)
    f = newfig([1150 420]);
    tiledlayout(f, 1, 3, 'Padding', 'compact', 'TileSpacing', 'compact');
    nexttile; panel_here(ratio, ok, ...
        'measured |E_y/E_x| after ONE mirror');
    nexttile; panel_here(abs(predY), ok, ...
        'closed form  sin^2(AOI)|sin 2\phi| / den');
    nexttile;
    loglog(frac, med, 'o-', 'LineWidth', 1.4, 'MarkerFaceColor', 'w'); hold on
    loglog(frac, med(end)*frac.^2, 'k--', 'LineWidth', 1.1);
    grid on; xlabel('\rho / \rho_{max}'); ylabel('median |E_y/E_x|');
    legend({'measured', '\rho^2'}, 'Location', 'northwest', 'Box', 'off');
    title(sprintf('radial law (max AOI %.1f\\circ)', rad2deg(max(aoi(ok)))));
    sgtitle(f, ['One mirror: cross-polarization is slope-driven and matches ' ...
                'the perfect-conductor closed form -- pre-fix it was flat at ~1'], ...
            'FontWeight', 'bold');
    savefig_(f, out);
end

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
    % TEXTBOOK Born & Wolf forms (ray-following p-hat), mirroring
    % tJonesPupil after mmacos 9bc2029:
    %   r_p = (N2 c_i - N1 c_t)/(N2 c_i + N1 c_t)
    %   r_s = (N1 c_i - N2 c_t)/(N1 c_i + N2 c_t)
    % This RPa was originally transcribed from the engine's own
    % expression, which made the phase comparison CIRCULAR in exactly
    % the r_p sign the 2022 import had flipped -- it agreed with the
    % pre-fix engine and disagrees by pi with the corrected one.  The
    % gate guard caught that staleness when the fix landed; do not
    % "restore" it.  See REVIEW_POL_SP_SIGN_2026-07-27.md.
    N1 = 1.0;  N2 = complex(nAl, -kAl);
    cthi = abs(kix.*nx + kiy.*ny + kiz.*nz);
    ctht = sqrt(1 - (N1/N2)^2*(1 - cthi.^2));
    RPa = (N2*cthi - N1*ctht)./(N2*cthi + N1*ctht);
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
    % textbook forms as in fold_fresnel; D is a magnitude ratio so the
    % r_p sign cancels here -- normalized anyway so the two copies of
    % the analytic cannot drift apart again.
    RP = (N2*c - N1*ct)./(N2*c + N1*ct);
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

function fig_zernike(pz, pm, out)
% Bar chart of the expansion (the literature comparison IS the bar chart:
% one term should stand up and the rest should be floor) + the measured
% map and its reconstruction, to show the fit is not hiding a residual.
    f = newfig([1150 700]);
    tiledlayout(f, 2, 3, 'Padding', 'compact', 'TileSpacing', 'compact');
    K = numel(pz.modes);
    nexttile([1 2]);
    b = bar(1:K, [pz.D(:,1), pz.D(:,2), pz.D(:,3)]);
    set(gca, 'XTick', 1:K, 'XTickLabel', pz.names, ...
        'XTickLabelRotation', 45, 'YScale', 'linear');
    grid on; ylabel('coefficient');
    legend({'s_1 (0/90 lin)', 's_2 (\pm45 lin)', 's_3 (circ)'}, ...
        'Location', 'best'); legend boxoff;
    title('diattenuation expansion -- only astigmatism stands up');
    nexttile;
    semilogy(1:K, max(abs(pz.D(:,1)), 1e-20), 'o-', 'LineWidth', 1); hold on;
    semilogy(1:K, max(abs(pz.D(:,2)), 1e-20), 's-', 'LineWidth', 1);
    semilogy(1:K, max(abs(pz.D(:,3)), 1e-20), '^-', 'LineWidth', 1);
    set(gca, 'XTick', 1:K, 'XTickLabel', pz.names, 'XTickLabelRotation', 45);
    grid on; ylabel('|coefficient|');
    title('same, log scale (floor = round-off)');
    panel_map(pm.Dvec(:,:,1), pm.mask, 'measured  D_{s1}');
    panel_map(pz.recon.Dvec(:,:,1), pm.mask, 'Zernike reconstruction');
    panel_map(pm.Dvec(:,:,1) - pz.recon.Dvec(:,:,1), pm.mask, ...
        'residual (higher order than mode 15)');
    sgtitle(f, ['Low-order expansion: the two-mirror system reduces to ' ...
                'POLARIZATION ASTIGMATISM'], 'FontWeight', 'bold');
    savefig_(f, out);
end

function v = zern_sum_(modes, coefs, rho, th)
    v = 0;
    for k = 1:numel(modes)
        v = v + coefs(k)*ansi_zern_(modes(k), rho, th);
    end
end

function Z = ansi_zern_(j, rho, th)
% ANSI mode on caller polar coords.  Duplicated from +macos/private/
% (not reachable from tools/) -- same table, same convention.
    jj = j - 1;
    n  = ceil((-3 + sqrt(9 + 8*jj)) / 2);
    m  = 2*jj - n*(n + 2);
    am = abs(m);
    R  = zeros(size(rho));
    for s = 0:((n - am)/2)
        c = (-1)^s * factorial(n - s) / ...
            (factorial(s) * factorial((n+am)/2 - s) * factorial((n-am)/2 - s));
        R = R + c * rho.^(n - 2*s);
    end
    if m >= 0, ang = cos(m*th); else, ang = sin(am*th); end
    P = [1, 2, 2, sqrt(6), sqrt(3), sqrt(6), sqrt(8), sqrt(8), sqrt(8), ...
         sqrt(8), sqrt(10), sqrt(10), sqrt(5), sqrt(10), sqrt(10)];
    Z = P(j) .* R .* ang;
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
%  Phase 2c -- contrast floor, exactness gates (model 256)
% =====================================================================
function V = phase2c_exact_gates(V, ~, mediaDir)
    fprintf('polval: Phase 2c exactness gates (model 256)\n');
    rxChain = polval_rx('Rx_VecChain.in');   % 2 MidStop, 4 Detector
    rxCass  = polval_rx('Rx_Cass_FarField.in');  % 5 ExitPupil, 6 Detector
    nAl = 1.45; kAl = 7.54; thkAl = 2.0e-7;  % Al at 632.8 nm, BaseUnits = m
    nMgF2 = 1.38; thkMgF2 = 1.1e-7;          % ~quarter wave in MgF2

    % ---- C2.1 the machinery invents no floor -------------------------
    % Rx_VecChain is collimated, on-axis, flat and uncoated: polarization
    % is a no-op by construction, so the co-polarized channel must BE the
    % scalar run -- contrast curve included -- and the cross channel must
    % be empty.  This is the "x-pol reduces to the scalar contrast curve
    % at round-off" gate, and it is what stops the decomposition from
    % manufacturing a floor out of its own reference-frame choices.
    macos.load_rx(rxChain);
    oc = macos.pol_contrast_floor(2, 4, 'input', 'x');
    macos.polarization('off');
    Is = macos.intensity(4);
    pk = max(Is(:));
    V = addval(V, 'C21_VC_CROSS', oc.floor.cross / oc.floor.co, '%.3e', 'relative', ...
        'cross-polarized power on a polarization-neutral train', ...
        'tPolContrast/test_reduction_to_scalar_contrast_curve');
    V = addval(V, 'C21_VC_SCALAR', max(abs(oc.I_co(:) - Is(:))) / pk, '%.3e', ...
        'relative to peak', 'co-polarized channel == the scalar run', ...
        'tPolContrast/test_reduction_to_scalar_contrast_curve');
    cc = radial_mean_(oc.I_co);  cs = radial_mean_(Is);
    g  = isfinite(cc) & isfinite(cs) & cs > 0;
    V = addval(V, 'C21_VC_CURVE', max(abs(cc(g) - cs(g)) ./ cs(g)), '%.3e', ...
        'relative', 'co-polarized contrast curve == the scalar contrast curve', ...
        'tPolContrast/test_reduction_to_scalar_contrast_curve');
    V = addval(V, 'C21_VC_BINS', nnz(g), '%d', 'radial bins', ...
        'contrast-curve bins compared', 'this driver');

    % ---- C2.2 the split is a unitary change of basis -----------------
    macos.load_rx(rxCass);
    o = macos.pol_contrast_floor(5, 6, 'input', 'x', 'dark_zone', [10 40]);
    V = addval(V, 'C22_PARSEVAL', o.checks.parseval, '%.3e', 'relative to peak', ...
        'co + cross == |Ex|^2 + |Ey|^2 pointwise', ...
        'tPolContrast/test_parseval_on_the_split');
    V = addval(V, 'C22_CLOSURE', o.checks.closure, '%.3e', 'relative to peak', ...
        'co + cross + longitudinal == the engine intensity', ...
        'tPolContrast/test_energy_bookkeeping');

    % ---- C2.3 the floor of the bare two-mirror train, by component ---
    V = addval(V, 'C23_CROSS_OVER_CO', o.floor.cross_over_co, '%.4e', 'relative', ...
        'cross-polarized fraction, uncoated Cassegrain', ...
        'tPolContrast/test_floor_reported_by_component');
    V = addval(V, 'C23_LONG_FRAC', o.floor.long / (o.floor.co + o.floor.cross ...
        + o.floor.long), '%.4e', 'relative', ...
        'longitudinal fraction (compare the Tranche-1 out-of-plane 1-f)', ...
        'tPolContrast/test_floor_reported_by_component');
    V = addval(V, 'C23_DZ_BARE', o.floor.dark_zone.cross.mean, '%.3e', 'contrast', ...
        'mean peak-normalized cross-polarized contrast, 10-40 px annulus', ...
        'tPolContrast/test_floor_reported_by_component');

    % analyzer tracking, including the circular state -- the ONLY input
    % that can see a conjugated coherency matrix (a linear state has a
    % real analyzer, for which conj() is the identity).
    vc = [1; 1i]/sqrt(2);
    ocirc = macos.pol_contrast_floor(5, 6, 'input', vc);
    a = ocirc.per_state(1).analyzer;
    V = addval(V, 'C23_ANALYZER', abs(1 - abs(a' * vc)), '%.3e', 'relative', ...
        'derived analyzer tracks an arbitrary input state', ...
        'tPolContrast/test_analyzer_tracks_input_state');
    macos.polarization('on', 'Ex', [real(vc(1)) imag(vc(1))], ...
                             'Ey', [real(vc(2)) imag(vc(2))]);
    macos.vector_diffraction(true);
    D1 = macos.complex_field(6, 'plane', 1);
    D2 = macos.complex_field(6, 'plane', 2, 'reset_trace', false);
    ac = conj(a);
    badco = abs(conj(ac(1))*D1 + conj(ac(2))*D2).^2;
    trans = abs(D1).^2 + abs(D2).^2;
    V = addval(V, 'C23_CIRC_ORTHO', ...
        (sum(trans(:)) - sum(badco(:))) / sum(badco(:)), '%.3e', 'relative', ...
        'a conjugated analyzer reports the whole beam as cross-polarized', ...
        'tPolContrast/test_analyzer_would_be_wrong_if_conjugated');
    V = addval(V, 'C23_CIRC_TRUE', ocirc.floor.cross_over_co, '%.3e', 'relative', ...
        'true cross fraction for a circular input', ...
        'tPolContrast/test_analyzer_tracks_input_state');

    % ---- C2.4 scope: this train carries the whole chain --------------
    V = addval(V, 'C24_CARRIED', o.scope.worst, '%.6f', 'ratio', ...
        'grid / ray cross-polarized fraction (full carry)', ...
        'tPolContrast/test_scope_reports_full_carry_on_a_single_leg_train');
    V = addval(V, 'C24_CARRIED_LO', o.scope.worst, '%.6f', 'ratio', ...
        'same, lower bound', ...
        'tPolContrast/test_scope_reports_full_carry_on_a_single_leg_train');
    V = addval(V, 'C24_RAYFRAC', o.scope.ray_cross_frac(1), '%.4e', 'relative', ...
        'ray-level cross fraction (non-vacuity of the carry check)', ...
        'tPolContrast/test_scope_reports_full_carry_on_a_single_leg_train');

    % ---- C2.5 coating sensitivity ------------------------------------
    macos.load_rx(rxCass);
    al  = struct('elt', {2, 3}, 'index', nAl, 'extinc', kAl, ...
                 'thickness', thkAl, 'label', 'bare Al');
    pal = struct('elt', {2, 3}, 'index', [nMgF2 nAl], 'extinc', [0 kAl], ...
                 'thickness', [thkMgF2 thkAl], 'label', 'MgF2 / Al');
    osw = macos.pol_contrast_floor(5, 6, 'input', 'x', 'dark_zone', [10 40], ...
                                   'coatings', {al, pal});
    V = addval(V, 'C25_AL_REL', osw.sweep(1).d_cross_rel, '%.1f', 'x baseline', ...
        'cross-polarized power, bare Al vs uncoated', ...
        'tPolContrast/test_coating_sensitivity');
    V = addval(V, 'C25_MGF2_REL', osw.sweep(2).d_cross_rel, '%.1f', 'x baseline', ...
        'cross-polarized power, MgF2/Al vs uncoated', ...
        'tPolContrast/test_coating_sensitivity');
    V = addval(V, 'C25_DZ_AL', osw.sweep(1).floor.dark_zone.cross.mean, '%.3e', ...
        'contrast', 'mean cross contrast in the annulus, bare Al', ...
        'tPolContrast/test_coating_sensitivity');
    V = addval(V, 'C25_DZ_MGF2', osw.sweep(2).floor.dark_zone.cross.mean, '%.3e', ...
        'contrast', 'mean cross contrast in the annulus, MgF2/Al', ...
        'tPolContrast/test_coating_sensitivity');
    mono = double(osw.sweep(2).floor.dark_zone.cross.mean > ...
                  osw.sweep(1).floor.dark_zone.cross.mean && ...
                  osw.sweep(1).floor.dark_zone.cross.mean > ...
                  osw.floor.dark_zone.cross.mean);
    V = addval(V, 'C25_SWEEP_MONO', mono, '%d', 'bool', ...
        'floor rises monotonically bare -> Al -> MgF2/Al', ...
        'tPolContrast/test_coating_sensitivity');
    V = addval(V, 'C25_THKAL', thkAl * 1e9, '%.0f', 'nm', ...
        'Al layer thickness used', 'this driver');
    V = addval(V, 'C25_THKMGF2', thkMgF2 * 1e9, '%.0f', 'nm', ...
        'MgF2 overcoat thickness used', 'this driver');

    fig_floor_channels(o, osw, fullfile(mediaDir, 'polval_2c_channels.png'));
end

% =====================================================================
%  Phase 2c -- the coronagraph chain (model 512)
% =====================================================================
function V = phase2c_coro_gates(V, ~, mediaDir)
    fprintf('polval: Phase 2c coronagraph gates (model 512)\n');
    rx = polval_rx('Rx_Coro.in');
    PUP = 20; DET = 21;  MIR = [1 4 7 12 15 17 18];
    nAl = 1.45; kAl = 7.54; thkAl = 2.0e-4;   % BaseUnits = mm here

    macos.load_rx(rx);
    w = warning('off', 'macos:pol_contrast_floor:tranche1');
    restore = onCleanup(@() warning(w));
    o = macos.pol_contrast_floor(PUP, DET, 'input', 'x', 'dark_zone', [20 80]);

    V = addval(V, 'C31_CROSS_OVER_CO', o.floor.cross_over_co, '%.4e', 'relative', ...
        'cross-polarized fraction, coaxial coronagraph chain', ...
        'tPolContrastCoro/test_floor_reported_by_component');
    V = addval(V, 'C31_PEAKCONTRAST', o.floor.contrast_cross_peak, '%.4e', ...
        'contrast', 'peak cross-polarized contrast at the focal plane', ...
        'tPolContrastCoro/test_floor_reported_by_component');
    V = addval(V, 'C31_DZ_CROSS', o.floor.dark_zone.cross.mean, '%.3e', 'contrast', ...
        'mean cross-polarized contrast, 20-80 px annulus', ...
        'tPolContrastCoro/test_floor_reported_by_component');
    V = addval(V, 'C31_DZ_CROSS_MED', o.floor.dark_zone.cross.median, '%.3e', ...
        'contrast', 'median cross-polarized contrast, same annulus', ...
        'tPolContrastCoro/test_floor_reported_by_component');
    V = addval(V, 'C31_DZ_CO', o.floor.dark_zone.co.mean, '%.3e', 'contrast', ...
        'mean co-polarized contrast, same annulus', ...
        'tPolContrastCoro/test_floor_reported_by_component');
    V = addval(V, 'C31_PARSEVAL', o.checks.parseval, '%.3e', 'relative to peak', ...
        'Parseval on the split at model 512', ...
        'tPolContrastCoro/test_parseval_and_closure_at_scale');
    V = addval(V, 'C31_CLOSURE', o.checks.closure, '%.3e', 'relative to peak', ...
        'energy closure at model 512', ...
        'tPolContrastCoro/test_parseval_and_closure_at_scale');
    V = addval(V, 'C32_DOP', o.per_state(1).dop, '%.8f', 'ratio', ...
        'pupil degree of polarization (the analyzer is well defined)', ...
        'tPolContrastCoro/test_analyzer_is_fully_polarized_and_axis_aligned');

    % ---- C3.3 the Tranche-1 shortfall, measured ----------------------
    V = addval(V, 'C33_GRIDFRAC', o.scope.grid_cross_frac(1), '%.4e', 'relative', ...
        'cross-polarized fraction the diffraction grid carries', ...
        'tPolContrastCoro/test_tranche1_shortfall_is_detected');
    V = addval(V, 'C33_RAYFRAC', o.scope.ray_cross_frac(1), '%.4e', 'relative', ...
        'cross-polarized fraction the RAYS carry (the full train)', ...
        'tPolContrastCoro/test_tranche1_shortfall_is_detected');
    V = addval(V, 'C33_CARRIED', o.scope.worst, '%.4f', 'ratio', ...
        'carried fraction, uncoated', ...
        'tPolContrastCoro/test_tranche1_shortfall_is_detected');

    al = struct('elt', num2cell(MIR), 'index', nAl, 'extinc', kAl, ...
                'thickness', thkAl, 'label', 'bare Al');
    osw = macos.pol_contrast_floor(PUP, DET, 'input', 'x', ...
                                   'dark_zone', [20 80], 'coatings', {al});
    V = addval(V, 'C33_CARRIED_COAT', osw.sweep(1).scope.worst, '%.4f', 'ratio', ...
        'carried fraction with all seven mirrors coated', ...
        'tPolContrastCoro/test_coating_sensitivity_is_not_trustworthy_here');
    V = addval(V, 'C33_GRID_DREL', osw.sweep(1).d_cross_rel, '%+.4f', 'x baseline', ...
        'coating sensitivity the GRID reports (wrong sign)', ...
        'tPolContrastCoro/test_coating_sensitivity_is_not_trustworthy_here');

    % the ray-level truth the grid is missing (the sweep leaves Al applied)
    macos.polarization('on', 'Ex', [1 0], 'Ey', [0 0]);
    macos.vector_diffraction(true);
    macos.trace(PUP);
    rf = macos.ray_field(PUP);
    k = rf.status == 0;
    raycoat = sum(abs(rf.Ey(k)).^2) / sum(abs(rf.Ex(k)).^2);
    V = addval(V, 'C33_RAY_COAT', raycoat, '%.4e', 'relative', ...
        'ray-level cross fraction with the mirrors coated', ...
        'tPolContrastCoro/test_coating_sensitivity_is_not_trustworthy_here');
    V = addval(V, 'C33_RAY_DREL', ...
        raycoat / o.scope.ray_cross_frac(1) - 1, '%+.3f', 'x baseline', ...
        'coating sensitivity the RAYS report (the truth the grid misses)', ...
        'tPolContrastCoro/test_coating_sensitivity_is_not_trustworthy_here');
    V = addval(V, 'C33_NMIRROR', numel(MIR), '%d', 'mirrors', ...
        'reflectors in the chain', 'this driver');

    fig_coro_floor(o, fullfile(mediaDir, 'polval_2c_coro.png'));
end

function c = radial_mean_(I)
    N = size(I, 1);  ctr = (N - 1)/2;
    [xx, yy] = meshgrid(0:N-1, 0:N-1);
    rr = round(hypot(xx - ctr, yy - ctr)) + 1;
    c = accumarray(rr(:), I(:), [], @mean) / max(I(:));
end

function fig_floor_channels(o, osw, out)
%FIG_FLOOR_CHANNELS  Cass-FF: the three channels, and the coating trade.
    f = newfig([1150 380]);
    tiledlayout(f, 1, 3, 'Padding', 'compact', 'TileSpacing', 'compact');
    m = true(size(o.I_co));
    nexttile; panel_here(log10(max(o.contrast_co,    1e-30)), m, ...
        'log_{10} co-polarized (peak-normalized)');
    nexttile; panel_here(log10(max(o.contrast_cross, 1e-30)), m, ...
        'log_{10} cross-polarized');
    nexttile;
    r = 0:(numel(radial_mean_(o.I_co)) - 1);
    semilogy(r, radial_mean_(o.I_cross) * max(o.I_cross(:)) / max(o.I_co(:)), ...
             '-', 'LineWidth', 1.2); hold on
    semilogy(r, radial_mean_(osw.sweep(1).I_cross) * ...
             max(osw.sweep(1).I_cross(:)) / max(osw.sweep(1).I_co(:)), '-', 'LineWidth', 1.2);
    semilogy(r, radial_mean_(osw.sweep(2).I_cross) * ...
             max(osw.sweep(2).I_cross(:)) / max(osw.sweep(2).I_co(:)), '-', 'LineWidth', 1.2);
    grid on; xlabel('radius (px)'); ylabel('cross-polarized contrast');
    legend({'uncoated', 'bare Al', 'MgF_2 / Al'}, 'Location', 'northeast');
    title('polarization floor vs coating'); xlim([0 60]);
    savefig_(f, out);
end

function fig_coro_floor(o, out)
%FIG_CORO_FLOOR  Rx_Coro: co / cross channels and the contrast curves.
    f = newfig([1150 380]);
    tiledlayout(f, 1, 3, 'Padding', 'compact', 'TileSpacing', 'compact');
    m = true(size(o.I_co));
    nexttile; panel_here(log10(max(o.contrast_co,    1e-30)), m, ...
        'log_{10} co-polarized (peak-normalized)');
    nexttile; panel_here(log10(max(o.contrast_cross, 1e-30)), m, ...
        'log_{10} cross-polarized');
    nexttile;
    cco = radial_mean_(o.I_co);
    ccr = radial_mean_(o.I_cross) * max(o.I_cross(:)) / max(o.I_co(:));
    r = 0:(numel(cco) - 1);
    semilogy(r, cco, '-', 'LineWidth', 1.2); hold on
    semilogy(r, ccr, '-', 'LineWidth', 1.2);
    grid on; xlabel('radius (px)'); ylabel('contrast');
    legend({'co-polarized', 'cross-polarized'}, 'Location', 'northeast');
    title('coronagraph contrast, by channel'); xlim([0 150]);
    savefig_(f, out);
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

function p = capture_provenance(model)
%CAPTURE_PROVENANCE  git/host/tool state, sampled BEFORE the run writes.
    p = struct();
    p.generated     = char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss'));
    [p.engine_sha, p.engine_branch, p.engine_dirty] = ...
        gitinfo(fullfile(getenv('HOME'), 'dev', 'macos'));
    [p.resources_sha, p.resources_branch, p.resources_dirty] = ...
        gitinfo(fullfile(getenv('HOME'), 'dev', 'MACOS_resources'));
    p.matlab     = version('-release');
    p.model_size = model;
    [~, h] = system('hostname'); p.host = strtrim(h);
end

function write_numbers(V, path, p)
    out = struct('provenance', p, 'values', V);
    fid = fopen(path, 'w');
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
