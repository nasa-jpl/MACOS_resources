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
            V = polelt_gates(V, MODEL, mediaDir);
            V = radiometric_gates(V, MODEL);
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
        'G51_MALUS',      '<',   1e-12
        'G51_DYNRANGE',   '<',   1e-25
        'G52_CROSSED',    '==',  1
        'G53_S3',         '<',   1e-14
        'G53_S12',        '<',   1e-14
        'G53_AB_S3',      '<',   1e-14
        'G54_SLOPE_RESID','<',   1e-10
        'G54_AB_SLOPE',   '<',   1e-10
        'G55_COMPOSE',    '<',   1e-15
        'G55_AB_SINGLE',  '>',   0.1
        'G56_UNITARY',    '<',   1e-14
        'G56_JUNITARY',   '<',   1e-15
        'G56_AB_POL',     '>',   0.5
        'G57_BITWISE',    '==',  1
        'G58_GRID_MALUS', '<',   1e-10
        'G60_AOI_RESID',  '<',   1e-12
        'G60_MATAXIS',    '<',   1e-12
        'G60_VS_PASSAXIS_RESID', '<', 1e-9
        'G61_DEGEN_SPREAD','<',  1e-15
        'G61_DEGEN_ENGINE','<',  1e-12
        'G62_NULL_SEP',   '>',   5
        'G62_GRID_NULL',  '<',   1e-25
        'G62_GRID_LEAK',  '>',   1e-3
        'G62_LEAK_RESID', '<',   1e-6
        'G71_UNC_NORMAL', '<',   1e-12
        'G71_UNC_P',      '<',   1e-12
        'G71_UNC_S',      '<',   1e-12
        'G71_SP_SPLIT',   '>',   0.08
        'G72_IDX_NORMAL', '<',   1e-13
        'G72_IDX_P',      '<',   1e-13
        'G72_IDX_S',      '<',   1e-13
        'G72_GRID',       '<',   1e-12
        'G73_MGF2_NORMAL','<',   1e-12
        'G73_MGF2_P',     '<',   1e-12
        'G73_MGF2_S',     '<',   1e-12
        'G73_MGF2_GAIN',  '>',   0.98
        'G74_CLOSURE',    '<',   1e-12
        'G74_TELESCOPE',  '<',   1e-12
        'G75_STATE',      '<',   1e-11
        'G75_STATE_SPLIT','>',   0.02
        'G76_POLOFF',     '==',  1
        'G77_LAMBDA',     '<',   1e-11
        'G77_FLATNESS',   '<',   1e-11
        'G77_CONTRAST',   '>',   0.01
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
        % the overcoat trade reverses across the quarter-wave condition --
        % the two sides, the reversal, and the achromatic control that
        % makes the pair non-vacuous.  The three "invariance" limits are
        % exact by construction (coating coefficients depend on lambda only
        % through n*d/lambda, indices fixed), so they sit at round-off.
          'C26_RATIO_633',     '<',   1
          'C26_RATIO_1000',    '>',   1
          'C26_REVERSAL',      '>',   5
          'C26_QW_RATIO',      '<',   0.1
          'C26_ACHROM',        '>',   1
          'C26_ACHROM_RESID',  '<',   1e-6
          'C26_QW_INVAR',      '<',   1e-6
          'C26_METAL_INVAR',   '<',   1e-6
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

    % ---- C2.6 the same ladder on BOTH sides of the quarter-wave ------
    % Companion evidence for section 8.3.  The 110 nm film above is 0.607
    % quarter-waves at THIS fixture's 1 um, not the 0.96 its "632.8 nm"
    % comment describes, and the overcoat trade REVERSES across the
    % quarter-wave condition.  Section 8.3 stated that from an independent
    % analytic; these are the ENGINE numbers on both sides.  The fixture
    % does not move -- the companion wavelength is applied at runtime with
    % macos.set_src_wvl.  Tool + rationale:
    % mmacos/tools/pol_overcoat_chromatic/README.md.
    addpath(fullfile(fileparts(mfilename('fullpath')), '..', ...
                     'pol_overcoat_chromatic'));
    ocl = oc_ladder([], false);          % engine already init'd at 256
    GATE26 = ['tPolContrast/' ...
              'test_overcoat_trade_reverses_across_the_quarter_wave_condition'];
    V = addval(V, 'C26_LAM_CMP', ocl.at633.lambda * 1e9, '%.1f', 'nm', ...
        'companion wavelength, applied at runtime (the fixture stays at 1 um)', ...
        GATE26);
    V = addval(V, 'C26_QW_FRAC_1000', ocl.at1000.qw_frac, '%.4f', 'quarter-waves', ...
        'the 110 nm MgF2 film, in quarter-waves at the fixture''s own 1 um', ...
        GATE26);
    V = addval(V, 'C26_QW_FRAC_633', ocl.at633.qw_frac, '%.4f', 'quarter-waves', ...
        'the same film, in quarter-waves at 632.8 nm', GATE26);
    V = addval(V, 'C26_RATIO_1000', ocl.at1000.ratio_mgf2, '%.4f', 'x', ...
        'ENGINE cross-polarized power, MgF2/Al over bare Al, at 1 um (COSTS)', ...
        GATE26);
    V = addval(V, 'C26_RATIO_633', ocl.at633.ratio_mgf2, '%.4f', 'x', ...
        'ENGINE cross-polarized power, same film, at 632.8 nm (SUPPRESSES)', ...
        GATE26);
    V = addval(V, 'C26_REVERSAL', ocl.reversal, '%.2f', 'x', ...
        'the reversal itself: the 1 um ratio over the 632.8 nm ratio', GATE26);
    V = addval(V, 'C26_EXCESS_1000', ocl.at1000.ratio_excess, '%.4f', 'x', ...
        'same ratio formed from the coating EXCESS over uncoated (compares with the pure-Fresnel analytic)', ...
        GATE26);
    V = addval(V, 'C26_EXCESS_633', ocl.at633.ratio_excess, '%.4f', 'x', ...
        'coating-excess ratio at 632.8 nm', GATE26);
    V = addval(V, 'C26_EXCESS_QW', ocl.at1000.ratio_excess_qw, '%.4f', 'x', ...
        'coating-excess ratio for a TRUE quarter-wave overcoat (the column that compares with the pure-Fresnel analytic)', ...
        GATE26);
    V = addval(V, 'C26_QW_RATIO', ocl.at1000.ratio_qw, '%.4f', 'x', ...
        'ENGINE cross-polarized power, a TRUE quarter-wave MgF2 overcoat over bare Al', ...
        GATE26);
    V = addval(V, 'C26_QW_FLOOR', ocl.at1000.qw_over_uncoated, '%.4f', 'x', ...
        'the true-quarter-wave total, over the UNCOATED geometric floor -- why it cannot follow the coating-only analytic down', ...
        GATE26);
    V = addval(V, 'C26_QW_INVAR', ...
        abs(ocl.at633.ratio_qw / ocl.at1000.ratio_qw - 1), '%.1e', 'relative', ...
        'the quarter-wave condition is wavelength-invariant (181.2 nm at 1 um vs 114.6 nm at 632.8 nm, same answer)', ...
        GATE26);
    V = addval(V, 'C26_THK_ACHROM', ocl.achromatic.thk_mgf2 * 1e9, '%.1f', 'nm', ...
        'the achromatic control film: same optical thickness in WAVES at 632.8 nm that 110 nm has at 1 um', ...
        GATE26);
    V = addval(V, 'C26_ACHROM', ocl.achromatic.ratio_mgf2, '%.4f', 'x', ...
        'NON-VACUITY -- the control shows NO reversal, so the reversal is the film''s optical thickness and nothing else about changing lambda', ...
        GATE26);
    V = addval(V, 'C26_ACHROM_RESID', ...
        abs(ocl.achromatic.ratio_mgf2 / ocl.at1000.ratio_mgf2 - 1), '%.1e', ...
        'relative', 'the control lands back on the 1 um answer', GATE26);
    V = addval(V, 'C26_METAL_INVAR', ...
        abs(ocl.at633.al_over_bare / ocl.at1000.al_over_bare - 1), '%.1e', ...
        'relative', ...
        'the metal-only leg (no film, fixed indices) does not move with wavelength', ...
        GATE26);

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

% =====================================================================
%  Phase 3 -- polarizing elements (TrPolarizer + WavePlate), model 128
%
%  Every prediction below is a closed-form Jones identity written from
%  the textbook, NOT transcribed from the engine -- the standing lesson
%  of section 4.  Each mechanism also carries an A/B in which it is
%  switched off, so a passing number cannot mean "the element quietly
%  did nothing" (which is precisely what these EltIDs did before this
%  phase: they were name-table-only stubs).
% =====================================================================
function V = polelt_gates(V, ~, mediaDir)
    fprintf('polval: Phase 3 polarizing-element gates\n');
    rx    = polval_rx('Rx_PolElt.in');
    rxRef = polval_rx('Rx_PolElt_Ref.in');
    POL1 = 2; WP1 = 3; WP2 = 4; ANAL = 5; DET = 7;
    ax = @(t) [cos(t) sin(t) 0];

    % ---- G5.7 first: polarization OFF must be bit-identical to a twin
    % prescription with plain Reference surfaces in the same places.
    macos.load_rx(rx);      macos.polarization('off');
    tA = macos.trace(DET);  WA = macos.opd();  IA = macos.intensity(DET);
    macos.load_rx(rxRef);   macos.polarization('off');
    tB = macos.trace(DET);  WB = macos.opd();  IB = macos.intensity(DET);
    bitok = isequal(WA, WB) && isequal(IA, IB) && (tA.nRays == tB.nRays);
    V = addval(V, 'G57_BITWISE', double(bitok), '%d', 'bool', ...
        'pol-off trace bit-identical to the Reference-surface twin', ...
        'tPolElement/test_unpolarized_bit_identical_to_reference_twin');

    macos.load_rx(rx);
    macos.polarization('on', 'Ex', [1 0], 'Ey', [0 0]);
    cfg = @(tp,t1,r1,t2,r2,ta) polelt_configure(POL1,WP1,WP2,ANAL, ...
                                                ax(tp),ax(t1),r1,ax(t2),r2,ax(ta));

    % ---- G5.1 Malus.  I(theta) = I0 cos^2(theta).
    th = linspace(0, pi, 25);   I = zeros(size(th));
    for k = 1:numel(th)
        cfg(0, 0, 0, 0, 0, th(k));
        macos.trace(ANAL);
        I(k) = polelt_power(ANAL);
    end
    pred = I(1) * cos(th).^2;
    V = addval(V, 'G51_MALUS', max(abs(I - pred))/I(1), '%.3e', 'relative', ...
        'Malus law vs analyzer angle', 'tPolElement/test_malus_law');
    V = addval(V, 'G51_DYNRANGE', min(I)/max(I), '%.3e', 'relative', ...
        'Malus curve dynamic range (non-vacuity: a pass-everything element is flat)', ...
        'tPolElement/test_malus_law');

    % ---- G5.2 crossed extinction, exactly-orthogonal axes -> exactly 0.
    macos.polarizer(POL1, 'axis', [1 0 0]);
    macos.polarizer(ANAL, 'axis', [0 1 0]);
    macos.waveplate(WP1, 'axis', [1 0 0], 'retardance', 0);
    macos.waveplate(WP2, 'axis', [1 0 0], 'retardance', 0);
    macos.trace(ANAL);
    V = addval(V, 'G52_CROSSED', double(polelt_power(ANAL) == 0), '%d', 'bool', ...
        'crossed-polarizer extinction is exactly zero', ...
        'tPolElement/test_crossed_polarizer_extinction');

    % ---- G5.3 QWP: linear in, circular out.  The SIGN of S3 is the
    % point -- it flips with the retardance convention, so a gate on
    % |S3| would accept either.
    cfg(0, pi/4, 0.25, 0, 0, 0);
    macos.trace(WP1);   S = polelt_stokes(WP1);
    V = addval(V, 'G53_S3', abs(S.S3/S.S0 + 1), '%.3e', 'relative', ...
        'QWP linear->circular: signed S3/S0 = -1', ...
        'tPolElement/test_qwp_linear_to_circular');
    V = addval(V, 'G53_S12', max(abs([S.S1 S.S2]))/S.S0, '%.3e', 'relative', ...
        'QWP linear->circular: residual linear Stokes', ...
        'tPolElement/test_qwp_linear_to_circular');
    cfg(0, pi/4, 0, 0, 0, 0);            % A/B: no retardance
    macos.trace(WP1);   Sab = polelt_stokes(WP1);
    V = addval(V, 'G53_AB_S3', abs(Sab.S3/Sab.S0), '%.3e', 'relative', ...
        'A/B -- the same rig at zero retardance leaves the state linear', ...
        'tPolElement/test_qwp_linear_to_circular');

    % Stokes sweep vs retardance, for the figure
    Rs = linspace(0, 1, 41);   S3s = zeros(size(Rs));   S1s = S3s;
    for k = 1:numel(Rs)
        cfg(0, pi/4, Rs(k), 0, 0, 0);
        macos.trace(WP1);  Sk = polelt_stokes(WP1);
        S3s(k) = Sk.S3/Sk.S0;   S1s(k) = Sk.S1/Sk.S0;
    end

    % ---- G5.4 HWP rotates linear input by 2*theta.
    thw = linspace(0, pi/2, 17);   ang = zeros(size(thw));
    for k = 1:numel(thw)
        cfg(0, thw(k), 0.5, 0, 0, 0);
        macos.trace(WP1);  Sk = polelt_stokes(WP1);
        ang(k) = 0.5*atan2(Sk.S2, Sk.S1);
    end
    angu = unwrap(2*ang)/2;
    pfit = polyfit(thw, angu, 1);
    V = addval(V, 'G54_SLOPE_RESID', abs(pfit(1) - 2), '%.3e', 'absolute', ...
        'HWP output orientation slope vs plate angle (theory 2)', ...
        'tPolElement/test_hwp_rotates_by_2theta');
    ang0 = zeros(size(thw));
    for k = 1:numel(thw)
        cfg(0, thw(k), 0, 0, 0, 0);
        macos.trace(WP1);  Sk = polelt_stokes(WP1);
        ang0(k) = 0.5*atan2(Sk.S2, Sk.S1);
    end
    p0 = polyfit(thw, unwrap(2*ang0)/2, 1);
    V = addval(V, 'G54_AB_SLOPE', abs(p0(1)), '%.3e', 'absolute', ...
        'A/B -- at zero retardance, rotating the plate does nothing (slope 0)', ...
        'tPolElement/test_hwp_rotates_by_2theta');

    % ---- G5.5 composition: two QWPs on a common axis == one HWP.
    thf = 0.4;
    cfg(0, thf, 0.25, thf, 0.25, 0);  macos.trace(WP2);
    [ExA, EyA] = polelt_field(WP2);
    cfg(0, thf, 0.5,  thf, 0,    0);  macos.trace(WP2);
    [ExB, EyB] = polelt_field(WP2);
    nrm = max(abs(ExA));
    V = addval(V, 'G55_COMPOSE', ...
        max([abs(ExA-ExB); abs(EyA-EyB)])/nrm, '%.3e', 'relative', ...
        'two quarter-wave plates on a common axis reproduce one half-wave plate', ...
        'tPolElement/test_two_qwp_equal_one_hwp');
    cfg(0, thf, 0.25, thf, 0,    0);  macos.trace(WP2);
    [ExC, ~] = polelt_field(WP2);
    V = addval(V, 'G55_AB_SINGLE', max(abs(ExA-ExC))/nrm, '%.3e', 'relative', ...
        'A/B -- a SINGLE quarter-wave plate at the same axis differs grossly', ...
        'tPolElement/test_two_qwp_equal_one_hwp');

    % ---- G5.6 unitarity, for linear AND circular input.
    cfg(0, 0.37, 0.31, 0, 0, 0);
    macos.trace(POL1);  P0 = polelt_power(POL1);
    macos.trace(WP1);   P1 = polelt_power(WP1);
    uLin = abs(P1/P0 - 1);
    Jw = macos.elt_jones(WP1);
    cfg(0, pi/4, 0.25, 0.19, 0.42, 0);      % circular onto WP2
    macos.trace(WP1);  Pc0 = polelt_power(WP1);
    macos.trace(WP2);  Pc1 = polelt_power(WP2);
    V = addval(V, 'G56_UNITARY', max(uLin, abs(Pc1/Pc0 - 1)), '%.3e', 'relative', ...
        'retarder conserves power, linear and circular input', ...
        'tPolElement/test_waveplate_is_unitary');
    V = addval(V, 'G56_JUNITARY', norm(Jw'*Jw - eye(2)), '%.3e', 'absolute', ...
        'retarder Jones satisfies J^H J = I', ...
        'tPolElement/test_waveplate_is_unitary');
    Jp = macos.elt_jones(POL1);
    V = addval(V, 'G56_AB_POL', norm(Jp'*Jp - eye(2)), '%.3e', 'absolute', ...
        'A/B -- the POLARIZER Jones must FAIL unitarity, so the test can fail', ...
        'tPolElement/test_waveplate_is_unitary');

    % ---- G5.8 the diffraction grid carries the polarizing train.  This
    % is the tripwire for the two-dispatch-chain trap: an element wired
    % only into tracesub.F satisfies every ray-level gate above and
    % leaves the detector plane completely unchanged.
    thg = [0 pi/6 pi/3 pi/2];   Ig = zeros(size(thg));
    for k = 1:numel(thg)
        cfg(0, 0, 0, 0, 0, thg(k));
        macos.trace(DET);
        Ig(k) = sum(macos.intensity(DET), 'all');
    end
    V = addval(V, 'G58_GRID_MALUS', ...
        max(abs(Ig - Ig(1)*cos(thg).^2))/Ig(1), '%.3e', 'relative', ...
        'detector-plane intensity obeys Malus (grid sees the train)', ...
        'tPolElement/test_grid_carries_the_polarizing_train');

    % ---- the size of the off-normal axis-convention choice, in closed
    % form (no engine): the two candidate constructions, and the two
    % azimuths at which they agree no matter which is right.
    V = addval(V, 'G59_AMBIG20', rad2deg(polelt_ambig(deg2rad(20), pi/4)), ...
        '%.2f', 'degrees', ...
        'off-normal pass-vs-material axis difference at 20 deg AOI, 45 deg azimuth', ...
        'tPolElement/test_offnormal_convention_magnitude');
    V = addval(V, 'G59_AMBIG20_INPLANE', ...
        rad2deg(polelt_ambig(deg2rad(20), 0)), '%.3e', 'degrees', ...
        'the same difference with the axis IN the plane of incidence (vanishes)', ...
        'tPolElement/test_offnormal_convention_magnitude');

    fig_polelt(th, I, pred, Rs, S3s, S1s, thw, angu, ...
               fullfile(mediaDir, 'polelt_gates.png'));

    % =================================================================
    % G6.0-6.2  OFF NORMAL: the settled material-axis rule, driven in
    % the engine.  Rx_PolElt_Tilt.in tilts the BEAM (a collimated on-axis
    % bundle does not care whether the ELEMENT is tilted), so the
    % polarizers see a true 20 deg AOI and the convention is exercised.
    % =================================================================
    macos.load_rx(polval_rx('Rx_PolElt_Tilt.in'));
    TPOL = 2;   TANAL = 3;   TDET = 5;
    macos.polarization('on', 'Ex', [1 0], 'Ey', [0 0]);
    axf = @(p) [cos(p) sin(p) 0];
    phi = pi/4;

    macos.polarizer(TPOL,  'axis', axf(phi));
    macos.polarizer(TANAL, 'axis', axf(phi));
    macos.trace(TPOL);
    [d, r, n] = polelt_axis_from_field(TPOL);

    % the fixture really is at 20 deg, ray by ray, out of the engine's own
    % direction cosines and surface normal
    aoi = rad2deg(acos(abs(sum(r .* n, 1))));
    V = addval(V, 'G60_AOI_RESID', max(abs(aoi - 20)), '%.1e', 'degrees', ...
        'measured AOI at the tilted polarizer, residual from 20 deg (all rays)', ...
        'tPolElement/test_offnormal_transmitted_axis_is_the_material_rule');

    tMat = polelt_material_axis(r(:,1).', n(:,1).', axf(phi));
    tFS  = polelt_passaxis_axis(r(:,1).', axf(phi));
    V = addval(V, 'G60_MATAXIS', max(polelt_axis_angle(d, tMat)), ...
        '%.1e', 'radians', ...
        'engine transmitted axis vs the MATERIAL-axis closed form, 20 deg AOI / 45 deg azimuth', ...
        'tPolElement/test_offnormal_transmitted_axis_is_the_material_rule');
    aFS = rad2deg(polelt_axis_angle(d, tFS));
    V = addval(V, 'G60_VS_PASSAXIS', mean(aFS), '%.4f', 'degrees', ...
        'engine transmitted axis vs the REJECTED pass-axis projection -- the full ambiguity', ...
        'tPolElement/test_offnormal_transmitted_axis_is_the_material_rule');
    V = addval(V, 'G60_VS_PASSAXIS_RESID', ...
        max(abs(aFS - rad2deg(polelt_ambig(deg2rad(20), pi/4)))), ...
        '%.1e', 'degrees', ...
        'and that miss equals the closed form acos(2cos a/(1+cos^2 a)) to', ...
        'tPolElement/test_offnormal_transmitted_axis_is_the_material_rule');

    % degenerate azimuths: BOTH constructions, and the engine, coincide --
    % so a gate written there cannot tell the two rules apart
    dg = 0;   dgm = 0;
    macos.polarization('on', 'Ex', [1/sqrt(2) 0], 'Ey', [1/sqrt(2) 0]);
    for p = [0 pi/2]
        macos.polarizer(TPOL, 'axis', axf(p));
        macos.trace(TPOL);
        [dd, rr, nn] = polelt_axis_from_field(TPOL);
        tM = polelt_material_axis(rr(:,1).', nn(:,1).', axf(p));
        tF = polelt_passaxis_axis(rr(:,1).', axf(p));
        dg  = max(dg,  polelt_axis_angle(tM.', tF));
        dgm = max([dgm, max(polelt_axis_angle(dd, tM)), ...
                        max(polelt_axis_angle(dd, tF))]);
    end
    V = addval(V, 'G61_DEGEN_SPREAD', dg, '%.1e', 'radians', ...
        'at azimuth 0 and 90 the two constructions coincide (the vacuity trap)', ...
        'tPolElement/test_offnormal_degenerate_azimuths_are_blind');
    V = addval(V, 'G61_DEGEN_ENGINE', dgm, '%.1e', 'radians', ...
        'and the engine sits on both of them there, so such a gate proves nothing', ...
        'tPolElement/test_offnormal_degenerate_azimuths_are_blind');

    % the same discrimination on the DETECTOR plane (the second dispatch
    % chain): the crossed-analyzer null sits at a different azimuth under
    % each rule, so total intensity alone separates them
    macos.polarization('on', 'Ex', [1 0], 'Ey', [0 0]);
    macos.polarizer(TPOL, 'axis', axf(phi));
    fM = @(p2) dot(polelt_material_axis(r(:,1).', n(:,1).', axf(p2)), tMat);
    fF = @(p2) dot(polelt_passaxis_axis(r(:,1).', axf(p2)), tFS);
    p2m = fzero(fM, [phi+pi/2-0.3, phi+pi/2+0.3]);
    p2f = fzero(fF, [phi+pi/2-0.3, phi+pi/2+0.3]);
    Pg = zeros(1,3);   pset = [phi, p2m, p2f];
    for k = 1:3
        macos.polarizer(TANAL, 'axis', axf(pset(k)));
        macos.trace(TDET);
        Pg(k) = sum(macos.intensity(TDET), 'all');
    end
    V = addval(V, 'G62_NULL_SEP', rad2deg(abs(p2m - p2f)), '%.2f', 'degrees', ...
        'separation of the crossed-analyzer null predicted by the two rules', ...
        'tPolElement/test_offnormal_grid_crossed_null');
    V = addval(V, 'G62_GRID_NULL', Pg(2)/Pg(1), '%.1e', 'relative', ...
        'detector-plane power at the MATERIAL-rule crossed setting (extinguishes)', ...
        'tPolElement/test_offnormal_grid_crossed_null');
    V = addval(V, 'G62_GRID_LEAK', Pg(3)/Pg(1), '%.3e', 'relative', ...
        'detector-plane power at the PASS-axis crossed setting (does not)', ...
        'tPolElement/test_offnormal_grid_crossed_null');
    leakPred = cos(polelt_axis_angle( ...
        polelt_material_axis(r(:,1).', n(:,1).', axf(p2f)).', tMat))^2;
    V = addval(V, 'G62_LEAK_RESID', abs(Pg(3)/Pg(1) - leakPred)/leakPred, ...
        '%.1e', 'relative', ...
        'and that leak is the predicted cos^2 of the residual axis mismatch, to', ...
        'tPolElement/test_offnormal_grid_crossed_null');
end

function polelt_configure(p1, w1, w2, an, ap, a1, r1, a2, r2, aa)
    macos.polarizer(p1, 'axis', ap);
    macos.waveplate(w1, 'axis', a1, 'retardance', r1);
    macos.waveplate(w2, 'axis', a2, 'retardance', r2);
    macos.polarizer(an, 'axis', aa);
end

function [Ex, Ey, Ez] = polelt_field(srf)
    rf = macos.ray_field(srf);   m = (rf.status == 0);
    Ex = rf.Ex(m);  Ey = rf.Ey(m);  Ez = rf.Ez(m);
end

function P = polelt_power(srf)
    [Ex, Ey, Ez] = polelt_field(srf);
    P = sum(abs(Ex).^2 + abs(Ey).^2 + abs(Ez).^2);
end

function S = polelt_stokes(srf)
%   Conjugation order written out explicitly: MATLAB's ' conjugates its
%   LEFT operand, and getting it backwards builds conj(C) -- the slip
%   that passed every linear gate vacuously in section 5.
    [Ex, Ey, ~] = polelt_field(srf);
    S.S0 = sum(abs(Ex).^2 + abs(Ey).^2);
    S.S1 = sum(abs(Ex).^2 - abs(Ey).^2);
    S.S2 = sum(2*real(Ex .* conj(Ey)));
    S.S3 = sum(2*imag(Ex .* conj(Ey)));
end

function dth = polelt_ambig(aoi, az)
%POLELT_AMBIG  Angle between the two candidate polarizer pass axes off
%   normal: project the declared PASS axis (Fainman and Shamir) versus
%   project the MATERIAL, absorbing axis and take the complement (Korger
%   et al., which is what PolElt does).
    r    = [sin(aoi) 0 cos(aoi)];
    pass = [cos(az)  sin(az) 0];
    blok = [-sin(az) cos(az) 0];
    prj  = @(v) (v - dot(v,r)*r) / norm(v - dot(v,r)*r);
    a1 = prj(pass);
    a2 = cross(r, prj(blok));  a2 = a2 / norm(a2);
    dth = polelt_axis_angle(a1(:), a2);
end

function [d, r, n] = polelt_axis_from_field(srf)
%POLELT_AXIS_FROM_FIELD  Per-ray transmitted axis of a polarizer, read off
%   the field, with the geometry taken from the engine (direction cosines
%   and surface normal), not from the fixture's declared numbers.
    rf = macos.ray_field(srf);   m = (rf.status == 0);
    E  = [rf.Ex(m).'; rf.Ey(m).'; rf.Ez(m).'];
    r  = [rf.kx(m).'; rf.ky(m).'; rf.kz(m).'];
    n  = [rf.nx(m).'; rf.ny(m).'; rf.nz(m).'];
    [~, im] = max(abs(E(:,1)));
    d = real(E ./ E(im,:));
    d = d ./ vecnorm(d);
end

function t = polelt_material_axis(r, n, a)
%POLELT_MATERIAL_AXIS  Transmitted axis under the SETTLED rule: project the
%   absorbing direction n x a, extinguish it, transmit its partner.
    r = r / norm(r);
    w = cross(n, a);
    b = w - dot(w, r)*r;   b = b / norm(b);
    t = cross(b, r);       t = t(:).' / norm(t);
end

function t = polelt_passaxis_axis(r, a)
%POLELT_PASSAXIS_AXIS  Transmitted axis under the REJECTED construction:
%   the declared pass axis projected orthographically. Kept as an explicit
%   non-target.
    r = r / norm(r);
    t = a - dot(a, r)*r;   t = t(:).' / norm(t);
end

function ang = polelt_axis_angle(d, t)
%POLELT_AXIS_ANGLE  Angle between axis T (1x3) and each column of D (3xN),
%   sign-insensitive and computed as atan2(|cross|,|dot|) -- an acos of a
%   near-unit dot product loses half its digits and reports 1e-8 rad for a
%   residual that is really 1e-16.
    t  = t(:) / norm(t);
    ct = abs(sum(d .* t, 1));
    st = vecnorm(cross(d, repmat(t, 1, size(d,2)), 1));
    ang = atan2(st, ct);
end

function fig_polelt(th, I, pred, Rs, S3s, S1s, thw, angu, out)
    f = newfig([1100 340]);
    tiledlayout(f, 1, 3, 'Padding', 'compact', 'TileSpacing', 'compact');

    nexttile;
    plot(rad2deg(th), pred/I(1), '-', 'LineWidth', 1.4); hold on
    plot(rad2deg(th), I/I(1), 'o', 'MarkerSize', 4);
    xlabel('analyzer angle (deg)'); ylabel('I / I_0');
    title('Malus: cos^2\theta'); grid on; xlim([0 180]);
    legend({'cos^2\theta', 'MACOS'}, 'Location', 'north', 'Box', 'off');

    nexttile;
    plot(Rs, cos(2*pi*Rs), '-', 'LineWidth', 1.4); hold on
    plot(Rs, S1s, 's', 'MarkerSize', 4);
    plot(Rs, -sin(2*pi*Rs), '-', 'LineWidth', 1.4);
    plot(Rs, S3s, 'o', 'MarkerSize', 4);
    xlabel('retardance (waves)'); ylabel('S_i / S_0');
    title('retarder, fast axis at 45\circ'); grid on;
    legend({'cos2\pi R', 'S_1', '-sin2\pi R', 'S_3'}, ...
           'Location', 'southeast', 'Box', 'off');

    nexttile;
    plot(rad2deg(thw), rad2deg(2*thw), '-', 'LineWidth', 1.4); hold on
    plot(rad2deg(thw), rad2deg(angu), 'o', 'MarkerSize', 4);
    xlabel('half-wave plate angle (deg)'); ylabel('output orientation (deg)');
    title('HWP: rotation by 2\theta'); grid on;
    legend({'2\theta', 'MACOS'}, 'Location', 'southeast', 'Box', 'off');

    savefig_(f, out);
end

function V = radiometric_gates(V, ~)
%RADIOMETRIC_GATES  Section 7 -- coated-Refractor transmission radiometry.
%   Mirrors tPolRadiometric.  The engine numbers come from the two Refract
%   fixtures; every truth value is the Abeles characteristic-matrix result
%   computed by radio_T/radio_t below, written from Macleod ch. 2 and not
%   from elemsub.F.
    fprintf('polval: coated-Refractor radiometric gates\n');
    rxN = polval_rx('Rx_Refract.in');    PRE = 2; F1 = 3; F2 = 4; DET = 6;
    rx45 = polval_rx('Rx_Refract45.in'); PRE45 = 2; F45 = 3;
    nA = 1.0; nG = 1.5; lam0 = 1.0e-6;
    nC = 1.38; dC = lam0/(4*nC);          % quarter-wave MgF2 at lam0
    th45 = pi/4;

    % ---- G7.1 the INCUMBENT convention: uncoated |t|^2 IS the power
    % transmittance.  Establishes from the textbook what the coated branch
    % is being brought to, instead of asserting it.
    macos.load_rx(rxN);  macos.polarization('on', 'Ex', [1 0], 'Ey', [0 0]);
    Tn = radio_T2(PRE, F1);
    V = addval(V, 'G71_UNC_NORMAL', ...
        abs(Tn/radio_T(nA, [], [], nG, lam0, 0, 's') - 1), '%.3e', 'relative', ...
        'uncoated transmission is the power transmittance (normal incidence)', ...
        'tPolRadiometric/test_uncoated_transmission_is_the_power_transmittance');

    [Tp, Ts] = radio_45(rx45, PRE45, F45, []);
    Tpa = radio_T(nA, [], [], nG, lam0, th45, 'p');
    Tsa = radio_T(nA, [], [], nG, lam0, th45, 's');
    V = addval(V, 'G71_UNC_P', abs(Tp/Tpa - 1), '%.3e', 'relative', ...
        'uncoated p transmittance at 45 deg', ...
        'tPolRadiometric/test_uncoated_transmission_oblique_s_and_p');
    V = addval(V, 'G71_UNC_S', abs(Ts/Tsa - 1), '%.3e', 'relative', ...
        'uncoated s transmittance at 45 deg', ...
        'tPolRadiometric/test_uncoated_transmission_oblique_s_and_p');
    V = addval(V, 'G71_SP_SPLIT', abs(Tpa - Tsa), '%.4f', 'absolute', ...
        'non-vacuity: s and p transmittances are genuinely split at 45 deg', ...
        'tPolRadiometric/test_uncoated_transmission_oblique_s_and_p');

    % ---- G7.2 an index-matched layer IS a bare interface.  The headline.
    macos.load_rx(rxN);  macos.polarization('on', 'Ex', [1 0], 'Ey', [0 0]);
    Tu = radio_T2(PRE, F1);
    macos.coating(F1, 'index', nG, 'extinc', 0, 'thickness', 1.0e-7);
    Tc = radio_T2(PRE, F1);
    V = addval(V, 'G72_IDX_NORMAL', abs(sqrt(Tc/Tu) - 1), '%.3e', 'relative', ...
        'index-matched layer reproduces the bare interface (normal incidence)', ...
        'tPolRadiometric/test_index_matched_layer_equals_bare_interface_normal');

    [Tp, Ts, Tpu, Tsu] = radio_45(rx45, PRE45, F45, {nG, 0, 1.0e-7});
    V = addval(V, 'G72_IDX_P', abs(sqrt(Tp/Tpu) - 1), '%.3e', 'relative', ...
        'index-matched layer, p at 45 deg (the cos_sub/cos_inc half of the factor)', ...
        'tPolRadiometric/test_index_matched_layer_equals_bare_interface_oblique');
    V = addval(V, 'G72_IDX_S', abs(sqrt(Ts/Tsu) - 1), '%.3e', 'relative', ...
        'index-matched layer, s at 45 deg', ...
        'tPolRadiometric/test_index_matched_layer_equals_bare_interface_oblique');

    % ---- and the same claim on the DIFFRACTION GRID (propsub's chain).
    macos.load_rx(rxN);  macos.polarization('on', 'Ex', [1 0], 'Ey', [0 0]);
    macos.trace(DET);  Iu = sum(macos.intensity(DET), 'all');
    macos.coating(F1, 'index', nG, 'extinc', 0, 'thickness', 1.0e-7);
    macos.trace(DET);  Ic = sum(macos.intensity(DET), 'all');
    V = addval(V, 'G72_GRID', abs(Ic/Iu - 1), '%.3e', 'relative', ...
        'index-matched layer at the detector plane (second dispatch chain)', ...
        'tPolRadiometric/test_index_matched_layer_at_the_detector_plane');

    % ---- G7.3 a real stack: quarter-wave MgF2 on glass vs the textbook T.
    macos.load_rx(rxN);  macos.polarization('on', 'Ex', [1 0], 'Ey', [0 0]);
    macos.coating(F1, 'index', nC, 'extinc', 0, 'thickness', dC);
    Tm = radio_T2(PRE, F1);
    V = addval(V, 'G73_MGF2_NORMAL', ...
        abs(Tm/radio_T(nA, nC, dC, nG, lam0, 0, 's') - 1), '%.3e', 'relative', ...
        'MgF2 quarter-wave on glass vs the characteristic-matrix T (normal)', ...
        'tPolRadiometric/test_mgf2_quarterwave_normal_incidence');
    V = addval(V, 'G73_MGF2_GAIN', Tm, '%.4f', 'transmittance', ...
        'non-vacuity: the AR stack really transmits (bare glass is 0.96)', ...
        'tPolRadiometric/test_mgf2_quarterwave_normal_incidence');

    [Tp, Ts] = radio_45(rx45, PRE45, F45, {nC, 0, dC});
    V = addval(V, 'G73_MGF2_P', ...
        abs(Tp/radio_T(nA, nC, dC, nG, lam0, th45, 'p') - 1), '%.3e', 'relative', ...
        'MgF2 stack, p at 45 deg -- both halves of the factor at once', ...
        'tPolRadiometric/test_mgf2_quarterwave_45deg_s_and_p');
    V = addval(V, 'G73_MGF2_S', ...
        abs(Ts/radio_T(nA, nC, dC, nG, lam0, th45, 's') - 1), '%.3e', 'relative', ...
        'MgF2 stack, s at 45 deg', ...
        'tPolRadiometric/test_mgf2_quarterwave_45deg_s_and_p');

    % ---- G7.4 air-to-air closure.  MIXED plate (front coated, back bare):
    % the both-coated case is invariant under the landing because the two
    % factors cancel, so it cannot serve as the defect gate -- it is
    % measured separately, below, as the composition identity it is.
    macos.load_rx(rxN);  macos.polarization('on', 'Ex', [1 0], 'Ey', [0 0]);
    macos.coating(F1, 'index', nC, 'extinc', 0, 'thickness', dC);
    Ttot = radio_T2(PRE, F2);
    T1 = radio_T(nA, nC, dC, nG, lam0, 0, 's');
    T2 = radio_T(nG, [], [], nA, lam0, 0, 's');
    V = addval(V, 'G74_CLOSURE', abs(Ttot/(T1*T2) - 1), '%.3e', 'relative', ...
        'air-to-air power closure, coated front face + bare back face', ...
        'tPolRadiometric/test_air_to_air_power_closure_mixed_plate');

    macos.load_rx(rxN);  macos.polarization('on', 'Ex', [1 0], 'Ey', [0 0]);
    macos.coating(F1, 'index', nC, 'extinc', 0, 'thickness', dC);
    macos.coating(F2, 'index', nC, 'extinc', 0, 'thickness', dC);
    amp = sqrt(radio_T2(PRE, F2));
    t1 = radio_t(nA, nC, dC, nG, lam0, 0, 's');
    t2 = radio_t(nG, nC, dC, nA, lam0, 0, 's');
    V = addval(V, 'G74_TELESCOPE', abs(amp/abs(t1*t2) - 1), '%.3e', 'relative', ...
        'fully coated plate: the two radiometric factors cancel air-to-air', ...
        'tPolRadiometric/test_air_to_air_factors_telescope');

    % ---- G7.5 what the landing did NOT move: a common real scalar cancels
    % in t_p/t_s, so the transmitted polarization STATE is untouched.
    macos.load_rx(rx45);
    macos.polarization('on', 'Ex', [1/sqrt(2) 0], 'Ey', [1/sqrt(2) 0]);
    macos.coating(F45, 'index', nC, 'extinc', 0, 'thickness', dC);
    g = radio_geom(PRE45, F45);
    sh = cross(g.ihat, g.nhat, 2);  sh = sh ./ vecnorm(sh, 2, 2);
    pih = cross(sh, g.ihat, 2);     prh = cross(sh, g.rhat, 2);
    rat = abs((sum(g.Eout.*prh,2)./sum(g.Ein.*pih,2)) ./ ...
              (sum(g.Eout.*sh, 2)./sum(g.Ein.*sh, 2)));
    tp = radio_t(nA, nC, dC, nG, lam0, th45, 'p');
    ts = radio_t(nA, nC, dC, nG, lam0, th45, 's');
    V = addval(V, 'G75_STATE', abs(median(rat)/abs(tp/ts) - 1), '%.3e', 'relative', ...
        'transmitted p/s ratio equals the factor-FREE Fresnel ratio', ...
        'tPolRadiometric/test_scalar_factor_leaves_the_polarization_state_alone');
    V = addval(V, 'G75_STATE_SPLIT', abs(abs(tp/ts) - 1), '%.4f', 'absolute', ...
        'non-vacuity: that ratio is not 1', ...
        'tPolRadiometric/test_scalar_factor_leaves_the_polarization_state_alone');

    % ---- G7.6 the factor lives inside ifPol.
    macos.load_rx(rxN);  macos.polarization('off');
    macos.trace(DET);  Wu = macos.opd();  Iu = macos.intensity(DET);
    macos.coating(F1, 'index', nC, 'extinc', 0, 'thickness', dC);
    macos.coating(F2, 'index', nC, 'extinc', 0, 'thickness', dC);
    macos.trace(DET);  Wc = macos.opd();  Ic = macos.intensity(DET);
    V = addval(V, 'G76_POLOFF', double(isequal(Wu,Wc) && isequal(Iu,Ic)), ...
        '%d', 'bool', 'polarization-off is bit-identical with and without a coating', ...
        'tPolRadiometric/test_pol_off_is_untouched_by_the_coating');

    % ---- G7.7 a wavelength sweep: a real scalar cannot have created,
    % shifted or flattened the quarter-wave interference structure.
    lam = [0.6 0.8 1.0 1.2 1.5] * 1e-6;
    macos.load_rx(rxN);  macos.polarization('on', 'Ex', [1 0], 'Ey', [0 0]);
    macos.coating(F1, 'index', nC, 'extinc', 0, 'thickness', dC);
    Te = zeros(size(lam));  Ta = zeros(size(lam));
    for i = 1:numel(lam)
        macos.set_src_wvl(lam(i));
        Te(i) = radio_T2(PRE, F1);
        Ta(i) = radio_T(nA, nC, dC, nG, lam(i), 0, 's');
    end
    macos.set_src_wvl(lam0);
    V = addval(V, 'G77_LAMBDA', max(abs(Te./Ta - 1)), '%.3e', 'relative', ...
        'T(lambda) tracks the characteristic-matrix T across the sweep', ...
        'tPolRadiometric/test_quarterwave_structure_survives_the_scalar_factor');
    r = Te ./ Ta;
    V = addval(V, 'G77_FLATNESS', max(r) - min(r), '%.3e', 'relative', ...
        'engine/analytic ratio is wavelength-INDEPENDENT, as a real scalar must be', ...
        'tPolRadiometric/test_quarterwave_structure_survives_the_scalar_factor');
    V = addval(V, 'G77_CONTRAST', max(Ta) - min(Ta), '%.4f', 'transmittance', ...
        'non-vacuity: there IS quarter-wave structure to preserve', ...
        'tPolRadiometric/test_quarterwave_structure_survives_the_scalar_factor');
end

function T = radio_T2(iPre, iFace)
%RADIO_T2  Engine power transmittance |E_face|^2/|E_pre|^2 between stations.
%   Formed against the MEASURED incident field at iPre, never an assumed
%   source amplitude (the IFO slice-1 finding-2 rule).  macos.ray_field
%   returns the CURRENT RayE -- iElt selects only the surface normal -- so
%   each station gets its own trace.
    macos.trace(iPre);   a = macos.ray_field(iPre);
    macos.trace(iFace);  b = macos.ray_field(iFace);
    m  = (a.status == 0) & (b.status == 0);
    Aa = sqrt(abs(a.Ex(m)).^2 + abs(a.Ey(m)).^2 + abs(a.Ez(m)).^2);
    Ab = sqrt(abs(b.Ex(m)).^2 + abs(b.Ey(m)).^2 + abs(b.Ez(m)).^2);
    T  = median(Ab ./ Aa).^2;
end

function g = radio_geom(iPre, iFace)
%RADIO_GEOM  Fields at both stations plus the geometry, read from the engine.
    macos.trace(iPre);   a = macos.ray_field(iPre);
    macos.trace(iFace);  b = macos.ray_field(iFace);
    m = (a.status == 0) & (b.status == 0);
    g.ihat = [a.kx(m) a.ky(m) a.kz(m)];
    g.rhat = [b.kx(m) b.ky(m) b.kz(m)];
    g.nhat = [b.nx(m) b.ny(m) b.nz(m)];
    g.Ein  = [a.Ex(m) a.Ey(m) a.Ez(m)];
    g.Eout = [b.Ex(m) b.Ey(m) b.Ez(m)];
end

function [Tp, Ts, Tpu, Tsu] = radio_45(rx45, iPre, iFace, coat)
%RADIO_45  p and s transmittance on the 45-degree fixture, optionally with a
%   coating; also returns the UNCOATED pair measured in the same run so a
%   caller can form the coated/uncoated ratio without a second load.
%   On that fixture x^ is exactly p_i^ and y^ exactly s^ -- see its header.
    Tpu = 0; Tsu = 0;
    for k = 1:2
        macos.load_rx(rx45);
        if k == 1, macos.polarization('on', 'Ex', [1 0], 'Ey', [0 0]);
        else,      macos.polarization('on', 'Ex', [0 0], 'Ey', [1 0]);
        end
        Tu = radio_T2(iPre, iFace);
        if isempty(coat)
            Tc = Tu;
        else
            macos.coating(iFace, 'index', coat{1}, 'extinc', coat{2}, ...
                                 'thickness', coat{3});
            Tc = radio_T2(iPre, iFace);
        end
        if k == 1, Tp = Tc; Tpu = Tu; else, Ts = Tc; Tsu = Tu; end
    end
end

function T = radio_T(n0, nL, dL, nsub, lambda, theta0, pol)
%RADIO_T  Abeles characteristic-matrix POWER transmittance of a stack.
%   Macleod, "Thin-Film Optical Filters", ch. 2 (= Born & Wolf 1.6.4),
%   typed from the tilted-admittance definitions -- NOT transcribed from
%   elemsub.F, which would make the check circular in exactly the quantity
%   it exists to test (the lesson of REVIEW_POL_SP_SIGN_2026-07-27.md).
%   Zero layers -> the identity matrix -> the bare Fresnel interface.
    [B, C, e0, es] = radio_charmat(n0, nL, dL, nsub, lambda, theta0, pol);
    T = 4*e0*real(es) / abs(e0*B + C)^2;
end

function t = radio_t(n0, nL, dL, nsub, lambda, theta0, pol)
%RADIO_T  Composed FIELD amplitude coefficient, ORDINARY Fresnel sense.
%   Macleod's 2*eta0/(eta0*B+C) is the TANGENTIAL coefficient; for p the
%   tangential component is E*cos(theta), so it exceeds the ordinary t_p by
%   cos(theta_sub)/cos(theta_inc) -- 1.2472 at 45 deg into n=1.5, i.e. the
%   size of a plausible radiometric error, so it must be converted, not
%   waved off.  (T is unaffected either way.)
    [B, C, e0, ~] = radio_charmat(n0, nL, dL, nsub, lambda, theta0, pol);
    t = 2*e0 / (e0*B + C);
    if strcmp(pol, 'p')
        s0 = n0*sin(theta0);
        t = t * sqrt(1 - (s0/n0)^2) / sqrt(1 - (s0/nsub)^2);
    end
end

function [B, C, e0, es] = radio_charmat(n0, nL, dL, nsub, lambda, theta0, pol)
    nL = nL(:).';  dL = dL(:).';
    s0 = n0*sin(theta0);                          % Snell invariant
    et = @(n) radio_eta(n, s0, pol);
    e0 = et(n0);  es = et(nsub);
    M = eye(2);
    for j = 1:numel(nL)                           % OUTERMOST layer first
        cj  = sqrt(1 - (s0/nL(j))^2);
        dlt = 2*pi*nL(j)*dL(j)*cj/lambda;
        ej  = et(nL(j));
        M   = M * [cos(dlt), 1i*sin(dlt)/ej; 1i*ej*sin(dlt), cos(dlt)];
    end
    BC = M * [1; es];  B = BC(1);  C = BC(2);
end

function e = radio_eta(n, s0, pol)
    c = sqrt(1 - (s0/n)^2);
    if strcmp(pol, 's'), e = n*c; else, e = n/c; end
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
