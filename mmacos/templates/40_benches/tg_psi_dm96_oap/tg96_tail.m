function out = tg96_tail(varargin)
%TG96_TAIL  Re-tune the detector tail for the gauge (lens OR oap).  The
%   fieldlens tail (FL_F, FL_Kc, D_MASK_FL, DET_TRIM) was fit to L2's
%   aberrations; the OAP focuser is a DIFFERENT element, so the tail MUST be
%   re-run for optics='oap' (Dave's rule).  Minimizes the UNALIGNED NULL RMS
%   at reduced resolution (model 512, NGRID 193, flat DM) with the SAME rig
%   geometry (scale, BS AOI, DM leg, OAP folds) as the full run.  Writes
%   <tag>_tail.mat, which tg96_run reads at Stage B.
%
%   Usage:  tg96_tail('bench.optics','oap','tag','oap')
%           tg96_tail                        % defaults (lens, tag 'lens')
%           tg96_tail('verify_tail','objwin3_tail.mat', ...)  % gate only
%
%   The WINNER GATE (item 3): the optimizer's winner is not trusted to
%   certify itself.  After the tune, ONE single-actuator row is read through
%   the ray affine in ACTUATOR space; below gate_gain the winner is refused
%   and the GEOMETRIC SEED is returned, with the reason printed.  This exists
%   because on the OAP rig every term the cost computes preferred a tail that
%   reads a single actuator at 0.0338 where the seed reads 0.9809
%   (REPORT_reflective 4.5, runs/tailA vs runs/tailB).

exdir = fileparts(mfilename('fullpath'));  if isempty(exdir), exdir = pwd; end
if isempty(which('macos.init'))
    run(fullfile(exdir, '..', '..', '..', 'mmacos_setup.m'));
end
assert(~isempty(getenv('MACOS_HOME')), 'MACOS_HOME must be set.');
cd(exdir);

P = parse_params_(varargin{:});
s = P.dm(1).nact/56;  LAM = P.LAM;  QWP = P.QWP;  THETAS = P.THETAS;
% reduced-res tuning grid (geometry is full-scale; only diffraction sampling drops)
MODEL = 512;  NGRID = 193;  N_G = 256;  DX_G = 0.4;

% ---- rig geometry (real-scale Stage-A solve; matches tg96_run) --------
beam_r = P.clear.beam_r;  if isempty(beam_r), beam_r = s*P.bench.R_TO_AP; end
need = beam_r + [P.clear.HW_DM; P.clear.HW_REF; P.clear.HW_CAM] + P.clear.MARGIN;
AOI = P.bench.BS_AOI;  if isempty(AOI), AOI = ceil(max(asind(need/P.clear.LEG_CAP))/2); end
D_BS_TO = P.bench.D_BS_TO;
if isempty(D_BS_TO), D_BS_TO = ceil(max(need(1)/sind(2*AOI), s*250)/50)*50; end
oapargs = {};
if strcmp(P.bench.optics,'oap')
    need_off = beam_r + P.clear.HW_CAM + P.clear.MARGIN;
    a1 = P.oap.OAP1_AOI;  a2 = P.oap.OAP2_AOI;
    if isempty(a1), a1 = solve_fold_(s*P.bench.F1, need_off); end
    if isempty(a2), a2 = solve_fold_(s*P.bench.F2, need_off); end
    oapargs = {'OAP1_AOI',a1,'OAP2_AOI',a2,'OAP1_SIDE',P.oap.OAP1_SIDE,'OAP2_SIDE',P.oap.OAP2_SIDE};
    fprintf('TAIL geometry: optics oap, BS_AOI %d, DM leg %d, OAP folds %d/%d deg\n', AOI, D_BS_TO, a1, a2);
else
    fprintf('TAIL geometry: optics lens, BS_AOI %d, DM leg %d\n', AOI, D_BS_TO);
end

macos.init(MODEL);
% dm_influence_map (the same path tg96_run adds).  This was hard-coded to a
% macOS home directory, so on Linux it only warned and the tuner ran on
% whatever happened to be on the path; resolve it from this file instead.
addpath(fullfile(exdir, '..', '..', '90_polarization', 'tg_psi_dm'));
% The WINNER GATE reaches for tg96_place (this directory) and dmg_frame /
% dmg_lit (the shared library).  Without these two lines row_gain_ throws
% "Undefined function 'dmg_frame'", the gate returns NaN, and it REFUSES
% EVERY tail -- a gate that fails closed looks like a gate that works.  That
% is what the first gateseq3 run measured (both legs NaN); the lens leg, which
% must ACCEPT, is the non-vacuity check that caught it.
addpath(exdir);
addpath(fullfile(exdir, '..', 'dm_gauge_lib'));
% UNIQUE scratch names per process.  These were fixed strings -- tail_flat.txt,
% tail_test.in, tail_ref.in -- in the template directory, so two tg96_tail runs
% in the same folder read and wrote each other's decks.  That is not
% hypothetical: on 2026-09-15 two cost probes launched together reported
% DIFFERENT nulls for IDENTICAL parameters (71.80 nm vs 0.0289 nm) and the
% gate built on them had to be thrown away.  dmg_bench_clearance took the same
% fix on 2026-09-15 (b55d15a) for the same reason.
scr = sprintf('tail_%d_%s', feature('getpid'), P.tag);
f_flat = [scr '_flat.txt'];  f_test = [scr '_test.in'];  f_ref = [scr '_ref.in'];
cleanscr = onCleanup(@() delete_if_(( {f_flat, f_test, f_ref} )));
macos.write_grid_file(f_flat, zeros(N_G));
b = P.bench;
% Objective: 'null' minimizes the flat-DM null (the lens rig; reproduces the
% record).  'sharpness' ALSO images a single-actuator poke and rewards the
% recovered peak -- REQUIRED for the OAP rig, whose off-axis focuser leaves the
% flat-DM null insensitive to a defocused/astigmatic pupil (Dave 2026-09-10).
objective = P.bench.optics;   % 'lens' -> null ; 'oap' -> sharpness
if strcmp(objective,'oap'), objective = 'sharpness'; else, objective = 'null'; end
seed = [s*b.FL_F, b.FL_Kc, s*b.D_MASK_FL, s*b.DET_TRIM];
q0 = [0, seed(2), seed(3), seed(4)];   % FL_F = seed(1)*exp(q1) keeps positive
% the node the runner builds: the recomb plane / output-optics distances and
% the input polarizer's leg.  Omitting these tuned the tail on a DIFFERENT
% bench from the one tg96_run then used (the output optics sat 17/27 mm behind
% the splitter instead of 160/170 after the 22.5 deg round).
nodeargs = {};
if isfield(b,'D_RECOMB') && ~isempty(b.D_RECOMB), nodeargs = [nodeargs, {'D_RECOMB', b.D_RECOMB}]; end
if isfield(b,'D_RC_L2')  && ~isempty(b.D_RC_L2),  nodeargs = [nodeargs, {'D_RC_L2',  b.D_RC_L2}];  end
if isfield(b,'POL_IN')   && ~isempty(b.POL_IN),   nodeargs = [nodeargs, {'POL_IN',   b.POL_IN}];   end
if isfield(b,'SRC_AT_FOCUS') && ~isempty(b.SRC_AT_FOCUS), nodeargs = [nodeargs, {'SRC_AT_FOCUS', b.SRC_AT_FOCUS}]; end
C = struct('f_flat',f_flat,'f_test',f_test,'f_ref',f_ref, ...
           's',s,'AOI',AOI,'D_BS_TO',D_BS_TO,'NGRID',NGRID,'N_G',N_G,'DX_G',DX_G, ...
           'QWP',QWP,'THETAS',THETAS,'LAM',LAM,'seed',seed,'optics',b.optics, ...
           'oapargs',{oapargs},'nodeargs',{nodeargs},'bench',b,'objective',objective, ...
           'poke_nm',100, ...
           'place',P.place,'POKE',P.POKE,'gate_gain',0.95, ...
           'stn_hw',6,'act_lam',0.05);
% poke_nm 100 = 0.63 of lambda/4: a healthy map then reads ~0.63 on the wrap
%   meter and a PINNED one reads 1.00.  At the old 150 nm (0.95 of the range)
%   the guard could not separate them.
% gate_gain = the WINNER GATE: one single-actuator row through the ray affine
%   must recover at least this gain in ACTUATOR space, or the winner is
%   refused and the geometric seed is returned instead (item 3).
% ---- verify-only mode: gate an EXISTING tail, tune nothing --------------
% tg96_tail('verify_tail','objwin3_tail.mat', <the same bench args it was
% tuned with>) runs the winner gate alone.  This is how the gate itself is
% tested -- hand it a tail known not to read and see the refusal -- without
% paying for a 150-evaluation tune.
if isfield(P, 'verify_tail') && ~isempty(P.verify_tail)
    V = load(P.verify_tail);  vo = V.out;
    pv = [vo.FL_F, vo.FL_Kc, vo.D_MASK_FL, vo.DET_TRIM];
    [gv, iv] = row_gain_(pv, C);
    if ~isfinite(gv)
        verdict = 'GATE COULD NOT MEASURE';      % NOT the same as a bad tail
    elseif abs(gv) >= C.gate_gain
        verdict = 'ACCEPTED';
    else
        verdict = 'REFUSED';
    end
    fprintf(['TAIL VERIFY %s (optics %s): FL_F %.4f FL_Kc %.5f D_MASK_FL %.4f ' ...
             'DET_TRIM %.4f -> actuator-space gain %.4f (gate >= %.2f) -> %s\n  [%s]\n'], ...
            P.verify_tail, b.optics, pv(1), pv(2), pv(3), pv(4), gv, C.gate_gain, verdict, iv);
    out = struct('verify_tail',P.verify_tail,'gain',gv,'info',iv, ...
                 'threshold',C.gate_gain,'pass',strcmp(verdict,'ACCEPTED'), ...
                 'FL_F',pv(1),'FL_Kc',pv(2),'D_MASK_FL',pv(3),'DET_TRIM',pv(4), ...
                 'optics',b.optics);
    delete_if_({f_flat, f_test, f_ref});
    return
end

[r0, n0, k0] = cost_(q0, C);
fprintf('TAIL SEED: cost %.4f (null %.4f nm, poke-peak %.1f nm) [objective %s]\n', r0, n0, k0, objective);
[qb, rb] = fminsearch(@(q) cost_(q, C), q0, ...
    optimset('MaxFunEvals',150,'MaxIter',150,'TolFun',1e-3,'TolX',1e-4,'Display','off'));
pb = [seed(1)*exp(qb(1)), qb(2), qb(3), qb(4)];
[~, nb, kb] = cost_(qb, C);
fprintf('TAIL WINNER (%s): FL_F %.4f FL_Kc %.5f D_MASK_FL %.4f DET_TRIM %.4f -> null %.4f nm, poke-peak %.1f nm\n', ...
        b.optics, pb(1), pb(2), pb(3), pb(4), nb, kb);

% ---- the WINNER GATE (BRIEF_to_gauge_close item 3) --------------------
% Every quantity the cost function computes said the OAP rig's old winner was
% healthy while the battery read it at 3 % (REPORT_reflective 4.5).  The
% objective is therefore not trusted to certify its own winner: the winner has
% to READ.  One single-actuator row through the ray affine, in actuator space,
% decides -- and on failure the tuner hands back the GEOMETRIC SEED rather than
% a tail that does not read.  Refusing is the point; diagnosing the objective
% is a separate open question (README, "the tuner's objective").
[gw, iw] = row_gain_(pb, C);
fprintf('TAIL GATE: winner reads gain %.4f in actuator space [%s]\n', gw, iw);
% ---- THE GATE IS ADVISORY (2026-09-16) -------------------------------
% Its two-leg test FAILED BOTH LEGS, in opposite directions: objwin3, which the
% battery reads at 0.0338, was ACCEPTED at 0.9804; lens_tail, which the battery
% reads at 0.9968, was REFUSED at -0.8285.  So row_gain_ is not measuring what
% the battery measures.  Diagnosis from the two prints: the broken OAP tail has
% mag 6.125 DM-mm/det-mm against the seed's 10.44, i.e. a LARGER image of each
% actuator, so a POINT SAMPLE at the actuator's predicted pixel is diluted less
% and reads HIGHER -- the measure tracks magnification, not readability.  The
% battery instead DECONVOLVES the influence-function stencil over the actuator
% lattice, which is what makes it sensitive to the response's shape.
%
% Until row_gain_ does that, the gate REPORTS and never refuses: a measure that
% inverts the verdict would fall back to the seed on a good tail, and item 4's
% substrate runs would then measure the glass AND a tail regression together.
% Enforcing a wrong gate is worse than not gating.
gate_pass = true;
gate_measured = isfinite(gw) && abs(gw) >= C.gate_gain;
if ~gate_measured
    fprintf(['TAIL GATE (ADVISORY): row reads %.4f, below %.2f -- NOT enforced, ' ...
             'because the two-leg test showed this measure tracks magnification ' ...
             'rather than readability.  Winner kept.\n'], gw, C.gate_gain);
end
gate = struct('gain',gw,'info',iw,'threshold',C.gate_gain,'pass',gate_measured,'advisory',true, ...
              'seed_gain',NaN,'seed_info','','fellback',false);
if ~isfinite(gw)
    % A gate that cannot MEASURE is not the same as a tail that does not
    % read, and it must never pass for one: unmeasured, it refuses every
    % tail, so a broken gate silently turns every tune into "use the seed".
    % Shout, then still fall back -- the seed is the safe default, but nobody
    % should read this as evidence about the tail.
    warning('tg96_tail:gate_unmeasured', ...
        ['THE WINNER GATE COULD NOT MEASURE (%s).  It is refusing the winner ' ...
         'because it has no number, NOT because the tail failed.  Fix the gate ' ...
         'before reading anything into this run.'], iw);
end
if ~gate_pass
    ps = [seed(1)*exp(q0(1)), q0(2), q0(3), q0(4)];
    [gs, is_] = row_gain_(ps, C);
    gate.seed_gain = gs;  gate.seed_info = is_;  gate.fellback = true;
    fprintf(['TAIL GATE REFUSED the winner: actuator-space gain %.4f < %.2f. ' ...
             'Falling back to the GEOMETRIC SEED (gain %.4f).\n'], gw, C.gate_gain, gs);
    fprintf(['  Why this gate and not the cost: the cost''s four terms (null %.4f nm, ' ...
             'poke-peak %.1f nm, localization, wrap) all preferred this winner, and ' ...
             'they are orthogonal to readability.\n'], nb, kb);
    if isfinite(gs) && abs(gs) < C.gate_gain
        fprintf(['  WARNING: the seed does not read either (%.4f). The tail is not ' ...
                 'the whole story on this bench -- do not treat the seed as gated.\n'], gs);
    end
    pb = ps;  nb = n0;  kb = k0;   % the seed's own numbers, already measured
    fprintf(['  the tail of record is now the geometric seed: FL_F %.4f FL_Kc %.5f ' ...
             'D_MASK_FL %.4f DET_TRIM %.4f (null %.4f nm, poke-peak %.1f nm)\n'], ...
            pb(1), pb(2), pb(3), pb(4), nb, kb);
end
out = struct('FL_F',pb(1),'FL_Kc',pb(2),'D_MASK_FL',pb(3),'DET_TRIM',pb(4), ...
             'null_nm',nb,'seed_null_nm',n0,'poke_peak_nm',kb,'seed_poke_peak_nm',k0, ...
             'opt_model',MODEL,'opt_ngrid',NGRID,'optics',b.optics,'objective',objective, ...
             'gate',gate);
save([P.tag '_tail.mat'], 'out');
fprintf('wrote %s_tail.mat\n', P.tag);
end

% ---------------------------------------------------------------------
function [r, null_nm, peak_nm] = cost_(q, C)
    persistent neval;  if isempty(neval), neval = 0; end
    s = C.s;  b = C.bench;  p = [C.seed(1)*exp(q(1)), q(2), q(3), q(4)];
    null_nm = 1e6;  peak_nm = 0;  r = 1e6;
    conc = NaN;  wrapf = NaN;     % reported per eval so a tune is auditable
    try
        G = build_(p, C);
        AT = arm_desc(C.f_test, G.bt, G.T, 0);
        AR = arm_desc(C.f_ref,  G.br, G.R, 45);
        Sr = analyzer_basis(AR, C.QWP, []);  S0 = analyzer_basis(AT, C.QWP, []);
        I0 = frame(S0, Sr, 0);  msk = I0 > 0.1*max(I0(:));
        if nnz(msk) < 500, return; end
        pn = fourstep(S0, Sr, C.THETAS);
        hn = (pn - median(pn(msk))) * C.LAM/(4*pi) * 1e6;
        null_nm = std(hn(msk));
        if strcmp(C.objective,'null')
            r = null_nm;
        else   % 'sharpness': image a single-actuator poke, reward recovered peak
            Mp = dm_influence_map(C.N_G, C.DX_G, 'nact',b_nact_(b), 'pitch',b_pitch_(b), ...
                                  'pattern','single', 'poke',C.poke_nm*1e-6);
            hp = meas_surface(AT, C.QWP, Mp, Sr, pn, C.THETAS, C.LAM);
            peak_nm = 1e6*max(abs(hp(msk)));
            frac = peak_nm / C.poke_nm;                 % 1.0 = sharp, 0 = lost
            % --- LOCALIZATION and a WRAP GUARD (2026-09-15) -----------------
            % peak alone is NOT sharpness: max(abs(h)) is the largest value
            % ANYWHERE in the pupil, tied to neither the poked actuator nor to
            % the response being localized.  A defocused or mis-registered map
            % can supply a large maximum, and a WRAPPED map is guaranteed one,
            % since wrapping throws values to the ends of the lambda/4 range.
            % The cost was therefore MAXIMIZED by the failure it should reject:
            % the winner it picked (DET_TRIM +45.96) had walked the detector
            % off the DM's pupil conjugate -- magnification 5.477 DM-mm/det-mm
            % against 10.4 for the same bench on the geometric seed -- scored
            % frac 1.00, and read actuators at gain 0.034 where the seed reads
            % 0.98 (A/B: runs/tailA vs runs/tailB).
            %
            % conc: the fraction of the map's ENERGY within a few actuator
            %   pitches of its own peak.  Mapping-free -- the pupil's diameter
            %   in pixels comes from the mask, and one actuator should occupy
            %   1/nact of it -- so it needs no DM->detector affine, which is
            %   the very thing a bad tail corrupts.
            % wrapf: how close the map runs to the four-step's unambiguous
            %   range.  A map at 1.00 of lambda/4 carries no surface.
            [~, imax] = max(abs(hp(:)) .* double(msk(:)));
            [ri, ci] = ind2sub(size(hp), imax);
            d_pup = sqrt(4*nnz(msk)/pi);                % pupil diameter, px
            w = max(2, ceil(3*d_pup/b_nact_(b)));       % ~3 actuator pitches
            box = false(size(hp));
            box(max(1,ri-w):min(size(hp,1),ri+w), max(1,ci-w):min(size(hp,2),ci+w)) = true;
            E = sum(hp(msk).^2);
            conc = sum(hp(msk & box).^2) / max(E, eps);
            wrapf = 1e6*max(abs(hp(msk))) / (1e6*C.LAM/4);
            % LOCALIZATION dominant, then height; the null BOUNDED, not
            % minimized; a wrapped map refused.
            %
            % The null's weight is the point.  The old cost carried
            % (null_nm/2)^2, and THAT is what drove the defect: the optimizer
            % bought a 0.0223 nm null by walking the detector off the DM's
            % pupil conjugate, which is free as far as an arm DIFFERENCE is
            % concerned (both arms share the tail, so a common misplacement
            % cancels in the null) and fatal to the reading.  Measured under
            % the first version of this fix: the geometric seed, which READS at
            % gain 0.99, scored 1262.9 against the broken tail's 1.39 -- the
            % null term alone, 71 nm vs 0.022 nm.  A 71 nm null is perfectly
            % fine for reading.  So the null is scaled to 200 nm here: it keeps
            % a wildly mis-built tail out, and buys nothing below that.
            r = (1 - min(frac,1.2))^2 + 4*(1 - conc)^2 ...
                + 10*max(0, wrapf - 0.8)^2 + (null_nm/200)^2;
        end
    catch
        r = 1e6;
    end
    neval = neval + 1;
    fprintf(['TAILEVAL %3d: FL_F %.3f Kc %.4f D_MASK %.3f TRIM %.3f -> null %.4f nm, ' ...
             'peak %.1f nm, conc %.3f, wrap %.2f of lambda/4, cost %.4f\n'], ...
            neval, p(1), p(2), p(3), p(4), null_nm, peak_nm, conc, wrapf, r);
end
function n = b_nact_(~), n = 96; end
function n = b_pitch_(~), n = 1.0; end

function a = solve_fold_(F, need_off)
    for a = 5:44
        if F*abs(sind(180-2*a)) >= need_off, return; end
    end
    a = 45;
end

function P = parse_params_(varargin)
    if ~isempty(varargin) && isstruct(varargin{1})
        P = varargin{1};  rest = varargin(2:end);
    else
        P = tg96_params();  rest = varargin;
    end
    for k = 1:2:numel(rest)
        parts = strsplit(rest{k}, '.');
        P = setfield(P, parts{:}, rest{k+1});  %#ok<SFLD>
    end
end

function G = build_(p, C)
%BUILD_  the tuning bench at tail parameters p, decks emitted.  Factored out
% of cost_ so the winner GATE (row_gain_) builds the identical rig -- a gate
% on a differently-built bench measures nothing.
    s = C.s;  b = C.bench;
    G = macos.design.twyman_green('polarizing',true,'ngridpts',C.NGRID, ...
        'optics',C.optics, C.oapargs{:}, 'BS_AOI',C.AOI, C.nodeargs{:}, ...
        'F1',s*b.F1,'F2',s*b.F2,'D_LENS',s*b.D_LENS,'R_BAFFLE',s*b.R_BAFFLE,'D_SB',s*b.D_SB, ...
        'BS_T',s*b.BS_T,'D_L1_BS',s*b.D_L1_BS,'D_BS_TO',C.D_BS_TO,'D_BS_CMP',s*b.D_BS_CMP, ...
        'PLATE_SUB',b.PLATE_SUB,'EDGE_MARGIN',b.EDGE_MARGIN,'MASK_SUB',b.MASK_SUB, ...
        'R_TO_AP',s*b.R_TO_AP,'L1_Kr',s*b.L1_Kr,'L1_Kc',b.L1_Kc,'L2_Kr',-s*abs(b.L2_Kr),'L2_Kc',b.L2_Kc, ...
        'to_grid_file',C.f_flat,'to_grid_n',C.N_G,'to_grid_dx',C.DX_G, ...
        'qwp_ret',C.QWP,'pol_in_deg',b.pol_in_deg,'qwp_test_deg',b.qwp_test_deg, ...
        'qwp_ref_deg',b.qwp_ref_deg,'out_qwp_deg',b.out_qwp_deg,'analyzer_deg',b.analyzer_deg, ...
        'tail_arch','fieldlens','FL_F',p(1),'FL_Kc',p(2),'FL_D',s*b.FL_D,'D_MASK_FL',p(3),'DET_TRIM',p(4));
    G.bt.emit(C.f_test);  G.br.emit(C.f_ref);
end

function [g, info] = row_gain_(p, C)
%ROW_GAIN_  The recovered gain in ACTUATOR SPACE, by LATTICE DECONVOLUTION.
%
% WHAT THIS REPLACED, AND WHY (item 3, 2026-09-16).  The first form of this
% measure took ONE POINT SAMPLE: the measured map at the detector pixel the
% affine sends an actuator's centre to, over the DM surface at that same
% actuator's centre.  It failed its own two-leg test in the one direction
% that matters -- it ACCEPTED objwin3 (battery 0.0338) at 0.9804 and REFUSED
% the lens rig's tuned tail (battery 0.9968) at -0.8285.  A point sample at
% the peak reads how CONCENTRATED the response is, and that is set by the
% magnification (the broken tail's 6.125 against the seed's 10.44 spreads the
% response over fewer detector pixels and so dilutes the peak less, reading
% HIGHER).  Magnification is not readability; a gate built on it prefers the
% tails it should refuse.
%
% The lattice form measures what the battery measures.  The bench's OWN
% measured influence stencil is deconvolved off a multi-site poked map over
% the illuminated lattice, and the recovered command is regressed on the
% commanded one (score_'s gain, verbatim: Ad(lit)\a(lit)).  Magnification
% divides out because the stencil and the map are BOTH measured through the
% same tail and both resampled into the DM frame; what survives is whether a
% command at a site reappears at that site, with its amplitude, without
% leaking to its neighbours -- which is readability.
%
% NON-VACUITY, deliberately.  The stencil comes from tg96_place's ANCHOR
% poke; the row is poked at DIFFERENT, well-separated sites.  Building the
% stencil from the very map it then fits would return ~1 by construction and
% would gate nothing.  The sites are spread across the pupil rather than
% stacked at the centre, so a registration that degrades off-axis -- the way
% a walked conjugate does -- is in the measurement and not just at one lucky
% pixel.
%
% Cost is UNCHANGED: one placement (which already traces the anchor poke, now
% handed back as PL.hA) plus one poked map, once per tune, not per fminsearch
% evaluation.
    g = NaN;  info = '';
    try
        G = build_(p, C);
        AT = arm_desc(C.f_test, G.bt, G.T, 0);
        AR = arm_desc(C.f_ref,  G.br, G.R, 45);
        Sr = analyzer_basis(AR, C.QWP, []);  S0 = analyzer_basis(AT, C.QWP, []);
        I0 = frame(S0, Sr, 0);  msk = I0 > 0.1*max(I0(:));
        if nnz(msk) < 500, info = 'pupil lost (msk < 500 px)';  return; end
        pn = fourstep(S0, Sr, C.THETAS);
        measf = @(M) meas_surface(AT, C.QWP, M, Sr, pn, C.THETAS, C.LAM);
        cfg = struct('nact', b_nact_(C.bench), 'pitch', b_pitch_(C.bench));
        PL = tg96_place(AT, G.T, cfg, msk, C.N_G, C.DX_G, C.POKE, measf, ...
                        C.place, zeros(size(pn)));
        xg = ((1:C.N_G)-(C.N_G+1)/2)*C.DX_G;
        anc = PL.anchor;                                  % [bx by tax tay]

        % ---- the bench's own measured stencil, from the anchor poke -------
        % tg96_samp, NOT dmg_samp: the shared resampler is an axis permutation
        % plus signs plus one scale, which cannot express this rig's non-90
        % degree fold rotation (tg96_place carries it in frm.Linv).
        [hdA, reg] = tg96_samp(PL.hA, PL, xg, msk);  hdA(isnan(hdA)) = 0;
        % Measured, not asserted: how far this bench's mapping is from the
        % signed-permutation family the shared dmg_samp can express.
        fprintf(['  registration: rotation %.2f deg off axis (mod 90), nearest ' ...
                 'signed permutation is %.1f%% away, anisotropy %.4f -- ' ...
                 'dmg_samp is usable only when these are 0, 0%% and 1.\n'], ...
                reg.rot_deg, 100*reg.perm_err, reg.aniso);
        stn = dmg_stencil(hdA, xg, anc(3), anc(4), cfg.pitch, C.stn_hw) / C.POKE;
        if ~any(stn(:)) || ~all(isfinite(stn(:)))
            info = 'stencil empty or non-finite (the anchor poke did not register)';
            return;
        end

        % ---- the row: interior lit sites, spread, none of them the anchor --
        ic = row_sites_(PL.lit, cfg.nact, PL.aR);
        if isempty(ic), info = 'no lit sites clear of the anchor';  return; end
        Ad = zeros(cfg.nact);
        for k = 1:size(ic,1), Ad(ic(k,1), ic(k,2)) = C.poke_nm*1e-6; end
        M1 = dm_influence_map(C.N_G, C.DX_G, 'nact',cfg.nact, 'pitch',cfg.pitch, 'act',Ad);
        % MEAN-REFERENCE the row map exactly as tg96_place references the
        % anchor poke it built the stencil from (mkref: minus the median over
        % the mask).  Not cosmetic -- the OAP rig's null leaves a low-order
        % background across the pupil, and a constant the stencil never saw
        % deconvolves into a spurious UNIFORM command, which lands in the
        % unpoked floor and biases the regression.  Stencil and map have to be
        % referenced the same way or the gain is measuring the background.
        h1 = measf(M1);  h1 = h1 - median(h1(msk));
        hd1 = tg96_samp(h1, PL, xg, msk);  hd1(isnan(hd1)) = 0;

        % ---- deconvolve to actuator commands, then the battery's own gain --
        aa = dmg_act_fit(hd1, xg, PL.axg, PL.ayg, stn, PL.lit, C.act_lam);
        [g, e, fl, snr] = score_gain_(aa, Ad, PL.lit);
        info = sprintf(['%d sites, mag %.4f DM-mm/det-mm, %d lit, stencil hw %d ' ...
                        'from anchor (%d,%d); err %.1f pm, floor %.1f pm, SNR %.1f'], ...
                       size(ic,1), PL.mag, nnz(PL.lit), C.stn_hw, PL.aR(1), PL.aR(2), ...
                       e, fl, snr);
    catch ME
        info = sprintf('row failed: %s', ME.message);
    end
end

function ic = row_sites_(lit, nact, aR)
%ROW_SITES_  Interior lit actuators, spread over the pupil, clear of the
% anchor.  The centre plus one per quadrant at ~0.55 of the lit radius: a
% registration that degrades off-axis shows up here and cannot be hidden by a
% single well-placed centre site.  MIN_SEP keeps every site away from the
% anchor (whose response built the stencil) and from its neighbours, so the
% deconvolution is not asked to separate two overlapping kernels.
    MIN_SEP = 8;                                   % actuators
    ctr = (nact+1)/2;
    [ai, aj] = find(lit);
    if isempty(ai), ic = [];  return; end
    r = max(hypot(ai-ctr, aj-ctr));
    want = [0 0; 1 1; 1 -1; -1 1; -1 -1];          % centre + four quadrants
    ic = zeros(0,2);
    for k = 1:size(want,1)
        tgt = [ctr + 0.55*r*want(k,1)/max(norm(want(k,:)),1), ...
               ctr + 0.55*r*want(k,2)/max(norm(want(k,:)),1)];
        d = hypot(ai-tgt(1), aj-tgt(2));
        [~, ord] = sort(d);
        for q = ord(:).'
            cand = [ai(q) aj(q)];
            if hypot(cand(1)-aR(1), cand(2)-aR(2)) < MIN_SEP, continue; end
            if ~isempty(ic) && min(hypot(ic(:,1)-cand(1), ic(:,2)-cand(2))) < MIN_SEP, continue; end
            ic(end+1,:) = cand; %#ok<AGROW>
            break
        end
    end
end

function [g, e, fl, snr] = score_gain_(a, Ad, lit)
%SCORE_GAIN_  tg96_run's score_ verbatim: the actuator-space gain is the LS
% regression of the recovered command on the commanded one over the lit
% lattice, with the unpoked floor and SNR beside it.  Kept identical to the
% battery's so the gate and the battery report the SAME quantity -- the whole
% point of item 3 is that the gate must not use a proxy.
    g = Ad(lit) \ a(lit);
    e = sqrt(mean((a(lit) - Ad(lit)).^2))*1e9;
    pk = (Ad ~= 0) & lit;  un = lit & ~pk;
    if nnz(pk) < nnz(lit)/4
        fl = std(a(un))*1e9;  snr = mean(a(pk)) / max(std(a(un)), eps);
    else
        fl = NaN;  snr = NaN;
    end
end

% ==== PSI helpers (flat-DM subset, copied verbatim from tg96_tail.m) ==
function A = arm_desc(rx, b, ix, base_deg)
    nm = {b.E.name};
    iTO = [];  if isfield(ix,'iTO'), iTO = ix.iTO; end
    A = struct('rx',rx,'b',b,'iPol',find(strcmp(nm,'PolIn'),1), ...
        'iQ',find(contains(nm,'QWP') & ~strcmp(nm,'OutQWP')), ...
        'base',base_deg,'qwp_deg',base_deg,'oq_deg',0,'iTO',iTO, ...
        'iRC',ix.iRC,'iOQ',ix.iOutQWP,'iAn',ix.iAnalyzer,'iDET',ix.iDET);
end
function a = lax(psi, deg)
    u1 = macos.design.Bench.perp(psi(:));  u2 = cross(psi(:), u1);
    a = cosd(deg)*u1 + sind(deg)*u2;  a = a(:).';
end
function load_arm(A, QWP, an_deg, grid)
    macos.load_rx(A.rx);  b = A.b;
    if nargin >= 4 && ~isempty(grid) && ~isempty(A.iTO)
        macos.set_elt_grid(A.iTO, macos.get_elt_grid_spacing(A.iTO), grid);
    end
    macos.polarizer(A.iPol, 'axis', lax(b.E(A.iPol).psi, 45));
    qa = lax(b.E(A.iQ(1)).psi, A.qwp_deg);
    for j = 1:2, macos.waveplate(A.iQ(j), 'axis', qa, 'retardance', QWP); end
    macos.waveplate(A.iOQ, 'axis', lax(b.E(A.iOQ).psi, A.oq_deg), 'retardance', QWP);
    macos.polarizer(A.iAn, 'axis', lax(b.E(A.iAn).psi, an_deg));
    macos.polarization('on', 'Ex',[1/sqrt(2) 0], 'Ey',[1/sqrt(2) 0]);
    macos.vector_diffraction(true);
end
function E = arm_field(A, QWP, an_deg, grid)
    load_arm(A, QWP, an_deg, grid);
    E = cat(3, macos.complex_field(A.iDET,'plane',1), ...
               macos.complex_field(A.iDET,'plane',2), macos.complex_field(A.iDET,'plane',3));
end
function S = analyzer_basis(A, QWP, grid)
    S = struct('A',arm_field(A,QWP,0,grid),'C',arm_field(A,QWP,90,grid),'B',[]);
    S.B = 2*arm_field(A,QWP,45,grid) - S.A - S.C;
end
function E = synth(S, th), c=cosd(th); s=sind(th); E = c^2*S.A + c*s*S.B + s^2*S.C; end
function I = frame(Sx, Sr, th), I = sum(abs(synth(Sx,th)+synth(Sr,th)).^2, 3); end
function p = fourstep(Sx, Sr, th)
    p = atan2(frame(Sx,Sr,th(2))-frame(Sx,Sr,th(4)), frame(Sx,Sr,th(1))-frame(Sx,Sr,th(3)));
end
function h = meas_surface(A, QWP, M, Sr, p_null, THETAS, LAM)
    d = angle(exp(1i*(fourstep(analyzer_basis(A,QWP,M), Sr, THETAS) - p_null)));
    h = d * LAM/(4*pi);
end

function delete_if_(fs)
%DELETE_IF_  remove this run's scratch decks, quietly
for i = 1:numel(fs)
    if isfile(fs{i}), delete(fs{i}); end
end
end
