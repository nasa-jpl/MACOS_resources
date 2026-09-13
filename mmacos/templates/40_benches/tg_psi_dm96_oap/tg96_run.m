function out = tg96_run(varargin)
%TG96_RUN  One parameterized runner for the 96x96 Twyman-Green DM gauge, lens
%   OR all-reflective (OAP).  Ports tg_psi_dm96/tg96.m Stage A-E into the
%   edit-and-rerun runner form (Dave's standing rule).  Flip bench.optics to
%   drive the refractive record (equivalence gate) or the reflective variant
%   through the SAME code path.
%
%   Usage (interactive; NO exit -- see tg96_run_batch for -batch):
%     tg96_run                              % defaults (tg96_params, lens)
%     tg96_run(tg96_params())               % explicit
%     tg96_run('bench.optics','oap','tag','oap')   % OAP rig
%     tg96_run('MODEL',256,'NGRID',63,'tag','smoke')   % fast code-path check
%
%   Writes runs/<tag>/: <tag>_report.txt (tee'd), <tag>.mat (out struct),
%   <tag>_{test,ref}.in, <tag>_layout.png, <tag>_closure.png,
%   <tag>_transfer.png.  Reuses ../dm_gauge_lib is NOT required -- tg96.m's
%   verbatim polarization-PSI helpers are file-local here (same as tg96.m).
%
%   Stage A  clearance solve (folded for OAP: source->OAP1 + OAP2->detector)
%   Stage A2 sampling budget (asserted)
%   Stage B  build via macos.design.twyman_green(..., 'optics', ...)
%   Stage C  null / piston / single-actuator / registration / closure
%   Stage D  transfer curve to the DM Nyquist + held-out random
%   Stage E  differential rows (the pm product; base x deviation)
%   Stage LOOP  closed-loop HOLD metric (D7): the DM held at the working
%     surface by a proportional loop closed through the four-step reading and
%     its measured matrix; the shared dmg_loop. stages {'bench','loop','figs'}.

exdir = fileparts(mfilename('fullpath'));  if isempty(exdir), exdir = pwd; end
if isempty(which('macos.init'))
    run(fullfile(exdir, '..', '..', '..', 'mmacos_setup.m'));
end
assert(~isempty(getenv('MACOS_HOME')), 'MACOS_HOME must be set.');
addpath(exdir);                                    % tg96_place (found after the cd into runs/<tag>)
addpath(fullfile(exdir, '..', '..', '90_polarization', 'tg_psi_dm'));  % dm_influence_map
addpath(fullfile(exdir, '..', 'dm_gauge_lib'));   % dmg_frame/dmg_anchor/dmg_lit (matrix calib)

% ---- parse: struct P and/or dotted-path name/value overrides ----------
P = parse_params_(exdir, varargin{:});
if isempty(P.outdir), P.outdir = fullfile(exdir, 'runs', P.tag); end
if ~exist(P.outdir,'dir'), mkdir(P.outdir); end
oldcd = cd(P.outdir);  cleaner = onCleanup(@() cd(oldcd));
% param-file (memory) : copy a trim table into the run dir, else clear stale
if ~isempty(P.param_file)
    pf = P.param_file;
    if ~isfile(pf), pf = fullfile(exdir, P.param_file); end
    if ~isfile(pf), pf = which(P.param_file); end
    assert(~isempty(pf) && isfile(pf), 'param_file %s not found', P.param_file);
    copyfile(pf, 'macos_param.txt');
elseif isfile('macos_param.txt')
    delete('macos_param.txt');
end
rep = fopen([P.tag '_report.txt'], 'w');
cleaner2 = onCleanup(@() fclose(rep));
say = @(varargin) say_(rep, varargin{:});
want = @(st) any(strcmp(P.stages, st));

s = P.dm(1).nact / 56;             % uniform scale off the 56 mm v1 rig
say('=== TG96 gauge (%s optics) : Xinetics %dx%d, model %d, grid %dx%.2f ===\n', ...
    P.bench.optics, P.dm(1).nact, P.dm(1).nact, P.MODEL, P.grid.N_G, P.grid.DX_G);
say('uniform scale s = %.4f; tag "%s"\n\n', s, P.tag);

% ---- Stage A: clearance solve (folded layout re-solved for OAP) -------
[geom, ~] = stage_A_(P, s, say);

% ---- Stage A2: sampling budget --------------------------------------
stage_A2_(P, say);

% ---- Stage B: build --------------------------------------------------
macos.init(P.MODEL);
assert(P.grid.N_G <= macos.grid_size_max(), ...
    'N_G %d exceeds mGridMat %d at model %d', P.grid.N_G, macos.grid_size_max(), P.MODEL);
macos.write_grid_file(P.grid.flat_file, zeros(P.grid.N_G));
bench = struct('geom', geom, 's', s);
if want('bench') || want('battery') || want('loop') || want('jones')
    [G, bench] = stage_B_(P, s, geom, say, exdir);
end

place = struct();
if want('place')
    place = stage_place_(P, s, G, bench, say);
end

battery = struct();
if want('battery')
    battery = stage_CDE_(P, s, G, bench, say, place);
end

loop = struct();
if want('loop')
    loop = stage_loop_(P, s, G, bench, say, place);
end

jones = struct();
if want('jones')
    jones = stage_jones_(P, s, G, bench, say);
end

if want('figs')
    draw_layout_(geom, s, P);
    if isfield(loop, 'res'), draw_loop_(loop, P); end
    % the real optics layout via the Bench renderer (chief-ray polyline through
    % every element, aperture-sized footprint bars, element names + leg lengths)
    if isfield(bench, 'G')
        f = bench.G.bt.sketch('title', sprintf('TG96 %s test arm -- optics layout (XY plane)', P.bench.optics));
        exportgraphics(f, [P.tag '_sketch.png'], 'Resolution', 140);  close(f);
        fr = bench.G.br.sketch('title', sprintf('TG96 %s reference arm -- optics layout (XY plane)', P.bench.optics));
        exportgraphics(fr, [P.tag '_sketch_ref.png'], 'Resolution', 140);  close(fr);
        fprintf('wrote %s_sketch.png + %s_sketch_ref.png (Bench.sketch)\n', P.tag, P.tag);
    end
end

out = struct('P', P, 'geom', geom, 'bench', bench, 'battery', battery, ...
    'place', place, 'loop', loop, 'jones', jones, 's', s);
save([P.tag '.mat'], 'out');
say('\nwrote %s_report.txt + %s.mat + figures in %s\n', P.tag, P.tag, P.outdir);
end

% =====================================================================
%  STAGES
% =====================================================================
function [geom, legs] = stage_A_(P, s, say)
    beam_r = P.clear.beam_r;  if isempty(beam_r), beam_r = s*P.bench.R_TO_AP; end
    MARGIN = P.clear.MARGIN;  LEG_CAP = P.clear.LEG_CAP;
    legs = {'DM arm vs ref beam',       P.clear.HW_DM;
            'ref arm vs DM beam',       P.clear.HW_REF;
            'camera leg vs source beam',P.clear.HW_CAM};
    say('Stage A -- clearance solve (beam_r %.1f, margin %.0f, leg cap %.0f):\n', ...
        beam_r, MARGIN, LEG_CAP);
    need = cellfun(@(h) beam_r + h + MARGIN, legs(:,2));
    th_min = max(asind(need/LEG_CAP))/2;
    AOI = P.bench.BS_AOI;  if isempty(AOI), AOI = ceil(th_min); end
    Lreq = need/sind(2*AOI);
    say('  binding angle %.2f deg -> BS_AOI = %d deg\n', th_min, AOI);
    for k = 1:size(legs,1)
        say('  %-27s need %6.1f mm sep -> leg >= %5.0f mm\n', legs{k,1}, need(k), Lreq(k));
    end
    D_BS_TO = P.bench.D_BS_TO;
    if isempty(D_BS_TO), D_BS_TO = ceil(max(Lreq(1), s*250)/50)*50; end
    say('  chosen DM leg %d mm; achieved separations and margins:\n', D_BS_TO);
    for k = 1:size(legs,1)
        m = D_BS_TO*sind(2*AOI) - (need(k) - MARGIN);
        say('  %-27s separation %6.1f mm, margin %+6.1f mm (spec >= %.0f)\n', ...
            legs{k,1}, D_BS_TO*sind(2*AOI), m, MARGIN);
    end
    geom.AOI = AOI;  geom.D_BS_TO = D_BS_TO;  geom.beam_r = beam_r;
    % ---- OAP folded-layout clearance: the source->OAP1 and OAP2->detector
    %  legs are new.  Near-normal preferred; solve the smallest fold AOI whose
    %  lateral source/detector offset clears the collimated beam + body inside
    %  LEG_CAP.  offset(AOI) = F*|sin(180-2*AOI)| at leg length F (=F1/F2).
    if strcmp(P.bench.optics, 'oap')
        need_off = beam_r + P.clear.HW_CAM + MARGIN;   % clear source/cam bodies
        a1 = P.oap.OAP1_AOI;  a2 = P.oap.OAP2_AOI;
        if isempty(a1), a1 = solve_fold_(s*P.bench.F1, need_off); end
        if isempty(a2), a2 = solve_fold_(s*P.bench.F2, need_off); end
        geom.OAP1_AOI = a1;  geom.OAP2_AOI = a2;
        off1 = s*P.bench.F1*abs(sind(180-2*a1));
        off2 = s*P.bench.F2*abs(sind(180-2*a2));
        say('  OAP fold solve (need lateral clearance %.1f mm):\n', need_off);
        say('  OAP1 (collimator F1=%.0f): AOI %2d deg -> lateral %6.1f mm, margin %+6.1f\n', ...
            s*P.bench.F1, a1, off1, off1-need_off);
        say('  OAP2 (focuser   F2=%.0f): AOI %2d deg -> lateral %6.1f mm, margin %+6.1f\n', ...
            s*P.bench.F2, a2, off2, off2-need_off);
    end
    say('\n');
end

function a = solve_fold_(F, need_off)
%SOLVE_FOLD_  smallest integer fold AOI (deg) whose lateral offset
%   F*sin(180-2*AOI) clears need_off; near-normal preferred but the source/
%   detector body must clear the beam.  180-2*AOI is the chief turn.
    for a = 5:44
        if F*abs(sind(180-2*a)) >= need_off, return; end
    end
    a = 45;   % fall back to the right-angle fold
end

function stage_A2_(P, say)
    act_nyq  = P.dm(1).nact/2;
    det_nyq  = P.NGRID/2;
    grid_ppa = P.dm(1).pitch/P.grid.DX_G;
    say('Stage A2 -- sampling budget (actuator Nyquist %.0f cyc/pupil):\n', act_nyq);
    say('  detector Nyquist %5.1f cyc/pup  margin %4.1fx (need >= 2)\n', det_nyq, det_nyq/act_nyq);
    say('  DM surface grid %.2f px/actuator             (need >= 3)\n', grid_ppa);
    say('  diffraction grid %d^2 padding %4.1fx\n\n', P.MODEL, P.MODEL/P.NGRID);
    chk = @(c, varargin) assert_or_warn_(P, c, varargin{:});
    chk(det_nyq >= 2*act_nyq, 'sampling: det Nyquist %.1f < 2x actuator %.1f', det_nyq, act_nyq);
    chk(grid_ppa >= 3, 'sampling: %.2f grid px/actuator < 3', grid_ppa);
    chk(P.MODEL >= 2*P.NGRID, 'sampling: grid %d < 2x %d-px image', P.MODEL, P.NGRID);
end

function assert_or_warn_(P, c, varargin)
%   Stage-A2 gate.  P.smoke true downgrades the assert to a warning so a
%   coarse-MODEL/NGRID code-path check can run (NOT a result).
    if isfield(P,'smoke') && P.smoke
        if ~c, warning('tg96_run:smoke_sampling', varargin{:}); end
    else
        assert(c, varargin{:});
    end
end

function [G, bench] = stage_B_(P, s, geom, say, exdir)
    % tail params.  Lookup order: <tag>_tail.mat (explicit per-tag override),
    % then <optics>_tail.mat (the tuned set of record, keyed by bench.optics),
    % else the geometric seed.  Keying by OPTICS means a non-canonical run tag
    % does NOT silently fall back to the seed on the wrong bench (item 2).
    b = P.bench;
    T_FL_F = s*b.FL_F;  T_FL_Kc = b.FL_Kc;  T_DMF = s*b.D_MASK_FL;  T_TRIM = s*b.DET_TRIM;
    cand = {fullfile(exdir,[P.tag '_tail.mat']), fullfile(exdir,[b.optics '_tail.mat'])};
    tailf = '';  for ci = 1:numel(cand), if isfile(cand{ci}), tailf = cand{ci}; break; end, end
    bench.expected_null = [];
    if ~isempty(tailf)
        tl = load(tailf);
        T_FL_F = tl.out.FL_F;  T_FL_Kc = tl.out.FL_Kc;
        T_DMF  = tl.out.D_MASK_FL;  T_TRIM = tl.out.DET_TRIM;
        bench.expected_null = tl.out.null_nm;
        say('Tail: RE-TUNED set from %s (null %.3f nm at opt res; seed %.3f)\n', ...
            tailf, tl.out.null_nm, tl.out.seed_null_nm);
    else
        say('Tail: geometrically-scaled seed (no %s_tail.mat / %s_tail.mat) -- RE-RUN tg96_tail for %s\n', ...
            P.tag, b.optics, b.optics);
    end
    oapargs = {};
    if strcmp(b.optics,'oap')
        oapargs = {'OAP1_AOI',geom.OAP1_AOI, 'OAP2_AOI',geom.OAP2_AOI, ...
                   'OAP1_SIDE',P.oap.OAP1_SIDE, 'OAP2_SIDE',P.oap.OAP2_SIDE};
    end
    mk = @(gf) macos.design.twyman_green('polarizing',b.polarizing, 'ngridpts',P.NGRID, ...
        'optics',b.optics, oapargs{:}, 'BS_AOI',geom.AOI, ...
        'F1',s*b.F1, 'F2',s*b.F2, 'D_LENS',s*b.D_LENS, 'R_BAFFLE',s*b.R_BAFFLE, ...
        'D_SB',s*b.D_SB, 'BS_T',s*b.BS_T, 'D_L1_BS',s*b.D_L1_BS, ...
        'D_BS_TO',geom.D_BS_TO, 'D_BS_CMP',s*b.D_BS_CMP, 'R_TO_AP',s*b.R_TO_AP, ...
        'L1_Kr',s*b.L1_Kr, 'L1_Kc',b.L1_Kc, 'L2_Kr',-s*abs(b.L2_Kr), 'L2_Kc',b.L2_Kc, ...
        'to_grid_file',gf, 'to_grid_n',P.grid.N_G, 'to_grid_dx',P.grid.DX_G, ...
        'qwp_ret',P.QWP, 'pol_in_deg',b.pol_in_deg, 'qwp_test_deg',b.qwp_test_deg, ...
        'qwp_ref_deg',b.qwp_ref_deg, 'out_qwp_deg',b.out_qwp_deg, 'analyzer_deg',b.analyzer_deg, ...
        'tail_arch',b.tail_arch, 'FL_F',T_FL_F, 'FL_Kc',T_FL_Kc, 'FL_D',s*b.FL_D, ...
        'D_MASK_FL',T_DMF, 'DET_TRIM',T_TRIM);
    G = mk(P.grid.flat_file);
    G.bt.emit([P.tag '_test.in']);  G.br.emit([P.tag '_ref.in']);
    say('Stage B -- %s rig built (BS_AOI %d); emitted %s_{test,ref}.in\n\n', ...
        b.optics, geom.AOI, P.tag);
    bench.G = G;  bench.geom = geom;  bench.s = s;
    bench.n_elt = numel(G.bt.E);
end

function C = arm_setup_(P, G)
%ARM_SETUP_  arm descriptors + analyzer bases + null + mask (shared by the
%   battery and the placement stage).
    C.N_G = P.grid.N_G;  C.DX_G = P.grid.DX_G;  C.LAM = P.LAM;  C.QWP = P.QWP;
    C.THETAS = P.THETAS;  C.NACT = P.dm(1).nact;  C.PITCH = P.dm(1).pitch;
    C.rxT = [P.tag '_test.in'];  C.rxR = [P.tag '_ref.in'];
    C.AT = arm_desc(C.rxT, G.bt, G.T, 0);
    C.AR = arm_desc(C.rxR, G.br, G.R, 45);
    % OAP coating (brief item B): attach to the OAPs of BOTH arms so the
    % reference/null below already carry it. load_arm applies it each trace.
    cs = oap_coat_stack_(P);
    if ~isempty(cs)
        C.AT = set_oap_coat_(C.AT, cs);  C.AR = set_oap_coat_(C.AR, cs);
    end
    C.Sr = analyzer_basis(C.AR, C.QWP, []);
    C.S0 = analyzer_basis(C.AT, C.QWP, []);
    C.I0 = frame(C.S0, C.Sr, 0);  C.msk = C.I0 > 0.1*max(C.I0(:));
    C.p_null = fourstep(C.S0, C.Sr, C.THETAS);
    C.measf = @(M) meas_surface(C.AT, C.QWP, M, C.Sr, C.p_null, C.THETAS, C.LAM);
end

function battery = stage_CDE_(P, s, G, bench, say, place)
    if nargin < 6, place = struct(); end
    if strcmp(P.battery.calib_mode, 'matrix')
        battery = stage_matrix_(P, s, G, bench, say, place);  return;   % Route 2 (D2)
    end
    ctx = arm_setup_(P, G);
    N_G = ctx.N_G;  DX_G = ctx.DX_G;  LAM = ctx.LAM;  QWP = ctx.QWP;
    THETAS = ctx.THETAS;  NACT = ctx.NACT;  PITCH = ctx.PITCH;
    AT = ctx.AT;  AR = ctx.AR;  Sr = ctx.Sr;  S0 = ctx.S0;  msk = ctx.msk;  p_null = ctx.p_null;
    say('Stage C -- battery (design azimuths, unaligned):\n');
    az_t = arm_azimuth(AT, QWP, 0);  az_r = arm_azimuth(AR, QWP, 45);
    dep  = wrap180(az_t - az_r - 90);
    say('  arm azimuths: test %+.4f, ref %+.4f -> departure %+.4f deg\n', az_t, az_r, dep);
    h_null = (p_null - median(p_null(msk))) * LAM/(4*pi) * 1e6;
    null_nm = std(h_null(msk));
    say('  null: %.4f nm rms surface (%.1f pm) with nothing aligned\n', null_nm, 1e3*null_nm);
    dp = angle(exp(1i*(fourstep(analyzer_basis(AT,QWP,P.battery.piston_nm*1e-6*ones(N_G)),Sr,THETAS) - p_null)));
    gain = median(dp(msk))/(4*pi*P.battery.piston_nm*1e-6/LAM);
    say('  %d nm piston: |gain| %.5f (unaligned scale error %+.3f%%)\n', ...
        P.battery.piston_nm, abs(gain), 100*(abs(gain)-1));
    % In-pupil calibration-poke placement.  A circular beam on the square DM
    % leaves the geometric-center actuator outside the illuminated footprint for
    % the OAP rig (smaller/shifted pupil; measured: center 0 nm, off-centre
    % 130 nm).  LENS keeps the record's fixed placement (equivalence-exact); OAP
    % places poke A at the DM footprint CENTROID and poke B a radial offset off
    % it -- both clearly in-pupil (Dave 2026-09-11).
    if strcmp(P.bench.optics,'oap')
        macos.load_rx(rxT);  sTO = macos.trace(AT.iTO);  rTO = macos.get_ray_info(sTO.nRays);
        okT = rTO.ok_trace(:) & rTO.ok_pass(:);
        psiTO = macos.get_elt_psi(AT.iTO);  vptTO = macos.get_elt_vpt(AT.iTO);
        u1 = macos.design.Bench.perp(psiTO);  u2 = cross(psiTO, u1);
        dd = rTO.pos(:,okT) - vptTO;
        au = (u1.'*dd)/PITCH + (NACT+1)/2;  av = (u2.'*dd)/PITCH + (NACT+1)/2;
        acc = median(au);  acr = median(av);
        % the EXACT footprint centre recovers 0 (four-step chief/central-pixel
        % reference), and the OAP footprint is elliptical (fold foreshortening),
        % so use the measured PER-AXIS half-extents and place both pokes
        % off-centre, non-colinear, well inside both extents (measured: in-pupil
        % actuators image ~130 nm).  clamp to [1,NACT].
        hw_c = 0.5*(max(au)-min(au));  hw_r = 0.5*(max(av)-min(av));
        cl = @(x) min(max(round(x),1),NACT);
        pokeA = [cl(acr - 0.30*hw_r), cl(acc + 0.30*hw_c)];
        pokeB = [cl(acr + 0.35*hw_r), cl(acc - 0.35*hw_c)];
        say('  OAP in-pupil pokes: centroid (%.0f,%.0f) half-extent (%.0f,%.0f) -> pokeA %s pokeB %s\n', ...
            acr, acc, hw_r, hw_c, mat2str(pokeA), mat2str(pokeB));
    else
        pokeA = [];  pokeB = P.battery.reg_act;   % [] => dm_influence_map center
    end
    if isempty(pokeA)
        Mp = dm_influence_map(N_G, DX_G, 'nact',NACT,'pitch',PITCH,'pattern','single','poke',P.battery.single_nm*1e-6);
    else
        AA = zeros(NACT);  AA(pokeA(1),pokeA(2)) = 1;
        Mp = dm_influence_map(N_G, DX_G, 'nact',NACT,'pitch',PITCH,'act',P.battery.single_nm*1e-6*AA);
    end
    hp = meas_surface(AT, QWP, Mp, Sr, p_null, THETAS, LAM);
    say('  single actuator at %d nm: recovered peak %.1f nm\n', P.battery.single_nm, 1e6*max(abs(hp(msk))));
    % registration (two pokes)
    A2c = zeros(NACT);  A2c(pokeB(1), pokeB(2)) = 1;
    Mp2 = dm_influence_map(N_G, DX_G, 'nact',NACT,'pitch',PITCH,'act',P.battery.single_nm*1e-6*A2c);
    hp2 = meas_surface(AT, QWP, Mp2, Sr, p_null, THETAS, LAM);
    Mdm = dm_influence_map(N_G, DX_G, 'nact',NACT,'pitch',PITCH,'pattern','checker','poke',P.POKE);
    hm  = meas_surface(AT, QWP, Mdm, Sr, p_null, THETAS, LAM);
    axs = ((1:N_G)-(N_G+1)/2)*DX_G;
    [map, reg] = register_two_pokes(AT, G.T, Mp, hp, Mp2, hp2, N_G, DX_G, msk);
    say('  registration: parity %d of 8, meas sign %+d; |pokeB corr| %.4f (runner-up %.4f)\n', ...
        reg.par, reg.sign, abs(reg.pokeB_corr), reg.runner_up);
    assert(abs(reg.pokeB_corr) >= 0.8, 'registration gate FAILED: |corr| %.3f', abs(reg.pokeB_corr));
    hm = reg.sign * hm;
    tt = interpn(axs, axs, Mdm, map.Xt, map.Yt, 'spline', 0);
    hv = 1e6*(hm(msk)-mean(hm(msk)));  tv = 1e6*(tt(msk)-mean(tt(msk)));
    cc = corrcoef(hv, tv);
    say('  %dx%d closure: truth %.3f, measured %.3f, residual %.3f nm rms, corr %.6f\n', ...
        NACT,NACT, std(tv), std(hv), std(hv-tv), cc(1,2));
    say('  registration: mag %.4f, anamorphism %.2f%%, nonlinearity %.4f mm\n\n', ...
        map.mag, map.anam_pct, map.nonlin_mm);
    % Stage D transfer
    say('Stage D -- transfer curve (Nyquist %d cyc/pupil):\n', NACT/2);
    PQ = P.battery.transfer_PQ;  nM = size(PQ,1);  iiv = (1:NACT)';
    Tt = zeros(nnz(msk), nM);  Mm = zeros(nnz(msk), nM);  frq = zeros(1,nM);
    for k = 1:nM
        p = PQ(k,1); q = PQ(k,2);
        Ak = cos(pi*p*iiv/NACT)*cos(pi*q*iiv/NACT)';  Ak = Ak/max(abs(Ak(:)));
        Mk = dm_influence_map(N_G, DX_G, 'nact',NACT,'pitch',PITCH,'act',P.POKE*Ak);
        hk = reg.sign * meas_surface(AT, QWP, Mk, Sr, p_null, THETAS, LAM);
        tk = interpn(axs, axs, Mk, map.Xt, map.Yt, 'spline', 0);
        Mm(:,k) = hk(msk)-mean(hk(msk));  Tt(:,k) = tk(msk)-mean(tk(msk));  frq(k) = hypot(p,q)/2;
    end
    Gt = Tt \ Mm;
    say('  %-9s %8s %8s %11s\n', 'mode(p,q)','cyc/pup','gain','cross-talk');
    for k = 1:nM
        x = Gt(:,k); x(k) = 0;
        say('  %3d,%-5d %8.1f %8.4f %11.4f\n', PQ(k,1),PQ(k,2),frq(k),Gt(k,k),norm(x));
    end
    Mrnd = dm_influence_map(N_G, DX_G, 'nact',NACT,'pitch',PITCH,'poke',P.POKE,'pattern','random','seed',P.battery.rand_seed);
    hv2 = reg.sign * meas_surface(AT, QWP, Mrnd, Sr, p_null, THETAS, LAM);
    tv2 = interpn(axs, axs, Mrnd, map.Xt, map.Yt, 'spline', 0);
    hv2 = hv2(msk)-mean(hv2(msk));  tv2 = tv2(msk)-mean(tv2(msk));
    r2 = corrcoef(hv2-tv2, tv2);
    say('  held-out random: %.4f nm rms resid (input %.2f, resid/truth corr %+.2f)\n\n', ...
        1e6*std(hv2-tv2), 1e6*std(tv2), r2(1,2));
    % Stage E differential
    say('Stage E -- differential rows (deviation about a working state):\n');
    d_sng = dm_influence_map(N_G, DX_G, 'nact',NACT,'pitch',PITCH,'pattern','single','poke',P.battery.diff_single_nm*1e-6);
    d_rnd = dm_influence_map(N_G, DX_G, 'nact',NACT,'pitch',PITCH,'pattern','random','seed',P.battery.diff_rand_seed,'poke',P.battery.diff_rand_nm*1e-6);
    bases = {'flat', zeros(N_G), S0; sprintf('random %dnm',P.battery.base_rand_nm), Mrnd, []};
    devs  = {sprintf('single %dnm',P.battery.diff_single_nm), d_sng; sprintf('random %dnm',P.battery.diff_rand_nm), d_rnd};
    say('  %-12s %-14s %8s %10s %8s\n','base','deviation','gain','resid pm','corr');
    diff_rows = {};
    for bi = 1:2
        if isempty(bases{bi,3}), Sb = analyzer_basis(AT, QWP, bases{bi,2}); else, Sb = bases{bi,3}; end
        p_B = fourstep(Sb, Sr, THETAS);
        for di = 1:2
            Sd = analyzer_basis(AT, QWP, bases{bi,2} + devs{di,2});
            dmeas = reg.sign * angle(exp(1i*(fourstep(Sd,Sr,THETAS) - p_B))) * LAM/(4*pi);
            dt = interpn(axs, axs, devs{di,2}, map.Xt, map.Yt, 'spline', 0);
            aa = 1e6*(dmeas(msk)-mean(dmeas(msk)));  bb = 1e6*(dt(msk)-mean(dt(msk)));
            g = bb\aa;  res = std(aa - g*bb);  cco = corrcoef(aa,bb);
            say('  %-12s %-14s %8.4f %10.1f %8.4f\n', bases{bi,1}, devs{di,1}, g, 1e3*res, cco(1,2));
            diff_rows(end+1,:) = {bases{bi,1}, devs{di,1}, g, 1e3*res, cco(1,2)}; %#ok<AGROW>
        end
    end
    battery = struct('dep',dep, 'null_nm',null_nm, 'piston_gain',abs(gain), ...
        'PQ',PQ, 'frq',frq, 'G',Gt, 'closure_resid_nm',std(hv-tv), 'closure_corr',cc(1,2), ...
        'reg',reg, 'map',map, 'diff_rows',{diff_rows}, 'msk',msk);
    % figures
    fig = figure('Visible','off','Position',[60 60 640 430]);
    dg = arrayfun(@(k) Gt(k,k), 1:min(11,nM));
    plot(frq(1:numel(dg)), dg, 'o-','LineWidth',1.6); grid on;
    xlabel('cycles across pupil'); ylabel('measured gain');
    title(sprintf('TG96 %s transfer: 1 mm actuators?', P.bench.optics));
    print(fig, [P.tag '_transfer.png'], '-dpng','-r140');
end

% =====================================================================
%  Stage MATRIX (D2/Route 2): the measured response matrix dw/da
% =====================================================================
function battery = stage_matrix_(P, s, G, bench, say, place)
% Calibration by the MEASURED response matrix (Dave 2026-09-10; the ZWFS S10
% default).  Poke every matrix_step-th actuator on a sparse grid, step through
% the offsets so every lit actuator is poked once, cut each response from its
% OWN detector-pixel window (placed by the affine map, Route 1), assemble J
% (detector px x lit act) and estimate commands by regularized least squares.
% Every reading is mean-referenced over the mask (the four-step map's chief-
% pixel reference); each column carries its own volume as a rank-one term.
% Comparisons are in ACTUATOR units (pm).
    if ~isfield(place, 'PL')
        Pp = P;  Pp.place.gate_assert = false;       % do not die on the OAP D1 gate here
        Pp.place.gate_max_states = P.place.boot_states;  % placement needs only a few states
        place = stage_place_(Pp, s, G, bench, say);
    end
    ctx = place.ctx;  PL = place.PL;  h0 = place.h0;  cfg = P.dm(1);
    msk = ctx.msk;  N_G = ctx.N_G;  DX_G = ctx.DX_G;  LAM = ctx.LAM;
    N = size(msk,1);  Am = nnz(msk);  nact = cfg.nact;  POKE = P.POKE;
    say('Stage MATRIX -- measured response matrix dw/da (%s rig, step %d, sign %s):\n', ...
        P.bench.optics, P.battery.matrix_step, P.battery.matrix_sign);
    if ~isempty(oap_coat_stack_(P))
        say('  OAP coating: %s on L1+L2, both arms (macos.coating)\n', P.bench.coat_oap);
    end
    % flat-DM null (the arm-difference the common tail cannot remove; 0.134 nm
    % lens / 12.9 nm OAP) -- reported beside the differential rows (Dave)
    hn = (ctx.p_null - median(ctx.p_null(msk))) * ctx.LAM/(4*pi) * 1e6;
    null_nm = std(hn(msk));
    exp_null = [];  if isstruct(bench) && isfield(bench,'expected_null'), exp_null = bench.expected_null; end
    if ~isempty(exp_null) && (null_nm > 10*exp_null || null_nm < 0.1*exp_null)
        warning('tg96_run:null_off', ['measured null %.4f nm is >10x off the tail''s %.4f nm -- ' ...
            'wrong tail? (tail keyed by bench.optics; a non-canonical tag with no <optics>_tail.mat runs the seed)'], null_nm, exp_null);
    end
    say('  flat-DM null: %.4f nm rms surface (%.1f pm)%s -- the same-plane-fold arm difference\n', ...
        null_nm, 1e3*null_nm, iff_(~isempty(exp_null), sprintf(' [tail expects %.4f]', exp_null), ''));
    % ---- build J over the multiplexed poke sets --------------------------
    [J, ilit, vcol, hw, ns, np] = build_J_(ctx, cfg, PL, msk, P, h0);
    nlit = numel(ilit);
    JtJ = full(J.'*J) - (vcol*vcol.')/Am;            % mean-referenced (piston-nulled)
    d = diag(JtJ);  l2 = P.battery.matrix_lam * median(d(d > 0));
    Rf = chol(JtJ + l2*eye(nlit));
    est = @(h) est_matrix_tg(h, J, vcol, Am, Rf, ilit, nact, msk);
    say('  J: %d states, %d columns (lit), window %d px, reg lambda %.2e (of median col energy)\n', ...
        ns, nlit, 2*hw+1, P.battery.matrix_lam);
    litmask = false(nact);  litmask(ilit) = true;
    measr = @(A) meanref_(ctx.measf(dm_influence_map(N_G,DX_G,'nact',nact,'pitch',cfg.pitch,'act',A)) - h0, msk);

    % ---- Stage C: single actuator gain + floor (actuator space) ----------
    ic = pick_lit_(PL, cfg, 0.0);                    % an in-pupil actuator
    Ac1 = zeros(nact);  Ac1(ic(1),ic(2)) = P.battery.single_nm*1e-6;
    ah = est(measr(Ac1));
    g_sng = ah(ic(1),ic(2)) / (P.battery.single_nm*1e-6);
    resid = ah;  resid(ic(1),ic(2)) = 0;
    fl_sng = 1e9*sqrt(mean(resid(litmask).^2));       % pm
    say('  Stage C single actuator @%dnm: gain %.4f, off-target floor %.1f pm\n', ...
        P.battery.single_nm, g_sng, fl_sng);

    % ---- Stage D: modal transfer (12 modes) ------------------------------
    PQ = P.battery.transfer_PQ;  nM = size(PQ,1);  iiv = (1:nact)';
    frq = zeros(1,nM);  gt = zeros(1,nM);  xt = zeros(1,nM);
    say('Stage D -- modal transfer (matrix estimator):\n');
    say('  %-9s %8s %8s %11s\n','mode(p,q)','cyc/pup','gain','cross-talk');
    for k = 1:nM
        p = PQ(k,1);  q = PQ(k,2);
        Ak = cos(pi*p*iiv/nact)*cos(pi*q*iiv/nact)';  Ak = Ak/max(abs(Ak(:)));
        Akc = P.POKE * (Ak .* litmask);
        ahk = est(measr(Akc));
        tvec = Akc(litmask);  avec = ahk(litmask);
        gt(k) = (tvec.'*avec)/(tvec.'*tvec);          % LS modal gain
        xt(k) = norm(avec - gt(k)*tvec)/norm(tvec);    % cross-talk / leakage
        frq(k) = hypot(p,q)/2;
        say('  %3d,%-5d %8.1f %8.4f %11.4f\n', p,q,frq(k),gt(k),xt(k));
    end

    % ---- Stage E: differential rows (deviation about a working state) ----
    say('Stage E -- differential rows (matrix estimator; actuator space, pm):\n');
    rng(P.battery.base_rand_seed);  base_r = zeros(nact);
    base_r(litmask) = P.battery.base_rand_nm*1e-6*randn(nnz(litmask),1);
    d_sng = zeros(nact);  d_sng(ic(1),ic(2)) = P.battery.diff_single_nm*1e-6;
    rng(P.battery.diff_rand_seed);  d_rnd = zeros(nact);
    d_rnd(litmask) = P.battery.diff_rand_nm*1e-6*randn(nnz(litmask),1);
    bases = {'flat', zeros(nact); sprintf('random %dnm',P.battery.base_rand_nm), base_r};
    devs  = {sprintf('single %dnm',P.battery.diff_single_nm), d_sng; ...
             sprintf('random %dnm',P.battery.diff_rand_nm),   d_rnd};
    say('  %-14s %-14s %8s %10s %8s\n','base','deviation','gain','resid pm','corr');
    rows = {};
    for bi = 1:2
        hb = measr(bases{bi,2});
        for di = 1:2
            hbd = measr(bases{bi,2} + devs{di,2});
            adev = est(hbd - hb);                     % linear estimator: differential
            tv = devs{di,2}(litmask);  av = adev(litmask);
            g = (tv.'*av)/(tv.'*tv);
            r = 1e9*sqrt(mean((av - tv).^2));          % pm
            cc = corrcoef(av, tv);
            say('  %-14s %-14s %8.4f %10.1f %8.4f\n', bases{bi,1}, devs{di,1}, g, r, cc(1,2));
            rows(end+1,:) = {bases{bi,1}, devs{di,1}, g, r, cc(1,2)}; %#ok<AGROW>
        end
    end
    % ---- break ladder: single-10nm differential vs increasing working state -
    say('Stage E break ladder -- single %dnm differential vs base rms (%s):\n', ...
        P.battery.diff_single_nm, P.bench.optics);
    say('  %-10s %8s %10s %8s  %s\n','base rms','gain','floor pm','corr','note');
    lad = P.battery.break_ladder;  brk = zeros(numel(lad),5);
    qwave = ctx.LAM/4;                               % per-pixel four-step wrap threshold (surface)
    for j = 1:numel(lad)
        rng(P.battery.base_rand_seed);  bb = zeros(nact);
        bb(litmask) = lad(j)*1e-6*randn(nnz(litmask),1);
        hb = measr(bb);  hbd = measr(bb + d_sng);
        adev = est(hbd - hb);  tv = d_sng(litmask);  av = adev(litmask);
        g = (tv.'*av)/(tv.'*tv);  r = 1e9*sqrt(mean((av-tv).^2));  cc = corrcoef(av,tv);
        pwrap = max(abs(hb(msk)));                    % how far the base reading reaches vs lambda/4
        broke = ~isfinite(g) || g < 0 || g > 3 || cc(1,2) < 0.3;   % estimator diverged (the WRAP symptom)
        note = iff_(broke, sprintf('BROKE (wrap: base reads %.2f of lambda/4)', pwrap/qwave), '');
        say('  %6.0f nm %8.4f %10.1f %8.4f  %s\n', lad(j), g, r, cc(1,2), note);
        brk(j,:) = [lad(j) g r cc(1,2) double(broke)];
    end
    % ---- item 3b: regularization sweep on the dense-random row; is the OAP
    %      dense loss the reg shrinking the DIM (dark ~25%) columns? report the
    %      gain over bright vs dark columns separately, for a few matrix_lam ----
    cn = sqrt(diag(JtJ));                             % per-lit-column energy
    md = median(cn(cn>0).^2);
    darkc = cn < prctile(cn, 25);                     % bottom-25% column energy = the dim columns
    dk = false(nact);  dk(ilit(darkc)) = true;  br = false(nact);  br(ilit(~darkc)) = true;
    cn_map = zeros(nact);  cn_map(ilit) = cn;          % column norms over the lattice (item 4 picture)
    hrnd = measr(d_rnd) - measr(zeros(nact));          % flat/random-10nm differential response
    lamsw = P.battery.matrix_lam_sweep;  regrows = zeros(numel(lamsw),4);
    say('Stage E reg sweep -- dense random %dnm gain vs matrix_lam (bright vs dark 25%% columns):\n', P.battery.diff_rand_nm);
    say('  %-10s %8s %8s %8s\n','lambda','all','bright','dark');
    for li = 1:numel(lamsw)
        Rf_l = chol(JtJ + lamsw(li)*md*eye(nlit));
        est_l = @(h) est_matrix_tg(h, J, vcol, Am, Rf_l, ilit, nact, msk);
        a_l = est_l(hrnd);
        gg = @(m) (d_rnd(m).'*a_l(m))/(d_rnd(m).'*d_rnd(m));
        regrows(li,:) = [lamsw(li) gg(litmask) gg(br) gg(dk)];
        say('  %-10.0e %8.4f %8.4f %8.4f\n', regrows(li,1),regrows(li,2),regrows(li,3),regrows(li,4));
    end
    % ---- Stage D4: OAP alignment sensitivity (perturb OAP1/OAP2, re-read) ---
    d4 = [];
    if P.battery.d4 && strcmp(P.bench.optics,'oap')
        d4 = stage_D4_(P, ctx, PL, h0, msk, d_sng, litmask, est, say);
    end
    battery = struct('mode','matrix', 'place',place, 'null_nm',null_nm, 'g_sng',g_sng, 'fl_sng_pm',fl_sng, ...
        'PQ',PQ, 'frq',frq, 'transfer_gain',gt, 'transfer_xtalk',xt, 'diff_rows',{rows}, ...
        'break_ladder',brk, 'reg_sweep',regrows, 'cn_map',cn_map, 'darkmask',{dk}, ...
        'window',P.battery.matrix_window, 'calib_surface',P.battery.calib_surface, ...
        'd4',d4, 'nlit',nlit, 'nstates',ns, 'hw',hw);
end

function d4 = stage_D4_(P, ctx, PL, h0, msk, d_sng, litmask, est, say)
% OAP1/OAP2 decenter + tilt sensitivity: perturb one element (SI m / rad, local
% frame) and re-read the flat-DM null shift + the single-actuator differential
% through the UNPERTURBED calibration.  Reports nm per um and per urad, and what
% the differential leaves.  measf reloads + re-applies A.pert each call.
    cfg = P.dm(1);  N_G = ctx.N_G;  DX_G = ctx.DX_G;  QWP = ctx.QWP;
    THETAS = ctx.THETAS;  LAM = ctx.LAM;
    iO1 = find(strcmp({ctx.AT.b.E.name},'L1'), 1);   % collimator OAP1
    iO2 = find(strcmp({ctx.AT.b.E.name},'L2'), 1);   % focuser   OAP2
    dec  = P.battery.d4_dec_um  * 1e-6;              % m
    tilt = P.battery.d4_tilt_urad * 1e-6;            % rad
    P4 = { 'OAP1 decenter', iO1, [0;0;0],    [dec;0;0],  P.battery.d4_dec_um,  'nm/um';
           'OAP1 tilt',     iO1, [0;tilt;0], [0;0;0],    P.battery.d4_tilt_urad,'nm/urad';
           'OAP2 decenter', iO2, [0;0;0],    [dec;0;0],  P.battery.d4_dec_um,  'nm/um';
           'OAP2 tilt',     iO2, [0;tilt;0], [0;0;0],    P.battery.d4_tilt_urad,'nm/urad' };
    say('Stage D4 -- OAP alignment sensitivity (%d um decenter, %d urad tilt, one at a time):\n', ...
        P.battery.d4_dec_um, P.battery.d4_tilt_urad);
    say('  %-14s %10s %12s %9s %10s\n','perturb','null nm','sens','sng gain','resid pm');
    rows = {};
    for k = 1:size(P4,1)
        Ap = ctx.AT;  Ap.pert = struct('iElt',P4{k,2}, 'rot',P4{k,3}, 'trans',P4{k,4});
        mp = @(Ac) meanref_(meas_surface(Ap, QWP, dm_influence_map(N_G,DX_G,'nact',cfg.nact,'pitch',cfg.pitch,'act',Ac), ...
                                         ctx.Sr, ctx.p_null, THETAS, LAM), msk);
        hflat = mp(zeros(cfg.nact));                 % perturbation-induced OPD (mm), piston-removed
        null_nm = 1e6*std(hflat(msk));
        adev = est(mp(d_sng) - hflat);               % single-act differential on the perturbed rig
        tv = d_sng(litmask);  av = adev(litmask);
        g = (tv.'*av)/(tv.'*tv);  r = 1e9*sqrt(mean((av-tv).^2));
        say('  %-14s %10.3f %8.3f %s %9.4f %10.1f\n', P4{k,1}, null_nm, null_nm/P4{k,5}, P4{k,6}, g, r);
        rows(end+1,:) = {P4{k,1}, null_nm, null_nm/P4{k,5}, g, r}; %#ok<AGROW>
    end
    d4 = struct('rows',{rows}, 'dec_um',P.battery.d4_dec_um, 'tilt_urad',P.battery.d4_tilt_urad);
end

% =====================================================================
%  Stage LOOP (D7): the closed-loop hold metric (Dave 2026-09-11)
% =====================================================================
function LO = stage_loop_(P, s, G, bench, say, place)
% The on-orbit servo mode (BRIEF_loop_metric): the DM held at the working
% surface by a proportional loop of gain g, closed through the four-step PSI
% reading and its measured response matrix (calibrated ON the working surface,
% S10). ONE reading here (the four-step map), so no readings dimension -- the
% ZWFS runs L/I+/S/V. The loop code is SHARED: ../dm_gauge_lib/dmg_loop.m,
% gated by tests/tDmgLoop.m on a synthetic instrument. This stage assembles the
% instrument handles + the run matrix EXACTLY as zwfs_run's stage_loop_ and the
% same drift seed, so the two gauges run the identical realizations. Numbers in
% actuator surface units (pm over lit).
t0 = tic;
cfg = P.dm(1);  nact = cfg.nact;
g = P.loop.g;  K = P.loop.K;  NPH = P.loop.nph(:).';  DR = P.loop.drifts;
say('\n---- loop: closed-loop hold, DM %dx%d, four-step reading ----\n', nact, nact);
say(['loop: gain %.2f, %d cycles (steady state = last %d), set point = %s, ' ...
    'reference frames %s, drift seed %d; each cycle = ONE measurement (the DM ' ...
    'shape traced once, the four frames with N photons split nph/4), differential ' ...
    'to the set point through the measured matrix\n'], g, K, floor(K/2), P.loop.surface, P.loop.ref, P.loop.seed);
say(['dynamics: r(k+1) = (1 - gG) r(k) - gG e(k) + d(k+1); noise-only rms = ' ...
    'sig_n sqrt(gG/(2-gG)); walk rms^2 = (sig_d^2 + g^2G^2 sig_n^2)/(gG(2-gG)); ' ...
    'ramp lag = rate/(gG)\n']);

% ---- placement (reuse or bootstrap the windows the matrix needs) ---------
if ~isfield(place, 'PL')
    Pp = P;  Pp.place.gate_assert = false;
    Pp.place.gate_max_states = P.place.boot_states;
    place = stage_place_(Pp, s, G, bench, say);
end
ctx = place.ctx;  PL = place.PL;  h0 = place.h0;
msk = ctx.msk;  N_G = ctx.N_G;  DX_G = ctx.DX_G;  LAM = ctx.LAM;  QWP = ctx.QWP;
THETAS = ctx.THETAS;  Am = nnz(msk);  AT = ctx.AT;  Sr = ctx.Sr;

% flat-DM null (the same-plane-fold arm difference; reported for context) --
hn = (ctx.p_null - median(ctx.p_null(msk))) * LAM/(4*pi) * 1e6;
null_nm = std(hn(msk));
say('  flat-DM null: %.4f nm rms surface (%.1f pm) -- the same-plane-fold arm difference\n', null_nm, 1e3*null_nm);

% ---- calibration ON the set point (matrix measured on the working surface) -
Pl = P;  Pl.battery.calib_surface = iff_(strcmp(P.loop.surface, 'base'), 'base', 'flat');
[J, ilit, vcol, hw, ns] = build_J_(ctx, cfg, PL, msk, Pl, h0);
nlit = numel(ilit);
JtJ = full(J.'*J) - (vcol*vcol.')/Am;            % mean-referenced (piston-nulled)
dd = diag(JtJ);  l2 = P.battery.matrix_lam * median(dd(dd > 0));
Rf = chol(JtJ + l2*eye(nlit));
est = @(h) est_matrix_tg(h, J, vcol, Am, Rf, ilit, nact, msk);
litmask = false(nact);  litmask(ilit) = true;
say(['calibration: measured response matrix on the %s (%s); lit actuators %d, ' ...
    'J %d states x %d columns, window %d px, reg lambda %.2e\n'], ...
    Pl.battery.calib_surface, iff_(strcmp(Pl.battery.calib_surface,'base'), ...
    sprintf('%g nm rms, seed %d', P.battery.base_rms*1e6, P.battery.seed_base), 'flat DM'), ...
    nnz(litmask), ns, nlit, 2*hw+1, P.battery.matrix_lam);

% ---- the set point command A0 (== the Abase build_J_ calibrated on) -------
if strcmp(Pl.battery.calib_surface, 'base')
    rng(P.battery.seed_base);  A0 = zeros(nact);
    A0(litmask) = P.battery.base_rms*randn(nnz(litmask), 1);   % matches build_J_ (same seed/rms)
else
    A0 = zeros(nact);
end

% ---- the instrument (four-step PSI reading; frame-level so noise injects) --
dmap = @(A) dm_influence_map(N_G, DX_G, 'nact',nact, 'pitch',cfg.pitch, 'act',A);
ins = struct('lit', litmask, ...
    'measure', @(cmd) frames4_(analyzer_basis(AT, QWP, dmap(cmd)), Sr, THETAS), ...
    'noisy',   @(F, nph, seed) noisy4_(F, nph, seed), ...
    'diff',    @(F1, F0) meanref_(fsdiff_(F1, F0, LAM), msk), ...
    'est',     est);
base = struct('A0',A0, 'g',g, 'K',K, 'seed',P.loop.seed, 'ref',P.loop.ref, 'rmax',P.loop.rmax);

% ---- the runs --------------------------------------------------------------
res = struct('drift',{}, 'nph',{}, 'amp',{}, 'L',{});
kinds = DR;  if P.loop.floor, kinds = [{'none'} DR]; end
nrun = numel(P.loop.steps) + numel(NPH)*numel(kinds);
say('%d loop runs of %d states each (%d traced states)\n', nrun, K+1, nrun*(K+1));
irun = 0;
% noiseless steps: time constant + dynamic range
for amp = P.loop.steps
    o = base;  o.nph = Inf;  o.drift = struct('kind','step', 'amp',amp);
    L = dmg_loop(ins, o);  irun = irun + 1;
    res(end+1) = struct('drift','step', 'nph',Inf, 'amp',amp, 'L',L); %#ok<AGROW>
    fprintf('[loop %d/%d] step %g nm: rho %.3f, residual at K %.2f pm%s (%.1f min)\n', ...
        irun, nrun, amp*1e6, L.rho, L.rms(L.k_end)*1e9, div_(L), toc(t0)/60);
end
for nph = NPH
    for kd = 1:numel(kinds)
        o = base;  o.nph = nph;
        switch kinds{kd}
            case 'none',    o.drift = struct('kind','none');  amp = 0;
            case 'walk',    o.drift = struct('kind','walk', 'sigma',P.loop.walk_sigma);  amp = P.loop.walk_sigma;
            case 'thermal', o.drift = struct('kind','thermal', 'rate',P.loop.thermal_rate);  amp = P.loop.thermal_rate;
            otherwise,      error('tg96_run: loop.drifts must be a subset of walk | thermal');
        end
        L = dmg_loop(ins, o);  irun = irun + 1;
        res(end+1) = struct('drift',kinds{kd}, 'nph',nph, 'amp',amp, 'L',L); %#ok<AGROW>
        fprintf('[loop %d/%d] %s @ %.0e photons: ss %.2f pm, bias %.2f pm, sig_n %.2f pm%s (%.1f min)\n', ...
            irun, nrun, kinds{kd}, nph, L.ss*1e9, L.bias*1e9, L.sig_n*1e9, div_(L), toc(t0)/60);
    end
end

% ---- tables ----------------------------------------------------------------
pm = @(x) x*1e9;
say(['\nstep response (noiseless; a step of the given rms at cycle 1). rho = fitted ' ...
    'per-cycle contraction (1 - gG), tau = cycles to 1/e, k1e = first cycle below 1/e, ' ...
    'r(K/2)/r(K) = residual (pm) at cycles %d and %d -- a residual that stops falling ' ...
    'is the noiseless bias floor on this surface\n'], floor(K/2), K);
say('  %6s %8s %8s %6s %11s %11s\n','step','rho','tau','k1e','r(K/2) pm','r(K) pm');
for amp = P.loop.steps
    i = find(strcmp({res.drift},'step') & [res.amp]==amp, 1);  L = res(i).L;
    if L.diverged
        say('  %4.0f nm  DIVERGED at cycle %d (%.0f pm)\n', amp*1e6, L.k_end, pm(L.rms(L.k_end)));
    else
        say('  %4.0f nm %8.3f %8.1f %6s %11.3f %11.3f\n', amp*1e6, L.rho, L.tau, fmt0_(L.k_1e), pm(L.rms(floor(K/2))), pm(L.rms(end)));
    end
end
for kd = 1:numel(kinds)
    switch kinds{kd}
        case 'none',    lab = 'noise only (drift 0): the G2 line, ss vs sig_n sqrt(g/(2-g))';
        case 'walk',    lab = sprintf('random walk, %g pm per actuator per cycle', P.loop.walk_sigma*1e9);
        case 'thermal', lab = sprintf('thermal ramp, %g pm rms per cycle (defocus + astigmatism)', P.loop.thermal_rate*1e9);
    end
    say(['\nhold error vs photons per cycle (= per measurement, one per cycle) -- %s. ' ...
        'ss = steady-state rms over lit (pm), bias = rms of the mean residual (noise averaged ' ...
        'out), sig_n = single-shot estimate noise (pm), th = the theory line from sig_n\n'], lab);
    say('  %9s %8s %8s %9s %8s\n','N/cycle','ss pm','bias pm','sig_n pm','th pm');
    for nph = NPH
        i = find(strcmp({res.drift},kinds{kd}) & [res.nph]==nph, 1);  L = res(i).L;
        switch kinds{kd}
            case 'none',    th = L.theory.ss_noise;
            case 'walk',    th = L.theory.ss_walk;
            case 'thermal', th = hypot(L.theory.lag_ramp, L.theory.ss_noise);
        end
        if L.diverged, say('  %9.1e  DIVERGED at cycle %d\n', nph, L.k_end);
        else, say('  %9.1e %8.2f %8.2f %9.2f %8.2f\n', nph, pm(L.ss), pm(L.bias), pm(L.sig_n), pm(th)); end
    end
end
% the one number: photons per cycle to hold the spec
spec = P.loop.hold_spec;
say(['\nphotons per cycle to hold %.1f pm rms (log-log interpolation of ss over N; ' ...
    '''floor x'' = not reached: the noise-free residual sits at x pm):\n'], spec*1e9);
n_hold = nan(numel(kinds),1);
for kd = 1:numel(kinds)
    ss = nan(1,numel(NPH));  bias = ss;  dv = false(1,numel(NPH));
    for q = 1:numel(NPH)
        i = find(strcmp({res.drift},kinds{kd}) & [res.nph]==NPH(q), 1);
        ss(q) = res(i).L.ss;  bias(q) = res(i).L.bias;  dv(q) = res(i).L.diverged;
    end
    if all(dv), txt = 'DIVERGED';  n_hold(kd) = NaN;
    else, [n_hold(kd), txt] = hold_photons_(NPH, ss, bias, spec); end
    say('  %-8s %s\n', kinds{kd}, txt);
end
% spectrum of the held residual at the highest photon level
say('\nspectrum of the held residual at %.0e photons per cycle: rms (pm) in [< 4, 4-12, > 12] cycles per aperture\n', NPH(end));
for kd = 1:numel(kinds)
    i = find(strcmp({res.drift},kinds{kd}) & [res.nph]==NPH(end), 1);
    say('  %-8s %6.2f %6.2f %6.2f\n', kinds{kd}, pm(res(i).L.spec.band));
end
say('loop stage %.1f min (%d traced states)\n', toc(t0)/60, nrun*(K+1));
LO = struct('drifts',{kinds}, 'nph',NPH, 'steps',P.loop.steps, 'g',g, 'K',K, ...
    'surface',P.loop.surface, 'hold_spec',spec, 'n_hold',n_hold, 'lit',litmask, ...
    'A0',A0, 'null_nm',null_nm, 'nlit',nlit, 'nstates',ns, 'res',res);
end

% =====================================================================
%  Stage JONES (item B): OAP-fold retardance + fringe visibility
% =====================================================================
function JZ = stage_jones_(P, s, G, bench, say) %#ok<INUSD>
% The mechanism run behind the D5 reframe (brief oap3 item B): is the OAP
% dense-loss null the perfect conductor's (a knife-edge idealization) or a real
% fold cost?  For ideal / bareAl / protectedAl OAPs, report (a) the retardance
% of L1 -- the collimating fold, element 2, BEFORE PolIn, so jones_pupil harvests
% the pure fold Jones -- via macos.jones_pupil + macos.pol_maps (double-pole
% basis; the mean is a state, only the VARIATION is an aberration), and (b) the
% four-step fringe visibility V = 2*AC/DC of the flat-DM gauge, in the central
% band (where the ideal null darkens the pupil, D5) vs the pupil edge.  A single
% opaque Al layer that fills the band settles the question numbers-first.
JZ = struct('case',{},'ret_mean_mrad',{},'ret_var_mrad',{},'V_band',{},'V_edge',{},'lit_band',{},'lit_edge',{});
if ~strcmp(P.bench.optics,'oap'), say('\nStage JONES: OAP rig only; skipped.\n'); return; end
QWP = P.QWP;  THETAS = P.THETAS;
rxT = [P.tag '_test.in'];  rxR = [P.tag '_ref.in'];
cases = {'ideal',[]; 'bareAl',P.bench.coat_bareAl; 'protectedAl',P.bench.coat_protectedAl};
say('\n---- Stage JONES (item B): OAP-fold retardance + fringe visibility ----\n');
say(['L1 (collimating OAP fold) retardance via jones_pupil+pol_maps (double-pole, mrad; ' ...
    'mean = a state, var = the aberration), and the four-step fringe visibility ' ...
    'V=2*AC/DC (flat DM) in the central band vs the pupil edge, plus the lit fraction there\n']);
say('  %-12s %10s %10s %9s %9s %9s %9s\n','coating','ret mean','ret var','V band','V edge','lit band','lit edge');
for c = 1:size(cases,1)
    cs = cases{c,2};
    AT = arm_desc(rxT, G.bt, G.T, 0);  AR = arm_desc(rxR, G.br, G.R, 45);
    if ~isempty(cs), AT = set_oap_coat_(AT,cs);  AR = set_oap_coat_(AR,cs); end
    Sr = analyzer_basis(AR, QWP, []);
    Sx = analyzer_basis(AT, QWP, []);
    Fr = frames4_(Sx, Sr, THETAS);
    I1 = Fr(:,:,1);  I2 = Fr(:,:,2);  I3 = Fr(:,:,3);  I4 = Fr(:,:,4);
    DC = I1+I2+I3+I4;  AC = sqrt((I1-I3).^2 + (I2-I4).^2);
    msk = DC > 0.1*max(DC(:));  V = 2*AC ./ max(DC, eps);
    [rr,cc] = find(msk);  r0 = mean(rr);  c0 = mean(cc);
    [CG,RG] = meshgrid(1:size(DC,2), 1:size(DC,1));  rad = hypot(CG-c0, RG-r0);
    R = max(rad(msk));  band = rad < 0.20*R;  edge = rad >= 0.80*R & rad <= R;
    litb = mean(msk(band));  lite = mean(msk(edge));
    Vb = mean(V(band & msk));  Ve = mean(V(edge & msk));
    load_arm(AT, QWP, 0, []);                       % load coated rx for jones_pupil
    iL1 = find(strcmp({AT.b.E.name},'L1'), 1);
    pm = macos.pol_maps(macos.jones_pupil(iL1));
    rm = 1e3*pm.mean.ret;  rv = 1e3*pm.var_rms.ret;
    say('  %-12s %10.2f %10.2f %9.3f %9.3f %9.2f %9.2f\n', cases{c,1}, rm, rv, Vb, Ve, litb, lite);
    JZ(end+1) = struct('case',cases{c,1}, 'ret_mean_mrad',rm, 'ret_var_mrad',rv, ...
        'V_band',Vb, 'V_edge',Ve, 'lit_band',litb, 'lit_edge',lite); %#ok<AGROW>
end
end

function Fr = frames4_(Sx, Sr, th)
% the four analyzer-step intensity frames (noiseless) for a test-arm state Sx
% against the fixed reference-arm basis Sr
Fr = cat(3, frame(Sx,Sr,th(1)), frame(Sx,Sr,th(2)), frame(Sx,Sr,th(3)), frame(Sx,Sr,th(4)));
end

function Fn = noisy4_(F, nph, seed)
% photon noise on the four captured frames: nph photons per MEASUREMENT split
% nph/4 over the four frames (the S5 shot model, lifted from zwfs noisy_frames_).
% nph = Inf returns F unchanged.
Fn = F;
if ~isfinite(nph), return; end
rs = RandStream('mt19937ar', 'Seed', seed);
for k = 1:size(F,3)
    I = F(:,:,k);
    Fn(:,:,k) = I .* (1 + randn(rs, size(I)) ./ sqrt(max(I / sum(I(:)) * (nph/4), 1)));
end
end

function d = fsdiff_(F1, F0, LAM)
% the four-step differential surface map (mm): the phase DIFFERENCE wrapped
% (the V1 lesson -- wrap the difference, not each absolute map at +/-pi).
p1 = atan2(F1(:,:,2)-F1(:,:,4), F1(:,:,1)-F1(:,:,3));
p0 = atan2(F0(:,:,2)-F0(:,:,4), F0(:,:,1)-F0(:,:,3));
d = angle(exp(1i*(p1 - p0))) * LAM/(4*pi);
end

function t = div_(L)
if L.diverged, t = sprintf(' DIVERGED at cycle %d', L.k_end); else, t = ''; end
end

function s = fmt0_(x)
if isnan(x), s = '-'; else, s = sprintf('%d', x); end
end

function [n, txt] = hold_photons_(NPH, ss, bias, spec)
% (verbatim from zwfs_run) photons per cycle at which the steady-state rms
% crosses spec (log-log interpolation); NaN + a reason when it never crosses
n = NaN;
if all(ss > spec)
    txt = sprintf('floor %.1f', min(ss)*1e9);
    if ss(end) > spec && bias(end) < spec, txt = sprintf('> %.0e', NPH(end)); end
    return
end
if ss(1) <= spec, n = NPH(1);  txt = sprintf('< %.0e', NPH(1));  return; end
q = find(ss <= spec, 1);                                     % first point at or under spec
x = log(NPH(q-1:q));  y = log(ss(q-1:q));
n = exp(x(1) + (log(spec) - y(1)) * (x(2)-x(1)) / (y(2)-y(1)));
txt = sprintf('%.1e', n);
end

function draw_loop_(LO, P)
% the loop figure (mirror of zwfs_run_figs's loop panels, single reading):
% residual per cycle per photon level (+ the noiseless step), and the hold
% error vs photons per cycle for each drift.
res = LO.res;  NPH = LO.nph;  kinds = LO.drifts;  K = LO.K;
kshow = 'walk';  if ~any(strcmp(kinds,'walk')), kshow = kinds{end}; end
ramp = [209 229 240; 146 197 222; 67 147 195; 33 102 172; 8 48 107]/255;   % ordinal blues
f = figure('Color','w','Position',[100 100 1400 560],'Visible','off');
tl = tiledlayout(1,2,'Padding','compact','TileSpacing','compact');
tl.Title.String = sprintf('%s: closed-loop hold at gain %.2f on the %s, %d cycles, matrix calibration', P.tag, LO.g, LO.surface, K);
tl.Title.Interpreter = 'none';
ax = nexttile;  hold(ax,'on');
for q = 1:numel(NPH)
    i = find(strcmp({res.drift},kshow) & [res.nph]==NPH(q), 1);
    if isempty(i), continue; end
    cc = ramp(max(1, round((q-1)/max(numel(NPH)-1,1)*(size(ramp,1)-1))+1), :);
    semilogy(ax, 1:K, res(i).L.rms*1e9, '-', 'Color',cc, 'LineWidth',1.8, 'DisplayName',sprintf('%.0e photons/cycle',NPH(q)));
end
i = find(strcmp({res.drift},'step'), 1);
if ~isempty(i), semilogy(ax, 1:K, res(i).L.rms*1e9, '--', 'Color',[.5 .5 .5], 'LineWidth',1.5, 'DisplayName',sprintf('noiseless %g nm step',res(i).amp*1e6)); end
set(ax,'YScale','log');  grid(ax,'on');
yline(ax, LO.hold_spec*1e9, ':', sprintf('%g pm hold spec',LO.hold_spec*1e9), 'HandleVisibility','off');
xlabel(ax,'cycle');  ylabel(ax,'residual surface error over lit, pm rms');
title(ax, sprintf('four-step reading, %s drift: residual per cycle', kshow));
legend(ax,'Location','northeast');
ax = nexttile;  hold(ax,'on');
lst = struct('none',':', 'walk','-', 'thermal','--', 'step','-.');
for kd = 1:numel(kinds)
    ss = nan(1,numel(NPH));
    for q = 1:numel(NPH)
        i = find(strcmp({res.drift},kinds{kd}) & [res.nph]==NPH(q), 1);
        if ~isempty(i), ss(q) = res(i).L.ss*1e9; end
    end
    loglog(ax, NPH, ss, [lst.(kinds{kd}) 'o'], 'LineWidth',2, 'MarkerSize',6, 'DisplayName',kinds{kd});
end
set(ax,'XScale','log','YScale','log');  grid(ax,'on');
yline(ax, LO.hold_spec*1e9, ':', sprintf('%g pm hold spec',LO.hold_spec*1e9), 'HandleVisibility','off');
xlabel(ax,'photons per cycle (one measurement per cycle; the four frames share it)');
ylabel(ax,'steady-state hold error over lit, pm rms');
title(ax,'hold error vs photons per cycle (dotted none, solid walk, dashed thermal)');
legend(ax,'Location','best');
fn = [P.tag '_loop.png'];
exportgraphics(f, fn, 'Resolution',110);  close(f);
fprintf('wrote %s\n', fn);
end

function [J, ilit, vcol, hw, ns, np] = build_J_(ctx, cfg, PL, msk, P, h0)
% assemble the sparse response matrix J (detector px x lit actuators): each
% multiplexed poke's response, per unit command, cut from its affine-placed
% window, zeroed outside the mask, mean-referenced over the mask.  Lift of the
% ZWFS calib_matrix_ inner loop, meas_surface as the map primitive.
%   P.battery.matrix_window 'box' (default; +/-half-step window) | 'voronoi'
%     (assign every mask pixel to its NEAREST poked actuator of the frame --
%     nothing truncated, nothing double-counted; item 3a truncation test).
%   P.battery.calib_surface 'flat' (default) | 'base' (pokes ride a base
%     working state, measured differentially -- S10 doctrine; item 6).
    step = P.battery.matrix_step;  N = size(msk,1);  nact = cfg.nact;  POKE = P.POKE;
    hw = max(3, floor(0.5*step*cfg.pitch/(PL.mag*PL.dxd_mm)));
    win = 'box';  if isfield(P.battery,'matrix_window'), win = P.battery.matrix_window; end
    lit = PL.lit;  ilit = find(lit);  nlit = numel(ilit);
    col_of = zeros(nact);  col_of(ilit) = 1:nlit;  U = PL.U;  V = PL.V;
    dmap = @(A) dm_influence_map(ctx.N_G,ctx.DX_G,'nact',nact,'pitch',cfg.pitch,'act',A);
    % base working surface for calib_surface 'base' (differential calibration)
    hbase = h0;
    if isfield(P.battery,'calib_surface') && strcmp(P.battery.calib_surface,'base')
        rng(P.battery.seed_base);  Abase = zeros(nact);
        Abase(lit) = P.battery.base_rms*randn(nnz(lit),1);
        hbase = ctx.measf(dmap(Abase));  base_cmd = Abase;
    else
        base_cmd = zeros(nact);
    end
    % mask pixel coordinates (col,row) for the voronoi assignment
    [mr, mc] = find(msk);  mUV = [mc, mr];  mlin = mr + (mc-1)*N;
    I = {};  Jc = {};  V3 = {};  ns = 0;  np = 0;  maxst = P.battery.matrix_states;
    for ox = 1:step
      for oy = 1:step
        if ns >= maxst, break; end
        A = zeros(nact);  A(ox:step:nact, oy:step:nact) = 1;  A = A .* lit;
        if ~any(A(:)), continue; end
        if strcmp(P.battery.matrix_sign,'alternate')
            [rr,cc] = find(A);  sg = 1 - 2*mod((rr-ox)/step + (cc-oy)/step, 2);
            A(sub2ind([nact nact],rr,cc)) = sg;
        end
        ns = ns + 1;
        h = ctx.measf(dmap(base_cmd + POKE*A)) - hbase;
        h = (h - median(h(msk))) / POKE;              % response per unit command
        [pr, pc] = find(A);  npk = numel(pr);
        if strcmp(win,'voronoi')
            pUV = [U(sub2ind([nact nact],pr,pc)), V(sub2ind([nact nact],pr,pc))];
            near = dsearchn(pUV, mUV);                % nearest poke per mask pixel
            for q = 1:npk
                sel = near == q;  if ~any(sel), continue; end
                r = pr(q);  c = pc(q);  np = np + 1;  s = A(r,c);
                I{end+1} = mlin(sel);  Jc{end+1} = col_of(r,c)*ones(nnz(sel),1); ...
                    V3{end+1} = h(mlin(sel))*s; %#ok<AGROW>
            end
        else
            for q = 1:npk
                r = pr(q);  c = pc(q);  np = np + 1;
                u0 = round(U(r,c));  v0 = round(V(r,c));
                rows = max(1,v0-hw):min(N,v0+hw);  cols = max(1,u0-hw):min(N,u0+hw);
                [CC, RR] = meshgrid(cols, rows);
                blk = h(rows,cols) * A(r,c);  blk(~msk(rows,cols)) = 0;   % per unit +command
                I{end+1} = RR(:) + (CC(:)-1)*N;  Jc{end+1} = col_of(r,c)*ones(numel(blk),1);  V3{end+1} = blk(:); %#ok<AGROW>
            end
        end
      end
    end
    J = sparse(vertcat(I{:}), vertcat(Jc{:}), vertcat(V3{:}), N*N, nlit);
    vcol = full(sum(J, 1)).';
end

function a = est_matrix_tg(h, J, v, Am, Rf, ilit, nact, msk)
% actuator commands from a detector-space map by the measured matrix (columns
% = local window - v/Am over the mask; J'm = Jl'm - v (1'm)/Am).  Verbatim
% from the ZWFS est_matrix_.
    h(~msk) = 0;
    b = J.' * h(:) - v * (sum(h(msk))/Am);
    x = Rf \ (Rf.' \ b);
    a = zeros(nact);  a(ilit) = x;
end

function h = meanref_(h, msk)
    h(~msk) = 0;  h = h - median(h(msk));
end

function s = iff_(c, a, b), if c, s = a; else, s = b; end, end

function cs = oap_coat_stack_(P)
% the selected OAP coating stack (struct .index/.extinc/.thickness), or [] for
% the ideal reflector / non-OAP rig (brief item B).
cs = [];
if ~strcmp(P.bench.optics,'oap') || ~isfield(P.bench,'coat_oap'), return; end
switch P.bench.coat_oap
    case {'none',''},   cs = [];
    case 'bareAl',      cs = P.bench.coat_bareAl;
    case 'protectedAl', cs = P.bench.coat_protectedAl;
    otherwise, error('tg96_run: bench.coat_oap must be none | bareAl | protectedAl');
end
end

function A = set_oap_coat_(A, cs)
% attach the coating stack cs to the arm's OAP mirrors L1 (collimator) and L2
% (focuser), for load_arm to apply each trace.
nm = {A.b.E.name};  iL = [find(strcmp(nm,'L1'),1), find(strcmp(nm,'L2'),1)];
A.coat = struct('iElt',{}, 'n',{}, 'k',{}, 't',{});
for j = 1:numel(iL)
    A.coat(end+1) = struct('iElt',iL(j), 'n',cs.index, 'k',cs.extinc, 't',cs.thickness); %#ok<AGROW>
end
end

function rc = pick_lit_(PL, cfg, ~)
% an in-pupil actuator OFF the footprint centroid (for the single-poke tests):
% the exact centre reads ~0 (four-step chief-pixel reference), so target a
% site ~30% of the half-extent off the centroid, snapped to the lit set.
    [rr,cc] = find(PL.lit);  r0 = mean(rr);  c0 = mean(cc);
    hr = 0.5*(max(rr)-min(rr));  hc = 0.5*(max(cc)-min(cc));
    tr = r0 + 0.30*hr;  tc = c0 + 0.30*hc;                 % off-centre target
    d = (rr-tr).^2 + (cc-tc).^2;  [~,i] = min(d);  rc = [rr(i) cc(i)];
end

% =====================================================================
%  Stage PLACE (D1): affine window placement + the CoM gate
% =====================================================================
function place = stage_place_(P, s, G, bench, say)
    ctx = arm_setup_(P, G);
    cfg = P.dm(1);  N_G = ctx.N_G;  DX_G = ctx.DX_G;  msk = ctx.msk;  nact = cfg.nact;
    say('Stage PLACE -- affine window placement (%s rig):\n', P.bench.optics);
    % flat (unpoked) frame: subtracted from every poke frame so the OAP's
    % 12.9 nm low-order null background does not bias a poke's CoM (differential)
    h0 = ctx.measf(zeros(N_G));
    if isfield(P.place,'diag') && P.place.diag
        ic = round(cfg.nact/2);  dmaps = {};  dsite = [ic ic; ic ic+16; ic ic+32];
        for j = 1:size(dsite,1)
            Ad = zeros(cfg.nact);  Ad(dsite(j,1),dsite(j,2)) = 1;
            dmaps{j} = ctx.measf(dm_influence_map(N_G,DX_G,'nact',cfg.nact,'pitch',cfg.pitch,'act',P.POKE*Ad)) - h0; %#ok<AGROW>
        end
        save(fullfile(P.outdir,[P.tag '_diag.mat']), 'dmaps','dsite','msk','-v7.3');
        say('  DIAG: saved %d single-poke maps + mask to %s_diag.mat\n', numel(dmaps), P.tag);
    end
    % ---- bootstrap: the ray affine (carries the fold rotation) --------------
    PL = tg96_place(ctx.AT, G.T, cfg, msk, N_G, DX_G, P.POKE, ctx.measf, P.place, h0);
    frm = PL.frm;  ang = atan2d(frm.Lm(2,1), frm.Lm(1,1));
    say('  ray affine: mag %.4f DM-mm/det-mm, det %.4f, in-plane rotation %+.2f deg, dxd %.4e mm\n', ...
        PL.mag, det(frm.Lm), ang, PL.dxd_mm);
    say('  anchor: blob px (%.2f,%.2f) <-> DM (%.3f,%.3f) mm; field parity [t=%d su=%+d sv=%+d] (ref-poke err %.2f px)\n', ...
        PL.anchor(1),PL.anchor(2),PL.anchor(3),PL.anchor(4), PL.parity(1),PL.parity(2),PL.parity(3), PL.ref.err(PL.ref.ib));
    say('  lit actuators: %d of %d\n', nnz(PL.lit), nact^2);
    % ---- ONE measurement pass: each lit actuator's response CoM, windows
    %      centred on the bootstrap placement (generous, half-inter-poke) -----
    mmpx = PL.mag * PL.dxd_mm;                          % DM-mm per detector pixel
    deg = P.place.poly_deg;
    % PASS 1: bootstrap (ray affine + parity) windows -> measure CoMs.  The
    % linear bootstrap places the pupil interior well but the fold's DISTORTION
    % pushes the edge off, so some edge windows miss (NaN).
    boot = min(P.place.gate_max_states, P.place.boot_states);
    [comU, comV, poked] = mux_com_(ctx, cfg, PL.U, PL.V, PL.lit, msk, P, mmpx, h0, boot);
    pk = poked(:);  gd = pk & isfinite(comU(:)) & isfinite(comV(:));
    assert(nnz(gd) >= 12, 'placement bootstrap caught only %d blobs -- affine/parity is off', nnz(gd));
    e0 = hypot(comU(pk)-PL.U(pk), comV(pk)-PL.V(pk));  frac0 = mean(e0 <= P.place.gate_px);
    say('  bootstrap (ray affine): %.2f%% within %g px, median %.2f px (%d of %d blobs caught)\n', ...
        100*frac0, P.place.gate_px, median(e0(isfinite(e0))), nnz(gd), nnz(pk));
    % how much of the residual is affine-inexpressible (distortion)?  compare
    % the deg-1 (affine) and deg-2 fit residuals on the SAME pass-1 CoMs.
    [~,~,~,~, ra1] = poly_place_(PL.axg, PL.ayg, comU, comV, gd, 1);
    [~,~,~,~, ra2] = poly_place_(PL.axg, PL.ayg, comU, comV, gd, 2);
    say('  fit residual: affine %.2f px, deg-2 %.2f px (the gap = fold distortion)\n', ra1, ra2);
    % degree-`deg` polynomial refit, then a SECOND window pass on it to recover
    % the edge actuators the linear bootstrap missed.
    [U, V, Cu, Cv, r1] = poly_place_(PL.axg, PL.ayg, comU, comV, gd, deg);
    say('  refined (deg-%d poly refit from %d CoMs): fit residual %.2f px\n', deg, nnz(gd), r1);
    [comU, comV, poked] = mux_com_(ctx, cfg, U, V, PL.lit, msk, P, mmpx, h0, P.place.gate_max_states);   % PASS 2 (gate: all)
    pk = poked(:);  gd = pk & isfinite(comU(:)) & isfinite(comV(:));
    [U, V, Cu, Cv, r2] = poly_place_(PL.axg, PL.ayg, comU, comV, gd, deg);  % final map
    e1 = hypot(comU(pk)-U(pk), comV(pk)-V(pk));         % NaN where still missed
    frac = mean(e1 <= P.place.gate_px);
    say('  refined pass 2 (deg-%d, %d of %d caught): fit residual %.2f px, gate %.2f%% within %g px, median %.2f px\n', ...
        deg, nnz(gd), nnz(pk), r2, 100*frac, P.place.gate_px, median(e1(isfinite(e1))));
    if isfield(P.place,'diag2') && P.place.diag2
        % decisive single-vs-multiplexed test: for the pokes of ONE state,
        % measure each SINGLY and compare CoM to the multiplexed CoM + the map.
        N = size(msk,1);  [cg,rg] = meshgrid(1:N,1:N);  hw = max(3, floor(0.5*P.battery.matrix_step*cfg.pitch/mmpx));
        Ac = zeros(cfg.nact);  Ac(1:P.battery.matrix_step:cfg.nact, 1:P.battery.matrix_step:cfg.nact) = 1;  Ac = Ac.*PL.lit;
        [pr,pc] = find(Ac);  ns = min(numel(pr), 60);  sU=nan(ns,1); sV=nan(ns,1);
        for q = 1:ns
            Aq = zeros(cfg.nact);  Aq(pr(q),pc(q)) = 1;
            hq = ctx.measf(dm_influence_map(N_G,DX_G,'nact',cfg.nact,'pitch',cfg.pitch,'act',P.POKE*Aq)) - h0;
            u0=round(U(pr(q),pc(q))); v0=round(V(pr(q),pc(q)));
            rows=max(1,v0-hw):min(N,v0+hw); cols=max(1,u0-hw):min(N,u0+hw);
            b=abs(hq(rows,cols)); b(~msk(rows,cols))=0;
            if max(b(:))>0, b(b<0.5*max(b(:)))=0; [CC,RR]=meshgrid(cols,rows);
                sU(q)=sum(CC(:).*b(:))/sum(b(:)); sV(q)=sum(RR(:).*b(:))/sum(b(:)); end
        end
        eS=hypot(sU-arrayfun(@(q)U(pr(q),pc(q)),(1:ns)'), sV-arrayfun(@(q)V(pr(q),pc(q)),(1:ns)'));
        say('  DIAG2 single-poke gate on %d actuators: within2px %.1f%%, dark %.1f%% (multiplexed was ~86%%/14%%)\n', ...
            ns, 100*mean(eS<=2), 100*mean(isnan(eS)));
    end
    PL.U = U;  PL.V = V;  PL.polyC = {Cu, Cv};  PL.poly_deg = deg;
    place = struct('PL',PL, 'frac',frac, 'frac_boot',frac0, 'med_err',median(e1(isfinite(e1))), ...
        'affine',struct('mag',PL.mag,'det',det(frm.Lm),'rot_deg',ang), 'poly_resid',r2, ...
        'parity',PL.parity, 'comU',{comU}, 'comV',{comV}, 'poked',{poked}, ...
        'ctx',ctx, 'h0',{h0}, 'mmpx',mmpx);
    say('  D1 GATE: %.2f%% of lit within %g px (need >= %.1f%%), median err %.2f px\n', ...
        100*frac, P.place.gate_px, 100*P.place.gate_frac, median(e1(isfinite(e1))));
    % radial diagnostic: fraction within gate + fraction dark(NaN), by radius
    rr = hypot(PL.axg(pk), PL.ayg(pk));  ee = e1;
    edges = linspace(0, max(rr)+eps, 6);
    say('  by radius (mm):        %s\n', sprintf('%6.1f ', edges(2:end)));
    for lab = ["within2px","dark(NaN)"]
        s = '';
        for b = 1:5
            sel = rr >= edges(b) & rr < edges(b+1);
            if lab=="within2px", val = mean(ee(sel) <= P.place.gate_px);
            else,                val = mean(isnan(ee(sel))); end
            s = [s sprintf('%6.2f ', val)]; %#ok<AGROW>
        end
        say('    %-10s %s\n', lab, s);
    end
    % report the ray affine's structure (SVD of Lm: the two axis scales + how
    % far off axis-aligned -- the fold shows as anamorphism + a rotation the
    % dihedral parity search cannot express; distortion beyond it needs the poly)
    [~, Sv, Vv] = svd(frm.Lm);  offax = atan2d(abs(Vv(2,1)), abs(Vv(1,1)));
    say('  ray-affine SVD: DM-mm/det-mm %.3f / %.3f, anamorphism %.2f%%, principal axis %+.2f deg off the DM axes\n', ...
        Sv(1,1), Sv(2,2), 100*(Sv(1,1)/Sv(2,2)-1), offax);
    % ---- non-vacuity: the best AXIS-ALIGNED (shear-free parity+scale) map --
    %  Fit it to the finite CoMs and gate over the SAME poked set (NaN=miss) so
    %  it is directly comparable to the affine's %frac.  This is what a
    %  parity+scale registration can express (register_two_pokes' family).
    %  VACUOUS on the lens (its mapping IS axis-aligned -> both pass); the
    %  meaningful comparison is on the OAP.  Note (item 5, item 4): for these
    %  near-normal OAP folds the map is near-axis-aligned (SVD anamorphism ~0),
    %  so the affine's edge over a WELL-ANCHORED parity map is modest -- the
    %  dominant reason register_two_pokes failed on the OAP was its CENTRE-poke
    %  anchor reading 0 (the four-step chief-pixel reference), which the affine
    %  route sidesteps with an off-centre anchor + the ray-fit linear part.
    [Ua, Va] = axis_aligned_map_(PL.axg, PL.ayg, comU, comV, gd);
    ea = hypot(comU(pk)-Ua(pk), comV(pk)-Va(pk));
    fracO = mean(ea <= P.place.gate_px);
    place.old = struct('frac',fracO, 'med_err',median(ea(isfinite(ea))));
    if strcmp(P.bench.optics,'lens')
        say('  NON-VACUITY (axis-aligned map): %.2f%% within %g px -- VACUOUS on the lens (its mapping is axis-aligned)\n', ...
            100*fracO, P.place.gate_px);
    else
        say('  NON-VACUITY (axis-aligned map, OAP): %.2f%% within %g px vs affine %.2f%% (same %d poked); register_two_pokes centre anchor reads 0\n', ...
            100*fracO, P.place.gate_px, 100*frac, nnz(pk));
    end
    if (~isfield(P.place,'gate_assert') || P.place.gate_assert)
        assert(frac >= P.place.gate_frac, ...
            'D1 window-placement gate FAILED (%s rig): %.2f%% < %.1f%%', ...
            P.bench.optics, 100*frac, 100*P.place.gate_frac);
    end
end

function [comU, comV, poked] = mux_com_(ctx, cfg, U, V, lit, msk, P, mmpx, h0, maxst)
% ONE multiplexed sweep: measure each poked actuator's response CoM (col,row)
% in a half-inter-poke window centred on the (coarse) prediction (U,V).
% mmpx = DM-mm per detector pixel (mag*dxd_mm) sizes the window.  h0 = the flat
% (unpoked) frame, subtracted so the null background does not bias the CoM.
% maxst caps the number of multiplexed states (offsets) swept.
    step = P.battery.matrix_step;  N = size(msk,1);  nact = cfg.nact;
    hw = max(3, floor(0.5*step*cfg.pitch/mmpx));       % half inter-poke spacing, px
    comU = nan(nact);  comV = nan(nact);  poked = false(nact);
    nst = 0;
    for ox = 1:step
      for oy = 1:step
        if nst >= maxst, break; end
        Ac = zeros(nact);  Ac(ox:step:nact, oy:step:nact) = 1;  Ac = Ac .* lit;
        if ~any(Ac(:)), continue; end
        if strcmp(P.battery.matrix_sign,'alternate')
            % zero-mean checkerboard of +/- pokes: the multiplexed pattern puts
            % no shared pedestal in the frame and neighbouring halos cancel
            % pairwise, so each poke's window CoM is clean (Dave 2026-09-10).
            [rr2,cc2] = find(Ac);
            sg = 1 - 2*mod((rr2-ox)/step + (cc2-oy)/step, 2);
            Ac(sub2ind([nact nact],rr2,cc2)) = sg;
        end
        nst = nst + 1;
        M = dm_influence_map(ctx.N_G, ctx.DX_G, 'nact',nact,'pitch',cfg.pitch,'act',P.POKE*Ac);
        h = ctx.measf(M) - h0;  h = h - median(h(msk));
        [pr, pc] = find(Ac);
        for q = 1:numel(pr)
            r = pr(q);  c = pc(q);  poked(r,c) = true;
            u0 = round(U(r,c));  v0 = round(V(r,c));
            rows = max(1,v0-hw):min(N,v0+hw);  cols = max(1,u0-hw):min(N,u0+hw);
            blk = abs(h(rows,cols));  mk = msk(rows,cols);  blk(~mk) = 0;
            if max(blk(:)) <= 0, continue; end
            blk(blk < 0.5*max(blk(:))) = 0;
            [CC, RR] = meshgrid(cols, rows);
            comU(r,c) = sum(CC(:).*blk(:))/sum(blk(:));
            comV(r,c) = sum(RR(:).*blk(:))/sum(blk(:));
        end
      end
    end
end

function [U, V, Cu, Cv, resid, inl] = poly_place_(axg, ayg, comU, comV, gd, deg)
% ROBUST degree-`deg` 2-D polynomial fit lattice(x,y) -> pixel(col,row) from
% the finite CoMs, evaluated over the whole lattice.  deg=1 is the affine;
% deg>=2 would capture smooth distortion (the OAP fold has none -- it is
% affine).  Coordinates are NORMALIZED to ~[-1,1] first (raw mm makes the
% deg-2 design matrix x^2~2300 ill-conditioned).  Outliers (wrong-blob CoMs
% from a loose bootstrap window catching a neighbour) are rejected by MAD so
% they do not drag the least-squares map -- this is what the loose OAP
% bootstrap needs and the tight lens bootstrap did not.
    idx = find(gd);  sc = max([abs(axg(gd)); abs(ayg(gd)); eps]);
    u = comU(idx);  v = comV(idx);
    Ba = poly_basis_(axg(idx)/sc, ayg(idx)/sc, deg);
    keep = true(size(idx));  Cu = Ba \ u;  Cv = Ba \ v;
    for it = 1:4
        r = hypot(Ba*Cu - u, Ba*Cv - v);
        thr = max(4*1.4826*median(abs(r(keep) - median(r(keep)))), 1.0);   % >=1px floor
        keep = r <= median(r(keep)) + thr;
        Cu = Ba(keep,:) \ u(keep);  Cv = Ba(keep,:) \ v(keep);
    end
    resid = sqrt(mean(hypot(Ba(keep,:)*Cu - u(keep), Ba(keep,:)*Cv - v(keep)).^2));
    Ball = poly_basis_(axg(:)/sc, ayg(:)/sc, deg);
    U = reshape(Ball*Cu, size(axg));  V = reshape(Ball*Cv, size(axg));
    inl = false(size(axg));  inl(idx(keep)) = true;
end

function B = poly_basis_(x, y, deg)
% 2-D polynomial design matrix up to total degree `deg` (columns 1,x,y,x^2,...)
    x = x(:);  y = y(:);  B = [];
    for d = 0:deg
        for i = 0:d
            B = [B, (x.^(d-i)).*(y.^i)]; %#ok<AGROW>
        end
    end
end

function [U, V] = axis_aligned_map_(axg, ayg, comU, comV, gd)
% best AXIS-ALIGNED shear-free map lattice(x,y) -> pixel(col,row): a per-axis
% scale + offset under the better of the two dihedral assignments (col<-x/row<-y
% or col<-y/row<-x); signs fall out of the linear fit.  Everything a parity +
% scale registration can express; the off-axis (rotation/shear) content is what
% it CANNOT.  Fit on the finite CoMs (gd); evaluate over the whole lattice.
    x = axg(gd);  y = ayg(gd);  u = comU(gd);  v = comV(gd);  o = ones(numel(x),1);
    best = inf;  U = nan(size(axg));  V = nan(size(axg));
    for asn = 1:2
        if asn == 1
            pu = [x o]\u;  pv = [y o]\v;  m = median(hypot([x o]*pu-u, [y o]*pv-v));
            Uc = pu(1)*axg + pu(2);  Vc = pv(1)*ayg + pv(2);
        else
            pu = [y o]\u;  pv = [x o]\v;  m = median(hypot([y o]*pu-u, [x o]*pv-v));
            Uc = pu(1)*ayg + pu(2);  Vc = pv(1)*axg + pv(2);
        end
        if m < best, best = m;  U = Uc;  V = Vc;  end
    end
end

% =====================================================================
%  parse + local helpers (PSI machinery copied verbatim from tg96.m)
% =====================================================================
function P = parse_params_(exdir, varargin)
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

function say_(rep, varargin)
    fprintf(1, varargin{:});  fprintf(rep, varargin{:});
end

function draw_layout_(geom, s, P)
    f = figure('Visible','off','Position',[40 40 940 640]); hold on; axis equal;
    a = 2*geom.AOI;  br = geom.beam_r;  L_dm = geom.D_BS_TO;
    L_ref = max(300, s*100+150);  L_out = max(350, s*200+150);  L_in = 400;
    Pdm=L_dm*[cosd(a) sind(a)]; Pref=[L_ref 0]; Pin=[-L_in 0]; Pout=-L_out*[cosd(a) sind(a)];
    db=@(p1,p2,c) patch('XData',[p1(1)+nrm(p1,p2,br,1) p2(1)+nrm(p1,p2,br,1) p2(1)-nrm(p1,p2,br,1) p1(1)-nrm(p1,p2,br,1)], ...
        'YData',[p1(2)+nrm(p1,p2,br,2) p2(2)+nrm(p1,p2,br,2) p2(2)-nrm(p1,p2,br,2) p1(2)-nrm(p1,p2,br,2)], ...
        'FaceColor',c,'FaceAlpha',0.30,'EdgeColor','none');
    db(Pin,[0 0],[.85 .45 .2]); db([0 0],Pref,[.85 .45 .2]);
    db([0 0],Pdm,[.25 .45 .8]); db([0 0],Pout,[.35 .65 .35]);
    rectangle('Position',[Pdm(1)-P.clear.HW_DM Pdm(2)-20 2*P.clear.HW_DM 40],'FaceColor',[.75 .82 1],'EdgeColor','k');
    plot(0,0,'ks','MarkerSize',12,'MarkerFaceColor','y');
    ttl = sprintf('TG96 %s layout: BS %d\\circ, DM leg %d mm, beam %.0f mm', P.bench.optics, geom.AOI, L_dm, 2*br);
    if strcmp(P.bench.optics,'oap'), ttl=[ttl sprintf(', OAP fold %d/%d\\circ',geom.OAP1_AOI,geom.OAP2_AOI)]; end
    title(ttl); xlabel('mm'); ylabel('mm'); grid on;
    print(f,[P.tag '_layout.png'],'-dpng','-r130');
end
function n = nrm(p1,p2,r,i)
    d = p2-p1;  v = [-d(2) d(1)]/norm(d)*r;  n = v(i);
end

function A = arm_desc(rx, b, ix, base_deg)
    nm = {b.E.name};
    A = struct('rx', rx, 'b', b, 'iPol', find(strcmp(nm,'PolIn'),1), ...
        'iQ', find(contains(nm,'QWP') & ~strcmp(nm,'OutQWP')), ...
        'base', base_deg, 'qwp_deg', base_deg, 'oq_deg', 0, 'iTO', [], ...
        'iRC', ix.iRC, 'iOQ', ix.iOutQWP, 'iAn', ix.iAnalyzer, 'iDET', ix.iDET);
    if isfield(ix,'iTO'), A.iTO = ix.iTO; end
end
function a = lax(psi, deg)
    u1 = macos.design.Bench.perp(psi(:));  u2 = cross(psi(:), u1);
    a = cosd(deg)*u1 + sind(deg)*u2;  a = a(:).';
end
function x = wrap180(x), x = mod(x + 90, 180) - 90; end
function load_arm(A, QWP, an_deg, grid)
    macos.load_rx(A.rx);  b = A.b;
    % D4: rigid-body perturbation of an element (OAP1/OAP2), applied AFTER the
    % reload so it persists through the measurement (macos.perturb, SI metres /
    % radians).  A.pert = struct('iElt',..,'rot',[3x1],'trans',[3x1]).
    if isfield(A,'pert') && ~isempty(A.pert)
        macos.perturb(A.pert.iElt, 'rotation',A.pert.rot, 'translation',A.pert.trans, 'frame','local');
    end
    if nargin >= 4 && ~isempty(grid)
        macos.set_elt_grid(A.iTO, macos.get_elt_grid_spacing(A.iTO), grid);
    end
    % OAP coating (item B): thin-film stack on the listed OAPs, applied after
    % the reload so it persists (active with polarization on). A.coat = struct
    % array {iElt, n[1xL], k[1xL], t[1xL]} outermost->innermost.
    if isfield(A,'coat') && ~isempty(A.coat)
        for cq = 1:numel(A.coat)
            macos.coating(A.coat(cq).iElt, 'index',A.coat(cq).n, 'extinc',A.coat(cq).k, 'thickness',A.coat(cq).t);
        end
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
    E0 = arm_field(A,QWP,0,grid); E45 = arm_field(A,QWP,45,grid); E90 = arm_field(A,QWP,90,grid);
    S = struct('A',E0,'C',E90,'B',2*E45-E0-E90);
end
function E = synth(S, th), c=cosd(th); s=sind(th); E = c^2*S.A + c*s*S.B + s^2*S.C; end
function e = arm_state(A, QWP, iElt)
    load_arm(A, QWP, 0);  macos.trace(iElt);  f = macos.ray_field(iElt);
    ok = f.status == 0;  psi = A.b.E(iElt).psi(:);
    u1 = macos.design.Bench.perp(psi);  u2 = cross(psi, u1);
    e1 = f.Ex*u1(1)+f.Ey*u1(2)+f.Ez*u1(3);  e2 = f.Ex*u2(1)+f.Ey*u2(2)+f.Ez*u2(3);
    r = e2(ok)./e1(ok);  a = median(abs(e1(ok)));
    e = [a; a*(median(real(r))+1i*median(imag(r)))];
end
function az = arm_azimuth(A, QWP, qwp_deg)
    A.qwp_deg = qwp_deg;  e = arm_state(A, QWP, A.iRC);
    az = 0.5*atan2d(2*real(conj(e(1))*e(2)), abs(e(1))^2 - abs(e(2))^2);
end
function I = frame(Sx, Sr, th), I = sum(abs(synth(Sx,th)+synth(Sr,th)).^2, 3); end
function p = fourstep(Sx, Sr, th)
    p = atan2(frame(Sx,Sr,th(2))-frame(Sx,Sr,th(4)), frame(Sx,Sr,th(1))-frame(Sx,Sr,th(3)));
end
function h = meas_surface(A, QWP, M, Sr, p_null, THETAS, LAM)
    d = angle(exp(1i*(fourstep(analyzer_basis(A,QWP,M), Sr, THETAS) - p_null)));
    h = d * LAM/(4*pi);
end
function [map, reg] = register_two_pokes(A, ix, MpA, hpA, MpB, hpB, N_G, DX_G, msk)
    macos.load_rx(A.rx);
    s1 = macos.trace(ix.iTO);   ito  = macos.get_ray_info(s1.nRays);
    s2 = macos.trace(ix.iDET);  idet = macos.get_ray_info(s2.nRays);
    okr = ito.ok_trace(:)&ito.ok_pass(:)&idet.ok_trace(:)&idet.ok_pass(:);
    psi1 = macos.get_elt_psi(ix.iTO);  vpt1 = macos.get_elt_vpt(ix.iTO);
    u1 = macos.design.Bench.perp(psi1);  v1 = cross(psi1,u1);
    xy_to = [u1.'; v1.']*(ito.pos - vpt1);
    psi2 = macos.get_elt_psi(ix.iDET);  u2 = macos.design.Bench.perp(psi2);  v2 = cross(psi2,u2);
    xy_d = [u2.'; v2.']*(idet.pos - idet.pos(:,1));
    xy_to = xy_to(:,okr);  xy_d = xy_d(:,okr);
    Aaf = [xy_d.' ones(nnz(okr),1)] \ xy_to.';  Lm = Aaf(1:2,:).';
    [~,Ss,~] = svd(Lm);  sm = diag(Ss);
    nl = xy_to - (Lm*xy_d + Aaf(3,:).');
    map = struct('mag',sqrt(abs(det(Lm))),'anam_pct',100*(sm(1)/sm(2)-1),'nonlin_mm',sqrt(mean(sum(nl.^2,1))));
    N = size(hpA,1);  [cg,rg] = meshgrid(1:N,1:N);
    cx = sum(cg(msk))/nnz(msk);  cy = sum(rg(msk))/nnz(msk);
    dxp = macos.dx_at(ix.iDET,'mm');  a1 = (cg-cx)*dxp;  a2 = (rg-cy)*dxp;
    axs = ((1:N_G)-(N_G+1)/2)*DX_G;  sc = map.mag;
    w = abs(hpA - median(hpA(msk)));  w(~msk) = 0;  w(w<0.05*max(w(:))) = 0;
    bx = sum(a1(:).*w(:))/sum(w(:));  by = sum(a2(:).*w(:))/sum(w(:));
    [~,iA] = max(abs(MpA(:)));  [rA,cA] = ind2sub(size(MpA),iA);  pA = [axs(cA) axs(rA)];
    hpB0 = hpB(msk)-mean(hpB(msk));
    cands = {a1,a2; a1,-a2; -a1,a2; -a1,-a2; a2,a1; a2,-a1; -a2,a1; -a2,-a1};
    bA = {[bx by];[bx -by];[-bx by];[-bx -by];[by bx];[by -bx];[-by bx];[-by -bx]};
    reg = struct('pokeB_corr',0,'par',0);  tab = zeros(1,8);  best = -1;
    for c = 1:8
        Xc = sc*cands{c,1};  Yc = sc*cands{c,2};
        Xc = Xc - (sc*bA{c}(1)-pA(1));  Yc = Yc - (sc*bA{c}(2)-pA(2));
        tps = interpn(axs,axs,MpB,Xc,Yc,'spline',0);
        cc = corrcoef(hpB0, tps(msk)-mean(tps(msk)));  tab(c) = cc(1,2);
        if abs(cc(1,2)) > best, best = abs(cc(1,2));  reg.pokeB_corr = cc(1,2); reg.par = c; reg.Xt = Xc; reg.Yt = Yc; end
    end
    st = sort(abs(tab),'descend');  reg.runner_up = st(2);  reg.table = tab;  reg.sign = sign(reg.pokeB_corr);
    map.Xt = reg.Xt;  map.Yt = reg.Yt;
end
