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

exdir = fileparts(mfilename('fullpath'));  if isempty(exdir), exdir = pwd; end
if isempty(which('macos.init'))
    run(fullfile(exdir, '..', '..', '..', 'mmacos_setup.m'));
end
assert(~isempty(getenv('MACOS_HOME')), 'MACOS_HOME must be set.');
addpath(fullfile(exdir, '..', '..', '90_polarization', 'tg_psi_dm'));  % dm_influence_map

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
if want('bench') || want('battery')
    [G, bench] = stage_B_(P, s, geom, say, exdir);
end

battery = struct();
if want('battery')
    battery = stage_CDE_(P, s, G, bench, say);
end

if want('figs')
    draw_layout_(geom, s, P);
end

out = struct('P', P, 'geom', geom, 'bench', bench, 'battery', battery, 's', s);
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
    assert(det_nyq >= 2*act_nyq, 'sampling: det Nyquist %.1f < 2x actuator %.1f', det_nyq, act_nyq);
    assert(grid_ppa >= 3, 'sampling: %.2f grid px/actuator < 3', grid_ppa);
    assert(P.MODEL >= 2*P.NGRID, 'sampling: grid %d < 2x %d-px image', P.MODEL, P.NGRID);
end

function [G, bench] = stage_B_(P, s, geom, say, exdir)
    % tail params: re-tuned set from <tag>_tail.mat / tg96_tail.mat if present
    b = P.bench;
    T_FL_F = s*b.FL_F;  T_FL_Kc = b.FL_Kc;  T_DMF = s*b.D_MASK_FL;  T_TRIM = s*b.DET_TRIM;
    tailf = fullfile(exdir, [P.tag '_tail.mat']);   % tg96_tail writes here
    if isfile(tailf)
        tl = load(tailf);
        T_FL_F = tl.out.FL_F;  T_FL_Kc = tl.out.FL_Kc;
        T_DMF  = tl.out.D_MASK_FL;  T_TRIM = tl.out.DET_TRIM;
        say('Tail: RE-TUNED set from %s (null %.3f nm at opt res; seed %.3f)\n', ...
            tailf, tl.out.null_nm, tl.out.seed_null_nm);
    else
        say('Tail: geometrically-scaled seed (%s not found -- RE-RUN tg96_tail for oap)\n', tailf);
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

function battery = stage_CDE_(P, s, G, bench, say)
    N_G = P.grid.N_G;  DX_G = P.grid.DX_G;  LAM = P.LAM;  QWP = P.QWP;
    THETAS = P.THETAS;  NACT = P.dm(1).nact;  PITCH = P.dm(1).pitch;
    rxT = [P.tag '_test.in'];  rxR = [P.tag '_ref.in'];
    AT = arm_desc(rxT, G.bt, G.T, 0);
    AR = arm_desc(rxR, G.br, G.R, 45);
    say('Stage C -- battery (design azimuths, unaligned):\n');
    az_t = arm_azimuth(AT, QWP, 0);  az_r = arm_azimuth(AR, QWP, 45);
    dep  = wrap180(az_t - az_r - 90);
    say('  arm azimuths: test %+.4f, ref %+.4f -> departure %+.4f deg\n', az_t, az_r, dep);
    Sr = analyzer_basis(AR, QWP, []);
    S0 = analyzer_basis(AT, QWP, []);
    I0 = frame(S0, Sr, 0);  msk = I0 > 0.1*max(I0(:));
    p_null = fourstep(S0, Sr, THETAS);
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
    if nargin >= 4 && ~isempty(grid)
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
