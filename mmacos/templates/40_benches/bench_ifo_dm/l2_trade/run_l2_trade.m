% run_l2_trade.m -- the L2 pupil-relay architecture trade (PLAN sec.5/6).
%
% Strategy (from run_mechanism.m findings): the retrace term is LINEAR in
% beam deflection -- 0.146 nm rms of non-tilt phase error per urad of DM
% tilt on the baseline singlet, which extrapolates to 6.2 nm at the
% checker's rms deflection vs 6.76 nm observed.  So the architectures are
% OPTIMIZED on that cheap physics coefficient (k_tilt: plane-fit residual
% of the differential phase under a 2e-6 rad DM tilt, ~15 s/eval) with the
% detector held at the DM conjugate by a lever-null trim (the thin-lens
% seed leaves ~4.6 mm), and only the finalists pay for the full M1 metric
% (ifo_l2_metric, ~2 min/eval).  Candidates, per the plan:
%   singlet        the committed baseline (reference row)
%   singlet+trim   baseline + DET_TRIM nulling the DM-tilt lever (C0)
%   fieldlens      C1: field lens behind the FocalMask, f_FL + trim tuned
%   doublet        C2: L2 split in two, conics spot-held, k_tilt-tuned
%
% Run:  matlab -batch "run('.../run_l2_trade.m'); exit(0)"

addpath(fullfile(getenv('HOME'), 'dev/MACOS_resources/mmacos/src'));
here = fileparts(mfilename('fullpath'));
if isempty(here), here = pwd; end
cd(here);
macos.init(256);
if ~isfile('dm_zero.txt'), macos.write_grid_file('dm_zero.txt', zeros(256)); end

res = struct('name',{}, 'tg',{}, 'k_tilt_nm',{}, 'lever_mm',{}, 'out',{});

% ---- 1. baseline ----------------------------------------------------------
tg = {};
fprintf('\n##### singlet (baseline) #####\n');
[k0, lev0] = k_tilt(tg);
fprintf('  k_tilt %.4f nm @2e-6 rad, lever %.2f mm/rad\n', k0, lev0);
res(end+1) = struct('name','singlet', 'tg',{tg}, 'k_tilt_nm',k0, ...
    'lever_mm',lev0, 'out',[]);

% ---- 2. singlet + conjugate trim (C0) -------------------------------------
fprintf('\n##### singlet + trim #####\n');
z = trim_null({}, 0);
tg = {'DET_TRIM', z};
[k1, lev1] = k_tilt(tg);
fprintf('  DET_TRIM %.3f mm -> k_tilt %.4f nm, lever %.2f mm/rad\n', z, k1, lev1);
res(end+1) = struct('name','singlet+trim', 'tg',{tg}, 'k_tilt_nm',k1, ...
    'lever_mm',lev1, 'out',[]);

% ---- 3. field lens (C1) ---------------------------------------------------
fprintf('\n##### fieldlens #####\n');
bestf = struct('k', inf);
for fFL = [50 70 100 140 200 300]
    tgf = {'tail_arch','fieldlens', 'FL_F',fFL};
    z = trim_null(tgf, 0);
    [k, lv] = k_tilt([tgf {'DET_TRIM', z}]);
    fprintf('  scan f_FL %g: trim %.3f, k_tilt %.4f nm, lever %.2f mm/rad\n', ...
        fFL, z, k, lv);
    if k < bestf.k, bestf = struct('k',k, 'f',fFL, 'z',z); end
end
% refine [log f_FL, FL_Kc, D_MASK_FL] at held trim (f bounded to a
% buildable [25,400], FL 1.5..20 mm behind the mask), re-null trim after
obj = @(p) fl_obj(p, bestf.z);
p = fminsearch(obj, [log(bestf.f) -2.25 5], ...
    optimset('Display','iter', 'MaxFunEvals',45, 'TolX',1e-3, 'TolFun',1e-4));
tgf = {'tail_arch','fieldlens', 'FL_F',exp(p(1)), 'FL_Kc',p(2), 'D_MASK_FL',p(3)};
z = trim_null(tgf, bestf.z);
tg = [tgf {'DET_TRIM', z}];
[k2, lev2] = k_tilt(tg);
fprintf('  refined: f_FL %.2f, FL_Kc %.4f, D_MASK_FL %.2f, trim %.3f -> k_tilt %.4f nm, lever %.2f mm/rad\n', ...
    exp(p(1)), p(2), p(3), z, k2, lev2);
res(end+1) = struct('name','fieldlens', 'tg',{tg}, 'k_tilt_nm',k2, ...
    'lever_mm',lev2, 'out',[]);

% ---- 4. doublet (C2) ------------------------------------------------------
fprintf('\n##### doublet #####\n');
tgd0 = {'tail_arch','doublet'};
% stage 1: hold focus -- MASK_TRIM (true focus vs thin-lens seed; conics
% cannot move the focus, so without this the spot floor is pure defocus)
% + conics against the mask spot (cheap trace probe).  MASK_TRIM is
% pre-scanned for a NONZERO fminsearch seed -- the default simplex
% perturbs a 0-valued parameter by 0.00025, freezing it (learned here:
% the first two trade runs sat at a 33 um defocus floor because of this).
mts = -2:1:10;
sps = arrayfun(@(mt) spot_probe([tgd0 {'MASK_TRIM',mt}]), mts);
[~, im] = min(sps);
fprintf('  MASK_TRIM pre-scan: best %.0f mm (spot %.1f um)\n', mts(im), 1e3*sps(im));
sobj = @(q) spot_probe([tgd0 {'MASK_TRIM',q(1), 'L2A_Kc',q(2), 'L2B_Kc',q(3)}]);
q = fminsearch(sobj, [mts(im) -2.25 -2.25], ...
    optimset('Display','iter', 'MaxFunEvals',80, 'TolX',1e-4, 'TolFun',1e-7));
fprintf('  spot-held: MASK_TRIM %.3f, L2A_Kc %.4f, L2B_Kc %.4f, spot %.4f um\n', ...
    q(1), q(2), q(3), 1e3*sobj(q));
% stage 2: k_tilt over conics (spot-penalized) at nulled trim
tgd1 = [tgd0 {'MASK_TRIM',q(1), 'L2A_Kc',q(2), 'L2B_Kc',q(3)}];
z = trim_null(tgd1, 0);
obj = @(p) k_tilt([tgd0 {'MASK_TRIM',q(1), 'L2A_Kc',p(1), 'L2B_Kc',p(2), 'DET_TRIM',z}]) ...
    + 100*max(0, 1e3*spot_probe([tgd0 {'MASK_TRIM',q(1), 'L2A_Kc',p(1), 'L2B_Kc',p(2)}]) - 1);
p = fminsearch(obj, q(2:3), ...
    optimset('Display','iter', 'MaxFunEvals',30, 'TolX',1e-4, 'TolFun',1e-4));
tgd = [tgd0 {'MASK_TRIM',q(1), 'L2A_Kc',p(1), 'L2B_Kc',p(2)}];
z = trim_null(tgd, z);
tg = [tgd {'DET_TRIM', z}];
[k3, lev3] = k_tilt(tg);
fprintf('  refined: Kc [%.4f %.4f], trim %.3f -> k_tilt %.4f nm, lever %.2f mm/rad, spot %.3f um\n', ...
    p(1), p(2), z, k3, lev3, 1e3*spot_probe(tgd));
res(end+1) = struct('name','doublet', 'tg',{tg}, 'k_tilt_nm',k3, ...
    'lever_mm',lev3, 'out',[]);

% ---- 5. full M1 metric on every candidate ---------------------------------
fprintf('\n##### full metric (M1) on all candidates #####\n');
for k = 1:numel(res)
    res(k).out = ifo_l2_metric(res(k).tg, 'workdir', here);
end
[~, iw] = min(arrayfun(@(r) r.out.m1_resid_nm + 1e6*(~r.out.pass), res));
fprintf('\nwinner: %s\n', res(iw).name);

% linearity: baseline + winner at POKE_NM = 5
fprintf('\n##### linearity @ POKE_NM = 5 #####\n');
lin_base = ifo_l2_metric(res(1).tg, 'workdir', here, 'poke_nm', 5);
lin_win  = ifo_l2_metric(res(iw).tg, 'workdir', here, 'poke_nm', 5);

% ---- 6. table + figure ----------------------------------------------------
fprintf('\n=========== L2 TRADE RESULTS (poke 50 nm checker, seed 7) ===========\n');
fprintf('%-14s %9s %9s %8s %8s %9s %9s %7s  %s\n', 'arch', 'M1(nm)', ...
    'k_tilt', 'corr', 'mag', 'nl(mm)', 'lever', 'guards', 'per-nm');
for k = 1:numel(res)
    o = res(k).out;
    if k == 1, pn = lin_base; elseif k == iw, pn = lin_win; else, pn = []; end
    if isempty(pn), pertxt = '--';
    else, pertxt = sprintf('%.3f nm @5nm poke', pn.m1_resid_nm); end
    fprintf('%-14s %9.3f %9.4f %8.4f %8.4f %9.4f %9.2f %7s  %s\n', ...
        res(k).name, o.m1_resid_nm, res(k).k_tilt_nm, o.m1_corr, ...
        o.map.mag, o.map.nl_rms_mm, res(k).lever_mm, pass_str(o.pass), pertxt);
end
fprintf('  (per-nm column: M1 @ POKE_NM=5 / truth rms -- linearity check)\n');

f = figure('Color','w','Position',[80 80 900 460]);
subplot(1,2,1);
bar(categorical({res.name}, {res.name}), arrayfun(@(r) r.out.m1_resid_nm, res));
ylabel('M1 residual (nm rms)'); grid on; hold on;
yline(1.0, 'r--', 'gate 1 nm'); yline(0.402, 'g--', 'recomb floor');
title('physical-instrument vs-truth residual');
subplot(1,2,2);
bar(categorical({res.name}, {res.name}), [res.k_tilt_nm]);
ylabel('k\_tilt (nm @ 2\murad)'); grid on;
title('tail slope-coupling coefficient');
print(f, fullfile(here,'l2_trade_results.png'), '-dpng', '-r140');

save(fullfile(here,'l2_trade_results.mat'), 'res', 'iw', 'lin_base', 'lin_win');
fprintf('\ndone -> l2_trade_results.{mat,png}\n');

% ===========================================================================
%  probes
% ===========================================================================
function [T, rx] = emit_base(tg)
%EMIT_BASE  Build + emit the zero-grid test arm for a candidate config.
%   Rejects configurations without mechanical clearance for the detector.
    G0 = macos.design.twyman_green('to_grid_file','dm_zero.txt', ...
        'to_grid_n',256, 'to_grid_dx',0.35, tg{:});
    assert(G0.det_leg >= 20, 'det_leg %.1f mm < 20 mm clearance', G0.det_leg);
    rx = fullfile(pwd, 'l2p_base.in');
    G0.bt.emit(rx);
    T = G0.T;
end

function k = fl_obj(p, z)
%FL_OBJ  Bounded field-lens objective: p = [log f_FL, FL_Kc, D_MASK_FL].
    if p(1) < log(25) || p(1) > log(400) || p(3) < 1.5 || p(3) > 20
        k = 1e3;  return;
    end
    k = k_tilt({'tail_arch','fieldlens', 'FL_F',exp(p(1)), ...
        'FL_Kc',p(2), 'D_MASK_FL',p(3), 'DET_TRIM',z});
end

function [k_nm, lever] = k_tilt(tg)
%K_TILT  Tail slope-coupling probe: plane-fit residual (nm rms) of the
%   differential detector field phase under a 2e-6 rad DM tilt, plus the
%   chief lever (mm/rad).  ~15 s.  Errors return a large penalty.
    LAM = 6.328e-4;  a = 2e-6;
    try
        [T, rx] = emit_base(tg);
        macos.load_rx(rx);
        psi0 = macos.get_elt_psi(T.iTO);
        u1 = macos.design.Bench.perp(psi0);
        Ku = [0 -u1(3) u1(2); u1(3) 0 -u1(1); -u1(2) u1(1) 0];
        Rt = eye(3) + sin(a)*Ku + (1-cos(a))*(Ku*Ku);
        vdet0 = macos.get_elt_vpt(T.iDET);
        cdet = macos.get_elt_psi(T.iDET);  cdet = cdet/norm(cdet);
        E0 = macos.complex_field(T.iDET);
        msk = abs(E0) > 0.1*max(abs(E0(:)));
        macos.load_rx(rx);
        macos.set_elt_psi(T.iTO, Rt*psi0);
        Et = macos.complex_field(T.iDET);
        hT = angle(Et .* conj(E0)) * LAM/(4*pi);
        [cg, rg] = meshgrid(1:size(hT,1), 1:size(hT,2));
        A = [ones(nnz(msk),1) cg(msk) rg(msk)];
        r = hT(msk) - A*(A \ hT(msk));
        k_nm = 1e6*std(r);
        % chief lever at the same state
        sc = macos.trace(T.iDET);  ic = macos.get_ray_info(sc.nRays);
        d = ic.pos(:,1) - vdet0;  d = d - cdet*(cdet.'*d);
        lever = norm(d)/a;
        if sc.nRays < 100 || nnz(msk) < 1000, k_nm = 1e3; end
    catch me
        fprintf('  [k_tilt penalty: %s]\n', me.message);
        k_nm = 1e3;  lever = nan;
    end
end

function z = trim_null(tg, zseed)
%TRIM_NULL  Secant-solve DET_TRIM so the DM-tilt chief lever vanishes.
%   ZSEED must be a trim at which the rig is known to build (short-f
%   field-lens tails have little clearance in trim).
    l = @(z) signed_lever([tg {'DET_TRIM', z}]);
    z0 = zseed;      l0 = l(z0);
    z1 = zseed + 2;  l1 = l(z1);
    for it = 1:5
        if abs(l1 - l0) < eps, break; end
        z2 = z1 - l1*(z1 - z0)/(l1 - l0);
        z0 = z1;  l0 = l1;  z1 = z2;  l1 = l(z1);
        if abs(l1) < 0.05, break; end               % < 0.05 mm/rad
    end
    z = z1;
end

function l = signed_lever(tg)
%SIGNED_LEVER  Chief lever (mm/rad, signed along the u2 axis) for a
%   +1e-5 rad DM tilt about u1.
    a = 1e-5;
    [T, rx] = emit_base(tg);
    macos.load_rx(rx);
    psi0 = macos.get_elt_psi(T.iTO);
    u1 = macos.design.Bench.perp(psi0);
    Ku = [0 -u1(3) u1(2); u1(3) 0 -u1(1); -u1(2) u1(1) 0];
    Rt = eye(3) + sin(a)*Ku + (1-cos(a))*(Ku*Ku);
    vdet0 = macos.get_elt_vpt(T.iDET);
    psid = macos.get_elt_psi(T.iDET);
    u2 = macos.design.Bench.perp(psid);  v2 = cross(psid, u2);
    macos.set_elt_psi(T.iTO, Rt*psi0);
    sc = macos.trace(T.iDET);  ic = macos.get_ray_info(sc.nRays);
    d = ic.pos(:,1) - vdet0;
    s2 = [u2.'; v2.']*d;
    [~, ii] = max(abs(s2));                        % dominant walk axis
    l = s2(ii)/a;
end

function r = spot_probe(tg)
%SPOT_PROBE  RMS transverse ray spread (mm) at the FocalMask, zero-grid arm.
    try
        [T, rx] = emit_base(tg);
        macos.load_rx(rx);
        s = macos.trace(T.iMASK);
        if s.nRays < 100, r = 1e3; return; end
        info = macos.get_ray_info(s.nRays);
        ok = info.ok_trace(:) & info.ok_pass(:);
        P = info.pos(:, ok);
        pch = info.pos(:,1);  dch = info.dir(:,1)/norm(info.dir(:,1));
        d = P - pch;  d = d - dch*(dch.'*d);
        r = sqrt(mean(sum(d.^2,1)));
    catch me
        fprintf('  [spot penalty: %s]\n', me.message);
        r = 1e3;
    end
end

function s = pass_str(p)
    if p, s = 'PASS'; else, s = 'FAIL'; end
end
