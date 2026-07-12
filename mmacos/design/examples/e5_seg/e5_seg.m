%E5_SEG  Segmented e5 primary with edge sensors + laser-MET truss:
%   mono .in -> segmented .in (incl. MET points) -> dedx, dldx, dwdx
%   -> MET metric performance (post-control wavefront residual).
%
% Sprint 2D worked example (PLAN_DESIGN_LAYER §6.6 / §8-2D).  Forward
% model on one per-body DOF vector x (SI; per body [rot_xyz|trans_xyz]
% in its LOCAL/TElt triad):
%     w = dwdx*x + w0     e = dedx*x + e0     l = dldx*x + l0
% Simple control dx = -pinv(dwdx)*(w_hat - wtarg) driven by the
% ESTIMATED wavefront cancels the correctable part exactly, so the
% configuration merit is the wavefront image of the estimation error:
%     E||w_post||^2 = trace(dwdx * P_dx * dwdx'),
%     P_dx = X - X*H'*(H*X*H' + R)^-1*H*X,   H = [dedx; dldx]
% reported against the prior baseline W = dwdx*X*dwdx' (Dave
% 2026-07-12).  A Monte-Carlo estimate/control loop cross-checks the
% analytic trace.
%
% Modify the knobs below and re-run.  Requires the SegMirMaker binary
% (segmirmaker/README.md) and a built mmacos mex.  Run from anywhere;
% outputs land beside this script.

%% ------------------------- knobs ----------------------------------
RINGS   = 1;          % 1 -> 7 segments, 2 -> 19
GRID    = 'Hex';      % segment tiling / ray grid (Hex|Pie|...)
GAP     = 50;         % inter-segment gap, mm
MEASCFG = 2;          % edge sensors: 1 inner edges, 2 all adjacencies
NF      = 3;          % MET fiducials on the hub (3..6)
R_FID   = 300;        % hub fiducial ring radius, mm
R_LFRAC = 0.7;        % launcher ring radius as a fraction of lMon
SIG_ROT   = 1e-6;     % prior (deploy) uncertainty: rad per rot DOF
SIG_TRANS = 1e-6;     % ... metres per trans DOF
SIG_EDGE  = 1e-9;     % edge-sensor noise, metres
SIG_MET   = 1e-9;     % MET gauge noise, metres
NMC     = 200;        % Monte-Carlo draws for the cross-check
MODEL   = 512;

here = fileparts(mfilename('fullpath'));
res_root = fileparts(fileparts(fileparts(fileparts(here))));  % MACOS_resources
tin = fullfile(res_root, 'segmirmaker', 'test_in');

%% ---------------- 1. segment the mono parent ----------------------
fprintf('[1] segmenting e5mono (rings=%d, %s, gap=%g)\n', RINGS, GRID, GAP);
seg = macos.design.segment_rx(fullfile(tin, 'e5mono.in'), 'elt', 1, ...
    'rings', RINGS, 'grid', GRID, 'gap', GAP, 'dofs', 6, ...
    'meas_config', MEASCFG);
nseg = seg.nseg;
hub   = nseg + 1;                 % m2 follows the segments (read order)
extra = seg.n_elt - 2;            % fpa (Return before exitpupil/FP)

%% ---------------- 2. MET truss (Stewart geometry) ------------------
fprintf('[2] MET truss: 6 launchers/seg + 6 around elt %d -> %d fiducials on elt %d\n', ...
    extra, NF, hub);
am = macos.design.add_met(seg.in, seg, 'hub', hub, 'r_fid', R_FID, ...
    'nf', NF, 'extra_sources', extra, 'r_launch_frac', R_LFRAC);

%% ---------------- 3. load + sensing Jacobians ----------------------
old = cd(seg.run.workdir);        % GridFile= resolves from cwd
restore = onCleanup(@() cd(old));
macos.init(MODEL);
macos.load_rx(am.in);
t0 = macos.trace();
fprintf('[3] segmented+met system: %d elts, %d rays, rmsWFE %.3g\n', ...
    macos.num_elt(), t0.nRays, t0.rmsWFE);

bodies = [seg.seg_elts, hub, extra];          % x = these, 6 DOF each
nx = 6*numel(bodies);

% dedx: SegMirMaker Hx (segments only; hub/extra columns zero), SI.
es = macos.design.edge_sensors(seg.hx);
dedx = zeros(es.nmeas, nx);
dedx(:, 1:6*nseg) = es.dedx * 1e-3;           % readings mm -> m
for s = 1:nseg                                % rot cols were mm/rad
    dedx(:, (s-1)*6+(4:6)) = dedx(:, (s-1)*6+(4:6)) * 1e3;  % trans: mm/mm -> m/m
end

% dldx: FD over the engine met beams (SI).
dm = macos.design.dmet_dx(bodies);
dldx = dm.dldx;

% dwdx: local FD, SAME column convention by construction (OPD at the
% focal plane, tilt-free enough for a demo; production path = the
% dw_dx channel harvest).
cbm = mmacos('base_unit_to_metres');
w0 = macos.opd(); w0 = w0(:);          % OPDMat, WaveUnits (= mm here)
mask = isfinite(w0) & (w0 ~= 0); w0 = w0(mask);
nw = numel(w0);
dwdx = zeros(nw, nx);
hrot = 1e-6; htr = 1e-6;
fprintf('[3b] dwdx FD over %d DOFs x %d OPD samples\n', nx, nw);
for k = 1:numel(bodies)
    for d = 1:6
        th = zeros(3,1); del = zeros(3,1);
        if d <= 3, th(d) = hrot; h = hrot; else, del(d-3) = htr; h = htr; end
        macos.perturb(bodies(k), 'rotation', th, 'translation', del, ...
                      'frame', 'local');
        macos.modify();                       % dirty the cached trace/OPD
        macos.trace();
        wp = macos.opd(); wp = wp(:); wp = wp(mask);
        macos.perturb(bodies(k), 'rotation', -th, 'translation', -del, ...
                      'frame', 'local');
        macos.modify();
        dwdx(:, (k-1)*6+d) = (wp - w0) * cbm / h;
    end
end
macos.modify(); macos.trace();                % restore baseline trace

%% ---------------- 4. MET metric performance ------------------------
X  = diag(repmat([SIG_ROT^2*ones(1,3), SIG_TRANS^2*ones(1,3)], 1, numel(bodies)));
Re = SIG_EDGE^2 * eye(es.nmeas);
Rl = SIG_MET^2  * eye(dm.nmeas);
post = @(H, R) X - X*H'*((H*X*H' + R) \ (H*X));
rmsw = @(P) sqrt(trace(dwdx*P*dwdx') / nw);
cases = { 'prior (no sensing)', X; ...
          'edge sensors only',  post(dedx, Re); ...
          'MET only',           post(dldx, Rl); ...
          'edge + MET',         post([dedx; dldx], blkdiag(Re, Rl)) };
fprintf('\n[4] post-control wavefront residual (rms, control = -pinv(dwdx)*w_hat):\n');
r = zeros(4,1);
for c = 1:4
    r(c) = rmsw(cases{c,2});
    fprintf('   %-20s %10.3f nm rms\n', cases{c,1}, r(c)*1e9);
end
fprintf('   sensing removes %.1f%% of the prior wavefront variance (edge+MET)\n', ...
    100*(1 - r(4)^2/r(1)^2));

% Monte-Carlo cross-check of the analytic trace (edge+MET case).
H = [dedx; dldx]; R = blkdiag(Re, Rl);
K = X*H'/(H*X*H' + R);                        % MMSE gain
rng(1);
acc = 0;
for q = 1:NMC
    x  = sqrt(diag(X)) .* randn(nx,1);
    y  = H*x + sqrt(diag(R)) .* randn(size(H,1),1);
    xh = K*y;
    wpost = dwdx*(x - xh);                    % full-authority control
    acc = acc + mean(wpost.^2);
end
mc = sqrt(acc/NMC);
fprintf('   Monte-Carlo (%d draws): %.3f nm rms  (analytic %.3f) [%.1f%%]\n', ...
    NMC, mc*1e9, r(4)*1e9, 100*abs(mc-r(4))/r(4));

%% ---------------- 5. artifacts -------------------------------------
copyfile(am.in, fullfile(here, 'e5_seg_met.in'));
copyfile(seg.hx, fullfile(here, 'e5_segHx.m'));
copyfile(fullfile(tin, 'flat.txt'), fullfile(here, 'flat.txt'));
save(fullfile(here, 'e5_seg.mat'), 'seg', 'am', 'dedx', 'dldx', 'dwdx', ...
     'w0', 'bodies', 'X', 'Re', 'Rl', 'r', 'mc', 'cases');
fig = figure('Visible', 'off');
bar(r*1e9); set(gca, 'XTickLabel', cases(:,1)); ylabel('w_{post} rms [nm]');
title(sprintf('e5\\_seg MET metric (%d segs, %d edge + %d MET meas)', ...
    nseg, es.nmeas, dm.nmeas)); grid on;
saveas(fig, fullfile(here, 'e5_seg_metric.png')); close(fig);
fprintf('\n[5] artifacts saved beside the script (e5_seg_met.in, .mat, metric png)\n');
