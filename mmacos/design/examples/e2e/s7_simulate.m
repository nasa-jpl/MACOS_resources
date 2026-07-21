%S7_SIMULATE  Stage 7: closed-loop drift simulation with WF maintenance.
%
% Stage 7 of the end-to-end worked example -- a THIN DRIVER over the
% simulate stage runner (design/runners/run_simulator.m).  It plays a
% 500 s disturbance history through the mmacos ENGINE showing BOTH the
% uncorrected and the corrected performance, with an image-based
% wavefront-control INITIALIZATION (delayed a couple frames -- "no
% system starts perfect", Dave 2026-07-20), an RBCS metrology loop
% holding the pose, and a periodic WF-MAINTENANCE recontrol at 400 s
% (Tesch's WF Maintenance Activity).
%
% TWO scenarios (Dave 2026-07-21), one movie each:
%   A) METROLOGY-BIAS drift: the metrology zero-point drifts slowly
%      (thermal).  The MET loop faithfully holds the BIASED reading, so
%      the true pose -- and the wavefront -- slowly walk off UNSEEN by
%      the loop.  The 400 s image-based recontrol re-references the
%      metrology and knocks the drift back: the reset visibly limits it.
%   B) FIGURE (focus/astigmatism) drift: a slow per-segment focus +
%      astig figure trend (the low-order modes that drift in a variable
%      thermal environment).  The MET truss reads RIGID POSE only
%      (figure is a separate WFS domain), so the figure accumulates
%      unseen.  Rigid-body control CAN counter segment focus (via
%      piston) and astig (via the LATERAL DOFs -- an x move changes a
%      segment's local best-fit radius on the parabolic parent, y/twist
%      add astig), so the 400 s recontrol (with a tight ridge that
%      engages those lateral DOFs) removes the focus and most of the
%      astig; higher orders it cannot (Dave 2026-07-21).
%
% Run AFTER s3_segmentation.m, s4_jacobians.m and s5_met.m.

addpath(fullfile(getenv('HOME'), 'dev/MACOS_resources/mmacos/src'));
addpath(fullfile(getenv('HOME'), 'dev/MACOS_resources/mmacos/design/src'));
addpath(fullfile(getenv('HOME'), 'dev/MACOS_resources/mmacos/design/runners'));
P = e2e_params();
here = fileparts(mfilename('fullpath'));  if isempty(here), here = pwd; end
v = char(P.seg.variant);
rx  = fullfile(here, sprintf('e2e_%s.in', v));
hx  = fullfile(here, sprintf('e2e_%sHx.m', v));
jac = fullfile(here, 's4_jacobians.mat');
met = fullfile(here, sprintf('e2e_%s_met.mat', v));
assert(isfile(rx) && isfile(hx), 's7 needs e2e_%s.in + Hx -- run s3 first', v);
assert(isfile(jac), 's7 needs s4_jacobians.mat -- run s4 first');
assert(isfile(met), 's7 needs e2e_%s_met.mat -- run s5 first', v);

J = load(jac);  M = load(met);
nb = numel(M.bodies);  nseg = M.seg.nseg;
nz = numel(J.oz.channel_names);  ng = numel(J.og.channel_names);
T  = 50;                    % 500 s at 10 s steps
DT = 10;  RESET_T = 400;  WOF = 2;   % reset @ 400 s; WFC on at frame 2
WFE_INIT = 100e-6;          % m RMS total initial WFE (as deployed)
rng(7);

% -- common rigid initial pose (PTT-dominant) + per-step drift std -----
rms_ = @(v) sqrt(mean(v(:).^2));
icfx = find(strcmp(J.ox.field_names, 'C'), 1);
icfz = find(strcmp(J.oz.field_names, 'C'), 1);
Bc = center_dwdx_(J.ox, icfx, M.bodies);      % OPD per rigid DOF (cbm=1, e2e)
Bz = J.oz.per_field_dwdz{icfz};               % OPD per MonZern coef
x0 = zeros(6*nb, 1);  qx = zeros(6*nb, 1);
for b = 1:nseg+1
    r = (b-1)*6;                              % [Rx Ry Rz Tx Ty Tz]:
    x0(r + (1:2)) = 1.0 * randn(2, 1);   x0(r + 3) = 0.1 * randn;   % tip/tilt, clock
    x0(r + (4:5)) = 0.1 * randn(2, 1);   x0(r + 6) = 1.0 * randn;   % lateral, piston
    qx(r + (1:2)) = 4e-9;  qx(r + 3) = 0.4e-9;
    qx(r + (4:5)) = 0.4e-9; qx(r + 6) = 4e-9;
end
x0 = x0 * (WFE_INIT / rms_(Bc * x0));         % scale to 100 um WFE
Xrig = x0 + cumsum(qx .* randn(6*nb, T), 2);  % rigid history

% -- dmdx (for the metrology-bias scenario) ----------------------------
es = macos.design.edge_sensors(hx);
dldx = M.dldx_opt;  if isempty(dldx), dldx = M.dldx; end
dedx = zeros(es.nmeas, 6*nb);  dedx(:, 1:6*nseg) = es.dedx;   % cbm=1 (e2e)
dmdx = [dldx; dedx];
vis = usejava('desktop');

%% ===== scenario A: metrology-bias drift ==============================
% A slow TRUE-pose drift p(t) (rigid, -> ~60 nm WFE by 500 s) is imposed
% purely through a metrology BIAS = -dmdx*p: the loop, holding the
% biased reading, drives the true pose to follow p unseen.  Figure = 0.
pdir = zeros(6*nb, 1);
for b = 1:nseg+1, r=(b-1)*6; pdir(r+(1:2)) = randn(2,1); pdir(r+6) = randn; end
pdir = pdir * (60e-9 / rms_(Bc * pdir));      % 60 nm WFE at full scale
pdrift = pdir * ((1:T) / T);                  % linear ramp, 6*nb x T
biasA = -dmdx * pdrift;                        % metrology bias hiding it
tsA = struct('dt', DT, 'x', Xrig, 'z', zeros(nz,T), 'g', zeros(ng,T));
artA = run_simulator(rx, 'hx', hx, 'jac', jac, 'met', met, 'ts', tsA, ...
    'wfc_on_frame', WOF, 'wfc_tol', 3e-4, ...
    'meas_bias', biasA, ...                    % the hidden drift
    'wfc_reset_times', RESET_T, 'wfc_reset_tol', 3e-4, ...
    'npix', 128, 'dwell', 0.8, 'visible', vis, ...
    'model_size', P.seg.model_size, 'out_dir', here, 'name', 's7A');
copyfile(artA.report, fullfile(here, 's7A_report.txt'));

%% ===== scenario B: focus/astig figure drift ==========================
% Per-segment focus (MonZern 5) + astig (4,6) trend -- the thermal low-
% order modes.  The loop reads RIGID POSE only (loop_senses_figure=0),
% so the figure accumulates unseen; the 400 s recontrol with a TIGHT
% ridge engages the lateral DOFs that counter segment astig.
focastig = false(nz, 1);
for k = 1:nz
    tk = regexp(J.oz.channel_names{k}, 'MonZern(\d+)$', 'tokens', 'once');
    if ~isempty(tk) && any(str2double(tk{1}) == [4 5 6]), focastig(k) = true; end
end
assert(any(focastig), 's7: no focus/astig MonZern modes in the harvest');
zdir = randn(nz,1) .* focastig;  zdir = zdir / rms_(Bz * zdir);
z0 = zdir * 10e-9;                             % 10 nm initial figure
zdir2 = randn(nz,1) .* focastig;  zdir2 = zdir2 / rms_(Bz * zdir2);
Zfig = z0 + zdir2 * (60e-9 * (1:T) / T);       % +60 nm focus/astig trend (2x)
icfg = find(strcmp(J.og.field_names, 'C'), 1);
Bg = J.og.per_field_dwdg{icfg};
gdir = randn(ng,1);  gdir = gdir / rms_(Bg * gdir);
g0 = gdir * 20e-9;                             % 20 nm static grid figure floor
tsB = struct('dt', DT, 'x', Xrig, 'z', Zfig, 'g', repmat(g0,1,T));
artB = run_simulator(rx, 'hx', hx, 'jac', jac, 'met', met, 'ts', tsB, ...
    'wfc_on_frame', WOF, 'wfc_tol', 3e-4, ...
    'loop_senses_figure', false, ...           % truss reads rigid pose only
    'wfc_reset_times', RESET_T, 'wfc_reset_tol', 1e-5, ...  % engage lateral DOFs
    'npix', 128, 'dwell', 0.8, 'visible', vis, ...
    'model_size', P.seg.model_size, 'out_dir', here, 'name', 's7B');
copyfile(artB.report, fullfile(here, 's7B_report.txt'));

%% ===== summary =======================================================
rp = @(a) reset_effect_(a, RESET_T);
fprintf('\nStage 7 complete: s7A (metrology bias) + s7B (focus/astig figure)\n');
[ba,aa] = rp(artA);
fprintf('A  metrology bias : corr %.3g -> %.3g nm over 500 s; reset@400 %.3g -> %.3g nm\n', ...
    artA.rms_wfe_corr(WOF+1), artA.rms_wfe_corr(end), ba, aa);
[bb,ab] = rp(artB);
fprintf('B  focus/astig fig: corr %.3g -> %.3g nm over 500 s; reset@400 %.3g -> %.3g nm\n', ...
    artB.rms_wfe_corr(WOF+1), artB.rms_wfe_corr(end), bb, ab);
fprintf('Next: upgrade the RBCS estimator to the Kalman form + figure states.\n');

% -- local helpers -----------------------------------------------------
function B = center_dwdx_(ox, icf, bodies)
Bf = ox.per_field_dwdx{icf};  cn = ox.channel_names;
dn = {'Rx','Ry','Rz','Tx','Ty','Tz'};  nb = numel(bodies);
B = zeros(size(Bf, 1), 6*nb);
for b = 1:nb
    for d = 1:6
        i = find(strcmp(cn, sprintf('Elt %d %s', bodies(b), dn{d})), 1);
        if ~isempty(i), B(:, (b-1)*6 + d) = Bf(:, i); end
    end
end
end

function [rb, ra] = reset_effect_(art, reset_t)
% rb = frame just BEFORE the reset, ra = the reset frame itself
ib = find(art.t < reset_t, 1, 'last');  ia = find(art.t >= reset_t, 1, 'first');
rb = art.rms_wfe_corr(ib);  ra = art.rms_wfe_corr(ia);
end
