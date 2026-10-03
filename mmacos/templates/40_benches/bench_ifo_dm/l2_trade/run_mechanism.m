% run_mechanism.m -- PLAN sec.5 "analysis first": decompose WHERE the
% baseline 6.76 nm detector-mode retrace comes from, before optimizing
% anything.  Three probes:
%
%   A. det-vs-rc decomposition.  The same measurement run at the RECOMB
%      plane (before L2/mask/detector) is the tail-immune gauge; the
%      det-minus-rc difference map, both resampled onto the DM plane, IS
%      the detector-leg retrace term.  Where does it live (rim?), and what
%      does it correlate with -- truth h (gain error), radius (field
%      aberration of the pupil relay), |grad h| (slope/walk coupling)?
%
%   B. DM-tilt phase response.  Tilt the zero-grid DM rigidly by alpha and
%      measure the differential phase: ideal = a pure tilt plane in the DM
%      coordinates; the plane-fit residual is the tail's phase error for a
%      uniform beam deflection 2*alpha.  Three alphas give the scaling law
%      (linear vs quadratic); scaled to the checker's actual slope range it
%      predicts the retrace magnitude if slope-coupling is the mechanism.
%
%   C. Chief-height mapping vs alpha (the plan's literal ask): where the
%      chief lands at the detector vs DM tilt -- the linear lever is the
%      pupil-conjugate error, the curvature is angular mapping distortion.
%
% Run:  matlab -batch "run('.../run_mechanism.m'); exit(0)"

addpath(fullfile(getenv('HOME'), 'dev/MACOS_resources/mmacos/src'));
here = fileparts(mfilename('fullpath'));
if isempty(here), here = pwd; end
cd(here);                                          % GridFile resolves from cwd
LAM = 6.328e-4;                                    % mm

% ---- A. det-vs-rc decomposition ------------------------------------------
od  = ifo_l2_metric({}, 'workdir', here);                          % detector
orc = ifo_l2_metric({}, 'workdir', here, 'plane', 'rc');           % recomb

% resample the rc measurement onto the det pixels' DM coordinates
Frc = scatteredInterpolant(orc.dbg.Xt(orc.dbg.msk), orc.dbg.Yt(orc.dbg.msk), ...
    orc.dbg.hm(orc.dbg.msk), 'linear', 'none');
m   = od.dbg.msk;
hrc_on_det = nan(size(od.dbg.hm));
hrc_on_det(m) = Frc(od.dbg.Xt(m), od.dbg.Yt(m));
mm = m & ~isnan(hrc_on_det);
tail = od.dbg.hm - hrc_on_det;                     % the retrace term, mm
tail(~mm) = nan;
tail_rms_nm = 1e6*std(tail(mm));

% what does the tail correlate with?
ht = od.dbg.ht;                                    % truth on det pixels
r  = sqrt(od.dbg.Xt.^2 + od.dbg.Yt.^2);            % DM-plane radius
[gx, gy] = gradient(od.dbg.Mdm, od.dbg.ax, od.dbg.ax);   % truth slope field
gmag = sqrt(gx.^2 + gy.^2);
Fg = griddedInterpolant({od.dbg.ax, od.dbg.ax}, gmag, 'linear', 'none');
gm_on_det = nan(size(ht));  gm_on_det(m) = Fg(od.dbg.Xt(m), od.dbg.Yt(m));
mg = mm & ~isnan(gm_on_det);
c2 = corrcoef(tail(mm), ht(mm));           c_ht = c2(1,2);
c2 = corrcoef(tail(mm), r(mm).^2);         c_r2 = c2(1,2);
c2 = corrcoef(tail(mg), gm_on_det(mg));    c_gm = c2(1,2);
kap = sum(tail(mm).*ht(mm)) / sum(ht(mm).^2);      % gain split
orth = tail - kap*ht;
rin = mm & r <= od.dbg.r_beam/sqrt(2);  rout = mm & r > od.dbg.r_beam/sqrt(2);
fprintf('\n=== A. det-vs-rc decomposition ===\n');
fprintf('  M1 det %.3f nm | M1 rc %.3f nm | tail map %.3f nm rms\n', ...
    od.m1_resid_nm, orc.m1_resid_nm, tail_rms_nm);
fprintf('  tail corr: vs truth %+.3f, vs r^2 %+.3f, vs |grad h| %+.3f\n', ...
    c_ht, c_r2, c_gm);
fprintf('  gain split: kappa %+.4f (%.3f nm), orthogonal %.3f nm rms\n', ...
    kap, abs(kap)*1e6*std(ht(mm)), 1e6*std(orth(mm)));
fprintf('  tail rms inner/outer equal-area halves: %.3f / %.3f nm (rim factor %.2f)\n', ...
    1e6*std(tail(rin)), 1e6*std(tail(rout)), std(tail(rout))/std(tail(rin)));

% in-beam slope statistics of the checker (sets the alpha range below)
inb = false(size(gmag));
[AXX, AYY] = ndgrid(od.dbg.ax, od.dbg.ax);
inb(sqrt(AXX.^2+AYY.^2) <= od.dbg.r_beam) = true;
slope_rms = sqrt(mean(gmag(inb).^2));  slope_max = max(gmag(inb));
fprintf('  checker in-beam slope: rms %.3e, max %.3e rad of surface\n', slope_rms, slope_max);

% ---- B. DM-tilt phase response -------------------------------------------
rx_base = fullfile(here, 'l2m_base_arm.in');
G0 = macos.design.twyman_green('to_grid_file', 'dm_zero.txt', ...
    'to_grid_n', 256, 'to_grid_dx', 0.35);         % same rig -> same indices
T = G0.T;
macos.load_rx(rx_base);
psi0 = macos.get_elt_psi(T.iTO);
u1 = macos.design.Bench.perp(psi0);
Ku = [0 -u1(3) u1(2); u1(3) 0 -u1(1); -u1(2) u1(1) 0];
Rtilt = @(t) eye(3) + sin(t)*Ku + (1-cos(t))*(Ku*Ku);
E0 = macos.complex_field(T.iDET);
mskb = abs(E0) > 0.1*max(abs(E0(:)));
X = od.dbg.Xt;  Y = od.dbg.Yt;                     % det pixel -> DM mm map
A_ls = [ones(nnz(mskb),1) X(mskb) Y(mskb)];
alphas = [5e-7 1e-6 2e-6];
fprintf('\n=== B. DM-tilt phase response (plane-fit residual = tail error) ===\n');
tilt_res_nm = zeros(size(alphas));  tilt_gain = zeros(size(alphas));
for ia = 1:numel(alphas)
    a = alphas(ia);
    macos.load_rx(rx_base);
    macos.set_elt_psi(T.iTO, Rtilt(a)*psi0);
    Et = macos.complex_field(T.iDET);
    hT = angle(Et .* conj(E0)) * LAM/(4*pi);       % mm
    cfs = A_ls \ hT(mskb);
    resb = hT(mskb) - A_ls*cfs;
    tilt_res_nm(ia) = 1e6*std(resb);
    tilt_gain(ia) = norm(cfs(2:3))/a;              % ideal = 1: h is surface-
                                                   % calibrated (phi*lam/4pi)
    fprintf('  alpha %.1e rad: fitted tilt gain %.4f (ideal 1), plane-residual %.4f nm rms\n', ...
        a, tilt_gain(ia), tilt_res_nm(ia));
end
pB = polyfit(log(alphas), log(tilt_res_nm), 1);
% predicted retrace if uniform-deflection coupling were the whole story --
% the checker's BEAM deflection is 2x its surface slope (reflection)
pred_nm = exp(polyval(pB, log(2*slope_rms)));
fprintf('  scaling: residual ~ alpha^%.2f;  extrapolated to checker rms beam deflection %.1e -> %.2f nm\n', ...
    pB(1), 2*slope_rms, pred_nm);

% ---- C. chief-height mapping vs alpha ------------------------------------
amax = 2*slope_max;                                % beam deflection range
al = linspace(-amax, amax, 9);
sh = zeros(2, numel(al));
macos.load_rx(rx_base);
vdet0 = macos.get_elt_vpt(T.iDET);
psid  = macos.get_elt_psi(T.iDET);
u2 = macos.design.Bench.perp(psid);  v2 = cross(psid, u2);
for ia = 1:numel(al)
    macos.load_rx(rx_base);
    macos.set_elt_psi(T.iTO, Rtilt(al(ia))*psi0);
    sc = macos.trace(T.iDET);  ic = macos.get_ray_info(sc.nRays);
    d = ic.pos(:,1) - vdet0;
    sh(:,ia) = [u2.'; v2.']*d;
end
lev = [ones(numel(al),1) al(:)] \ sh.';            % linear part per axis
shfit = ([ones(numel(al),1) al(:)]*lev).';
shres = sh - shfit;
lever_mm = norm(lev(2,:));
fprintf('\n=== C. chief-height mapping vs alpha (+/-%.1e rad) ===\n', amax);
fprintf('  lever %.3f mm/rad (conjugate error ~ lever/2 = %.1f mm of leg)\n', ...
    lever_mm, lever_mm/2);
fprintf('  max nonlinear chief walk %.4f um over the range\n', 1e3*max(vecnorm(shres)));

% ---- figure ---------------------------------------------------------------
f = figure('Color','w','Position',[60 60 1700 900]);
sub = @(k) subplot(2,3,k);
sub(1);
q = 1e6*tail;  imagesc(q, 'AlphaData', ~isnan(q)); axis image; colorbar;
title(sprintf('retrace term (det - rc), %.2f nm rms', tail_rms_nm));
sub(2);
q = nan(size(ht)); q(m) = 1e6*ht(m); imagesc(q, 'AlphaData', ~isnan(q));
axis image; colorbar; title('truth (nm) on det pixels');
sub(3);
q = nan(size(ht)); q(mg) = 1e6*gm_on_det(mg)*od.dbg.r_beam;  % scaled for vis
imagesc(q, 'AlphaData', ~isnan(q)); axis image; colorbar;
title(sprintf('|grad h| (corr with tail %+.2f)', c_gm));
sub(4);
rb = linspace(0, max(r(mm)), 24);  rc_ = 0.5*(rb(1:end-1)+rb(2:end));
trms = arrayfun(@(k) 1e6*std(tail(mm & r>=rb(k) & r<rb(k+1))), 1:numel(rc_));
plot(rc_, trms, 'o-'); grid on;
xlabel('DM-plane radius (mm)'); ylabel('tail rms (nm)');
title('retrace vs pupil radius');
sub(5);
loglog(alphas, tilt_res_nm, 'o-'); grid on; hold on;
loglog(2*slope_rms, pred_nm, 'rs', 'MarkerFaceColor','r');
xlabel('DM tilt \alpha (rad)'); ylabel('plane-fit residual (nm rms)');
title(sprintf('uniform-tilt tail error ~ \\alpha^{%.2f}; red = checker rms deflection', pB(1)));
sub(6);
plot(al, 1e3*sh(1,:), 'o-', al, 1e3*sh(2,:), 's-'); grid on; hold on;
plot(al, 1e3*shres(1,:), '^--', al, 1e3*shres(2,:), 'v--');
xlabel('DM tilt \alpha (rad)'); ylabel('chief shift at det (\mum)');
legend({'u','v','u nonlin','v nonlin'}, 'Location','best');
title(sprintf('chief mapping: lever %.2f mm/rad', lever_mm));
print(f, fullfile(here,'mech_figure.png'), '-dpng', '-r140');

save(fullfile(here,'mech.mat'), 'od', 'orc', 'tail', 'tail_rms_nm', ...
    'c_ht', 'c_r2', 'c_gm', 'kap', 'slope_rms', 'slope_max', ...
    'alphas', 'tilt_res_nm', 'tilt_gain', 'pB', 'pred_nm', ...
    'al', 'sh', 'shres', 'lever_mm');
fprintf('\nmechanism analysis done -> mech_figure.png, mech.mat\n');
