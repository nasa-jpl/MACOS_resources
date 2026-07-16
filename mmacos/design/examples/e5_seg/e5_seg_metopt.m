%E5_SEG_METOPT  Tier-3 MET-layout optimization on the e5_seg model.
%
% Run AFTER e5_seg.m (loads its saved workspace; GRID='Hex').
% Optimizes the segment-truss layout against the post-control
% wavefront residual trace(dwdx*P_dx*dwdx') using the ANALYTIC gauge
% Jacobian (macos.design.dldx_analytic == engine FD, tMet); the winner
% is realized with add_met('launch_pts',...) and re-validated with the
% engine-FD dmet_dx.
%
% Launcher placement model (Dave 2026-07-12):
%   - launchers must NOT obscure the reflecting surface: they sit on
%     the segment-boundary hexagon OFFSET OUTWARD by EDGE_OFF (5 mm
%     nominal) — i.e. always exactly the specified clearance off the
%     optical edge, in the inter-segment gap zone;
%   - 6 launchers = 3 MIRROR PAIRS about the segment's RADIAL
%     CENTERLINE (center segment uses its xhat), pair angles free —
%     the optimizer explores positions, not just a clocked ring.
% Search: combinations of 3 pair angles x fiducial ring (nf, radius,
% clocking) step-and-evaluate; worst-mode guard; engine validation.

EDGE_OFF = 5;                      % launcher clearance off the optical edge, mm
PHI_GRID = deg2rad(10:10:170);     % candidate pair angles off the radial line
RFID_GRID = [150 300 600 1200 2400];
FCLK_GRID = deg2rad(0:15:105);
NF_GRID  = [3 6];

here = fileparts(mfilename('fullpath'));
S = load(fullfile(here, 'e5_seg.mat'));
seg = S.seg; nseg = seg.nseg; n42 = 6*nseg;
D = S.dwdx(:, 1:n42); E = S.dedx(:, 1:n42); X = S.X(1:n42, 1:n42);
sige = sqrt(S.Re(1,1)); sigl = sqrt(S.Rl(1,1));
nw = size(D, 1); G = D'*D;
bodies = struct('rpt', {}, 'T', {});
for s2 = 1:nseg
    f = seg.frames(s2);
    bodies(s2) = struct('rpt', f.rpt, 'T', [f.xhat f.yhat f.zhat]);
end
% hub fiducial plane (same construction as add_met)
L = readlines(seg.in); tl = strtrim(L);
g3 = @(key,i0) str2double(string(regexp(L(find(startsWith(tl(i0:end), ...
    key+"="),1)+i0-1), key+'=\s*(\S+)\s+(\S+)\s+(\S+)','tokens','once')))';
ihub = find(tl == "EltName=  m2", 1);
pv = g3("VptElt", ihub); ps = g3("psiElt", ihub); ps = ps/norm(ps);
[~,imin] = min(abs(ps)); e0 = zeros(3,1); e0(imin) = 1;
xh = cross(ps,e0); xh = xh/norm(xh); yh = cross(ps,xh);

% boundary-true hex geometry, offset outward by the edge clearance
% (macos.design.hex_tile: apothem = width/2, ONE global tiling clocking
% -- NOT per-segment face-frame angles), and the per-segment radial
% centerline (pair-symmetry axis) in the TILING plane.
T = macos.design.hex_tile(seg, EDGE_OFF);
C2 = [T.u.'; T.v.'] * ([seg.frames.rpt] - T.c0);
rad_ang = atan2(C2(2,:), C2(1,:));
rad_ang(vecnorm(C2) < 1e-6) = 0;                   % center segment
ctx = struct('pv',pv,'xh',xh,'yh',yh,'seg',seg,'nseg',nseg, ...
    'bodies',bodies,'E',E,'X',X,'G',G,'nw',nw,'sige',sige,'sigl',sigl, ...
    'rad_ang',rad_ang,'T',T,'C2',C2);

% baseline = the as-built clocked ring re-expressed in this model:
% pairs at 30/90/150 deg on the offset boundary
base = struct('phis', deg2rad([30 90 150]), 'nf', 3, 'rfid', 300, 'fclock', 0);
[r0, w0m] = metric_(base, ctx);
fprintf('baseline (edge ring, %g mm clearance): rms %.3f nm, worst %.3f nm\n', ...
    EDGE_OFF, r0*1e9, w0m*1e9);

combos = nchoosek(1:numel(PHI_GRID), 3);
best = base; rb = r0; wb = w0m; nev = 0; tic;
for nf = NF_GRID
  for ci = 1:size(combos,1)
    for rf = RFID_GRID
      for fc = FCLK_GRID
        lay = struct('phis', PHI_GRID(combos(ci,:)), 'nf', nf, ...
                     'rfid', rf, 'fclock', fc);
        [r1, w1] = metric_(lay, ctx); nev = nev + 1;
        if r1 < rb, best = lay; rb = r1; wb = w1; end
      end
    end
  end
end
fprintf('%d layouts evaluated in %.1f s (analytic)\n', nev, toc);
fprintf('best: pair angles [%s] deg, nf=%d, rfid=%g, fclock=%.0f deg\n', ...
    join(string(round(rad2deg(best.phis))), ' '), best.nf, best.rfid, ...
    rad2deg(best.fclock));
fprintf('      rms %.3f nm (was %.3f), worst-mode %.3f nm (was %.3f)\n', ...
    rb*1e9, r0*1e9, wb*1e9, w0m*1e9);

%% engine validation of the winner (realized via explicit launch_pts)
[~, ~, LP] = metric_(best, ctx);
am2 = macos.design.add_met(seg.in, seg, 'hub', nseg+1, ...
    'r_fid', best.rfid, 'nf', best.nf, 'fid_clock', best.fclock, ...
    'launch_pts', LP, 'extra_sources', seg.n_elt-2, ...
    'out_in', fullfile(seg.run.workdir, 'e5_seg_metopt.in'));
old = cd(seg.run.workdir); restore = onCleanup(@() cd(old));
macos.init(512); macos.load_rx(am2.in); macos.trace();
dm2 = macos.design.dmet_dx(seg.seg_elts);
H = [E; dm2.dldx]; R = blkdiag(sige^2*eye(size(E,1)), ...
                               sigl^2*eye(size(dm2.dldx,1)));
P = X - X*H'*((H*X*H' + R) \ (H*X));
rfd = sqrt(trace(P*G)/nw);
fprintf('engine-FD validation of winner: rms %.3f nm (analytic %.3f, %.2f%%)\n', ...
    rfd*1e9, rb*1e9, 100*abs(rfd-rb)/rb);
copyfile(am2.in, fullfile(here, 'e5_seg_metopt.in'));
save(fullfile(here, 'e5_seg_metopt.mat'), 'base', 'best', 'r0', 'w0m', ...
     'rb', 'wb', 'rfd', 'nev', 'EDGE_OFF');

% MET setup view of the WINNER (engine holds am2's Rx + trace): optimized
% launchers filled, baseline edge-ring launchers as open circles.
[~, ~, LP0] = metric_(base, ctx);
fv = macos.design.met_view(seg, am2, 'visible', false, ...
    'overlay_pts', [LP0{:}], 'edge_off', EDGE_OFF, ...
    'title', sprintf(['e5_seg optimized MET layout: %.3f -> %.3f nm rms ' ...
                      '(pairs [%s] deg, nf=%d, rfid=%g, fclock=%.0f deg; open circles = baseline)'], ...
                     r0*1e9, rb*1e9, ...
                     join(string(round(rad2deg(best.phis))), ' '), ...
                     best.nf, best.rfid, rad2deg(best.fclock)), ...
    'save', fullfile(here, 'e5_seg_metopt_layout.png'));
close(fv);
fprintf('artifacts: e5_seg_metopt.in / .mat / _layout.png beside the script\n');

function [rms_w, worst, LP] = metric_(lay, c)
% 3 mirror pairs about each segment's radial centerline, ON the
% boundary-true hex offset outward by the edge clearance (c.T =
% macos.design.hex_tile(seg, EDGE_OFF): apothem = width/2 + off, one
% GLOBAL tiling clocking).  Angles are tiling-plane angles.
if lay.nf == 3, pair = [1 2 2 3 3 1]; else, pair = 1:6; end
thf = lay.fclock + 2*pi*(0:lay.nf-1)/lay.nf;
fid = c.pv + lay.rfid*(c.xh*cos(thf) + c.yh*sin(thf));
src = zeros(3, 6*c.nseg); tgt = zeros(3, 6*c.nseg);
LP = cell(1, c.nseg);
for s3 = 1:c.nseg
    phi = c.rad_ang(s3) + [lay.phis, -lay.phis];      % 6 angles, mirror pairs
    r  = c.T.boundary(phi, s3);                       % offset-hex boundary
    P2 = c.C2(:, s3) + r.*[cos(phi); sin(phi)];
    P6 = c.T.c0 + c.T.u*P2(1,:) + c.T.v*P2(2,:);
    LP{s3} = P6;
    src(:, (s3-1)*6+(1:6)) = P6;
    tgt(:, (s3-1)*6+(1:6)) = fid(:, pair);
end
Hl = macos.design.dldx_analytic(c.bodies, src, tgt, ...
                                repelem(1:c.nseg,6), zeros(1,6*c.nseg));
H = [c.E; Hl];
R = blkdiag(c.sige^2*eye(size(c.E,1)), c.sigl^2*eye(size(Hl,1)));
P = c.X - c.X*H'*((H*c.X*H' + R) \ (H*c.X));
rms_w = sqrt(trace(P*c.G)/c.nw);
worst = sqrt(max(real(eig(P*c.G))));
end
