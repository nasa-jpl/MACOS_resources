%E5_SEG_METOPT  Tier-3 MET-layout optimization on the e5_seg model (v2).
%
% Run AFTER e5_seg.m (loads its saved workspace; GRID='Hex').
% Minimizes the post-control wavefront residual trace(dwdx*P_dx*dwdx')
% using the ANALYTIC gauge Jacobian (macos.design.dldx_analytic ==
% engine FD, tMet); the winner is realized with
% add_met('launch_pts',...,'pair_map',...) and re-validated with the
% engine-FD dmet_dx.
%
% Search space (Dave 2026-07-12 + 2026-07-16):
%   - launchers ON the segment's TRUE boundary (macos.design.hex_tile:
%     apothem = width/2, one global tiling clocking) offset OUTWARD by
%     EDGE_OFF so they never obscure the reflecting surface;
%   - HARD separation constraint: no two launchers anywhere in the
%     array (same or adjacent segments -- corner junctions!) closer
%     than MIN_SEP (~50 mm hardware envelope);
%   - TWO placement families, both mirror-symmetric about each
%     segment's radial centerline:
%       'spread'  3 pairs at free angles +/-phi_k (v1 model)
%       'cluster' 3 closely-spaced PAIRS (intra-pair arc delta) at
%                 {c0, +psi, -psi}, c0 = 0 or 180 deg -- clustered
%                 launchers on edges/corners whose two beams aim at
%                 DIFFERENT fiducials (maximizes beam-angle diversity
%                 from a common origin);
%   - beam-to-fiducial ASSIGNMENT enumerated per layout (not fixed
%     Stewart crossing struts): 3 patterns for 'spread', all ordered
%     per-cluster fiducial pairs for 'cluster';
%   - hub fiducial ring radius up to the SIDES of M2 (parsed from the
%     hub element's lMon), free clocking.
% Hierarchical: feasibility gate -> coarse assignment sweep -> refine
% the shortlist over the fiducial grids; worst-mode guard; engine
% validation of the winner.

EDGE_OFF = 5;                      % launcher clearance off the optical edge, mm
MIN_SEP  = 50;                     % min launcher-launcher separation, mm
PATTERN_FRAME = 'radial';          % launcher-pattern reference per segment:
                                   %  'radial'  = its radial centerline (max
                                   %              symmetry wrt the array)
                                   %  'segment' = its OWN face-frame x-axis --
                                   %              every segment in a ring gets
                                   %              the IDENTICAL pattern wrt its
                                   %              clocking (builder-uniform
                                   %              parts, Dave 2026-07-16)
FAMILIES = ["spread" "cluster"];   % keep BOTH: clustered pairs stay a
                                   % first-class option -- co-located pairs
                                   % help decouple segment DEFORMATION from
                                   % rigid-body DOFs in the estimator (Dave)
PHI_GRID  = deg2rad(10:10:170);    % 'spread': candidate pair angles
PSI_GRID  = deg2rad(20:20:160);    % 'cluster': mirrored cluster centers
C0_GRID   = [0 pi];                % 'cluster': on-centerline cluster
DELTA_GRID = deg2rad([4 10 20]);   % 'cluster': intra-pair arc separation
FCLK_COARSE = deg2rad(0:30:90);
FCLK_FINE   = deg2rad(0:15:105);
NF_GRID   = [3 6];                 % fiducial count (Dave: >=3, likely 6)
NREFINE   = 12;                    % layouts refined over the fine grids

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
% hub fiducial plane + M2 size (same construction as add_met)
L = readlines(seg.in); tl = strtrim(L);
g3 = @(key,i0) str2double(string(regexp(L(find(startsWith(tl(i0:end), ...
    key+"="),1)+i0-1), key+'=\s*(\S+)\s+(\S+)\s+(\S+)','tokens','once')))';
g1 = @(key,i0) str2double(string(regexp(L(find(startsWith(tl(i0:end), ...
    key+"="),1)+i0-1), key+'=\s*(\S+)','tokens','once')));
ihub = find(tl == "EltName=  m2", 1);
pv = g3("VptElt", ihub); ps = g3("psiElt", ihub); ps = ps/norm(ps);
[~,imin] = min(abs(ps)); e0 = zeros(3,1); e0(imin) = 1;
xh = cross(ps,e0); xh = xh/norm(xh); yh = cross(ps,xh);
% fiducials must MOUNT ON M2, near its edge (~25 mm inside the rim;
% Dave 2026-07-16: no structure beyond the mirror) -- ring radius zone
% = [r_ap - 25, r_ap] from the hub's circular aperture
r_ap = g1("ApVec", ihub);
if ~isfinite(r_ap), r_ap = g1("lMon", ihub); end
RFID_GRID   = round([r_ap-25, r_ap-12, r_ap]);
RFID_COARSE = RFID_GRID(1);
fprintf('hub m2 aperture radius %.0f mm -> fiducial rim zone [%s]\n', ...
    r_ap, join(string(RFID_GRID), ' '));

% boundary-true hex geometry (apothem = width/2 + clearance, ONE global
% tiling clocking) + per-segment radial centerline in the TILING plane
T = macos.design.hex_tile(seg, EDGE_OFF);
C2 = [T.u.'; T.v.'] * ([seg.frames.rpt] - T.c0);
rad_ang = atan2(C2(2,:), C2(1,:));
rad_ang(vecnorm(C2) < 1e-6) = 0;
% launcher-pattern reference angle per segment (PATTERN_FRAME)
if strcmpi(PATTERN_FRAME, 'segment')
    ref_ang = zeros(1, nseg);            % segment face-frame x in tiling plane
    for s2 = 1:nseg
        xs = seg.frames(s2).xhat;
        ref_ang(s2) = atan2(dot(xs, T.v), dot(xs, T.u));
    end
else
    ref_ang = rad_ang;
end
ctx = struct('pv',pv,'xh',xh,'yh',yh,'seg',seg,'nseg',nseg, ...
    'bodies',bodies,'E',E,'X',X,'G',G,'nw',nw,'sige',sige,'sigl',sigl, ...
    'ref_ang',ref_ang,'T',T,'C2',C2,'min_sep',MIN_SEP);

% baseline = edge ring, Stewart crossing struts, rim-zone fiducials
base = struct('family',"spread", 'angs',deg2rad([30 90 150 -30 -90 -150]), ...
              'pmap',[1 2 2 3 3 1], 'nf',3, 'rfid',RFID_COARSE, 'fclock',0);
[~, LPb, srcb] = place_(base, ctx);
fprintf('baseline launcher min separation: %.1f mm (MIN_SEP %g)\n', ...
    minsep_(srcb), MIN_SEP);
[r0, w0m] = metric_(base, ctx);
fprintf('baseline (edge ring, %g mm clearance): rms %.3f nm, worst %.3f nm\n', ...
    EDGE_OFF, r0*1e9, w0m*1e9);

%% ---- enumerate candidate layouts --------------------------------------
cands = {};
% 'spread': pair-angle combos x canonical assignment patterns per nf
% (global fiducial-index rotation is redundant with the ring symmetry)
spread_pmaps = { ...
    3, [1 2 2 3 3 1; 1 1 2 2 3 3; 1 2 3 1 2 3]; ...
    6, [1 2 3 4 5 6; 1 4 2 5 3 6; 2 1 4 3 6 5]};
combos = nchoosek(1:numel(PHI_GRID), 3);
if ~any(FAMILIES == "spread"), combos = combos([], :); end
for ci = 1:size(combos,1)
    phis = PHI_GRID(combos(ci,:));
    for nfi = 1:size(spread_pmaps,1)
        nf = spread_pmaps{nfi,1};
        if ~any(NF_GRID == nf), continue; end
        pm = spread_pmaps{nfi,2};
        for pi_ = 1:size(pm,1)
            cands{end+1} = struct('family',"spread", ...
                'angs',[phis, -phis], 'pmap',pm(pi_,:), 'nf',nf); %#ok<SAGROW>
        end
    end
end
n_spread = numel(cands);
% 'cluster': each cluster's 2 beams aim at DIFFERENT fiducials.
% Structured assignment (tractable, symmetry-aware): cluster k gets the
% pair (1+o_k, 1+mod(o_k+d, nf)) -- one index STRIDE d shared by all
% clusters, free per-cluster offsets o_k, both beam orders.
if ~any(FAMILIES == "cluster"), C0_GRID = []; end
for c0v = C0_GRID
  for psi = PSI_GRID
    for dl = DELTA_GRID
      h = dl/2;
      angs = [c0v-h, c0v+h, psi-h, psi+h, -psi-h, -psi+h];
      for nf = NF_GRID
        for d = 1:floor(nf/2)
          for o1 = 0:nf-1, for o2 = 0:nf-1, for o3 = 0:nf-1 %#ok<ALIGN>
            pa = [1+o1, 1+mod(o1+d,nf), 1+o2, 1+mod(o2+d,nf), ...
                  1+o3, 1+mod(o3+d,nf)];
            for ord = 0:1
              if ord, pm = pa([2 1 4 3 6 5]); else, pm = pa; end
              cands{end+1} = struct('family',"cluster", 'angs',angs, ...
                  'pmap',pm, 'nf',nf); %#ok<SAGROW>
            end
          end, end, end
        end
      end
    end
  end
end
fprintf('%d candidate layouts (%d spread + %d cluster)\n', numel(cands), ...
    n_spread, numel(cands) - n_spread);

%% ---- pass 0: geometric feasibility (placement only, no fiducials) -----
% Separation depends only on the launcher angles, so gate once per
% distinct placement, not per (assignment x fiducial) combination.
tic;
feas = false(numel(cands), 1);
sep_cache = containers.Map('KeyType','char', 'ValueType','logical');
for q = 1:numel(cands)
    key = sprintf('%.6f,', cands{q}.angs);
    if ~isKey(sep_cache, key)
        [~, ~, srcq] = place_(cands{q}, ctx);
        sep_cache(key) = minsep_(srcq) >= MIN_SEP;
    end
    feas(q) = sep_cache(key);
end
fprintf('pass 0: %d/%d layouts pass the %g mm separation gate (%.1f s)\n', ...
    nnz(feas), numel(cands), MIN_SEP, toc);

%% ---- pass 1: coarse sweep ---------------------------------------------
tic;
rc = inf(numel(cands), 1);
for q = find(feas).'
    lay = cands{q}; lay.rfid = RFID_COARSE; best_f = inf;
    for fc = FCLK_COARSE
        lay.fclock = fc;
        r1 = metric_(lay, ctx);
        best_f = min(best_f, r1);
    end
    rc(q) = best_f;
end
fprintf('pass 1: %d layouts x %d clockings in %.1f s (analytic)\n', ...
    nnz(feas), numel(FCLK_COARSE), toc);

%% ---- pass 2: refine the shortlist over the full fiducial grids --------
[~, order] = sort(rc);
best = base; rb = r0; wb = w0m; tic;
for q = order(1:min(NREFINE, numel(order))).'
    for rf = RFID_GRID
        for fc = FCLK_FINE
            lay = cands{q}; lay.rfid = rf; lay.fclock = fc;
            [r1, w1] = metric_(lay, ctx);
            if r1 < rb, best = lay; rb = r1; wb = w1; end
        end
    end
end
fprintf('pass 2: %d refined in %.1f s\n', NREFINE, toc);
fprintf('best (%s): angs [%s] deg, pmap [%s], rfid=%g, fclock=%.0f deg\n', ...
    best.family, join(string(round(rad2deg(best.angs))), ' '), ...
    join(string(best.pmap), ' '), best.rfid, rad2deg(best.fclock));
fprintf('      rms %.3f nm (baseline %.3f), worst-mode %.3f nm (was %.3f)\n', ...
    rb*1e9, r0*1e9, wb*1e9, w0m*1e9);

%% ---- engine validation of the winner ----------------------------------
[~, LP] = place_(best, ctx);
am2 = macos.design.add_met(seg.in, seg, 'hub', nseg+1, ...
    'r_fid', best.rfid, 'nf', best.nf, 'fid_clock', best.fclock, ...
    'launch_pts', LP, 'pair_map', best.pmap, ...
    'extra_sources', seg.n_elt-2, 'r_extra', 100, ...  % aft Return: no ApVec
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
     'rb', 'wb', 'rfd', 'EDGE_OFF', 'MIN_SEP', 'RFID_GRID');

% MET setup view of the WINNER: optimized launchers filled, baseline
% edge ring as open circles.
fv = macos.design.met_view(seg, am2, 'visible', false, ...
    'overlay_pts', [LPb{:}], 'edge_off', EDGE_OFF, ...
    'title', sprintf(['e5_seg optimized MET layout (%s): %.3f -> %.3f nm rms ' ...
                      '(pmap [%s], rfid=%g, fclock=%.0f deg; open circles = baseline)'], ...
                     best.family, r0*1e9, rb*1e9, ...
                     join(string(best.pmap), ' '), best.rfid, ...
                     rad2deg(best.fclock)), ...
    'save', fullfile(here, 'e5_seg_metopt_layout.png'));
close(fv);
fprintf('artifacts: e5_seg_metopt.in / .mat / _layout.png beside the script\n');

% ---------------------------------------------------------------------------
function [ok, LP, src] = place_(lay, c)
%PLACE_  Launcher positions for a layout: ON the boundary-true offset
%   hex (c.T) at tiling-plane angles rad_ang(s) + lay.angs.
src = zeros(3, 6*c.nseg);
LP = cell(1, c.nseg);
for s3 = 1:c.nseg
    phi = c.ref_ang(s3) + lay.angs;                   % 6 tiling angles
    r  = c.T.boundary(phi, s3);                       % offset-hex boundary
    P2 = c.C2(:, s3) + r.*[cos(phi); sin(phi)];
    P6 = c.T.c0 + c.T.u*P2(1,:) + c.T.v*P2(2,:);
    LP{s3} = P6;
    src(:, (s3-1)*6+(1:6)) = P6;
end
ok = minsep_(src) >= c.min_sep;
end

function d = minsep_(src)
%MINSEP_  Minimum pairwise distance among all launcher positions.
n = size(src, 2);
D = squeeze(vecnorm(reshape(src, 3, 1, n) - reshape(src, 3, n, 1)));
D(1:n+1:end) = inf;
d = min(D(:));
end

function [rms_w, worst] = metric_(lay, c)
%METRIC_  Post-control wavefront residual for a layout (Inf when the
%   launcher separation constraint is violated).
[ok, ~, src] = place_(lay, c);
if ~ok, rms_w = inf; worst = inf; return; end
thf = lay.fclock + 2*pi*(0:lay.nf-1)/lay.nf;
fid = c.pv + lay.rfid*(c.xh*cos(thf) + c.yh*sin(thf));
tgt = repmat(fid(:, lay.pmap), 1, c.nseg);
Hl = macos.design.dldx_analytic(c.bodies, src, tgt, ...
                                repelem(1:c.nseg,6), zeros(1,6*c.nseg));
H = [c.E; Hl];
R = blkdiag(c.sige^2*eye(size(c.E,1)), c.sigl^2*eye(size(Hl,1)));
P = c.X - c.X*H'*((H*c.X*H' + R) \ (H*c.X));
rms_w = sqrt(trace(P*c.G)/c.nw);
worst = sqrt(max(real(eig(P*c.G))));
end
