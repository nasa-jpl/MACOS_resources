%E5_SEG_METOPT  Tier-3 MET-layout optimization on the e5_seg model.
%
% Run AFTER e5_seg.m (loads its saved workspace).  Optimizes the
% segment-truss layout against the post-control wavefront residual
% trace(dwdx*P_dx*dwdx') using the ANALYTIC gauge Jacobian
% (macos.design.dldx_analytic — validated == engine FD in tMet), so
% the search costs linear algebra only; the winner is then realized
% with add_met and validated with the engine-FD dmet_dx.
%
% Search structure (recorded in CURRENT_SLICE): SYMMETRY (one
% launcher pattern per segment, replicated) -> small COMBINATORIC set
% (nf + pairing) x STEP-AND-EVALUATE grid over the continuous knobs
% (launcher radius/clocking, fiducial radius/clocking) -> worst-mode
% guard alongside the trace -> engine validation of the winner.
% Scope: x = the 7x6 SEGMENT DOFs (hub/extra bodies need engine
% TElt/RptElt frames for the analytic rows — queued extension).

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

ctx = struct('pv',pv,'xh',xh,'yh',yh,'seg',seg,'nseg',nseg, ...
    'bodies',bodies,'E',E,'X',X,'G',G,'nw',nw,'sige',sige,'sigl',sigl);

base = struct('rl',0.7,'lclock',pi/6,'nf',3,'rfid',300,'fclock',0);
[r0, w0m] = metric_(base, ctx);
fprintf('baseline (as-built): rms %.3f nm, worst-mode %.3f nm\n', r0*1e9, w0m*1e9);

best = base; rb = r0; wb = w0m; nev = 0; tic;
for nf = [3 6]
  for rl = 0.3:0.05:0.95
    for lc = deg2rad(0:5:55)
      for rf = [150 300 600 1200 2400]
        for fc = deg2rad(0:10:110)
          lay = struct('rl',rl,'lclock',lc,'nf',nf,'rfid',rf,'fclock',fc);
          [r1, w1] = metric_(lay, ctx); nev = nev + 1;
          if r1 < rb, best = lay; rb = r1; wb = w1; end
        end
      end
    end
  end
end
fprintf('%d layouts evaluated in %.1f s (analytic)\n', nev, toc);
fprintf('best: rl=%.2f lclock=%.0f deg nf=%d rfid=%g fclock=%.0f deg\n', ...
    best.rl, rad2deg(best.lclock), best.nf, best.rfid, rad2deg(best.fclock));
fprintf('      rms %.3f nm (was %.3f), worst-mode %.3f nm (was %.3f)\n', ...
    rb*1e9, r0*1e9, wb*1e9, w0m*1e9);

%% engine validation of the winner
res_root = fileparts(fileparts(fileparts(fileparts(here))));
am2 = macos.design.add_met(seg.in, seg, 'hub', nseg+1, ...
    'r_fid', best.rfid, 'nf', best.nf, 'r_launch_frac', best.rl, ...
    'launch_clock', best.lclock, 'fid_clock', best.fclock, ...
    'extra_sources', seg.n_elt-2, ...
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
     'rb', 'wb', 'rfd', 'nev');
fprintf('artifacts: e5_seg_metopt.in / .mat beside the script\n');

function [rms_w, worst] = metric_(lay, c)
if lay.nf == 3, pair = [1 2 2 3 3 1]; else, pair = 1:6; end
thf = lay.fclock + 2*pi*(0:lay.nf-1)/lay.nf;
fid = c.pv + lay.rfid*(c.xh*cos(thf) + c.yh*sin(thf));
tl6 = lay.lclock + 2*pi*(0:5)'/6;
src = zeros(3, 6*c.nseg); tgt = zeros(3, 6*c.nseg);
for s3 = 1:c.nseg
    f3 = c.seg.frames(s3);
    src(:, (s3-1)*6+(1:6)) = f3.rpt + lay.rl*f3.lmon* ...
        (f3.xhat*cos(tl6') + f3.yhat*sin(tl6'));
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
