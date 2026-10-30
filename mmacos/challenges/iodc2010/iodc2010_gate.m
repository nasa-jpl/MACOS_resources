function G = iodc2010_gate(p, L, theta_deg)
%IODC2010_GATE  The evaluator argued: engine wavefront vs the independent tracer.
%   G = iodc2010_gate(p) runs the gate lens (the symmetric pair, stop in the
%   middle, 10 deg) ; G = iodc2010_gate(p, L, theta) any lens / field.
%
%   On the SAME rays the engine launched (its first-surface crossings backed
%   up along the field direction), iodc2010_reftrace computes the optical
%   path to the sphere FEX placed; the gate reports
%     G.dW_max_waves   max |W_engine - W_ref| after removing each map's mean
%     G.drms_waves     |RMS_engine - RMS_ref|
%     G.rms_engine_waves, G.rms_ref_waves
%     G.dP1_max_mm     max distance between the engine's and the tracer's
%                      first-surface crossing (the ray start / intersection)
%     G.dQ_max_mm      the same on the reference sphere
%     G.chief_stop_mm  where the chief crosses the stop plane (0 = aimed)
%     G.yR_engine_mm, G.yR_ref_mm   chief image height both ways
%     G.efl_paraxial_mm, G.efl_engine_mm
%     G.rms_vs_grid    RMS at p.nGridpts and at two denser grids (convergence)
%     G.rms_vs_radius  RMS with the sphere radius changed by +-10 % (the
%                      sensitivity of the convention to the exit-pupil distance)
%   and G.pass_1e6 = (dW_max and drms below 1e-6 wave, dP1/dQ below 1e-9 mm).
arguments
    p struct
    L struct = struct()
    theta_deg (1,1) double = 10
end
if isempty(fieldnames(L))
    n  = iodc2010_glass(p.glass, p.lambda_nm);
    R2 = 80;
    mk2 = @(d) iodc2010_lens('R', R2, 'sgn', [1 0 0 -1], 't', [6 d 6 NaN], ...
        'glass', {'N-BK7', 'air', 'N-BK7', 'air'}, 'stop', [6 + d/2, 7], 'name', 'gate_pair');
    f1 = R2 / (n - 1);  d0 = 2 * f1 - f1^2 / p.efl_mm;
    d  = fzero(@(dd) abs(iodc2010_paraxial(mk2(dd), p.lambda_nm).efl) - p.efl_mm, d0);
    L  = mk2(d);
end
if isempty(L.semidia), L.semidia = repmat(p.semidia_default, 1, L.N); end
P = iodc2010_paraxial(L, p.lambda_nm);
macos.init(p.model_size);
deck = fullfile(tempdir, [L.name '_gate.in']);
E = iodc2010_emit(L, P, p, deck);
macos.load_rx(E.path);
F = iodc2010_eval(E, P, p, theta_deg, 'rays', true);
lam = E.lambda_mm;

ok = F.ray.ok_e1 & F.ray.ok_xp & F.ray.ok_fp;
d  = F.ray.dir_in;
% every ray starts on ONE plane perpendicular to the field direction (the
% collimated object wavefront), 50 mm upstream of the chief's first crossing:
% backing each crossing up by the same distance would copy the first
% surface's sag into the start points.
P1e = F.ray.pos_e1(:, ok);
s0  = d.' * (P1e - P1e(:, 1)) + 50;             % along-d distance from that plane
p0  = P1e - d * s0;
sph = struct('C', F.ray.chief_img, 'R', F.xp.rad, 'V', F.xp.vpt);
T = iodc2010_reftrace(L, P, p0, d, sph);
assert(~T.lost(1) && ~any(T.lost), 'gate: the reference tracer lost a ray the engine kept');

We = F.ray.opl_xp(ok) - F.ray.opl_xp(1);   We = We - mean(We);
Wr = T.W - mean(T.W);
G.dW_max_waves = max(abs(We - Wr)) / lam;
G.rms_engine_waves = sqrt(mean(We.^2)) / lam;
G.rms_ref_waves    = sqrt(mean(Wr.^2)) / lam;
G.drms_waves       = abs(G.rms_engine_waves - G.rms_ref_waves);
G.rms_eval_waves   = F.rms_waves;            % what the scorer uses (same computation)
% the engine's opd() map omits the chief: its RMS must equal the per-ray set
% without ray 1 -- the identity that pins what the map is
Wn = We(2:end);  Wn = Wn - mean(Wn);
G.rms_opd_waves    = F.rms_waves_opd;
G.drms_opd_nochief = abs(sqrt(mean(Wn.^2)) / lam - F.rms_waves_opd);
G.dP1_max_mm = max(vecnorm(F.ray.pos_e1(:, ok) - T.P1));
G.dQ_max_mm  = max(vecnorm(F.ray.pos_xp(:, ok) - T.Q));
G.chief_stop_mm = hypot(T.Pstop(1, 1), T.Pstop(2, 1));
G.yR_engine_mm = F.yR;  G.yR_ref_mm = T.Pimg(2, 1);  G.yP_mm = F.yP;
G.dist_pct = F.dist_pct;
G.nRays = nnz(ok);  G.nLost_engine = F.nLost;

% EFL: paraxial vs the engine's near-axis ray
F0 = iodc2010_eval(E, P, p, 0);
r  = macos.srt(1e-4, 0, E.iFP);
h  = r.pos(1, E.iSurf(1));                  % the ray's actual height (srt's scale is not assumed)
u  = atan2(r.dir(1, E.iSurf(end)), r.dir(3, E.iSurf(end)));
G.efl_paraxial_mm = P.efl;  G.efl_engine_mm = -h / tan(u);
G.rms_axis_waves  = F0.rms_waves;
macos.modify();

% sampling convergence of the RMS at the gate field
grids = [p.nGridpts, 2 * p.nGridpts - 1, 3 * p.nGridpts - 2];
grids = grids(grids <= p.model_size);
rmsg = zeros(size(grids));
for i = 1:numel(grids)
    pg = p;  pg.nGridpts = grids(i);
    Eg = iodc2010_emit(L, P, pg, deck);  macos.load_rx(Eg.path);
    Fg = iodc2010_eval(Eg, P, pg, theta_deg);  rmsg(i) = Fg.rms_waves;
end
G.rms_vs_grid = [grids; rmsg].';
macos.load_rx(E.path);

% sensitivity to the reference-sphere radius (same centre, same rays)
rr = [0.9 1 1.1] * F.xp.rad;  rmsr = zeros(size(rr));
for i = 1:numel(rr)
    Ti = iodc2010_reftrace(L, P, p0, d, struct('C', sph.C, 'R', rr(i), 'V', sph.C - rr(i) * F.ray.chief_dir_img));
    Wi = Ti.W - mean(Ti.W);  rmsr(i) = sqrt(mean(Wi.^2)) / lam;
end
G.rms_vs_radius = [rr; rmsr].';

G.theta_deg = theta_deg;  G.lens = L.name;  G.deck = deck;
G.pass_1e6 = G.dW_max_waves < 1e-6 && G.drms_waves < 1e-6 && G.dP1_max_mm < 1e-9 && G.dQ_max_mm < 1e-9 && G.drms_opd_nochief < 1e-9;

fprintf('GATE %s @ %.1f deg, %d rays: RMS engine %.6f / ref %.6f waves (|d| %.2e), max|dW| %.2e waves\n', ...
    L.name, theta_deg, G.nRays, G.rms_engine_waves, G.rms_ref_waves, G.drms_waves, G.dW_max_waves);
fprintf('     first-surface crossings agree to %.2e mm, sphere points to %.2e mm; chief at the stop %.2e mm off centre\n', ...
    G.dP1_max_mm, G.dQ_max_mm, G.chief_stop_mm);
fprintf('     yR engine %.6f / ref %.6f mm, yP %.6f -> distortion %+.4f %%; EFL paraxial %.6f / engine %.6f mm\n', ...
    G.yR_engine_mm, G.yR_ref_mm, G.yP_mm, G.dist_pct, G.efl_paraxial_mm, G.efl_engine_mm);
fprintf('     RMS vs grid: %s | vs sphere radius x0.9/1/1.1: %s\n', ...
    sprintf('%d->%.6f ', G.rms_vs_grid.'), sprintf('%.6f ', G.rms_vs_radius(:, 2)));
fprintf('     opd() map RMS %.6f = per-ray set without the chief to %.1e; pass at 1e-6 wave: %d\n', G.rms_opd_waves, G.drms_opd_nochief, G.pass_1e6);
end
