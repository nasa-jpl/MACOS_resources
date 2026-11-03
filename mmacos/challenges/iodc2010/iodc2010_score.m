function S = iodc2010_score(L, p, opts)
%IODC2010_SCORE  Score a green lens on the IODC 2010 merit, metric first.
%   S = iodc2010_score(L, p) emits the deck, loads it, checks the paraxial
%   focal length, sizes the clear apertures (optional), scans the semi-field
%   until the first failure and bisects back, and returns
%
%     S.efl_mm        paraxial EFL (textbook y-nu, iodc2010_paraxial)
%     S.efl_engine_mm the engine's near-axis EFL (a 1e-4-pupil ray, macos.srt)
%     S.efl_ok        |efl| within p.efl_mm +- p.efl_tol_mm
%     S.D_mm          paraxial entrance-pupil diameter
%     S.theta_lim_deg the semi-field: the largest field (to p.theta_tol_deg)
%                     at which RMS <= limit AND |distortion| <= limit AND no
%                     ray is lost; the first criterion to fail is S.limit_reason
%     S.merit         D_mm * theta_lim_deg   (0 when the lens fails on axis
%                     or the EFL check)
%     S.fields        table of every field evaluated (theta, rms, dist, lost)
%     S.rules         edge-thickness / air-space check at the clear apertures
%     S.deck          the emitted .in;  S.E, S.P the element map and paraxials
%
%   Everything here is what the README states as the metric; the report it
%   writes (<outdir>/<name>_report.txt) prints the convention before the
%   numbers.
arguments
    L struct
    p struct
    opts.quiet (1,1) logical = false
    opts.deck  (1,:) char = ''
end
say = @(varargin) iif_(~opts.quiet, varargin{:});

P = iodc2010_paraxial(L, p.lambda_nm);
S.P = P;  S.L = L;
S.efl_mm = P.efl;
S.efl_ok = abs(abs(P.efl) - p.efl_mm) <= p.efl_tol_mm;
S.D_mm   = P.D;
S.image_real = P.z_img > L.z(end);

if isempty(opts.deck), opts.deck = fullfile(p.outdir, [L.name '.in']); end
if ~macos.has_rx() || true, macos.init(p.model_size); end

% ---- the scan deck: the lens's own apertures, else OPEN (only the stop clips).
%      The semi-field limit is then set by the RMS and the distortion -- the
%      clear apertures are sized to the footprint at that limit afterwards (the
%      "no mounting material beyond the clear aperture" loophole the paper
%      names) and the sized deck is verified there.
auto_ap = p.size_apertures && isempty(L.semidia);
Lw = L;
if isempty(L.semidia), Lw.semidia = inf(1, L.N); end
E = iodc2010_emit(Lw, P, p, opts.deck);
macos.load_rx(E.path);

% engine near-axis EFL: a ray at 1e-4 of the pupil, on axis
F0 = iodc2010_eval(E, P, p, 0);              % sets the on-axis source + stop
r  = macos.srt(1e-4, 0, E.iFP);
h  = r.pos(1, E.iSurf(1));                  % the ray's actual height at surface 1
u  = atan2(r.dir(1, E.iSurf(end)), r.dir(3, E.iSurf(end)));
S.efl_engine_mm = -h / tan(u);
macos.modify();

% ---- the field scan: stop at the first failure, bisect back
rows = row_(F0);
say('  theta %6.2f deg  rms %.4f waves  lost %d/%d  %s\n', 0, F0.rms_waves, F0.nLost, F0.nLaunched, F0.reason);
lo = NaN;  hi = NaN;  reason = F0.reason;
if F0.pass
    lo = 0;
    for th = p.field_scan_deg(p.field_scan_deg > 0)
        F = iodc2010_eval(E, P, p, th);  rows = [rows; row_(F)];
        say('  theta %6.2f deg  rms %.4f waves  dist %+7.3f %%  lost %d/%d  %s\n', ...
            th, F.rms_waves, F.dist_pct, F.nLost, F.nLaunched, F.reason);
        if F.pass, lo = th; else, hi = th; reason = F.reason; break; end
    end
    if isnan(hi)
        say('  every scanned field passes; the scan ends at %.2f deg (extend p.field_scan_deg)\n', lo);
        reason = 'scan-end';
    else
        while hi - lo > p.theta_tol_deg
            mid = 0.5 * (lo + hi);
            F = iodc2010_eval(E, P, p, mid);  rows = [rows; row_(F)];
            if F.pass, lo = mid; else, hi = mid; reason = F.reason; end
        end
    end
end
S.theta_lim_deg = max(lo, 0);  if isnan(lo), S.theta_lim_deg = 0; end
S.limit_reason  = reason;
S.fields = cell2table(rows, 'VariableNames', {'theta_deg', 'rms_waves', 'dist_pct', 'nLost', 'nLaunched', 'pass', 'reason'});
S.fields = sortrows(S.fields, 'theta_deg');

% ---- clear apertures = the footprint at the semi-field limit; verify there
if auto_ap
    fp = zeros(1, L.N);
    for th = unique([0, S.theta_lim_deg])
        iodc2010_eval(E, P, p, th);              % source + stop at this field
        for k = 1:L.N
            tk = macos.trace(E.iSurf(k));  rk = macos.get_ray_info(tk.nRays);
            okk = rk.ok_trace(:).';
            fp(k) = max(fp(k), max(hypot(rk.pos(1, okk), rk.pos(2, okk))));
        end
    end
    Lw.semidia = fp * (1 + p.aperture_margin);
    E = iodc2010_emit(Lw, P, p, opts.deck);
    macos.load_rx(E.path);
    Fv = iodc2010_eval(E, P, p, S.theta_lim_deg);
    S.verify = struct('theta_deg', S.theta_lim_deg, 'rms_waves', Fv.rms_waves, 'nLost', Fv.nLost, ...
        'pass', Fv.pass, 'reason', Fv.reason);
    say('  apertures sized to the footprint at %.2f deg: [%s] mm; verified: %s\n', ...
        S.theta_lim_deg, sprintf('%.3f ', Lw.semidia), tern_(Fv.pass, 'pass', ['FAIL ' Fv.reason]));
    if ~Fv.pass && F0.pass          % an on-axis failure keeps its own reason (the verify at 0 repeats it)
        S.limit_reason = ['sized-deck ' Fv.reason];  S.theta_lim_deg = 0;
    end
end
S.E = E;  S.deck = E.path;  S.semidia = Lw.semidia;  S.L = L;

% ---- rules: positive edge thickness in glass, positive edge air space
S.rules = edge_rules_(Lw, P);

S.merit = 0;
if S.efl_ok && S.image_real, S.merit = S.D_mm * S.theta_lim_deg; end

% ---- report
S.report = fullfile(p.outdir, [L.name '_report.txt']);
write_report_(S, p);
say('  %s: |EFL| %.4f mm (%s), D %.3f mm, theta_lim %.2f deg (%s) -> merit %.1f\n', ...
    L.name, abs(S.efl_mm), tern_(S.efl_ok, 'ok', 'OUT OF TOLERANCE'), S.D_mm, ...
    S.theta_lim_deg, S.limit_reason, S.merit);
end

% -------------------------------------------------------------------------
function R = edge_rules_(L, P)
% Sag of surface k at its clear semi-diameter, and the axial/edge thickness
% of every gap before the image.  Glass gaps need a positive edge thickness,
% air gaps a positive edge air space (the problem's "glass constraint").
N = L.N;
sag = zeros(1, N);
for k = 1:N
    if P.c(k) ~= 0
        a = L.semidia(k);  Rk = 1 / P.c(k);
        if a > abs(Rk), sag(k) = NaN; else, sag(k) = Rk - sign(Rk) * sqrt(Rk^2 - a^2); end
    end
end
edge = zeros(1, N - 1);  ok = true(1, N - 1);
for k = 1:N - 1
    edge(k) = L.t(k) - sag(k) + sag(k + 1);
    ok(k)   = isfinite(edge(k)) && edge(k) > 0;
end
R = struct('sag', sag, 'edge_thickness', edge, 'ok', all(ok), 'per_gap_ok', ok, ...
    'semidia', L.semidia);
end

function write_report_(S, p)
fid = fopen(S.report, 'w');  c = onCleanup(@() fclose(fid));
fprintf(fid, 'IODC 2010 "green lens" -- %s\n\n', S.L.name);
fprintf(fid, 'METRIC (stated before the numbers; challenges/iodc2010/README.md):\n');
fprintf(fid, '  RMS wavefront error in waves at %g nm on the exit-pupil sphere centred on the chief\n', p.lambda_nm);
fprintf(fid, '  ray''s image-plane point (chief-ray reference; tilt and focus not removed), piston\n');
fprintf(fid, '  removed, %d rays across the paraxial entrance pupil, every launched ray must pass;\n', p.nGridpts);
fprintf(fid, '  distortion 100 (yR - yP)/yP at the image plane used; EFL paraxial (y-nu).\n');
fprintf(fid, '  Semi-field = the largest field (to %.2f deg) with RMS <= %.3f, |dist| <= %.1f %%, no ray lost.\n\n', ...
    p.theta_tol_deg, p.rms_limit, p.dist_limit);
fprintf(fid, 'PRESCRIPTION: R = %.6f mm, signs [%s], t = [%s] mm, glass {%s}, stop z = %.4f mm a = %.4f mm\n', ...
    S.L.R, sprintf('%+d ', S.L.sgn), sprintf('%.4f ', S.L.t), strjoin(S.L.glass, ','), S.L.stop(1), S.L.stop(2));
fprintf(fid, '  clear semi-diameters [%s] mm; image plane at z = %.6f mm (paraxial focus %.6f)\n\n', ...
    sprintf('%.4f ', S.semidia), S.P.z_img, S.P.z_focus);
fprintf(fid, 'FIRST ORDER: EFL %.6f mm (engine near-axis %.6f; %s), D %.4f mm (f/%.3f), EP at z = %.4f, XP at z = %.4f mm\n', ...
    S.efl_mm, S.efl_engine_mm, tern_(S.efl_ok, 'within 0.01 mm of 100', 'OUT OF TOLERANCE'), S.D_mm, S.P.fno, S.P.z_ep, S.P.z_xp);
fprintf(fid, 'RULES: edge thickness/air space per gap [%s] mm -> %s\n\n', ...
    sprintf('%.4f ', S.rules.edge_thickness), tern_(S.rules.ok, 'all positive', 'VIOLATION'));
fprintf(fid, 'FIELDS:\n  theta_deg  rms_waves  dist_pct  lost/launched  result\n');
T = S.fields;
for i = 1:height(T)
    fprintf(fid, '  %8.3f   %8.5f  %+8.3f   %4d/%-4d      %s\n', T.theta_deg(i), T.rms_waves(i), ...
        T.dist_pct(i), T.nLost(i), T.nLaunched(i), tern_(T.pass(i), 'pass', ['FAIL ' T.reason{i}]));
end
fprintf(fid, '\nRESULT: semi-field %.2f deg (limited by %s), D %.3f mm, MERIT D*theta = %.1f\n', ...
    S.theta_lim_deg, S.limit_reason, S.D_mm, S.merit);
fprintf(fid, 'Published 2010 winner: 81.30 mm x 43.50 deg = 3537 (Bates); 10th place 1094.\n');
end

function r = row_(F)
r = {F.theta_deg, F.rms_waves, F.dist_pct, F.nLost, F.nLaunched, F.pass, F.reason};
end

function out = tern_(c, a, b)
if c, out = a; else, out = b; end
end

function iif_(c, varargin)
if c, fprintf(varargin{:}); end
end
