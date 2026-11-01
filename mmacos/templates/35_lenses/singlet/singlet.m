% singlet.m  (mmacos/templates/35_lenses/singlet/ -- a parameterized design runner)
% =====================================================================
%  BEST-FORM SINGLET: the bending that minimizes spherical aberration
% =====================================================================
%  A singlet of given focal length, f-number and glass can still be bent:
%  the two radii trade against each other at constant power, and the
%  spherical aberration depends on the split.  Coddington's shape factor
%  X = (c1 + c2)/(c1 - c2) names the bending (X = +1 convex-plano with the
%  curved face toward the object, 0 equi-convex, -1 plano-convex the other
%  way round); the thin-lens third-order theory gives the minimum-SA shape
%  at the position factor p = (m + 1)/(m - 1) of the conjugates as
%
%      X_min = -2 (n^2 - 1) p / (n + 2)        (p = -1 for a distant object)
%
%  (Kingslake & Johnson, Lens Design Fundamentals, 2nd ed., Sec. 4.3;
%  Smith, Modern Optical Engineering, 4th ed., Sec. 6.3).  This driver
%  builds the plano singlet the benches use (Bench.add_lens's form, both
%  ways round), the textbook best form, then lets the ENGINE find the
%  bending -- MACOS CALIB on both radii from an equi-convex start -- and
%  reports the two shape factors side by side with a bending scan.
%
%  METRIC (stated before the numbers): RMS wavefront error in waves at
%  LAMBDA_NM over the full aperture, on the exit-pupil reference sphere
%  centred on the CHIEF ray's image point (macos.opd_ref('chief')), piston
%  removed, tilt and focus KEPT, with the image plane at the lens's own
%  PARAXIAL focus; "best focus" lines move the image plane to the RMS
%  minimum (CALIB on the image plane alone).
%
%  Run:  >> singlet                      the defaults below
%        >> out = singlet('FNUM', 4, 'GLASS', 'N-SF6')   any choice overridden
%  Returns OUT (also saved as OUTDIR/singlet.mat) and writes
%  OUTDIR/singlet_report.txt and OUTDIR/singlet_bending.png.
% =====================================================================
function out = singlet(varargin)
exdir = fileparts(mfilename('fullpath'));

% ====================  USER DESIGN CHOICES  ==========================
o.EFL_MM    = 100;        % focal length (mm)
o.FNUM      = 5;          % f-number EFL / EPD (the stop at the first surface)
o.GLASS     = 'N-BK7';    % any name in macos_f90/macos_glass_list.txt
o.LAMBDA_NM = 587.5618;   % the design wavelength (He d line)
o.THICK_MM  = 4;          % centre thickness (mm)
o.OBJECT_MM = Inf;        % object distance in front of the lens (Inf = collimated)
% ---------------------------------------------------------------------
o.NGRID     = 63;         % rays across the entrance pupil
o.MODEL     = 128;        % engine model size
o.NSCAN     = 41;         % lenses in the bending scan (0 = skip the scan and its figure)
o.OUTDIR    = fullfile(exdir, 'out');
% =====================================================================
for i = 1:2:numel(varargin)
    assert(isfield(o, varargin{i}), 'singlet: unknown choice %s', varargin{i});
    o.(varargin{i}) = varargin{i+1};
end
EFL_MM = o.EFL_MM;  FNUM = o.FNUM;  GLASS = o.GLASS;  LAMBDA_NM = o.LAMBDA_NM;  THICK_MM = o.THICK_MM;
OBJECT_MM = o.OBJECT_MM;  NGRID = o.NGRID;  MODEL = o.MODEL;  OUTDIR = o.OUTDIR;

if ~exist(OUTDIR, 'dir'), mkdir(OUTDIR); end
macos.init(MODEL);
n   = macos.design.glass(GLASS, LAMBDA_NM);
EPD = EFL_MM / FNUM;
if isfinite(OBJECT_MM)
    s_i = 1 / (1 / EFL_MM - 1 / OBJECT_MM);  m = -s_i / OBJECT_MM;
else
    m = 0;
end
p_pos = (m + 1) / (m - 1);
fprintf('====================================================================\n');
fprintf(' Best-form singlet | f = %g mm | f/%g (EPD %.3g mm) | %s n = %.6f at %g nm | t = %g mm\n', ...
    EFL_MM, FNUM, EPD, GLASS, n, LAMBDA_NM, THICK_MM);
fprintf(' object %s (magnification %.4g, position factor p = %.4g)\n', ...
    tern(isfinite(OBJECT_MM), sprintf('at %g mm', OBJECT_MM), 'at infinity'), m, p_pos);
fprintf('====================================================================\n');

ev = @(L, tag) eval_singlet(L, EPD, LAMBDA_NM, OBJECT_MM, NGRID, fullfile(OUTDIR, [tag '.in']));

%% -- Stage 1 -- the plano singlet, both ways round ---------------------
Lcp = shape_lens(1, EFL_MM, THICK_MM, GLASS, LAMBDA_NM, EPD, 'convex_plano');
Lpc = Lcp.flip();  Lpc.name = 'plano_convex';  Lpc.surf_names = {};
r_cp = ev(Lcp, 'convex_plano');
r_pc = ev(Lpc, 'plano_convex');
fprintf('\n[1] the plano singlet (R = %.4f mm, the benches'' add_lens form with a sphere):\n', Lcp.R(1));
fprintf('      curved face to the object (X = +1): RMS %.4f waves\n', r_cp.rms);
fprintf('      flat face to the object   (X = -1): RMS %.4f waves  (%.1fx worse)\n', r_pc.rms, r_pc.rms / r_cp.rms);

%% -- Stage 2 -- Coddington's best form (thin-lens third order) ---------
X_tb = -2 * (n^2 - 1) * p_pos / (n + 2);
Ltb  = shape_lens(X_tb, EFL_MM, THICK_MM, GLASS, LAMBDA_NM, EPD, 'textbook');
r_tb = ev(Ltb, 'textbook');
fprintf('\n[2] textbook minimum-SA bending: X = -2(n^2-1)p/(n+2) = %.4f\n', X_tb);
fprintf('      R1 = %.4f  R2 = %.4f mm (R2/R1 = %.3f), scaled to EFL %.6f mm\n', Ltb.R, Ltb.R(2) / Ltb.R(1), ...
    Ltb.paraxial(LAMBDA_NM).efl);
fprintf('      RMS %.4f waves at the paraxial focus (%.2fx better than convex-plano)\n', r_tb.rms, r_cp.rms / r_tb.rms);

%% -- Stage 3 -- the engine finds the bending: CALIB on both radii -------
L0  = shape_lens(0, EFL_MM, THICK_MM, GLASS, LAMBDA_NM, EPD, 'equiconvex');
r0  = ev(L0, 'equiconvex');
P0  = macos.design.lens_pupils(L0, LAMBDA_NM, 'stop', [0 NaN], 'epd', EPD, 'object', OBJECT_MM);
cal = struct('var_R', [1 2], 'max_iters', 3000, 'ray_grid', 31);
E0  = macos.design.lens_emit(L0, P0, fullfile(OUTDIR, 'calib_bending.in'), 'nGridpts', NGRID, 'calib', cal);
macos.load_rx(E0.path);
res = macos.calib();
Rs  = [-macos.get_elt_kr(E0.iSurf(1)), -macos.get_elt_kr(E0.iSurf(2))];
macos.modify();
Lc  = L0;  Lc.R = Rs;  Lc.name = 'calib';
X_c = shape_factor(Lc);
efl_c = Lc.paraxial(LAMBDA_NM).efl;
% CALIB held the image plane, so it could trade a little power for focus;
% put the solved bending back at the asked EFL and re-score it like the others
Lcs = shape_lens(X_c, EFL_MM, THICK_MM, GLASS, LAMBDA_NM, EPD, 'calib');
r_c = ev(Lcs, 'calib');
fprintf('\n[3] CALIB on R1, R2 from the equi-convex start (RMS %.4f waves at X = 0):\n', r0.rms);
fprintf('      %d iterations (%s), image plane held at the start''s paraxial focus\n', res.n_iter, res.stop_reason);
fprintf('      solved R1 = %.4f  R2 = %.4f mm, EFL %.4f mm -> X = %.4f\n', Rs, efl_c, X_c);
fprintf('      re-scaled to EFL %g: RMS %.4f waves at the paraxial focus\n', EFL_MM, r_c.rms);
fprintf('      shape factor: engine %.4f vs textbook %.4f (difference %+.4f)\n', X_c, X_tb, X_c - X_tb);

%% -- Stage 4 -- the bending scan ----------------------------------------
Xs = linspace(-1.5, 2.5, o.NSCAN);  rs = zeros(size(Xs));  rb = zeros(size(Xs));
for i = 1:numel(Xs)
    r = ev(shape_lens(Xs(i), EFL_MM, THICK_MM, GLASS, LAMBDA_NM, EPD, 'scan'), 'scan');
    rs(i) = r.rms;  rb(i) = r.rms_best;
end
if isempty(Xs), ib = []; ibb = []; else, [~, ib] = min(rs);  [~, ibb] = min(rb); end
if ~isempty(Xs)
fprintf('\n[4] bending scan, X = %.2f .. %.2f (%d lenses at EFL %g):\n', Xs(1), Xs(end), numel(Xs), EFL_MM);
fprintf('      minimum on the grid: X = %.2f at the paraxial focus, X = %.2f at best focus\n', Xs(ib), Xs(ibb));
end
fprintf('      best focus: textbook %.4f, convex-plano %.4f, plano-convex %.4f waves\n', r_tb.rms_best, r_cp.rms_best, r_pc.rms_best);
if ~isempty(Xs)
fig = figure('Visible', 'off', 'Position', [100 100 640 420]);
semilogy(Xs, rs, 'b-', Xs, rb, 'b--', 'LineWidth', 1.4);  hold on;
semilogy(X_tb, r_tb.rms, 'ko', 'MarkerFaceColor', 'k');
semilogy(X_c, r_c.rms, 'rs', 'MarkerSize', 9, 'LineWidth', 1.4);
semilogy([1 -1], [r_cp.rms r_pc.rms], 'g^', 'MarkerFaceColor', 'g');
grid on;  xlabel('Coddington shape factor X = (c_1 + c_2)/(c_1 - c_2)');
ylabel(sprintf('RMS wavefront error (waves at %g nm)', LAMBDA_NM));
legend({'paraxial focus', 'best focus', sprintf('textbook X = %.3f', X_tb), ...
    sprintf('CALIB X = %.3f', X_c), 'plano (X = \pm1)'}, 'Location', 'north');
title(sprintf('%s singlet, f = %g mm, f/%g', GLASS, EFL_MM, FNUM));
exportgraphics(fig, fullfile(OUTDIR, 'singlet_bending.png'), 'Resolution', 110);  close(fig);
end

%% -- Report --------------------------------------------------------------
out = struct('n', n, 'X_textbook', X_tb, 'X_calib', X_c, 'rms_textbook', r_tb.rms, ...
    'rms_calib', r_c.rms, 'rms_convex_plano', r_cp.rms, 'rms_plano_convex', r_pc.rms, ...
    'rms_best_textbook', r_tb.rms_best, 'R_textbook', Ltb.R, 'R_calib', Rs, 'calib', res, ...
    'scan_X', Xs, 'scan_rms', rs, 'scan_rms_best', rb, 'lens_textbook', Ltb, 'lens_calib', Lcs);
fid = fopen(fullfile(OUTDIR, 'singlet_report.txt'), 'w');
fprintf(fid, 'Best-form singlet: %s, f = %g mm, f/%g, %g nm, t = %g mm, object %g mm\n\n', GLASS, EFL_MM, FNUM, LAMBDA_NM, THICK_MM, OBJECT_MM);
fprintf(fid, 'METRIC: RMS WFE (waves) on the chief-ray-centred exit-pupil sphere, piston removed,\n');
fprintf(fid, '        tilt and focus kept, image at the paraxial focus; best focus = RMS-minimum image plane.\n\n');
fprintf(fid, '  lens            X        R1 (mm)     R2 (mm)    RMS paraxial  RMS best focus\n');
rows = {'convex-plano', 1, Lcp, r_cp; 'plano-convex', -1, Lpc, r_pc; 'textbook', X_tb, Ltb, r_tb; 'CALIB', X_c, Lcs, r_c};
for i = 1:size(rows, 1)
    fprintf(fid, '  %-13s %7.4f  %10.4f  %10.4f    %9.5f     %9.5f\n', rows{i, 1}, rows{i, 2}, rows{i, 3}.R, rows{i, 4}.rms, rows{i, 4}.rms_best);
end
fprintf(fid, '\nshape factor: CALIB %.4f vs textbook %.4f (%+.4f); CALIB %d iterations, %s\n', X_c, X_tb, X_c - X_tb, res.n_iter, res.stop_reason);
fclose(fid);
save(fullfile(OUTDIR, 'singlet.mat'), 'out');
fprintf('\nDONE.  Report %s\n', fullfile(OUTDIR, 'singlet_report.txt'));
end

% =====================================================================
function L = shape_lens(X, f, t, glass, lam, epd, name)
% the singlet of shape factor X and thickness t at EFL f exactly: thin-lens
% curvatures c1 = (X+1)/2 k, c2 = (X-1)/2 k, then k solved on the thick lens
cc = @(k) [(X + 1) / 2, (X - 1) / 2] * k;
mk = @(k) macos.design.Lens('R', 1 ./ cc(k), 't', t, 'glass', {glass}, 'D', 1.05 * epd, 'name', name);
n  = macos.design.glass(glass, lam);
k0 = 1 / ((n - 1) * f);
k  = fzero(@(k) mk(k).paraxial(lam).efl - f, k0);
L  = mk(k);
end

function X = shape_factor(L)
c = L.curv();  X = (c(1) + c(2)) / (c(1) - c(2));
end

function r = eval_singlet(L, epd, lam, obj, ngrid, deck)
% RMS on axis at the paraxial focus, then at best focus (CALIB on the image plane)
P = macos.design.lens_pupils(L, lam, 'stop', [0 NaN], 'epd', epd, 'object', obj);
E = macos.design.lens_emit(L, P, deck, 'nGridpts', ngrid);
macos.load_rx(E.path);
F = macos.design.lens_eval(E, P, 0);
assert(F.nLost == 0, 'singlet: %s lost %d rays', L.name, F.nLost);
r.rms = F.rms_waves;
Eb = macos.design.lens_emit(L, P, deck, 'nGridpts', ngrid, 'calib', struct('var_fp', true, 'max_iters', 50));
macos.load_rx(Eb.path);
macos.calib();
zb = macos.get_elt_vpt(Eb.iFP);
macos.modify();
Pb = macos.design.lens_pupils(L, lam, 'stop', [0 NaN], 'epd', epd, 'object', obj, 'image', zb(3));
Eb = macos.design.lens_emit(L, Pb, deck, 'nGridpts', ngrid);   % the image plane (and its Return) at best focus
macos.load_rx(Eb.path);
Fb = macos.design.lens_eval(Eb, Pb, 0);
r.rms_best = Fb.rms_waves;
r.dz_best = zb(3) - P.z_img;
r.F = F;
end

function s = tern(c, a, b)
if c, s = a; else, s = b; end
end
