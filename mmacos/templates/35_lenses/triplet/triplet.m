% triplet.m  (mmacos/templates/35_lenses/triplet/ -- a parameterized design runner)
% =====================================================================
%  COOKE TRIPLET: the first lens that carries a field
% =====================================================================
%  A singlet or a cemented doublet corrects the axis; off axis their field
%  is set by astigmatism and the Petzval curvature, which no bending
%  removes.  H. Dennis Taylor's triplet (Cooke, 1893) spreads the power over
%  a positive - negative - positive train with air spaces: the separation
%  lets the negative element flatten the Petzval sum while the outer pair
%  keep the power, and the near-symmetry about the central stop cancels coma,
%  distortion and lateral color to first order.
%
%  The thin-lens SEED is the textbook construction (Kingslake & Johnson,
%  Lens Design Fundamentals, 2nd ed., Ch. 12; Smith, Modern Optical
%  Engineering, 4th ed., Sec. 12.6): outer crowns of equal power phi_a, a
%  central flint phi_b at the stop, two equal air spaces d, solved for
%      the total power             phi(phi_a, phi_b, d) = 1/EFL
%      zero axial color            sum y_i^2 phi_i / V_i = 0
%      the Petzval sum             sum phi_i / n_i = PETZVAL / EFL
%  (lateral color and distortion vanish by the symmetry).  Then the thick
%  lens: each element at its centre thickness, crowns equi-convex, the flint
%  equi-concave, scaled to EFL_MM; then MACOS CALIB on the six radii and the
%  axial positions of the last four surfaces at three fields, d light; then
%  the field scan.
%
%  Lessons carried from the IODC 2010 lane (challenges/iodc2010): CALIB has
%  no distortion row, so distortion is REPORTED, never fitted; the seed's air
%  spaces sit well off any floor (a gap on its floor clamps the first move).
%
%  METRIC (stated before the numbers): per field, RMS wavefront error in
%  waves at LAMBDA_NM over the full entrance pupil, on the exit-pupil sphere
%  centred on the chief ray's image point (macos.opd_ref('chief')), piston
%  removed, tilt and focus kept, image plane fixed; distortion
%  100 (yR - yP)/yP with yP the paraxial chief height.  Rays are uniform
%  over the paraxial entrance pupil; a lost ray is reported.
%
%  Run:  >> triplet                      the defaults below
%        >> out = triplet('HALF_FIELD', 15, 'FNUM', 4)
%  Returns OUT (also OUTDIR/triplet.mat); writes OUTDIR/triplet_report.txt and
%  OUTDIR/triplet_field.png.
% =====================================================================
function out = triplet(varargin)
exdir = fileparts(mfilename('fullpath'));

% ====================  USER DESIGN CHOICES  ==========================
o.EFL_MM     = 100;          % focal length (mm)
o.FNUM       = 5;            % f-number
o.HALF_FIELD = 20;           % semi-field (deg)
o.CROWN      = 'SK16';       % outer elements (Taylor's classic pair: SK16 / F4)
o.FLINT      = 'F4';         % central element
o.PETZVAL    = 0.2;          % Petzval sum x EFL (0 = flat; 0.2-0.3 is the classic compromise)
o.T_MM       = [5 2 5];      % centre thicknesses (mm)
o.LAMBDA_NM  = 587.5618;     % design wavelength (d); color is set by the seed
o.CAL_FIELDS = [0 0.7 1];    % CALIB fields as fractions of HALF_FIELD
o.N_CYCLES   = 3;            % CALIB cycles (a radii round + a spacings round each)
o.FLOOR_MM   = 1.0;          % smallest axial gap, glass or air (mm)
% ---------------------------------------------------------------------
o.NGRID  = 45;
o.MODEL  = 128;
o.OUTDIR = fullfile(exdir, 'out');
% =====================================================================
for i = 1:2:numel(varargin)
    assert(isfield(o, varargin{i}), 'triplet: unknown choice %s', varargin{i});
    o.(varargin{i}) = varargin{i+1};
end
EFL_MM = o.EFL_MM;  FNUM = o.FNUM;  HALF_FIELD = o.HALF_FIELD;  CROWN = o.CROWN;  FLINT = o.FLINT;
PETZVAL = o.PETZVAL;  T_MM = o.T_MM;  LAMBDA_NM = o.LAMBDA_NM;  CAL_FIELDS = o.CAL_FIELDS;
NGRID = o.NGRID;  MODEL = o.MODEL;  OUTDIR = o.OUTDIR;  N_CYCLES = o.N_CYCLES;  FLOOR_MM = o.FLOOR_MM;

if ~exist(OUTDIR, 'dir'), mkdir(OUTDIR); end
macos.init(MODEL);
lF = 486.1327;  lC = 656.2725;  ld = LAMBDA_NM;
nn = @(g, l) macos.design.glass(g, l);
na = nn(CROWN, ld);  nb = nn(FLINT, ld);
Va = (na - 1) / (nn(CROWN, lF) - nn(CROWN, lC));  Vb = (nb - 1) / (nn(FLINT, lF) - nn(FLINT, lC));
EPD = EFL_MM / FNUM;
fprintf('====================================================================\n');
fprintf(' Cooke triplet | f = %g mm f/%g | +/-%g deg | %s (V %.1f) / %s (V %.1f) / %s\n', ...
    EFL_MM, FNUM, HALF_FIELD, CROWN, Va, FLINT, Vb, CROWN);
fprintf('====================================================================\n');

%% -- Stage 1 -- the thin-lens power split --------------------------------
x = cooke_thin(na, nb, Va, Vb, EFL_MM, PETZVAL);
fprintf('\n[1] thin-lens seed: f_a = f_c = %.3f mm, f_b = %.3f mm, air spaces %.3f mm\n', 1/x(1), 1/x(2), x(3));
fprintf('      (power 1/%g, axial color 0, Petzval sum %.2f/f)\n', EFL_MM, PETZVAL);

%% -- Stage 2 -- the thick seed, scaled to the EFL --------------------------
L0 = thick_seed(x, T_MM, CROWN, FLINT, na, nb, EFL_MM, ld, EPD);
z_s = L0.z(3) + T_MM(2) / 2;                        % the stop mid-flint
z_img = macos.design.lens_pupils(L0, ld, 'stop', [z_s NaN], 'epd', EPD).z_img;   % the image plane, held by every solve
S0 = scan(L0, z_s, z_img, EPD, ld, HALF_FIELD, NGRID, fullfile(OUTDIR, 'seed.in'));
fprintf('\n[2] thick seed: R = [%s] mm, gaps [%s] mm\n', sprintf('%.3f ', L0.R), sprintf('%.3f ', L0.t));
print_scan(S0);

%% -- Stage 3 -- CALIB on the radii and the spacings, in alternating rounds --
% The engine cannot take a radius and a position DOF on the SAME element in
% one solve (calib_run fails at its first step; MACOS_sandbox/
% calib_pist_roc_2026-11-01), and its WFE merit does not charge lost rays, so
% a free spacing can collapse an air gap onto its neighbour and "win" on the
% few rays left.  So the solve alternates, as challenges/iodc2010's solve does:
% a round on the six radii, then a round on the axial positions of surfaces
% 3-6 (surface 1 and the stop stay put), each round ACCEPTED only when every
% gap clears its floor, every edge thickness at the clear radius is positive,
% the full field loses no ray, and the worst field's RMS (lens_eval, the
% independent score) went down.
cal_f = CAL_FIELDS * HALF_FIELD;
L1 = L0;  z_s1 = z_s;  z_img1 = z_img;
best = merit(L1, z_s1, z_img1, EPD, ld, cal_f, HALF_FIELD, NGRID, fullfile(OUTDIR, 'round.in'));
merit0 = best;
rounds = struct('kind', {}, 'n_iter', {}, 'stop', {}, 'accepted', {}, 'merit', {}, 'why', {}, 'clamped', {});
for cyc = 1:N_CYCLES
    for kind = {'radii', 'spacings'}
        [Lt, res] = calib_round(L1, z_s1, z_img1, EPD, ld, cal_f, kind{1}, NGRID, fullfile(OUTDIR, ['calib_' kind{1} '.in']));
        % back to the asked EFL at once (the image plane and the stop scale with it)
        sc = EFL_MM / Lt.paraxial(ld).efl;
        Lt = Lt.scale(sc);  zst = z_s1 * sc;  zit = z_img1 * sc;
        nclamp = 0;
        if strcmp(kind{1}, 'spacings')             % a gap under its floor is CLAMPED there (the IODC rule)
            nclamp = nnz(Lt.t < FLOOR_MM);  Lt.t = max(Lt.t, FLOOR_MM);
        end
        why = check_lens(Lt, FLOOR_MM, 1.3 * EPD / 2);
        if isempty(why)
            [mt, lost] = merit(Lt, zst, zit, EPD, ld, cal_f, HALF_FIELD, NGRID, fullfile(OUTDIR, 'round.in'));
            if lost, why = sprintf('%d rays lost', lost);
            elseif mt >= best, why = sprintf('merit %.4f not below %.4f', mt, best); end
        end
        ok = isempty(why);
        if ok, L1 = Lt;  z_s1 = zst;  z_img1 = zit;  best = mt; end
        rounds(end+1) = struct('kind', kind{1}, 'n_iter', res.n_iter, 'stop', res.stop_reason, ...
            'accepted', ok, 'merit', best, 'why', why, 'clamped', nclamp); %#ok<AGROW>
        fprintf('      cycle %d %-8s: %4d it (%s) -> %s\n', cyc, kind{1}, res.n_iter, res.stop_reason, ...
            [tern(ok, sprintf('accepted, merit %.4f waves', best), ['rejected: ' why]), ...
             tern(nclamp > 0, sprintf(' (%d gap(s) clamped to the floor)', nclamp), '')]);
    end
end
S1 = scan(L1, z_s1, z_img1, EPD, ld, HALF_FIELD, NGRID, fullfile(OUTDIR, 'triplet.in'));
res = rounds;
fprintf('\n[3] CALIB rounds at %s deg, EFL held at %g: %d of %d accepted; merit (RMS over the CALIB fields) %.3f -> %.4f waves\n', ...
    mat2str(cal_f), EFL_MM, nnz([rounds.accepted]), numel(rounds), merit0, best);
fprintf('      R = [%s] mm, gaps [%s] mm\n', sprintf('%.3f ', L1.R), sprintf('%.3f ', L1.t));
print_scan(S1);
fprintf('      edge of the field: RMS %.3f waves, distortion %+.3f %%, reported not fitted\n', S1.rms(end), S1.dist(end));

%% -- Figure + report -------------------------------------------------------
fig = figure('Visible', 'off', 'Position', [100 100 720 360]);
subplot(1, 2, 1);  semilogy(S0.theta, S0.rms, 'b--o', S1.theta, S1.rms, 'r-s', 'LineWidth', 1.3);  grid on;
xlabel('field (deg)');  ylabel(sprintf('RMS WFE (waves at %g nm)', ld));  legend('thick seed', 'CALIB', 'Location', 'northwest');
subplot(1, 2, 2);  plot(S0.theta, S0.dist, 'b--o', S1.theta, S1.dist, 'r-s', 'LineWidth', 1.3);  grid on;
xlabel('field (deg)');  ylabel('distortion (%)');
sgtitle(sprintf('Cooke triplet %s/%s/%s, f = %g mm, f/%g', CROWN, FLINT, CROWN, EFL_MM, FNUM));
exportgraphics(fig, fullfile(OUTDIR, 'triplet_field.png'), 'Resolution', 110);  close(fig);
out = struct('thin', x, 'lens_seed', L0, 'lens', L1, 'stop_z', z_s1, 'image_z', z_img1, 'scan_seed', S0, 'scan', S1, 'rounds', res, ...
    'merit_seed', merit0, 'merit', best, 'efl', L1.paraxial(ld).efl);
save(fullfile(OUTDIR, 'triplet.mat'), 'out');
fid = fopen(fullfile(OUTDIR, 'triplet_report.txt'), 'w');
fprintf(fid, 'Cooke triplet %s/%s/%s, f = %g mm, f/%g, semi-field %g deg, %g nm\n\n', CROWN, FLINT, CROWN, EFL_MM, FNUM, HALF_FIELD, ld);
fprintf(fid, 'METRIC: per field, RMS WFE (waves) on the chief-ray-centred exit-pupil sphere, piston removed,\n');
fprintf(fid, '        tilt and focus kept, image plane fixed; distortion 100 (yR - yP)/yP.  CALIB has no\n');
fprintf(fid, '        distortion row: distortion is reported, not fitted.\n\n');
fprintf(fid, 'seed   R [%s] t [%s]\n', sprintf('%.4f ', L0.R), sprintf('%.4f ', L0.t));
fprintf(fid, 'CALIB  R [%s] t [%s] stop z %.4f\n\n', sprintf('%.4f ', L1.R), sprintf('%.4f ', L1.t), z_s1);
fprintf(fid, '  field   RMS seed   RMS CALIB   dist seed %%  dist CALIB %%\n');
for i = 1:numel(S1.theta)
    fprintf(fid, '  %5.1f   %8.4f   %9.4f   %+10.4f  %+11.4f\n', S1.theta(i), S0.rms(i), S1.rms(i), S0.dist(i), S1.dist(i));
end
fclose(fid);
fprintf('\nDONE.  Report %s\n', fullfile(OUTDIR, 'triplet_report.txt'));
end

% =====================================================================
function x = cooke_thin(na, nb, Va, Vb, f, ptz)
% [phi_a phi_b d]: symmetric thin triplet (phi_c = phi_a, equal spaces), Newton
F = @(x) eqs_(x, na, nb, Va, Vb, f, ptz);
x = [1.1 / f, -2 / f, 0.11 * f].';
for it = 1:60
    r = F(x);  if norm(r) < 1e-13, break; end
    J = zeros(3);
    for j = 1:3
        h = 1e-7 * max(abs(x(j)), 1e-6);  e = zeros(3, 1);  e(j) = h;
        J(:, j) = (F(x + e) - F(x - e)) / (2 * h);
    end
    x = x - J \ r;
end
assert(norm(F(x)) < 1e-10 && x(3) > 0, 'triplet: the thin-lens solve failed');
x = x.';
end

function r = eqs_(x, na, nb, Va, Vb, f, ptz)
p = [x(1) x(2) x(1)];  y = zeros(1, 3);  u = 0;  yy = 1;
for k = 1:3, y(k) = yy;  u = u - yy * p(k);  if k < 3, yy = yy + x(3) * u; end, end
r = [-u * f - 1; f * sum(y.^2 .* p ./ [Va Vb Va]); f * sum(p ./ [na nb na]) - ptz];
end

function L = thick_seed(x, t, crown, flint, na, nb, f, lam, epd)
% crowns equi-convex, flint equi-concave at the thin powers; vertex gaps
% = the thin spaces less half of each neighbour's thickness; scaled to f
ca = x(1) / (na - 1) / 2;  cb = x(2) / (nb - 1) / 2;
R = 1 ./ [ca -ca cb -cb ca -ca];
g = x(3) - [t(1) + t(2), t(2) + t(3)] / 2;
assert(all(g > 0.1 * x(3)), 'triplet: the thicknesses eat the air spaces');
mk = @(s) macos.design.Lens('R', R * s, 't', [t(1) g(1) * s t(2) g(2) * s t(3)], ...
    'glass', {crown, 'air', flint, 'air', crown}, 'D', 1.3 * epd, 'name', 'triplet');
s = fzero(@(s) mk(s).paraxial(lam).efl - f, 1);
L = mk(s);
end

function S = scan(L, z_s, z_img, epd, lam, th_max, ngrid, deck)
th = linspace(0, th_max, 6);
P = macos.design.lens_pupils(L, lam, 'stop', [z_s NaN], 'epd', epd, 'image', z_img);
E = macos.design.lens_emit(L, P, deck, 'nGridpts', ngrid);
macos.load_rx(E.path);
S = struct('theta', th, 'rms', zeros(size(th)), 'dist', zeros(size(th)), 'lost', zeros(size(th)), 'P', P);
for i = 1:numel(th)
    F = macos.design.lens_eval(E, P, th(i));
    S.rms(i) = F.rms_waves;  S.dist(i) = F.dist_pct;  S.lost(i) = F.nLost;
end
S.dist(1) = 0;
end

function print_scan(S)
fprintf('      field (deg):  %s\n', sprintf('%7.1f', S.theta));
fprintf('      RMS (waves):  %s\n', sprintf('%7.3f', S.rms));
fprintf('      dist (%%):     %s\n', sprintf('%+7.3f', S.dist));
if any(S.lost), fprintf('      rays lost:    %s\n', sprintf('%7d', S.lost)); end
end

function [Lt, res] = calib_round(L, z_s, z_img, epd, lam, fields, kind, ngrid, deck)
% one CALIB solve on the radii OR the positions of surfaces 3-6 (image plane
% held at z_img); the lens read back
P = macos.design.lens_pupils(L, lam, 'stop', [z_s NaN], 'epd', epd, 'image', z_img);
c = struct('fields_deg', fields, 'max_iters', 3000, 'ray_grid', 21);
if strcmp(kind, 'radii'), c.var_R = 1:L.N; else, c.var_pos = 3:L.N; end
E = macos.design.lens_emit(L, P, deck, 'nGridpts', ngrid, 'calib', c);
macos.load_rx(E.path);
try
    res = macos.calib();
catch err
    res = struct('n_iter', -1, 'stop_reason', ['failed: ' err.message]);
    macos.modify();  Lt = L;  return
end
R = arrayfun(@(k) -macos.get_elt_kr(E.iSurf(k)), 1:L.N);
z = arrayfun(@(k) macos.get_elt_vpt(E.iSurf(k)), 1:L.N, 'UniformOutput', false);
z = cellfun(@(v) v(3), z);
macos.modify();
Lt = L;  Lt.R = R;  Lt.t = diff(z);
end

function why = check_lens(L, floor_mm, r_clear)
% '' when every axial gap clears the floor and every edge thickness is positive
why = '';
if any(L.t < floor_mm), why = sprintf('gap %.3f mm below the %.1f mm floor', min(L.t), floor_mm); return; end
e = L.edge_thickness(r_clear);
if any(~(e > 0)), why = sprintf('edge thickness %.3f mm at r = %.1f mm', min(e), r_clear); end
end

function s = tern(c, a, b)
if c, s = a; else, s = b; end
end

function [m, lost] = merit(L, z_s, z_img, epd, lam, fields, th_max, ngrid, deck)
% CALIB's objective scored independently: the RMS over FIELDS of each field's
% RMS WFE (lens_eval).  LOST = the most rays lost at any of FIELDS or th_max.
P = macos.design.lens_pupils(L, lam, 'stop', [z_s NaN], 'epd', epd, 'image', z_img);
E = macos.design.lens_emit(L, P, deck, 'nGridpts', ngrid);
macos.load_rx(E.path);
ths = unique([fields th_max]);  r = zeros(size(ths));  lost = 0;
for i = 1:numel(ths)
    F = macos.design.lens_eval(E, P, ths(i));
    r(i) = F.rms_waves;  lost = max(lost, F.nLost);
end
m = sqrt(mean(r(ismember(ths, fields)).^2));
end
