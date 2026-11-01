% doublet.m  (mmacos/templates/35_lenses/doublet/ -- a parameterized design runner)
% =====================================================================
%  CEMENTED ACHROMAT: the Fraunhofer doublet from its glass pair
% =====================================================================
%  Two glasses of different dispersion bring two wavelengths to one focus.
%  With Abbe numbers V = (n_d - 1)/(n_F - n_C) the thin-lens element powers
%  that make the F and C foci coincide at total power phi are
%
%      phi_1 = phi V1/(V1 - V2),     phi_2 = -phi V2/(V1 - V2)
%
%  and what is left, the d focus off the common F/C focus, is the SECONDARY
%  SPECTRUM, f (P1 - P2)/(V1 - V2) with the partial dispersion
%  P = (n_d - n_C)/(n_F - n_C) (Kingslake & Johnson, Lens Design
%  Fundamentals, 2nd ed., Sec. 5.3; Smith, Modern Optical Engineering,
%  4th ed., Sec. 6.5).  A cemented pair has three radii: power and color
%  fix two, the third -- the BENDING -- is free, and it sets the spherical
%  aberration.  This driver
%    [1] takes the powers from the Abbe numbers (catalog glasses, the
%        engine's own table),
%    [2] solves the THICK lens paraxially (y-nu) so that the EFL is EFL_MM
%        at d and the F and C foci coincide exactly, for an equi-convex
%        crown -- the textbook achromat,
%    [3] lets the ENGINE find the bending: MACOS CALIB on the three radii,
%        with the three wavelengths F, d, C in the solve so the color the
%        paraxial step set is held (a one-wavelength solve frees the color:
%        see the README),
%    [4] measures the focal shift across F, d, C from THREE engine traces
%        (one wavelength per trace) and the secondary spectrum against the
%        textbook value,
%    [5] places a VENDOR achromat from the lens catalog
%        (macos.design.lens_catalog) on the same bench and reports its
%        EFL / BFL against the vendor's numbers.
%
%  METRIC (stated before the numbers): RMS wavefront error in waves at the
%  wavelength named, over the full aperture, on the exit-pupil sphere
%  centred on the chief ray's image point (macos.opd_ref('chief')), piston
%  removed, tilt and focus kept, image plane at the d-line paraxial focus.
%  Focal positions are the engine's near-axis back focal distances.
%
%  Run:  >> doublet                      the defaults below
%        >> out = doublet('FLINT', 'SF2', 'FNUM', 4)
%  Returns OUT (also OUTDIR/doublet.mat) and writes OUTDIR/doublet_report.txt and
%  OUTDIR/doublet_focus.png (the chromatic focal-shift curve).
% =====================================================================
function out = doublet(varargin)
exdir = fileparts(mfilename('fullpath'));

% ====================  USER DESIGN CHOICES  ==========================
o.EFL_MM   = 100;          % focal length at d (mm)
o.FNUM     = 5;            % f-number (stop at the first surface)
o.CROWN    = 'N-BK7';      % front element (positive)
o.FLINT    = 'F2';         % rear element (negative); any catalog name
o.T_MM     = [6 2.5];      % centre thicknesses crown, flint (mm)
o.LAM_NM   = [486.1327 587.5618 656.2725];   % F, d, C (the achromat's pair is F and C)
o.VENDOR_PART = 'PAC095AR.14';               % a lens_catalog part ('' = skip stage 5)
% ---------------------------------------------------------------------
o.NGRID    = 63;
o.MODEL    = 128;
o.OUTDIR   = fullfile(exdir, 'out');
% =====================================================================
for i = 1:2:numel(varargin)
    assert(isfield(o, varargin{i}), 'doublet: unknown choice %s', varargin{i});
    o.(varargin{i}) = varargin{i+1};
end
EFL_MM = o.EFL_MM;  FNUM = o.FNUM;  CROWN = o.CROWN;  FLINT = o.FLINT;  T_MM = o.T_MM;
LAM_NM = o.LAM_NM;  VENDOR_PART = o.VENDOR_PART;  NGRID = o.NGRID;  MODEL = o.MODEL;  OUTDIR = o.OUTDIR;

if ~exist(OUTDIR, 'dir'), mkdir(OUTDIR); end
macos.init(MODEL);
lF = LAM_NM(1);  ld = LAM_NM(2);  lC = LAM_NM(3);
EPD = EFL_MM / FNUM;
nn = @(g, l) macos.design.glass(g, l);
V  = @(g) (nn(g, ld) - 1) / (nn(g, lF) - nn(g, lC));
Pd = @(g) (nn(g, ld) - nn(g, lC)) / (nn(g, lF) - nn(g, lC));
V1 = V(CROWN);  V2 = V(FLINT);  P1 = Pd(CROWN);  P2 = Pd(FLINT);
fprintf('====================================================================\n');
fprintf(' Cemented achromat | f = %g mm f/%g | %s (V %.2f) + %s (V %.2f) | t = [%g %g] mm\n', ...
    EFL_MM, FNUM, CROWN, V1, FLINT, V2, T_MM);
fprintf('====================================================================\n');

%% -- Stage 1 -- thin-lens powers from the Abbe numbers -----------------
ph1 = V1 / (V1 - V2) / EFL_MM;  ph2 = -V2 / (V1 - V2) / EFL_MM;
ss_tb = EFL_MM * (P1 - P2) / (V1 - V2);
fprintf('\n[1] thin lens: phi1 = %.6f, phi2 = %.6f /mm (f1 = %.3f, f2 = %.3f mm)\n', ph1, ph2, 1/ph1, 1/ph2);
fprintf('      partial dispersions P = (n_d - n_C)/(n_F - n_C): %.4f, %.4f\n', P1, P2);
fprintf('      textbook secondary spectrum f (P1 - P2)/(V1 - V2) = %.4f mm\n', ss_tb);

%% -- Stage 2 -- the thick lens, solved paraxially at F and C -----------
k1 = ph1 / (nn(CROWN, ld) - 1);  k2 = ph2 / (nn(FLINT, ld) - 1);
mk = @(c) macos.design.Lens('R', 1 ./ c, 't', T_MM, 'glass', {CROWN, FLINT}, 'D', 1.1 * EPD, 'name', 'ach');
c0 = achromat(k1 / 2, [k1 / 2 - k1, k1 / 2 - k1 - k2], mk, EFL_MM, lF, ld, lC);
L0 = mk(c0);
z0 = arrayfun(@(l) L0.paraxial(l).bfl, LAM_NM);
z_img = L0.z(end) + z0(2);                       % the d paraxial focus: the image plane every lens is scored at
r0 = eval_rms(L0, EPD, LAM_NM, NGRID, z_img, fullfile(OUTDIR, 'achromat_paraxial.in'));
fprintf('\n[2] thick-lens achromat, equi-convex crown (y-nu, EFL %.6f at d, BFL_F - BFL_C = %.1e mm):\n', ...
    L0.paraxial(ld).efl, z0(1) - z0(3));
fprintf('      R = [%.4f %.4f %.4f] mm\n', L0.R);
fprintf('      RMS at F / d / C: %.4f / %.4f / %.4f waves\n', r0);

%% -- Stage 3 -- the engine finds the bending: CALIB at F, d, C -----------
P0 = macos.design.lens_pupils(L0, ld, 'stop', [0 NaN], 'epd', EPD);
E0 = macos.design.lens_emit(L0, P0, fullfile(OUTDIR, 'calib_bending.in'), 'nGridpts', NGRID, ...
    'calib', struct('var_R', [1 2 3], 'max_iters', 3000, 'ray_grid', 21, 'wavelengths_nm', LAM_NM));
macos.load_rx(E0.path);
res = macos.calib();
Rc = arrayfun(@(k) -macos.get_elt_kr(E0.iSurf(k)), 1:3);
macos.modify();
Lc = L0;  Lc.R = Rc;  Lc.name = 'calib';
rc = eval_rms(Lc, EPD, LAM_NM, NGRID, z_img, fullfile(OUTDIR, 'achromat_calib.in'));
fprintf('\n[3] CALIB on R1..R3 at F, d, C (%d iterations, %s):\n', res.n_iter, res.stop_reason);
fprintf('      R = [%.4f %.4f %.4f] mm, EFL %.4f at d\n', Lc.R, Lc.paraxial(ld).efl);
fprintf('      RMS at F / d / C: %.4f / %.4f / %.4f waves (paraxial achromat: %.4f / %.4f / %.4f)\n', rc, r0);

% for contrast: the same solve at d alone -- it nulls the spherical
% aberration and lets the color go (the three radii have one direction the
% d light cannot see: the color)
E1 = macos.design.lens_emit(L0, P0, fullfile(OUTDIR, 'calib_bending_d.in'), 'nGridpts', NGRID, ...
    'calib', struct('var_R', [1 2 3], 'max_iters', 3000, 'ray_grid', 21));
macos.load_rx(E1.path);
res1 = macos.calib();
R1 = arrayfun(@(k) -macos.get_elt_kr(E1.iSurf(k)), 1:3);
macos.modify();
L1 = L0;  L1.R = R1;
z1 = arrayfun(@(l) L1.paraxial(l).bfl, LAM_NM);
r1 = eval_rms(L1, EPD, LAM_NM, NGRID, z_img, fullfile(OUTDIR, 'achromat_calib_d.in'));
fprintf('      contrast, CALIB at d alone (%d iterations, %s): RMS F/d/C %.4f / %.4f / %.4f, BFL_F - BFL_C %+.4f mm\n', ...
    res1.n_iter, res1.stop_reason, r1, z1(1) - z1(3));

%% -- Stage 4 -- focal shift across the band, one trace per wavelength ---
% the engine's near-axis back focal distance at F, d, C (three decks, three
% traces) for the paraxial achromat -- whose F and C foci coincide, so the
% d focus's offset IS the secondary spectrum the textbook formula predicts --
% and for the CALIB lens, whose solve traded a little of that color for the
% spherochromatism
zE0 = zeros(1, 3);  zEc = zeros(1, 3);  zPc = zeros(1, 3);
for i = 1:3
    zE0(i) = engine_bfl(L0, LAM_NM(i), fullfile(OUTDIR, 'focus.in'));
    zEc(i) = engine_bfl(Lc, LAM_NM(i), fullfile(OUTDIR, 'focus.in'));
    zPc(i) = Lc.paraxial(LAM_NM(i)).bfl;
end
ss_e = zE0(2) - mean(zE0([1 3]));
fprintf('\n[4] focal shift, back focal distance minus its d value (mm), one engine trace per wavelength:\n');
fprintf('      paraxial achromat:  F %+.5f   C %+.5f   (y-nu: F %+.5f  C %+.5f)\n', zE0(1) - zE0(2), zE0(3) - zE0(2), z0(1) - z0(2), z0(3) - z0(2));
fprintf('      CALIB lens:         F %+.5f   C %+.5f   (y-nu: F %+.5f  C %+.5f)\n', zEc(1) - zEc(2), zEc(3) - zEc(2), zPc(1) - zPc(2), zPc(3) - zPc(2));
fprintf('      secondary spectrum (paraxial achromat, d focus - F/C focus): engine %+.5f mm, textbook -%.5f mm (thin lens)\n', ss_e, ss_tb);
zs_e = zEc;  zs_p = zPc;
fprintf('      worst RMS across the band: CALIB %.4f, paraxial achromat %.4f waves\n', max(rc), max(r0));
% the chromatic focal-shift curve, paraxial (y-nu) over the band, the engine's three traces marked
lam_c = linspace(450, 700, 51);
f0 = arrayfun(@(l) L0.paraxial(l).bfl, lam_c) - z0(2);
fc = arrayfun(@(l) Lc.paraxial(l).bfl, lam_c) - zPc(2);
fig = figure('Visible', 'off', 'Position', [100 100 640 400]);
plot(lam_c, 1e3 * f0, 'b-', lam_c, 1e3 * fc, 'r-', 'LineWidth', 1.4);  hold on;
plot(LAM_NM, 1e3 * (zE0 - zE0(2)), 'bo', LAM_NM, 1e3 * (zEc - zEc(2)), 'rs', 'LineWidth', 1.3, 'MarkerSize', 8);
grid on;  xlabel('wavelength (nm)');  ylabel('back focal distance minus its d value (\mum)');
legend({'paraxial achromat (y-nu)', 'CALIB at F, d, C (y-nu)', 'engine, one trace per wavelength', ''}, 'Location', 'north');
title(sprintf('%s + %s achromat, f = %g mm: chromatic focal shift', CROWN, FLINT, EFL_MM));
exportgraphics(fig, fullfile(OUTDIR, 'doublet_focus.png'), 'Resolution', 110);  close(fig);

%% -- Stage 5 -- a vendor achromat from the catalog ------------------------
vend = struct();
if ~isempty(VENDOR_PART)
    [Lv, iv] = macos.design.lens_catalog(VENDOR_PART);
    Pv = Lv.paraxial(iv.lambda_nm);
    efl_e = engine_efl(Lv, iv.lambda_nm, fullfile(OUTDIR, 'vendor.in'));
    fprintf('\n[5] %s (%s): R = [%s] mm, t = [%s] mm, %s\n', iv.part, iv.vendor, ...
        sprintf('%.3f ', Lv.R), sprintf('%.2f ', Lv.t), strjoin(cellfun(@gname, Lv.glass, 'UniformOutput', false), ' / '));
    fprintf('      at %g nm: EFL paraxial %.3f / engine %.3f / vendor %g mm; BFL paraxial %.3f / vendor %g mm\n', ...
        iv.lambda_nm, Pv.efl, efl_e, iv.efl, Pv.bfl, iv.bfl);
    vend = struct('part', iv.part, 'efl', Pv.efl, 'efl_engine', efl_e, 'bfl', Pv.bfl, 'vendor', iv);
end

%% -- Report --------------------------------------------------------------
out = struct('V', [V1 V2], 'P', [P1 P2], 'phi', [ph1 ph2], 'R_paraxial', L0.R, 'rms_paraxial', r0, ...
    'R_calib', Lc.R, 'rms_calib', rc, 'calib', res, 'bfl_engine', zs_e, 'bfl_paraxial', zs_p, ...
    'bfl_engine_achromat', zE0, 'bfl_paraxial_achromat', z0, 'z_img', z_img, ...
    'secondary_engine', ss_e, 'secondary_textbook', ss_tb, 'vendor', vend, 'lens', Lc, ...
    'lens_paraxial', L0, 'R_calib_d', R1, 'rms_calib_d', r1, 'FC_calib_d', z1(1) - z1(3), 'calib_d', res1);
save(fullfile(OUTDIR, 'doublet.mat'), 'out');
fid = fopen(fullfile(OUTDIR, 'doublet_report.txt'), 'w');
fprintf(fid, 'Cemented achromat: %s + %s, f = %g mm, f/%g, t = [%g %g] mm\n\n', CROWN, FLINT, EFL_MM, FNUM, T_MM);
fprintf(fid, 'METRIC: RMS WFE (waves at each wavelength) on the chief-ray-centred exit-pupil sphere, piston\n');
fprintf(fid, '        removed, tilt and focus kept, image at the d paraxial focus; one trace per wavelength.\n\n');
fprintf(fid, '  lens        R1 (mm)    R2 (mm)    R3 (mm)     RMS F     RMS d     RMS C\n');
fprintf(fid, '  paraxial %9.4f  %9.4f  %9.4f   %7.4f   %7.4f   %7.4f\n', L0.R, r0);
fprintf(fid, '  CALIB    %9.4f  %9.4f  %9.4f   %7.4f   %7.4f   %7.4f\n', Lc.R, rc);
fprintf(fid, '  CALIB d  %9.4f  %9.4f  %9.4f   %7.4f   %7.4f   %7.4f   (d alone: BFL_F - BFL_C %+.4f mm)\n', R1, r1, z1(1) - z1(3));
fprintf(fid, '\nfocal shift (engine, one trace per wavelength, mm, re d):\n');
fprintf(fid, '  paraxial achromat F %+.5f  C %+.5f;  CALIB lens F %+.5f  C %+.5f\n', zE0(1) - zE0(2), zE0(3) - zE0(2), zEc(1) - zEc(2), zEc(3) - zEc(2));
fprintf(fid, 'secondary spectrum (paraxial achromat): engine %+.5f mm, textbook (thin lens) -%.5f mm\n', ss_e, ss_tb);
if ~isempty(fieldnames(vend))
    fprintf(fid, '%s: EFL paraxial %.3f, engine %.3f, vendor %g; BFL paraxial %.3f, vendor %g (at %g nm)\n', ...
        vend.part, vend.efl, vend.efl_engine, vend.vendor.efl, vend.bfl, vend.vendor.bfl, vend.vendor.lambda_nm);
end
fclose(fid);
fprintf('\nDONE.  Report %s\n', fullfile(OUTDIR, 'doublet_report.txt'));
end

% =====================================================================
function c = achromat(c1, x0, mk, f, lF, ld, lC)
% the thick-lens achromat with front curvature c1: (c2, c3) so that EFL(d) = f and BFL(F) = BFL(C)
res = @(x) [mk([c1 x]).paraxial(ld).efl - f; ...
            mk([c1 x]).paraxial(lF).bfl - mk([c1 x]).paraxial(lC).bfl];
x = x0(:);
for it = 1:50                                   % Newton on two equations, central differences
    r = res(x.');
    if max(abs(r)) < 1e-12, break; end
    J = zeros(2);  h = 1e-7 * max(abs(x));
    for j = 1:2
        e = zeros(2, 1);  e(j) = h;
        J(:, j) = (res((x + e).') - res((x - e).')) / (2 * h);
    end
    x = x - J \ r;
end
assert(max(abs(res(x.'))) < 1e-9, 'doublet: the achromat solve did not converge');
c = [c1 x.'];
end

function r = eval_rms(L, epd, lams, ngrid, z_img, deck)
% RMS at each wavelength on the image plane z_img (the paraxial achromat's d
% focus, the plane CALIB held), one trace per wavelength
P = macos.design.lens_pupils(L, lams(2), 'stop', [0 NaN], 'epd', epd, 'image', z_img);
E = macos.design.lens_emit(L, P, deck, 'nGridpts', ngrid);
macos.load_rx(E.path);
r = zeros(1, numel(lams));
for i = 1:numel(lams)
    macos.set_src_wvl(lams(i));
    F = macos.design.lens_eval(E, P, 0, 'lambda_nm', lams(i));
    assert(F.nLost == 0, 'doublet: rays lost at %g nm', lams(i));
    r(i) = F.rms_waves;
end
end

function z = engine_bfl(L, lam, deck)
% the engine's near-axis back focal distance at ONE wavelength: the lens on a
% collimated bench (surface 1 at z = 50), two srt heights, the axis crossing
% behind the last vertex extrapolated to zero height (the h^2 term out)
O = lens_bench_(L, lam, deck);
[~, zc] = near_axis_(O);
z = zc - (50 + O.thickness);
end

function efl = engine_efl(L, lam, deck)
O = lens_bench_(L, lam, deck);
efl = near_axis_(O);
end

function O = lens_bench_(L, lam, deck)
b = macos.design.Bench('lens', 'dir', [0; 0; 1], 'zsource', 1e22, 'aperture', 0.5 * L.D, ...
    'wavelen', lam * 1e-6, 'ngridpts', 21);
O = b.add_optic(L, 50);  b.add_detector(L.paraxial(lam).bfl, 'FP');  b.emit(deck);
macos.load_rx(deck);
end

function [efl, zc] = near_axis_(O)
% EFL and axis crossing from two srt heights, Richardson on h^2 (tLens's gate)
hs = [1e-4 2e-4];  E = zeros(1, 2);  Z = zeros(1, 2);  ax = [0; 0; 1];
for i = 1:2
    r  = macos.srt(hs(i), 0, O.i_last);
    p1 = r.pos(:, O.i_first);  hv = p1 - ax * (ax.' * p1);  e1 = hv / norm(hv);
    d  = r.dir(:, O.i_last);  u = (e1.' * d) / (ax.' * d);
    pl = r.pos(:, O.i_last);
    E(i) = -norm(hv) / u;
    Z(i) = ax.' * pl - (e1.' * pl) / u;
end
efl = (4 * E(1) - E(2)) / 3;
zc  = (4 * Z(1) - Z(2)) / 3;
end

function s = gname(g)
if isstruct(g), s = g.name; elseif isnumeric(g), s = sprintf('n=%g', g); else, s = g; end
end
