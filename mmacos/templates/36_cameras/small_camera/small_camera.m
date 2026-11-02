% small_camera.m  (mmacos/templates/36_cameras/small_camera/ -- a parameterized design runner)
% =====================================================================
%  THE SMALL DEDICATED IMAGER: a Cassegrain with a refractive field corrector
% =====================================================================
%  A single-purpose camera whose telescope and instrument are ONE design: two
%  mirrors, a corrector behind the primary's hole, a filter, one detector.  The
%  worked case is the Mars Observer Camera's narrow-angle channel, whose deck
%  the MACOS manual carries as ImageDemo.in (Appendix A): published as "a 35-cm
%  aperture, 3.5-m focal length (f/10) Ritchey-Chretien telescope", a 2048 x 1
%  line array of 13 um pixels, 3.7 urad per pixel, a 0.44 deg field and a
%  500-900 nm band (Malin et al. 1992, J. Geophys. Res. 97(E5) 7699, Table 1).
%
%  The stages:
%    1  the deck as found, STRIPPED: the primary's Zernike figure removed, the
%       secondary, exit pupil and image plane put back on the axis; the builder
%       (macos.design.cass_camera) rebuilds it from the deck's own numbers and
%       the two trace alike (the A/B)
%    2  refocus: the figure carried the focus, so without it the secondary's
%       spacing brings the image back to the deck's detector plane (CALIB on
%       that one DOF); the field at the design wavelength, the clear apertures
%       the beam needs, the primary's hole
%    3  the glasses: the deck's fixed indices matched to the engine's catalog,
%       and the band traced with the matched glasses' dispersion
%    4  the bare telescope: the closed-form Ritchey-Chretien of the same m and
%       beta without the relay -- what the corrector is for
%    5  the camera from its SPEC (aperture, focal length, pixel, band): the RC
%       closed form, the found corrector scaled and in catalog glass, the
%       first-order solve that sets the focal length and puts the focus behind
%       the filter, then CALIB rounds over the band and the field
%    6  the sampling: lambda F / pixel at each wavelength -- the camera is
%       undersampled by design, and by how much
%
%  METRIC (stated before the numbers): per field and wavelength, RMS wavefront
%  error in waves over the annular pupil, on the exit-pupil sphere centred on
%  the chief ray's image point (macos.opd_ref('chief')), piston removed, tilt
%  and focus kept, image plane fixed (macos.design.lens_eval); distortion
%  100 (yR - yP)/yP with yP the paraxial chief height on that plane.  The
%  diffraction limit quoted is Marechal's 0.0714 wave.
%
%  Run:  >> small_camera                          the defaults (the MOC narrow angle)
%        >> out = small_camera('D_MM', 200, 'EFL_MM', 2400, 'PIXEL_UM', 9)
%  Returns OUT (also OUTDIR/small_camera.mat); writes OUTDIR/small_camera_report.txt
%  and OUTDIR/small_camera_field.png.
% =====================================================================
function out = small_camera(varargin)
exdir = fileparts(mfilename('fullpath'));

% ====================  USER DESIGN CHOICES  ==========================
o.DECK         = '';           % the found deck ('' = the macos manual's ImageDemo.in beside this checkout)
o.D_MM         = 350;          % aperture (mm)
o.EFL_MM       = 3500;         % camera focal length (mm)
o.PRIMARY_FNUM = NaN;          % primary f-number (NaN = the found deck's)
o.PIXEL_UM     = 13;           % detector pitch (um)
o.NPIX         = 2048;         % pixels across the field (a line array)
o.BAND_NM      = [500 700 900];% band; the middle entry is the design wavelength
o.GLASSES      = {};           % corrector + filter glasses ({} = the catalog matches of the found indices)
o.CAL_FIELDS   = [0 0.5 1];    % CALIB fields as fractions of the half field
o.N_ROUNDS     = 4;            % stage-5 rounds (CALIB, then the first-order re-solve; the best is kept)
% ---------------------------------------------------------------------
o.NGRID  = 33;
o.MODEL  = 128;
o.OUTDIR = fullfile(exdir, 'out');
% =====================================================================
for i = 1:2:numel(varargin)
    assert(isfield(o, varargin{i}), 'small_camera: unknown choice %s', varargin{i});
    o.(varargin{i}) = varargin{i+1};
end
OUTDIR = o.OUTDIR;  NGRID = o.NGRID;  lams = o.BAND_NM(:).';  l0 = lams(2);
if isempty(o.DECK)
    res = fileparts(fileparts(fileparts(fileparts(exdir))));          % .../MACOS_resources
    o.DECK = fullfile(fileparts(res), 'macos', 'docs', 'macos-manual', 'examples', 'ImageDemo.in');
end
assert(isfile(o.DECK), 'small_camera: the found deck %s is not there (set DECK)', o.DECK);
if ~exist(OUTDIR, 'dir'), mkdir(OUTDIR); end
macos.init(o.MODEL);
mar = 0.0714;
buf = containers.Map('KeyType', 'double', 'ValueType', 'char');
say = @(varargin) say_(buf, sprintf(varargin{:}));
fprintf('====================================================================\n');
fprintf(' Small dedicated imager | the found deck %s\n', o.DECK);
fprintf(' spec: D %g mm, f %g mm (f/%.4g), %d x %g um pixels, band %s nm\n', o.D_MM, o.EFL_MM, ...
    o.EFL_MM / o.D_MM, o.NPIX, o.PIXEL_UM, mat2str(lams));
fprintf('====================================================================\n');

%% -- Stage 1 -- the deck as found, stripped, and rebuilt -------------------
[found, info] = strip_deck(o.DECK, fullfile(OUTDIR, 'found_stripped.in'), fullfile(OUTDIR, 'found_ab.in'));
F0 = found;                                  % the found numbers (lambda of the deck)
lf = F0.lambda_nm;
C0 = macos.design.cass_camera('mirrors', F0.M, 'D', F0.D, 'relay', relay_(F0, F0.n), ...
    'obscuration', F0.obs, 'image', F0.z_img, 'lambda_nm', lf);
half0 = atand(o.NPIX * o.PIXEL_UM * 1e-3 / 2 / C0.par.yP_unit);
abf = [0 0.5 1] * half0;
Eb = macos.design.camera_emit(C0, fullfile(OUTDIR, 'found_built.in'), 'nGridpts', info.nGridpts);
ab = zeros(2, numel(abf));
macos.load_rx(Eb.path);
for j = 1:numel(abf), ab(1, j) = macos.design.lens_eval(Eb, Eb.P, abf(j)).rms_waves; end
Ef = struct('iStop', info.iM1, 'iImgRet', info.iImgRet, 'iXP', info.iXP, 'iFP', info.iFP, 'finite', false, ...
    'standoff', -info.z_src, 'z_src', info.z_src, 'lambda_mm', lf * 1e-6);
macos.load_rx(fullfile(OUTDIR, 'found_ab.in'));
for j = 1:numel(abf), ab(2, j) = macos.design.lens_eval(Ef, Eb.P, abf(j)).rms_waves; end
say('\n[1] the deck as found, stripped (%d Zernike terms removed from %s; %s put back on the axis):', ...
    info.n_zern, info.zern_elt, strjoin(info.recentred, ', '));
say('      mirrors R1 %.7g K1 %.7g, R2 %.7g K2 %.7g, separation %.7g mm; D %g, obscuration %.4f', ...
    F0.M.R1, F0.M.K1, F0.M.R2, F0.M.K2, F0.M.sep, F0.D, F0.obs);
say('      paraxial at %g nm: f %.2f mm (f/%.3f), chief height %.2f mm per tan(theta) on the deck''s plane; two-mirror f %.2f, relay x%.4f', ...
    lf, C0.par.efl, C0.par.fno, C0.par.yP_unit, C0.par.telescope.efl, C0.par.telescope.relay_mag);
say('      the paraxial focus is %.4f mm %s the deck''s image plane (the figure carried the focus)', ...
    abs(C0.par.z_paraxial - F0.z_img), ternary_(C0.par.z_paraxial < F0.z_img, 'in front of', 'behind'));
say('      builder vs deck (spiders and the overfilled beam dropped from both), RMS waves at %s deg:', mat2str(abf, 3));
say('        deck %s | builder %s | max |diff| %.2g', mat2str(ab(2, :), 6), mat2str(ab(1, :), 6), max(abs(diff(ab))));
[m0, b0] = mbeta_(F0.M);

%% -- Stage 2 -- refocus by the secondary; the field; the apertures ---------
calf = o.CAL_FIELDS * half0;
E2 = macos.design.camera_emit(C0, fullfile(OUTDIR, 'refocus.in'), 'nGridpts', NGRID, ...
    'calib', struct('fields_deg', calf, 'var_sep', true, 'ray_grid', 21));
macos.load_rx(E2.path);
r2 = macos.calib();
v = macos.get_elt_vpt(E2.iM2);
M1 = F0.M;  M1.sep = -v(3);
C1 = macos.design.cass_camera('mirrors', M1, 'D', F0.D, 'relay', relay_(F0, F0.n), ...
    'obscuration', F0.obs, 'image', F0.z_img, 'lambda_nm', lf, 'field_deg', half0);
scf = linspace(0, half0, 5);
S1 = scan_(C1, fullfile(OUTDIR, 'refocused.in'), lf, scf, NGRID);
say('\n[2] refocus by the secondary (CALIB, %d iterations, %s): separation %+.2f um', r2.n_iter, r2.stop_reason, 1e3 * (M1.sep - F0.M.sep));
say('      f %.2f mm, chief height %.2f mm per tan(theta); the paraxial focus now %.4f mm from the detector plane', ...
    C1.par.efl, C1.par.yP_unit, C1.par.z_paraxial - F0.z_img);
print_scan_(say, scf, lf, S1);
semi = abs(C1.par.y) + abs(C1.par.yc) * tand(half0);
say('      clear semi-diameters the beam needs at %.3f deg (paraxial): %s', half0, ...
    strjoin(arrayfun(@(k) sprintf('%s %.2f', C1.surf(k).name, semi(k)), 1:numel(semi), 'UniformOutput', false), ', '));
say('      the primary''s hole: %.2f mm radius needed at %.3f deg, inside the %.2f mm shadow (%s)', C1.par.hole.r_need, ...
    half0, C1.par.hole.r_shadow, ternary_(C1.par.hole.clears, 'clears', 'DOES NOT CLEAR'));

%% -- Stage 3 -- the glasses and the band ------------------------------------
if isempty(o.GLASSES)
    [G, gm] = match_glasses_(F0.n, lf);
else
    G = o.GLASSES;  gm = {};
end
C3 = macos.design.cass_camera('mirrors', M1, 'D', F0.D, 'relay', relay_(F0, G), ...
    'obscuration', F0.obs, 'image', F0.z_img, 'lambda_nm', l0);
S3 = scan_(C3, fullfile(OUTDIR, 'found_glass.in'), lams, scf, NGRID);
zc3 = arrayfun(@(l) macos.design.cass_camera('mirrors', M1, 'D', F0.D, 'relay', relay_(F0, G), ...
    'obscuration', F0.obs, 'lambda_nm', l).par.z_paraxial, lams);
say('\n[3] the glasses: the deck''s indices at %g nm against the engine''s catalog:', lf);
for k = 1:numel(gm), say('      %s', gm{k}); end
say('      the refocused design in %s over the band:', strjoin(G, ' / '));
print_scan_(say, scf, lams, S3);
say('      paraxial focus %s mm at %s nm: spread %.4f mm against the +/-2 lambda N^2 = %.4f mm depth of focus at %g nm', ...
    mat2str(zc3, 7), mat2str(lams), max(zc3) - min(zc3), 2 * l0 * 1e-6 * C3.par.fno^2, l0);

%% -- Stage 4 -- the bare telescope ------------------------------------------
Ct = macos.design.cass_camera('family', 'rc', 'D', F0.D, 'primary_fnum', F0.M.R1 / 2 / F0.D, ...
    'telescope_fnum', m0 * F0.M.R1 / 2 / F0.D, 'bfd', b0 * F0.M.R1 / 2, 'obscuration', F0.obs, 'lambda_nm', l0);
St = scan_(Ct, fullfile(OUTDIR, 'bare_rc.in'), l0, scf * C1.par.yP_unit / Ct.par.yP_unit, NGRID);
say('\n[4] the bare Ritchey-Chretien (closed form at the found m = %.4f, beta = %.4f: K1 %.6f, K2 %.6f; the deck''s %.6f, %.6f)', ...
    m0, b0, Ct.mirrors.K1, Ct.mirrors.K2, F0.M.K1, F0.M.K2);
say('      f %.2f mm at its own paraxial focus, the same image heights:', Ct.par.efl);
print_scan_(say, scf * C1.par.yP_unit / Ct.par.yP_unit, l0, St);

%% -- Stage 5 -- the camera from its spec -------------------------------------
D = o.D_MM;  s = D / F0.D;
P1 = o.PRIMARY_FNUM;  if isnan(P1), P1 = F0.M.R1 / 2 / F0.D; end
F5 = F0;  sc = @(c) cellfun(@(x) x * s, c, 'UniformOutput', false);
F5.R = sc(F0.R);  F5.t = sc(F0.t);  F5.z0 = F0.z0 * s;
gap = (F0.z_img - (F0.z0(end) + sum(F0.t{end}))) * s;          % filter to detector, scaled
half = atand(o.NPIX * o.PIXEL_UM * 1e-3 / 2 / o.EFL_MM);
calf5 = o.CAL_FIELDS * half;  scf5 = linspace(0, half, 5);
% Each round: CALIB from the previous round's design, then the first-order
% re-solve (focal length and focus back where the spec puts them).  CALIB's
% radii move the relay's magnification, so the first re-solves shift the
% two-mirror m and undo part of the balance; the rounds settle as the re-solve's
% move shrinks.  The design KEPT is the best one seen -- the independent merit,
% the worst lens_eval RMS over the band at the CALIB fields and the full field,
% lowest -- among those that lose no ray beyond the shadow (the LM merit does
% not charge lost rays: the lens templates' lesson).  A failed CALIB, a failed
% re-solve or a round that loses rays ends the rounds.
K = [NaN NaN];                                                  % the seed: the RC's own conics
[C5, Nt, bfd] = first_order_(D, P1, K, F5, G, o.EFL_MM, gap, l0, F0.obs, half);
C5seed = C5;
mfld = unique([calf5 half]);
[best, okb] = merit_(C5, fullfile(OUTDIR, 'design_merit.in'), lams, mfld, NGRID);
merit_seed = best;
rounds = struct('n_iter', {}, 'stop', {}, 'calib_worst', {}, 'merit', {}, 'accepted', {}, 'why', {});
Cw = C5;  Fw = F5;                                              % the working design
for rr = 1:o.N_ROUNDS
    E5 = macos.design.camera_emit(Cw, fullfile(OUTDIR, 'design_calib.in'), 'nGridpts', NGRID, ...
        'calib', struct('fields_deg', calf5, 'var_R', 1:3, 'var_K', [1 2], 'ray_grid', 21, 'wavelengths_nm', lams));
    macos.load_rx(E5.path);
    try
        r5 = macos.calib();
    catch err
        rounds(rr) = struct('n_iter', NaN, 'stop', 'failed', 'calib_worst', NaN, 'merit', NaN, 'accepted', false, ...
            'why', ['CALIB failed (' err.message ')']);
        break
    end
    Ft = Fw;  Ft.R{1} = arrayfun(@(k) -macos.get_elt_kr(E5.iSurf(k)), 1:3);
    Kt = [macos.get_elt_kc(E5.iM1) macos.get_elt_kc(E5.iM2)];
    why = '';  mt = NaN;
    try
        [Ct, Ntt, bft] = first_order_(D, P1, Kt, Ft, G, o.EFL_MM, gap, l0, F0.obs, half);
        [mt, okt] = merit_(Ct, fullfile(OUTDIR, 'design_merit.in'), lams, mfld, NGRID);
        if ~okt, why = 'rays lost beyond the shadow';
        elseif ~(mt < best), why = 'not the best yet';
        end
    catch err
        why = ['the first-order re-solve failed: ' err.message];
    end
    acc = isempty(why);
    rounds(rr) = struct('n_iter', r5.n_iter, 'stop', r5.stop_reason, 'calib_worst', max(r5.new_wfe(:)) / (l0 * 1e-6), ...
        'merit', mt, 'accepted', acc, 'why', why);
    if startsWith(why, 'the first-order') || startsWith(why, 'rays lost'), break; end
    Cw = Ct;  Fw = Ft;
    if acc, C5 = Ct;  F5 = Ft;  K = Kt;  Nt = Ntt;  bfd = bft;  best = mt;  end
end
S5s = scan_(C5seed, fullfile(OUTDIR, 'design_seed.in'), lams, scf5, NGRID);
S5 = scan_(C5, fullfile(OUTDIR, 'design.in'), lams, scf5, NGRID);
say('\n[5] the camera from its spec: D %g mm, f %g mm, primary f/%.4g, %s, the found corrector x%.4g', D, o.EFL_MM, P1, strjoin(G, ' / '), s);
say('      merit = the worst RMS over %s nm at %s deg; seed (RC conics, the found corrector) %.4f waves%s', ...
    mat2str(lams), mat2str(mfld, 3), merit_seed, ternary_(okb, '', ', RAYS LOST'));
for rr = 1:numel(rounds)
    q = rounds(rr);
    if q.accepted, v = 'KEPT (best so far)'; else, v = q.why; end
    if isnan(q.n_iter), say('      round %d: %s -- the rounds end', rr, q.why);  continue; end
    say('      round %d: CALIB (corrector radii, both conics) %d iterations, %s; after the re-solve %.4f waves -- %s', ...
        rr, q.n_iter, q.stop, q.merit, v);
end
say('      result: two-mirror f/%.5g, BFD %.4f mm -> separation %.4f, R1 %.4f, R2 %.4f, K1 %.6f, K2 %.6f', Nt, bfd, ...
    C5.mirrors.sep, C5.mirrors.R1, C5.mirrors.R2, C5.mirrors.K1, C5.mirrors.K2);
say('      corrector R %s mm; f %.4f mm (asked %g), f/%.4f, image %.4f mm behind the filter', mat2str(F5.R{1}, 7), ...
    C5.par.efl, o.EFL_MM, C5.par.fno, C5.par.z_img - (F5.z0(end) + sum(F5.t{end})));
print_scan_(say, scf5, lams, S5);

%% -- Stage 6 -- the sampling ------------------------------------------------
p_mm = o.PIXEL_UM * 1e-3;
Q = lams * 1e-6 * C5.par.fno / p_mm;
say('\n[6] sampling: pixel %g um = %.3f urad (f %.1f mm); field %d pixels = %.4f deg full', o.PIXEL_UM, ...
    1e3 * p_mm / C5.par.efl * 1e3, C5.par.efl, o.NPIX, 2 * half);
say('      Q = lambda N / p at %s nm: %s; Nyquist sampling is Q = 2, so the camera is undersampled x%s', ...
    mat2str(lams), mat2str(Q, 3), mat2str(2 ./ Q, 3));
say('      (lambda/D at %g nm is %.3f urad: the pixel is %.2f lambda/D)', l0, l0 * 1e-6 / D * 1e6, p_mm / C5.par.efl / (l0 * 1e-6 / D));

%% -- the figure --------------------------------------------------------------
fig = figure('Visible', 'off', 'Units', 'inches', 'Position', [0 0 10 5.6]);
set(fig, 'PaperPositionMode', 'auto');
subplot(1, 2, 1);  hold on;  cols = lines(numel(lams));
for i = 1:numel(lams)
    plot(scf, S3.rms(i, :), '-o', 'Color', cols(i, :), 'DisplayName', sprintf('found, refocused, in %s, %g nm', strjoin(G, '/'), lams(i)));
    plot(scf5, S5.rms(i, :), '--s', 'Color', cols(i, :), 'DisplayName', sprintf('from the spec, %g nm', lams(i)));
end
plot(scf * C1.par.yP_unit / Ct.par.yP_unit, St.rms(1, :), ':k^', 'DisplayName', sprintf('bare RC (same image heights), %g nm', l0));
yline(mar, 'k-', 'Marechal', 'HandleVisibility', 'off');
set(gca, 'YScale', 'log');  grid on;  box on;
xlabel('field angle (deg)');  ylabel('RMS wavefront error (waves)');
title('Wavefront error across the field');  legend('Location', 'southoutside', 'FontSize', 7);
subplot(1, 2, 2);  hold on;
plot(scf, S1.dist, '-o', 'DisplayName', sprintf('found, refocused (%g nm)', lf));
plot(scf5, S5.dist(2, :), '--s', 'DisplayName', sprintf('from the spec (%g nm)', l0));
grid on;  box on;  xlabel('field angle (deg)');  ylabel('distortion (%)');
title('Distortion');  legend('Location', 'southoutside', 'FontSize', 7);
print(fig, fullfile(OUTDIR, 'small_camera_field.png'), '-dpng', '-r150');
close(fig);

out = struct('found', F0, 'info', info, 'C0', C0, 'ab', ab, 'ab_fields', abf, 'half0', half0, ...
    'refocus_um', 1e3 * (M1.sep - F0.M.sep), 'refocus_calib', rmfield(r2, intersect(fieldnames(r2), {'jacobian'})), ...
    'C1', C1, 'scan_refocused', S1, 'glasses', {G}, 'glass_match', {gm}, 'C3', C3, 'scan_glass', S3, ...
    'zpar_band', zc3, 'm', m0, 'beta', b0, 'Ct', Ct, 'scan_bare', St, 'C5seed', C5seed, 'scan_seed', S5s, ...
    'C5', C5, 'scan_design', S5, 'rounds', rounds, 'merit_seed', merit_seed, 'merit', best, 'merit_fields', mfld, 'corrector_R', F5.R{1}, 'half', half, 'fields', scf, ...
    'fields5', scf5, 'Q', Q, 'lams', lams, 'opts', o);
save(fullfile(OUTDIR, 'small_camera.mat'), 'out');
writelines(string(buf.values(num2cell(1:buf.Count))), fullfile(OUTDIR, 'small_camera_report.txt'));
end

function say_(buf, s)
% print a report line and keep it for small_camera_report.txt
fprintf('%s\n', s);
buf(buf.Count + 1) = s;
end

% =========================================================================
function [F, info] = strip_deck(src, out_stripped, out_ab)
% Strip the found deck: drop the Zernike figure, put the secondary, the exit
% pupil and the image plane back on the axis, and mirror z so the light travels
% +z (cass_camera's frame; a reflection, so a symmetric trace is unchanged).
% Then read the numbers the builder needs from the stripped deck itself.
% OUT_AB is the same deck with the spider vanes and the outer baffle ring
% removed and the beam cut to the aperture -- the deck the builder's is A/B'd
% against (camera_emit writes neither).
L = readlines(src);
num = '[-+]?[0-9.]+(?:[dDeE][-+]?[0-9]+)?';
o = strings(0);  elt = 0;  skip = false;  nz = 0;  zelt = '';  names = {};  rec = {};
for i = 1:numel(L)
    ln = L(i);  s = strtrim(ln);
    t = regexp(s, '^iElt=\s*(\d+)', 'tokens', 'once');
    if ~isempty(t), elt = str2double(t{1}); end
    t = regexp(s, '^EltName=\s*(\S+)', 'tokens', 'once');
    if ~isempty(t), names{elt} = t{1}; end %#ok<AGROW>
    if startsWith(s, 'ZernCoef=')
        skip = true;  zelt = names{elt};
        nz = nz + numel(regexp(extractAfter(s, '='), num, 'match'));
        continue
    end
    if skip
        if s == "" || ~isempty(regexp(s, ['^(' num '\s*)+$'], 'once'))
            nz = nz + numel(regexp(s, num, 'match'));  continue
        end
        skip = false;
    end
    if any(startsWith(s, ["pMon=" "xMon=" "yMon=" "zMon=" "lMon="])), continue; end
    if startsWith(s, 'Surface=') && contains(s, 'Zernike'), ln = " Surface= Conic"; end
    key = extractBefore(s, '=');
    if any(strcmp(key, {'VptElt', 'RptElt', 'psiElt', 'ChfRayPos', 'ChfRayDir'}))
        v = str2double(strrep(strrep(regexp(extractAfter(s, '='), num, 'match'), 'd', 'e'), 'D', 'e'));
        nmk = '';  if elt > 0, nmk = names{elt}; end
        if any(strcmp(key, {'VptElt', 'RptElt'})) && any(strcmp(nmk, {'SEC_MIR', 'EXIT_RET', 'IMG_PL'}))
            if any(v(1:2) ~= 0) && ~any(strcmp(rec, nmk)), rec{end + 1} = nmk; end %#ok<AGROW>
            v(1:2) = 0;
        end
        if strcmp(key, 'psiElt') && strcmp(nmk, 'EXIT_RET'), v = [0 0 -1]; end
        v(3) = -v(3);
        ln = sprintf(' %s= %.15E  %.15E  %.15E', key, v);
    end
    o(end + 1) = ln; %#ok<AGROW>
end
writelines(o, out_stripped);
% the A/B copy: no spiders, no baffle ring, the beam at the aperture
B = parse_(o);
D = 2 * B.SEC_OBS.ApVec(1);
a = strings(0);  elt = 0;  drop = 0;
for i = 1:numel(o)
    s = strtrim(o(i));  ln = o(i);
    t = regexp(s, '^iElt=\s*(\d+)', 'tokens', 'once');
    if ~isempty(t), elt = str2double(t{1}); end
    if drop > 0, drop = drop - 1;  continue; end
    if elt == 0 && startsWith(s, 'Aperture='), ln = sprintf('Aperture= %.15E', D); end
    if elt > 0 && any(strcmp(names{elt}, {'SEC_OBS', 'SPIDER_2', 'SPIDER_3'}))
        if startsWith(s, 'ObsType=') && contains(s, 'Rectangle'), drop = 2;  continue; end   % ObsVec + xObs
        if startsWith(s, 'nObs='), ln = sprintf('    nObs= %d', strcmp(names{elt}, 'SEC_OBS')); end
        if startsWith(s, 'ApType='), ln = "  ApType= None"; end
        if startsWith(s, 'ApVec='), continue; end
    end
    a(end + 1) = ln; %#ok<AGROW>
end
writelines(a, out_ab);
% the numbers, from the stripped deck
M = struct('R1', abs(B.PRI_MIR.KrElt), 'K1', B.PRI_MIR.KcElt, 'R2', abs(B.SEC_MIR.KrElt), ...
    'K2', B.SEC_MIR.KcElt, 'sep', -B.SEC_MIR.VptElt(3));
cr = {'COR_S1', 'COR_S2', 'COR_S3'};  fl = {'FLT_S1', 'FLT_S2'};
Rc = cellfun(@(nm) -B.(nm).KrElt * B.(nm).psiElt(3), cr);      % CODE V radius along +z
zc = cellfun(@(nm) B.(nm).VptElt(3), cr);
zf = cellfun(@(nm) B.(nm).VptElt(3), fl);
F = struct('M', M, 'D', D, 'obs', B.SEC_OBS.ObsVec(1) / (D / 2), ...
    'R', {{Rc, [Inf Inf]}}, 't', {{diff(zc), diff(zf)}}, 'z0', [zc(1) zf(1)], ...
    'n', {{B.COR_S1.IndRef, B.COR_S2.IndRef, B.FLT_S1.IndRef}}, ...
    'z_img', B.IMG_RET.VptElt(3), 'lambda_nm', B.src.Wavelen * 1e6);
info = struct('n_zern', nz, 'zern_elt', zelt, 'recentred', {rec}, 'nGridpts', B.src.nGridpts, ...
    'iM1', B.PRI_MIR.iElt, 'iImgRet', B.IMG_RET.iElt, 'iXP', B.EXIT_RET.iElt, 'iFP', B.IMG_PL.iElt, ...
    'z_src', B.src.ChfRayPos(3), 'stripped', out_stripped, 'ab', out_ab);
end

function B = parse_(lines)
% element name -> numeric values of its keys (the first ObsVec only); 'src' = the header
num = '[-+]?[0-9.]+(?:[dDeE][-+]?[0-9]+)?';
B = struct('src', struct());  cur = 'src';  ie = 0;
for i = 1:numel(lines)
    s = strtrim(lines(i));
    if ~contains(s, '='), continue; end
    key = char(strtrim(extractBefore(s, '=')));  val = strtrim(extractAfter(s, '='));
    if strcmp(key, 'iElt'), ie = str2double(val); continue; end
    if strcmp(key, 'EltName'), cur = char(val);  B.(cur) = struct('iElt', ie);  continue; end
    if ~isvarname(key) || isfield(B.(cur), key), continue; end
    v = str2double(strrep(strrep(regexp(val, num, 'match'), 'd', 'e'), 'D', 'e'));
    if ~isempty(v) && all(isfinite(v)), B.(cur).(key) = v; end
end
end

function rl = relay_(F, g)
% the corrector + filter as Lens objects at their stations; G = media
% (numeric indices or catalog names): corrector 1-2, filter
Lc = macos.design.Lens('R', F.R{1}, 't', F.t{1}, 'glass', g(1:2), 'name', 'COR');
Lf = macos.design.Lens('R', F.R{2}, 't', F.t{2}, 'glass', g(3), 'name', 'FLT');
rl = {Lc, F.z0(1); Lf, F.z0(2)};
end

function [G, txt] = match_glasses_(n, lam)
% the nearest catalog glass (macos_glass_list.txt) to each fixed index at LAM
T = readlines(fileparts_glass_());
names = {};
for i = 1:numel(T)
    tok = strsplit(strtrim(T(i)));
    if numel(tok) >= 7 && all(isfinite(str2double(tok(2:7)))), names{end + 1} = char(tok(1)); end %#ok<AGROW>
end
nv = cellfun(@(g) macos.design.glass(g, lam), names);
G = cell(1, numel(n));  txt = cell(1, numel(n));
for k = 1:numel(n)
    [d, ix] = sort(abs(nv - n{k}));
    G{k} = names{ix(1)};
    txt{k} = sprintf('n = %.6f: %s %.6f (%+.1e), then %s %.6f (%+.1e)', n{k}, names{ix(1)}, nv(ix(1)), ...
        nv(ix(1)) - n{k}, names{ix(2)}, nv(ix(2)), nv(ix(2)) - n{k});
end
end

function f = fileparts_glass_()
[~, S] = macos.design.glass('N-BK7', 587.56);
f = S.file;
end

function [m, b] = mbeta_(M)
% the two-mirror (m, beta) of given mirrors: sep = f1 (m - beta)/(m + 1),
% R2 = 2 m f1 (1 + beta)/(m^2 - 1) (Schroeder's Cassegrain relations)
f1 = M.R1 / 2;
g = @(m) 2 * m * f1 * (1 + (m - M.sep * (m + 1) / f1)) / (m^2 - 1) - M.R2;
m = fzero(g, [1.01 100]);
b = m - M.sep * (m + 1) / f1;
end

function [C, Nt, bfd] = first_order_(D, P1, K, F, G, efl, gap, lam, obs, half)
% the RC with primary f/P1 whose camera focal length is EFL with the focus GAP
% behind the filter: the two-mirror f-number Nt and back focal distance bfd,
% one nested 1-D solve each (bfd for the focus inside, Nt for the focal length)
rl = relay_(F, G);
z_t = F.z0(end) + sum(F.t{end}) + gap;
mk = @(Nt, bfd) macos.design.cass_camera('family', 'rc', 'D', D, 'primary_fnum', P1, 'telescope_fnum', Nt, ...
    'bfd', bfd, 'K1', K(1), 'K2', K(2), 'relay', rl, 'obscuration', obs, 'lambda_nm', lam, ...
    'image', NaN, 'field_deg', half);
fb = @(Nt) fzero(@(b) mk(Nt, b).par.z_paraxial - z_t, z_t * [0.5 1.2]);
Nt = fzero(@(Nt) mk(Nt, fb(Nt)).par.efl - efl, efl / D * [0.8 0.98]);
bfd = fb(Nt);
C = mk(Nt, bfd);
C = macos.design.cass_camera('family', 'rc', 'D', D, 'primary_fnum', P1, 'telescope_fnum', Nt, ...
    'bfd', bfd, 'K1', K(1), 'K2', K(2), 'relay', rl, 'obscuration', obs, 'lambda_nm', lam, ...
    'image', z_t, 'field_deg', half);
end

function [m, ok] = merit_(C, path, lams, ths, ngrid)
% the acceptance merit: worst RMS over LAMS x THS, and no ray lost beyond the
% on-axis count (the shadow)
S = scan_(C, path, lams, ths, ngrid);
m = max(S.rms(:));
ok = all(isfinite(S.rms(:))) && all(S.lost(:) == S.lost(1, 1));
end

function S = scan_(C, path, lams, ths, ngrid)
% RMS (waves) per wavelength x field on C's image plane, and distortion
E = macos.design.camera_emit(C, path, 'nGridpts', ngrid);
macos.load_rx(E.path);
S.rms = zeros(numel(lams), numel(ths));  S.dist = S.rms;  S.lost = S.rms;
for i = 1:numel(lams)
    macos.set_src_wvl(lams(i));
    for j = 1:numel(ths)
        F = macos.design.lens_eval(E, E.P, ths(j), 'lambda_nm', lams(i));
        S.rms(i, j) = F.rms_waves;  S.dist(i, j) = F.dist_pct;  S.lost(i, j) = F.nLost;
    end
end
S.dist(:, 1) = 0;  S.launched = F.nLaunched;
end

function print_scan_(say, ths, lams, S)
say('        field (deg)   %s', sprintf('%8.4f', ths));
for i = 1:numel(lams)
    say('        RMS %4g nm   %s', lams(i), sprintf('%8.4f', S.rms(i, :)));
end
say('        distortion %%  %s', sprintf('%8.3f', S.dist(min(2, end), :)));
say('        rays lost     %s of %d (the shadow)', sprintf('%8d', S.lost(1, :)), S.launched);
end

function s = ternary_(c, a, b)
if c, s = a; else, s = b; end
end
