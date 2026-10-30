function T = pupil_blur_lam_m(varargin)
%PUPIL_BLUR_LAM_M  The record estimator's sigma = 0 floor vs lambda_m WITH noise (CC 2026-10-09, an error-budget row).
%   T = PUPIL_BLUR_LAM_M() runs pupil_blur_demo's matrix floor sweep (lambda_m 1e-3 ... 1e-7,
%   the record's 1e-3 first) on the record's surfaces (the random 30 nm working surface and the
%   +/-50 nm checkerboard) at four per-pixel noise levels: none (the bias alone), the demo's 20 pm, and the bench's
%   shot-noise level at 1e13 and 1e14 photons per measurement (the deck's convention).
%   Per-pixel surface noise at N photons per measurement: N_pix = N / (pi/4 * 385^2) lit pixels
%   (the bench's 385-px pupil image; its pixel is 0.25 mm in the DM frame, the demo's dx),
%   sigma_phi = 1/sqrt(N_pix) (the shot-noise phase limit, to a factor sqrt(2) for the four-step's
%   exact visibility/bucket weighting), surface = LAM/(4 pi) sigma_phi (reflection).
%   Writes runs/pupil_blur_demo/pupil_blur_lam_m_report.txt.  Engine-free, ~30 s.
o = struct('nph', [1e13 1e14], 'npix_pupil', 385, 'lam_nm', 632.8, 'lam_m', [1e-3 1e-4 1e-5 1e-6 1e-7], 'outdir', '');
for k = 1:2:numel(varargin), o.(varargin{k}) = varargin{k+1}; end
here = fileparts(mfilename('fullpath'));
if isempty(o.outdir), o.outdir = fullfile(here, 'runs', 'pupil_blur_demo'); end
Npx = pi/4*o.npix_pupil^2;
pm_px = o.lam_nm*1e3/(4*pi) ./ sqrt(o.nph/Npx);                % pm of surface per pixel
lev = [0 20 pm_px];  labs = [{'noise-free', 'demo 20 pm'}, arrayfun(@(n) sprintf('%.0e ph/meas', n), o.nph, 'UniformOutput', false)];
f = fopen(fullfile(o.outdir, 'pupil_blur_lam_m_report.txt'), 'w');  cl = onCleanup(@() fclose(f));
say = @(varargin) [fprintf(varargin{:}), fprintf(f, varargin{:})];
say('=== pupil_blur_lam_m: the matrix estimator''s sigma = 0 floor vs lambda_m, with noise (%s) ===\n', datestr(now, 'yyyy-mm-dd HH:MM'));
say('per-pixel surface noise: %s; bench shot-noise levels from %d-px pupil (%.0f lit px), sigma_phi = 1/sqrt(N_pix) (to a factor sqrt 2), LAM %.1f nm\n', ...
    strjoin(arrayfun(@(k) sprintf('%s = %.2f pm', labs{k}, lev(k)), 1:numel(lev), 'UniformOutput', false), ', '), o.npix_pupil, Npx, o.lam_nm);
say('error = rms(recovered - true) over the 2852 lit actuators, piston removed / rms(true); pm = that x the command rms\n\n');
say('%-16s %8s %8s | %-24s | %-24s\n', 'noise', 'pm/px', 'lambda_m', 'checker (50 nm): % / pm', 'random (30 nm): % / pm');
T = struct('noise', {}, 'pm_px', {}, 'lam_m', {}, 'checker', {}, 'random', {});
od = tempname;  rmo = onCleanup(@() rmdir(od, 's'));
for k = 1:numel(lev)
    for lm = o.lam_m
        [~, out] = evalc('pupil_blur_demo(''kernel'',''gauss'',''sig_pitch'',0,''lam_sweep'',0.05,''lam_m'',lm,''lam_m_sweep'',lm,''noise_pm'',lev(k),''figures'',false,''outdir'',od)');
        e = out.floor.mat;  rc = out.cmd_rms;
        say('%-16s %8.2f %8.0e | %9.4f%% / %8.2f pm | %9.4f%% / %8.2f pm\n', labs{k}, lev(k), lm, 100*e(1), e(1)*rc(1)*1e9, 100*e(2), e(2)*rc(2)*1e9);
        T(end+1) = struct('noise', labs{k}, 'pm_px', lev(k), 'lam_m', lm, 'checker', e(1), 'random', e(2)); %#ok<AGROW>
    end
end
say('\nthe floor per noise level at the smallest lambda_m swept (bias -> 0; the noise part then PLATEAUS, it does not grow):\n');
for k = 1:numel(lev)
    r = T(strcmp({T.noise}, labs{k}));  [~, ic] = min([r.checker]);  [~, ir] = min([r.random]);
    say('  %-16s checker %.4f%% at %.0e | random %.4f%% (%.2f pm) at %.0e%s\n', labs{k}, 100*r(ic).checker, r(ic).lam_m, 100*r(ir).random, ...
        r(ir).random*rc(2)*1e9, r(ir).lam_m, iff_(abs(r(end).random - r(end-1).random) > 0.1*r(end).random, '  (still moving at the sweep edge)', '  (plateau)'));
end
say('read: (1) at the record''s lambda_m 1e-3 the floor is the regularization BIAS, 306 pm (1.03 %%) on the random 30 nm surface whatever the\n');
say('  photons -- the actuator-Nyquist roll-off (bias ~ lambda_m: x10 per decade until the noise part shows);\n');
say('  (2) the noise part is the UNREGULARIZED least-squares floor: this demo''s matrix is exact and well conditioned (no noise gain as\n');
say('  lambda_m -> 0), so the plateau is the photon floor -- random %.2f pm at 1e14 ph/meas, %.2f pm at 1e13 -- the deck''s measured 1.4 / 2.3 pm\n', ...
    T(find(strcmp({T.noise}, labs{end}), 1, 'last')).random*rc(2)*1e9, ...
    T(find(strcmp({T.noise}, labs{end-1}), 1, 'last')).random*rc(2)*1e9);
say('  class on a 10 nm change;\n');
say('  (3) the bench''s MEASURED matrix carries noise and model error in its columns, which is what lambda_m guards there: lowering it on the\n');
say('  bench must be re-gated (its Stage E reg sweep), not read from this ideal-matrix bound.\n');
end

function s = iff_(c, a, b), if c, s = a; else, s = b; end, end
