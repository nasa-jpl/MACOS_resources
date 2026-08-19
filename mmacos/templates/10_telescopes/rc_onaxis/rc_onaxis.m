% rc_onaxis.m  (mmacos/templates/10_telescopes/ -- a parameterized design driver)
% =====================================================================
%  PARAMETERIZED ON-AXIS RITCHEY-CHRETIEN DESIGNER
% =====================================================================
%  The classical Cassegrain-form RC: a two-mirror APLANAT (zero 3rd-order
%  spherical AND coma by construction), so it is diffraction-limited
%  on-axis.  Its field is limited by ASTIGMATISM, which two conics cannot
%  null -- that limit is precisely why wide-field telescopes add a third
%  mirror (a TMA).  This driver builds the RC from your aperture + f-
%  numbers, reports its first-order properties (including the central
%  obscuration the secondary casts), shows WFE vs field, then field-balances
%  the conics for a WIDER corrected field and saves that as the deliverable
%  (better-across-the-field usually beats perfect-on-axis for a real
%  instrument).
%
%  This is the OBSCURED baseline.  For the unobscured (coronagraph-feed)
%  version that decenters to an eccentric pupil, see ../rc_unobscured.
%
%  Run:  >> run('.../templates/10_telescopes/rc_onaxis/rc_onaxis.m')
% =====================================================================

addpath('~/dev/MACOS_resources/mmacos/src');
exdir = fileparts(mfilename('fullpath'));  if isempty(exdir), exdir = pwd; end

% ====================  USER DESIGN CHOICES  ==========================
APERTURE_M   = 1.0;     % aperture diameter D (m)
PRIMARY_FNUM = 2.0;     % M1 f/#
SYSTEM_FNUM  = 10.0;    % system f/#
BFD_M        = 0.30;    % back focal distance: focus this far behind M1 (m)
FOV_ARCMIN   = 4.0;     % evaluate WFE out to this field half-angle (arcmin)
BALANCE_FIELD = true;   % field-balance the conics (radius+conic) for a WIDER
                        % corrected field -- giving up a little on-axis WFE to
                        % gain performance across the field is usually the better
                        % deal for a real instrument than a perfect-on-axis
                        % aplanat.  This balanced design is the saved deliverable.
BALANCE_FOV  = 4.0;     % balance over +-this; keep it SHORT of FOV_ARCMIN
% ---------------------------------------------------------------------
LAM   = 633e-9;         % wavelength (m)
MODEL = 256;            % diffraction grid model size
DIFFRACTION_LIMIT = 0.07;   % ~lambda/14 (waves)
% =====================================================================

D = APERTURE_M;
fprintf('====================================================================\n');
fprintf(' On-axis Ritchey-Chretien | D=%.2f m | primary f/%.1f | system f/%.1f | BFD=%.0f mm\n', ...
        D, PRIMARY_FNUM, SYSTEM_FNUM, BFD_M*1000);
fprintf('====================================================================\n');

%% -- Stage 1 -- build the on-axis RC ---------------------------------
fov = linspace(0, FOV_ARCMIN, 7).';                % +y field samples (rot. symmetric)
scanF = [zeros(numel(fov),1), deg2rad(fov/60)];
t = macos.design.Telescope('family','RC', 'aperture_diameter_m',D, ...
        'primary_fnum',PRIMARY_FNUM, 'system_fnum',SYSTEM_FNUM, 'BFD_m',BFD_M, ...
        'wavelength_m',LAM, 'model_size',MODEL);
t.set_field_points(scanF);
nE = numel(t.spec.elt);
t.build();
macos.trace(nE);  wfe0 = rms_waves(macos.opd(), LAM);
fprintf('\n[1] on-axis RC built (%d elements): RMS WFE = %.4f waves (aplanat)\n', nE, wfe0);

%% -- Stage 2 -- first-order properties -------------------------------
EFL   = SYSTEM_FNUM * D;                            % effective focal length (m)
m     = SYSTEM_FNUM / PRIMARY_FNUM;                 % secondary magnification
pscl  = 206265 / (EFL*1000);                        % plate scale (arcsec/mm)
rep   = t.check_clipping('noload', true, 'quiet', true);
iM2   = find(strcmp({rep.name},'M2'), 1);
eps_o = NaN;  if ~isempty(iM2), eps_o = 2*rep(iM2).foot_r / D; end
fprintf('[2] first-order properties:\n');
fprintf('      EFL = %.3f m   (f/%.1f)        secondary mag m = %.2f\n', EFL, SYSTEM_FNUM, m);
fprintf('      plate scale = %.2f arcsec/mm   back focal dist = %.0f mm (behind M1)\n', pscl, BFD_M*1000);
fprintf('      central obscuration ~ %.2f (secondary diam / aperture)\n', eps_o);

%% -- Stage 3 -- WFE vs field (as-built RC) ---------------------------
scan = t.realize_apertures('fields', scanF, 'margin', 0.05, 'quiet', true);
w_rc = scan.wfe(:).';                               % waves, per field
fld  = scan.fields(:,2).';                          % arcmin
dl_rc = max([0, fld(w_rc <= DIFFRACTION_LIMIT)]);
fprintf('\n[3] WFE vs field (as-built RC, +y):\n');
fprintf('      field(arcmin):'); fprintf(' %5.1f', fld); fprintf('\n');
fprintf('      WFE  (waves) :'); fprintf(' %5.3f', w_rc); fprintf('\n');
fprintf('      diffraction-limited (< %.2f lambda) out to %.1f arcmin\n', DIFFRACTION_LIMIT, dl_rc);

%% -- Stage 4 -- optionally field-balance the conics ------------------
w_bal = [];  dl_bal = dl_rc;
if BALANCE_FIELD
    bfov = linspace(0, BALANCE_FOV, 4);                % balance points, short of FOV
    t.optimize('fields_arcmin', bfov(2:end), 'dofs', [0 0 0 0 0 0 1 1], 'max_iters', 80);
    sb = t.realize_apertures('fields', scanF, 'margin', 0.05, 'quiet', true);
    w_bal = sb.wfe(:).';
    dl_bal = max([0, fld(w_bal <= DIFFRACTION_LIMIT)]);
    fprintf('[4] field-balanced (radius+conic over +-%g''):\n', BALANCE_FOV);
    fprintf('      WFE  (waves) :'); fprintf(' %5.3f', w_bal); fprintf('\n');
    fprintf('      diffraction-limited out to %.1f arcmin (was %.1f)\n', dl_bal, dl_rc);
end

%% -- Stage 5 -- save deliverable + figures ---------------------------
t.add_pupil(numel(t.spec.elt));
rxfile = fullfile(exdir,'rc_onaxis.in');  matfile = fullfile(exdir,'rc_onaxis.mat');
t.save(rxfile);  t.save_spec(matfile);
fprintf('\n[5] saved: %s\n           + %s\n', rxfile, matfile);
try
    f1 = figure('Position',[60 60 660 430]);
    ir = ~isnan(w_rc);  plot(fld(ir), w_rc(ir), '-o', 'LineWidth',1.5); hold on; grid on;
    if ~isempty(w_bal)
        ib = ~isnan(w_bal);  plot(fld(ib), w_bal(ib), '-s', 'LineWidth',1.5);
    end
    yline(DIFFRACTION_LIMIT, '--', '\lambda/14', 'Color',[.6 0 0]);
    xlabel('field half-angle (arcmin)'); ylabel('RMS WFE (waves)');
    if ~isempty(w_bal), legend('as-built RC','field-balanced','Location','northwest');
    else, legend('as-built RC','Location','northwest'); end
    title(sprintf('On-axis RC -- WFE vs field (D=%.1f m, f/%.0f, obsc %.2f)', D, SYSTEM_FNUM, eps_o));
    saveas(f1, fullfile(exdir,'rc_onaxis_wfe.png'));
    fprintf('    WFE-vs-field figure: %s\n', fullfile(exdir,'rc_onaxis_wfe.png'));
catch ME, fprintf('    WFE plot skipped (%s)\n', ME.message); end
try
    f2 = t.view_layout('YZ', 'nrays', 11);             % Cassegrain side view (M2 = obscuration)
    saveas(f2, fullfile(exdir,'rc_onaxis_layout.png'));
    fprintf('    layout (YZ side view): %s\n', fullfile(exdir,'rc_onaxis_layout.png'));
catch ME, fprintf('    layout skipped (%s)\n', ME.message); end

%% -- Summary ---------------------------------------------------------
fprintf('\n--------------------------------------------------------------------\n');
fprintf(' D=%.2f m | f/%.0f | EFL %.2f m | mag %.1f | central obscuration %.2f\n', ...
        D, SYSTEM_FNUM, EFL, m, eps_o);
fprintf(' aplanatic: diffraction-limited on-axis; field-limited by astigmatism\n');
fprintf(' diffraction-limited out to %.1f arcmin%s\n', dl_bal, ...
        ternary(BALANCE_FIELD, ' (field-balanced)', ' (as-built RC)'));
fprintf('====================================================================\n');

% ---- helpers --------------------------------------------------------
function w = rms_waves(W, lam)
    v = W(isfinite(W) & W ~= 0);
    if isempty(v), w = NaN; else, w = std(v) / lam; end
end
function s = ternary(c, a, b), if c, s = a; else, s = b; end, end
