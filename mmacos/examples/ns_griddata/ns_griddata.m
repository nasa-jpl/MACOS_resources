% ns_griddata.m -- how ZrnGrData composes: Zernike + GridData on an NSReflector.
% =====================================================================
%  Answers the iris double-pass "ZGD" questions (Luis): segments A1/A3/A5 are
%  Element=NSReflector surfaces hit twice (iElt 17/19/21 pass 1, 35/37/39
%  pass 2), each carrying BOTH a Zernike figure (ZernCoef, ANSI modes
%  14 15 19 38) and a grid figure (GridFile=zern41em5z155em3.txt,
%  GridSrfdx=1.1e-2).  Five prescriptions differ ONLY in the Surface= type
%  of those six element entries:
%
%    iris_dp_conic.in     Surface=Conic      no figure        (baseline)
%    iris_dp_Zern.in      Surface=Zernike    Zernike only     (grid ignored)
%    iris_dp_GD.in        Surface=GridData   grid only        (ZernCoef ignored)
%    iris_dp_ZGD_flat.in  Surface=ZrnGrData  Zernike + FLAT grid (flat.txt)
%    iris_dp_ZGD.in       Surface=ZrnGrData  Zernike + grid   (both applied)
%
%  Each variant's ExitPupil OPD minus the conic baseline isolates that
%  surface type's figure contribution.  The script then checks:
%    (1) SUPERPOSITION:  dZGD == dZern + dGD  (ZrnGrData adds the two legs)
%    (2) FLAT GRID INERT: ZGD_flat == Zern    (an all-zero grid contributes
%        nothing; ZrnGrData's Zernike leg == Surface=Zernike)
%
%  SETUP: run mmacos_setup once per MATLAB session first.  GridFile= entries
%  resolve from the CWD -- opd_psf cd's to this dir so the grid files load.
% =====================================================================
here = fileparts(mfilename('fullpath'));  if isempty(here), here = pwd; end
run(fullfile(here, '..', '..', 'mmacos_setup.m'));

WF    = 55;    % ExitPupil (Return element) -- OPD reference
MODEL = 512;   % Rx has nGridpts=512

variants = {'conic', 'Zern', 'GD', 'ZGD_flat', 'ZGD'};
R = struct();
for k = 1:numel(variants)
    v = variants{k};
    R.(v) = macos.opd_psf(fullfile(here, ['iris_dp_' v '.in']), ...
              'wf_elt', WF, 'model_size', MODEL, 'show', false);
    fprintf('%-9s rms(OPD) = %.6e  nRays = %d\n', v, R.(v).rmsWFE, R.(v).nRays);
end

% common valid-ray mask (a pixel must be traced in EVERY variant)
m = true(size(R.conic.opd));
for k = 1:numel(variants), m = m & (R.(variants{k}).opd ~= 0); end
zed  = @(A) A .* m;                          % zero outside the common mask
rms_ = @(A) sqrt(mean(A(m).^2));

dZern = zed(R.Zern.opd     - R.conic.opd);   % Zernike-figure contribution
dGD   = zed(R.GD.opd       - R.conic.opd);   % grid-figure contribution
dFlat = zed(R.ZGD_flat.opd - R.conic.opd);   % Zernike + flat grid
dZGD  = zed(R.ZGD.opd      - R.conic.opd);   % Zernike + grid, one surface

sup   = dZGD - (dZern + dGD);                % superposition residual
inert = dFlat - dZern;                       % flat-grid-inertness residual

fprintf('\n=== figure contributions at ExitPupil (elt %d), WaveUnits ===\n', WF);
fprintf('rms(dZern) = %.4e   Zernike leg alone   (Surface=Zernike)\n', rms_(dZern));
fprintf('rms(dGD)   = %.4e   grid leg alone      (Surface=GridData)\n', rms_(dGD));
fprintf('rms(dZGD)  = %.4e   both, one surface   (Surface=ZrnGrData)\n', rms_(dZGD));
fprintf('\n(1) superposition  dZGD - (dZern + dGD):  rms = %.3e  max|.| = %.3e\n', ...
        rms_(sup), max(abs(sup(:))));
fprintf('(2) flat grid inert  ZGD_flat - Zern:     rms = %.3e  max|.| = %.3e\n', ...
        rms_(inert), max(abs(inert(:))));
fprintf(['--> both residuals at numerical noise means ZrnGrData = ' ...
         'Zernike + GridData, applied independently and added.\n']);

% ---- 2x3 panel figure -----------------------------------------------
f = figure('Visible', 'off', 'Position', [40 40 1500 900]);
tiles = {dZern, 'dZern = Zernike leg (Surface=Zernike)'; ...
         dGD,   'dGD = grid leg (Surface=GridData)'; ...
         dZGD,  'dZGD = both (Surface=ZrnGrData)'; ...
         dZern + dGD, 'dZern + dGD'; ...
         sup,   'superposition residual dZGD-(dZern+dGD)'; ...
         inert, 'flat-grid residual ZGD\_flat-Zern'};
for k = 1:6
    subplot(2, 3, k);  A = tiles{k, 1};  A(~m) = NaN;
    h = imagesc(A);  set(h, 'AlphaData', ~isnan(A));
    axis image off;  set(gca, 'Color', 'w');  colormap(gca, parula);  colorbar;
    title(tiles{k, 2});
end
sgtitle(sprintf(['iris\\_dp A1/A3/A5 NSReflector figure decomposition @ ' ...
                 'ExitPupil (elt %d)'], WF));
print(f, fullfile(here, 'ns_griddata_decomposition.png'), '-dpng', '-r140');
close(f);

save(fullfile(here, 'ns_griddata.mat'), 'R', 'dZern', 'dGD', 'dFlat', ...
     'dZGD', 'sup', 'inert', 'm', 'WF', 'MODEL');
fprintf('wrote ns_griddata_decomposition.png + ns_griddata.mat\n');
