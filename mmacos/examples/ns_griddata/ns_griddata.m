% ns_griddata.m -- GridData (Zernike-grid) figure on a NON-SEQUENTIAL reflector.
% =====================================================================
%  Repro for the iris double-pass "ZGD" case (Luis): Segment A1 is an
%  Element=NSReflector, Surface=ZrnGrData at iElt 17 (pass 1) and iElt 35
%  (pass 2), carrying GridFile=zern41em5z155em3.txt.  We compare the ExitPupil
%  OPD with the grid figure ON vs OFF (flat.txt) -- the DIFFERENCE is the grid
%  figure's contribution to the wavefront.  If it is ~zero, the non-sequential
%  reflector's grid figure is not being applied.
%
%  Uses the reusable macos.opd_psf utility (reads a .in, optional FEX, OPD +
%  optional PSF, save png/mat).
%
%  SETUP: run mmacos_setup once per MATLAB session first.  GridFile= entries
%  resolve from the CWD -- opd_psf cd's to this dir so the grid files load.
% =====================================================================
here = fileparts(mfilename('fullpath'));  if isempty(here), here = pwd; end
run(fullfile(here, '..', '..', 'mmacos_setup.m'));

WF    = 55;    % ExitPupil (Return element) -- OPD reference
MODEL = 512;   % Rx has nGridpts=512

on  = macos.opd_psf(fullfile(here, 'iris_dp_ZGD.in'), ...
        'wf_elt', WF, 'model_size', MODEL, 'show', false);
off = macos.opd_psf(fullfile(here, 'iris_dp_ZGD_flat.in'), ...
        'wf_elt', WF, 'model_size', MODEL, 'show', false);

D  = on.opd - off.opd;
m  = (on.opd ~= 0) & (off.opd ~= 0);
rms_ = @(A) sqrt(mean(A(A ~= 0).^2));
fprintf('\n=== grid figure ON vs OFF at ExitPupil (elt %d) ===\n', WF);
fprintf('rms(OPD_on) = %.4e\n', rms_(on.opd));
fprintf('rms(OPD_off)= %.4e\n', rms_(off.opd));
fprintf('rms(diff)   = %.4e   max|diff| = %.4e   (WaveUnits)\n', ...
        sqrt(mean(D(m).^2)), max(abs(D(m))));
fprintf(['--> diff ~ 0 means the NSReflector grid figure is NOT applied ' ...
         '(the bug); a z41 shape on the A1 footprint means it IS.\n']);

% ---- 3-panel figure: OPD on / off / difference ---------------------
f = figure('Visible', 'off', 'Position', [40 40 1500 480]);
tiles = {on.opd, 'OPD  grid ON (zern41)'; off.opd, 'OPD  grid OFF (flat)'; D, 'difference = grid figure'};
for k = 1:3
    subplot(1, 3, k);  A = tiles{k, 1};  A(A == 0) = NaN;
    h = imagesc(A);  set(h, 'AlphaData', ~isnan(A));
    axis image off;  set(gca, 'Color', 'w');  colormap(gca, parula);  colorbar;
    title(tiles{k, 2}, 'Interpreter', 'none');
end
sgtitle(sprintf('iris\\_dp\\_ZGD  A1 NSReflector ZrnGrData grid @ ExitPupil (elt %d)', WF));
print(f, fullfile(here, 'ns_griddata_opd_on_off_diff.png'), '-dpng', '-r140');
close(f);

save(fullfile(here, 'ns_griddata.mat'), 'on', 'off', 'D', 'WF', 'MODEL');
fprintf('wrote ns_griddata_opd_on_off_diff.png + ns_griddata.mat\n');
