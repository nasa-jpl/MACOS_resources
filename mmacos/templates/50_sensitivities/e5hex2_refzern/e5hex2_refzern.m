%E5HEX2_REFZERN  A conforming Reference surface (passive) for GS Zernike basis work.
%
%   Demonstrates the "conforming Reference" engine capability: an
%   Element=Reference / Surface=Zernike surface that CARRIES a Zernike basis
%   definition (segment shapes: modes, coefficients, normalization radius,
%   conforming aperture) while having NO effect on the light.
%
%   A Reference surface is passive -- the ray passes straight through, so the
%   total optical path to the next real element is unchanged.  This surface is
%   used only to *establish the segment shapes* over which a Gram-Schmidt
%   Zernike basis is developed (MATLAB-side, e.g. gs_zernike_segment_basis);
%   the engine's job is simply to load it and stay out of the light path.
%
%   Before this feature the engine REJECTED Reference+Zernike ("Invalid
%   Element/Surface Combination"), so a prescription carrying the basis on a
%   conforming reference could not be loaded at all.
%
%   This example proves the surface is passive: e5hex2grid.in (WITH the
%   conforming Reference at element 1) and e5hex2.in (the same system with the
%   reference removed) produce the SAME exit-pupil OPD.
%
%   Run (after mmacos_setup):
%       mmacos_setup
%       run e5hex2_refzern
%
%   Writes e5hex2_refzern_passive.png and e5hex2_refzern.mat.

here = fileparts(mfilename('fullpath'));
cd(here);

MS = 512;   % model size (nGridMat=256 grids need >=512)

fprintf('\n== Conforming Reference (passive) : with-ref vs no-ref OPD ==\n');

% wf_elt = -1 -> num_elt-1 (the exit-pupil slot) for each system
withRef = macos.opd_psf('e5hex2grid.in', 'model_size', MS, 'wf_elt', -1, ...
                        'show', false, 'save_png', false, 'save_mat', false);
noRef   = macos.opd_psf('e5hex2.in',     'model_size', MS, 'wf_elt', -1, ...
                        'show', false, 'save_png', false, 'save_mat', false);

OPD_withRef = withRef.opd;
OPD_noRef   = noRef.opd;
diffOPD     = OPD_withRef - OPD_noRef;    % should be ~0 : the reference is passive

rmsOf = @(A) sqrt(mean(A(isfinite(A) & A~=0).^2, 'omitnan'));
fprintf('  exit-pupil RMS  WITH ref = %.6g waves  (nRays %d)\n', rmsOf(OPD_withRef), withRef.nRays);
fprintf('  exit-pupil RMS  NO   ref = %.6g waves  (nRays %d)\n', rmsOf(OPD_noRef),   noRef.nRays);
fprintf('  RMS( with - no ref )     = %.4g waves  (== 0 => reference has no effect)\n', rmsOf(diffOPD));

% ---- 3-panel figure: with ref / no ref / difference --------------------
f = figure('Visible','off','Position',[100 100 1200 380],'Color','w');
tl = tiledlayout(f,1,3,'Padding','compact','TileSpacing','compact');
panels = {OPD_withRef, 'OPD: WITH conforming Reference'; ...
          OPD_noRef,   'OPD: reference removed'; ...
          diffOPD,     'difference (== 0, passive)'};
for k = 1:3
    ax = nexttile(tl);
    A = panels{k,1};  A(A==0) = NaN;
    imagesc(ax, A); axis(ax,'image','off'); colorbar(ax);
    title(ax, panels{k,2}, 'Interpreter','none');
end
title(tl, 'Conforming Reference is passive: no effect on the light (e5hex2grid vs e5hex2)', ...
      'Interpreter','none');
png = 'e5hex2_refzern_passive.png';
exportgraphics(f, png, 'Resolution', 130);
fprintf('  wrote %s\n', png);

save('e5hex2_refzern.mat', 'OPD_withRef', 'OPD_noRef', 'diffOPD', 'MS');
fprintf('  wrote e5hex2_refzern.mat\n');

% ---- pass/fail: the reference must be passive to roundoff --------------
assert(rmsOf(diffOPD) < 1e-6 * max(rmsOf(OPD_noRef), 1), ...
    'conforming Reference is NOT passive -- it is affecting the light path');
fprintf('  PASS: conforming Reference loads, carries the Zernike basis, and is passive.\n\n');

exit(0);
