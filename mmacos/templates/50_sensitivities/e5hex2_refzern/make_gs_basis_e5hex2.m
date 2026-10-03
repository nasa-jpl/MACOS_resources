% make_gs_basis_e5hex2.m -- PART 1: build + SAVE the per-segment GS Zernike basis
% for the e5hex2 segmented telescope, using the conforming Reference at elt 1.
% =====================================================================
%  The passive conforming Reference (Element=Reference / Surface=Zernike at elt
%  1 of e5hex2grid.in) is the near-pupil trace target that ESTABLISHES THE
%  SEGMENT SHAPES.  From those footprints macos.segment_grid_basis builds, in
%  each segment's own clocked (xData,yData) frame, a bespoke aperture mask plus
%  a Gram-Schmidt-orthonormalized stack of Zernike figure modes (Z4..Z15).
%
%  This is the MAKE-AND-SAVE half of the example: it writes the basis to a
%  .mat that run_dwdgrid_multi_e5hex2.m (PART 2) loads to build the dW/d(grid)
%  sensitivities -- so the (slow) basis build runs once and the dW step can be
%  re-run cheaply.
%
%  The Reference is PASSIVE (no effect on the light -- see e5hex2_refzern.m);
%  it exists only to give segment_grid_basis a valid trace target carrying the
%  segmented footprint.  Before this engine feature a Reference could not carry
%  a Zernike surface, so e5hex2grid.in could not be loaded at all.
%
%  Outputs (this directory):
%    gridmat_e5hex2grid_ansi_gs.mat        per-segment B + masks + frames
%    gridmat_e5hex2grid_ansi_gs_basis.png  per-segment x mode montage
%
%  Run (after mmacos_setup):  run make_gs_basis_e5hex2
% =====================================================================

here = fileparts(mfilename('fullpath'));  if isempty(here), here = pwd; end
cd(here);

% ========================  CONFIG  =========================
RX            = fullfile(here, 'e5hex2grid.in'); % 19-hex telescope + conf. Ref
MODEL         = 512;        % model size (>= nGridMat=256; e5hex2 loads at 512)
PM_REF_ELT    = 1;          % the conforming Reference (near-pupil trace target)
MODES         = 4:15;       % Zernike figure modes per segment
ORTHOGONALIZE = true;       % true = Gram-Schmidt; false = circular
ZERN_TYPE     = 'ansi';     % engine ZerntoMon1 / NormANSI
% ===========================================================

if ORTHOGONALIZE, orthlabel = 'Gram-Schmidt'; orthtag = 'gs';
else,             orthlabel = 'circular';     orthtag = 'circ'; end
[~, rxstem] = fileparts(RX);
tag = sprintf('%s_%s', ZERN_TYPE, orthtag);
fprintf('=== GS basis (make + save): %s (model %d, %s, %s) ===\n', ...
        rxstem, MODEL, ZERN_TYPE, orthlabel);

m   = macos.Session(MODEL);
out = macos.segment_grid_basis(m, RX, ...
    'pm_ref_elt', PM_REF_ELT, 'modes', MODES, ...
    'orthogonalize', ORTHOGONALIZE, 'zern_type', ZERN_TYPE);
ns  = numel(out.seg);   nm = numel(out.modes);
fprintf('built %d segments, %d modes each, N=%d grid, dx=%.4g\n', ns, nm, out.N, out.gdx);
for s = 1:ns
    fprintf('  seg elt %2d (%-9s): %5d mask px, %5d rays, R=%.3f\n', ...
        out.seg(s).iElt, out.seg(s).name, out.seg(s).mask_px, ...
        out.seg(s).n_rays, out.seg(s).R_seg);
end

% ---- montage of the per-segment mode basis (mask shapes visible) --------
basis_montage_(out, here, sprintf('gridmat_%s_%s_basis.png', rxstem, tag), ...
    sprintf('%s -- per-segment %s Zernike basis (%s)', rxstem, ZERN_TYPE, orthlabel));

% ---- SAVE the per-segment basis (consumed by run_dwdgrid_multi_e5hex2) ---
matf = fullfile(here, sprintf('gridmat_%s_%s.mat', rxstem, tag));
save(matf, '-struct', 'out', '-v7.3');
fprintf('saved GS basis: %s\n', matf);
fprintf('=== done: %d-segment x %d-mode %s/%s ===\n', ns, nm, ZERN_TYPE, orthlabel);

exit(0);

% =====================================================================
%  Local helpers (copied from gen_segment_gridmat)
% =====================================================================
function basis_montage_(out, here, png, ttl)
ns = numel(out.seg);   nm = numel(out.modes);
f = figure('Visible', 'off', 'Position', [40 40 130*nm+120 130*ns+90]);
t = tiledlayout(ns, nm, 'TileSpacing', 'compact', 'Padding', 'compact');
for s = 1:ns
    for k = 1:nm
        nexttile;   M = out.seg(s).B(:, :, k);   M(~out.seg(s).mask) = NaN;
        tile_(M);
        if s == 1, title(sprintf('z%d', out.modes(k)), 'FontSize', 7); end
        if k == 1, ylabel(sprintf('e%d', out.seg(s).iElt), 'FontSize', 7); end
    end
end
colormap(f, parula);
title(t, ttl, 'Interpreter', 'none');
print(f, fullfile(here, png), '-dpng', '-r140');   close(f);
fprintf('wrote %s\n', png);
end

function tile_(M)
h = imagesc(M);   set(h, 'AlphaData', ~isnan(M));
axis image off;   set(gca, 'Color', 'w');
end
