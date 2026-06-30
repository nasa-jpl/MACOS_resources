% gen_segment_gridmat.m -- STAND-ALONE per-segment GridMat (Zernike) generator.
% =====================================================================
%  For every grid-bearing segment of a SEGMENTED prescription, builds a bespoke
%  aperture mask + an ARRAY of Zernike mode grids GridMat(:,:,ii), ii over MODES
%  (e.g. Z4..Z15), in that segment's OWN clocked (xData,yData) frame.
%
%  These per-segment mode grids are the INFLUENCE BASIS consumed by
%  run_dwdgrid* to build the linear dW/d(grid) model (the effect of each mode on
%  each segment).  They are NOT written into the prescription -- collapsing the
%  modes into a single per-segment figure (mode x coef) belongs in the engine
%  and is a separate, later step.
%
%  Outputs (this directory):
%    gridmat_<rx>_<type>_<gs|circ>.mat    per-segment B + masks + frames
%    gridmat_<rx>_<type>_<gs|circ>_basis.png   per-segment x mode montage
%                                              (each tile shows the mask shape too)
%
%  Option axes:
%    ORTHOGONALIZE  true = Gram-Schmidt over each segment aperture; false =
%                   plain circular Zernikes confined to the segment.
%    ZERN_TYPE      'ansi' (engine ZerntoMon1/NormANSI) | 'noll' (zernike_mode).
%
%  Rx requirement: a near-pupil Reference ahead of the segments (trace target).
%  SegDemo3conic.in has it as element 1 (PM-conic Reference).
% =====================================================================
here = fileparts(mfilename('fullpath'));  if isempty(here), here = pwd; end

% ========================  CONFIG  =========================
RX            = fullfile(here, 'SegDemo3conic.in');  % self-contained Rx
MODEL         = 256;        % model size -- must be >= the Rx's nGridMat
PM_REF_ELT    = 1;          % near-pupil Reference (just before the PM)
MODES         = 4:15;       % Zernike figure modes per segment
ORTHOGONALIZE = true;       % true = Gram-Schmidt; false = circular Zernikes
ZERN_TYPE     = 'ansi';     % 'ansi' | 'noll'
% ===========================================================

run(fullfile(here, '..', '..', '..', 'mmacos_setup.m'));   % path setup (once/session)

if ORTHOGONALIZE, orthlabel = 'Gram-Schmidt'; orthtag = 'gs';
else,             orthlabel = 'circular';     orthtag = 'circ'; end
[~, rxstem] = fileparts(RX);
tag = sprintf('%s_%s', ZERN_TYPE, orthtag);
fprintf('=== GridMat generator: %s (model %d, %s, %s) ===\n', ...
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

% ---- show + save the per-segment mode basis (each tile is a mode on its
%      segment's mask, so the mask shapes are visible here too) -----------
basis_montage_(out, here, sprintf('gridmat_%s_%s_basis.png', rxstem, tag), ...
    sprintf('%s -- per-segment %s Zernike basis (%s)', rxstem, ZERN_TYPE, orthlabel));

% ---- save the per-segment GridMat arrays (for run_dwdgrid*) ----------
matf = fullfile(here, sprintf('gridmat_%s_%s.mat', rxstem, tag));
save(matf, '-struct', 'out', '-v7.3');      % compress the smooth mode stacks
fprintf('saved GridMat arrays: %s\n', matf);
fprintf('=== done: %d-segment x %d-mode %s/%s ===\n', ns, nm, ZERN_TYPE, orthlabel);

% =====================================================================
%  Local helpers
% =====================================================================
function basis_montage_(out, here, png, ttl)
% Segment x mode montage: each tile is mode k on segment s, masked to that
% segment's aperture -- so the per-segment mask shapes show here too.
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
