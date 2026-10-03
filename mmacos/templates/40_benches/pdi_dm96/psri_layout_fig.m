function G = psri_layout_fig(varargin)
%PSRI_LAYOUT_FIG  The buildable P/SRI on the DM-gauge bench: build both arms
%   (macos.design.psri_bench), put the pinhole seat at the reference lens's
%   TRUE focus (a scan of REF_TRIM, as the ZWFS's MASK_TRIM was found), emit
%   the two decks, trace them, and draw:
%     psri_layout.png   the sketch: the test deck's chief with the reference
%                       arm overlaid (blue), every element named, legs in mm
%     psri_render.png   the traced rig from above the bench and in ISO view,
%                       both arms' rays (green test, blue reference) and the
%                       optics as solid bodies (the ZWFS deck's recipe)
%   Prints the balance: the two chiefs' optical paths, the compensator
%   thickness that equalizes them, the exit-chief coincidence, and the
%   chief / footprint overlap at the camera from the traces.
%   Dev resolution (model 512, 65 rays); any psri_bench option overrides.
%   Run from this dir:  matlab -batch "psri_layout_fig"
exdir = fileparts(mfilename('fullpath'));  if isempty(exdir), exdir = pwd; end
if isempty(which('macos.init')), run(fullfile(exdir, '..', '..', '..', 'mmacos_setup.m')); end
addpath(fullfile(exdir, '..', 'zwfs_dm96'));                % zwfs_params, zwfs_mask
cd(exdir);
MODEL = 512;  NGRID = 65;  N_G = 256;  DX_G = 0.42;
% 'solve', false takes the SOLVED values of record from pdi_params
% (P.pdi.psri) instead of re-running the search -- the figure then costs one
% bench build instead of ~90, which is what makes a layout iterable.  The
% solve is the source of those values; keep them in step.
dosolve = true;
k = find(strcmp(varargin, 'solve'), 1);
if ~isempty(k), dosolve = varargin{k+1};  varargin(k:k+1) = []; end
macos.init(MODEL);
macos.write_grid_file('zwfs_flat.txt', zeros(N_G));
base = [{'ngridpts', NGRID, 'to_grid_file', 'zwfs_flat.txt', 'to_grid_n', N_G, 'to_grid_dx', DX_G}, varargin];
% ---- the pinhole seat at the true focus ----------------------------------
% The diffraction focus of an F/2.9 lens is ~20 um deep (2 lambda F^2), so a
% coarse scan of the mask-plane peak straddles it (the ZWFS stage-1 trap).
% Find the RAY focus first (the rms ray radius at the seat is smooth in the
% trim), then maximize the physical-optics peak within +-0.3 mm of it.
% The reference lenses' conics are SOLVED on the traced bench (the add_lens
% seed for a plano-convex focusing lens is far from stigmatic at F/2.9: 0.31
% mm rms ray blur at the best focus; the ZWFS's L2 carries a tuned conic for
% the same reason): Lr1's conic and the seat trim by the ray blur at the
% pinhole, then Lr2's conic by the reference arm's wavefront at the camera.
if dosolve
J1 = @(x) ref_rayrms(base, x(1), x(2), NaN);
x1 = fminsearch(J1, [3, -2.25], optimset('TolX', 1e-3, 'TolFun', 1e-7, 'MaxFunEvals', 90, 'Display', 'off'));
trim_ray = x1(1);  kc1 = x1(2);
% Lr2 = Lr1's mirror image about the pinhole (same conic, symmetric
% placement): the reverse of Lr1's path, so it recollimates a point at the
% pinhole exactly.  It must NOT be solved on the recollimated wavefront:
% the collimated beam at the recombination plane carries the front end's
% own residual (L1's collimation error, 5.8 um rms, which the tuned tail
% lens cancels for the test arm), and a "solved" Lr2 would half-cancel
% that instead of being right for the pinhole-filtered wave.  With the
% same conic the check plane reproduces the recombination plane's
% wavefront to the digit (the reversibility identity, printed).
kc2 = kc1;
pk = @(trim) -ref_peak(base, trim, kc1, kc2);
trim = fminbnd(pk, trim_ray - 0.3, trim_ray + 0.3, optimset('TolX', 2e-3, 'MaxFunEvals', 30, 'Display', 'off'));
[w_rc, w_chk, w_pin] = ref_wfe(base, trim, kc1, kc2);
fprintf('P/SRI bench: Lr1 conic %.4f, ray focus at REF_TRIM %+.3f mm (rms ray radius %.4f mm); Lr2 = Lr1 mirrored (conic %.4f); diffraction focus at %+.3f mm\n', kc1, trim_ray, J1(x1), kc2, trim);
fprintf('  geometric wavefront (WaveUnits mm -> nm rms): at the recombination plane %.1f nm (the front end''s collimation residual, filtered by the pinhole), recollimated after Lr2 %.1f nm (the same wave: reversibility), at the pinhole %.1f nm\n', w_rc*1e6, w_chk*1e6, w_pin*1e6);
else
    P0 = pdi_params();
    kc1 = P0.pdi.psri.LR1_Kc;  kc2 = P0.pdi.psri.LR2_Kc;  trim = P0.pdi.psri.REF_TRIM;
    fprintf('P/SRI bench: the SOLVED values of record, from pdi_params (Lr1 = Lr2 conic %.4f, REF_TRIM %+.4f mm) -- re-solve with psri_layout_fig(''solve'', true)\n', kc1, trim);
end
base = [base, {'LR1_Kc', kc1, 'LR2_Kc', kc2}];
G = macos.design.psri_bench(base{:}, 'REF_TRIM', trim);
G.bt.emit('psri_test.in');  G.br.emit('psri_ref.in');
B = G.balance;
if dosolve, fprintf('P/SRI bench: REF_TRIM %+.3f mm (mask-plane peak/sum %.3e)\n', trim, -pk(trim)); end
fprintf('  chief optical path, source -> camera: test %.4f mm, reference %.4f mm, difference %.2e mm\n', B.opl_test_mm, B.opl_ref_mm, B.dopl_mm);
fprintf('  lens-glass compensator in the test arm: %.3f mm of n = %.2f glass\n', B.t_comp_mm, G.P.N_GLASS);
fprintf('  exit chiefs after BS3: transverse offset %.2e mm; directions test %s / ref %s\n', B.exit_offset_mm, mat2str(B.test_dir_at_exit, 6), mat2str(B.ref_dir_at_exit, 6));
fprintf('  camera planes: offset %.2e mm\n', B.det_offset_mm);
% ---- trace both decks: chief and footprint at the camera ------------------
c = zeros(3, 2);  rf = zeros(1, 2);  decks = {'psri_test.in', 'psri_ref.in'};  idet = [G.T.iDET, G.R.iDET];
for k = 1:2
    macos.load_rx(decks{k});  t = macos.trace(idet(k));  ri = macos.get_ray_info(t.nRays);
    ok = ri.ok_trace(:) & ri.ok_pass(:);  c(:,k) = ri.pos(:,1);
    rf(k) = max(vecnorm(ri.pos(:,ok) - c(:,k)));
    fprintf('  %s: %d of %d rays reach the camera; footprint radius %.3f mm\n', decks{k}, nnz(ok), t.nRays, rf(k));
end
fprintf('  chief rays at the camera: separation %.3e mm\n', norm(c(:,1) - c(:,2)));
% ---- the figures, in the deck recipe (pdi_vfig_util) -----------------------
% Panel 1: the whole bench from above the fold plane, both arms' own traces
% overlaid (green test, blue reference), passive planes hidden, hardware
% named with leader lines.  Panel 2: the reference arm's node -- Lr1, the
% pinhole seat, Lr2, M3 -- cropped at full width, the crowded part of the
% bench a builder has to get right.
Et = G.bt.E;  Er = G.br.E;
green = [0 0.62 0.10];  blue = [0.10 0.35 0.80];  ink = [11 11 11]/255;
pass_t = find(strcmp({Et.element}, 'Reference'));
pass_r = find(strcmp({Er.element}, 'Reference'));
f = figure('Color', 'w', 'Position', [40 40 1800 1010], 'Visible', 'off');
tl = tiledlayout(f, 5, 1, 'Padding', 'compact', 'TileSpacing', 'compact');
ax1 = nexttile(tl, [2 1]);  ax2 = nexttile(tl, [3 1]);
for ax = [ax1 ax2]
    macos.load_rx('psri_test.in');
    macos.view_rx('ax', ax, 'ray_color', green, 'title', '', 'labels', false, 'hide', pass_t);
    macos.load_rx('psri_ref.in');
    macos.view_rx('ax', ax, 'ray_color', blue,  'title', '', 'labels', false, 'hide', pass_r);
end
TO = Et(G.T.iTO).vpt;  BS1 = Et(4).vpt;  M1 = Et(G.T.iM1).vpt;
BS2 = Et(G.T.iBS2).vpt;  BS3 = Et(G.T.iBS3).vpt;  CAM = Et(G.T.iDET).vpt;
SRC = Et(1).vpt;  L2 = Et(G.T.iL2).vpt;
LR1 = Er(G.R.iLR1).vpt;  PIN = Er(G.R.iPIN).vpt;  LR2 = Er(G.R.iLR2).vpt;  M3 = Er(G.R.iM3).vpt;
pdi_vfig_util('flat', ax1, ...
    'The P/SRI on the DM-gauge bench: the shared front end, then two balanced arms -- the test arm unfiltered (green) and the reference arm through its own pinhole (blue) -- recombined into one camera', 15);
pdi_vfig_util('frame', ax1, [SRC TO CAM BS2 PIN M1 M3], [0.02 0.02 0.06 0.12]);
axis(ax1, 'off');
pdi_vfig_util('label', ax1, { ...
    TO,  [-0.06 -0.26], '96 mm deformable mirror (retro)'; ...
    BS1, [-0.03  0.15], 'beamsplitter, 7 deg'; ...
    BS2, [ 0.13  0.07], 'BS2: the split'; ...
    PIN, [-0.03 -0.27], 'pinhole + phase shifter'; ...
    CAM, [ 0.03  0.14], 'camera at the pupil image'}, 15);
% panel 2: the reference arm's node, the crowded part a builder has to get
% right -- Lr1, the pinhole seat at the true focus, Lr2, and the fold onto BS3
pdi_vfig_util('flat', ax2, sprintf(['The reference arm: Lr1 (f %.0f mm, F/%.1f) focuses onto the pinhole seat in its near-field sphere bracket, the phase shifter steps it, ' ...
    'Lr2 -- Lr1 mirrored about the pinhole -- recollimates, and M3 folds it onto BS3.  The test arm carries %.1f mm of compensating glass, which equalizes the two chief paths to %.0e mm.  (The fold plane, turned 90 deg in the page)'], ...
    G.P.F_REF, G.P.F_REF/(2*G.P.R_TO_AP), B.t_comp_mm, max(abs(B.dopl_mm), 1e-16)), 15, 90);
pdi_vfig_util('frame', ax2, [LR1 PIN LR2 M3], [0.10 0.10 0.45 0.45]);
pdi_vfig_util('label', ax2, { ...
    LR1, [-0.02 -0.26], 'Lr1'; ...
    PIN, [ 0.00  0.26], 'pinhole seat (the true focus)'; ...
    LR2, [ 0.02 -0.26], 'Lr2'; ...
    M3,  [ 0.03  0.24], 'M3, onto BS3'}, 16);
print(f, 'psri_layout.png', '-dpng', '-r130');  close(f);
fprintf('wrote psri_layout.png\n');
% ---- the traced render: the same two decks, from above and in perspective --
% Plain view() + axis tight, NOT a hand-placed camera: the hand-placed one
% left both panels a tenth of their tile (the first pass), which is exactly
% the failure the recipe is about.
VW = {[0 90],  'The bench from above the table'; ...
      [-35 22], 'The same rig in perspective'};
f = figure('Color','w', 'Position',[40 40 1800 760], 'Visible','off');
tl2 = tiledlayout(f, 1, 2, 'Padding','tight', 'TileSpacing','tight');
for q = 1:size(VW,1)
    ax = nexttile(tl2);
    macos.load_rx('psri_test.in');  macos.view_rx('ax', ax, 'title', '', 'labels', false, 'ray_color', green);
    macos.load_rx('psri_ref.in');   macos.view_rx('ax', ax, 'title', '', 'labels', false, 'ray_color', blue);
    view(ax, VW{q,1}(1), VW{q,1}(2));
    axis(ax, 'equal');  axis(ax, 'tight');  axis(ax, 'off');
    title(ax, VW{q,2}, 'Color', ink, 'FontWeight', 'normal', 'FontSize', 15);
end
print(f, 'psri_render.png', '-dpng', '-r130');  close(f);
fprintf('wrote psri_render.png\n');
macos.unload();
end

function r = ref_rayrms(base, trim, kc1, kc2)
% rms ray radius about the chief at the pinhole seat (geometric focus)
G = macos.design.psri_bench(base{:}, 'REF_TRIM', trim, 'LR1_Kc', kc1, 'LR2_Kc', kc2);
G.br.emit('psri_ref_scan.in');  macos.load_rx('psri_ref_scan.in');
t = macos.trace(G.R.iPIN);  ri = macos.get_ray_info(t.nRays);
ok = ri.ok_trace(:) & ri.ok_pass(:);  c = ri.pos(:,1);
r = sqrt(mean(vecnorm(ri.pos(:,ok) - c).^2));
end

function [w_rc, w, w_pin] = ref_wfe(base, trim, kc1, kc2)
% the reference arm's wavefront error where it is COLLIMATED again -- at the
% RefCollimated plane after Lr2 (normal to the beam; at the tilted fold the
% engine's OPD reads the mirror's geometry, and at the camera the tail's
% converging beam dominates -- both merits are meaningless)
G = macos.design.psri_bench(base{:}, 'REF_TRIM', trim, 'LR1_Kc', kc1, 'LR2_Kc', kc2);
G.br.emit('psri_ref_scan.in');  macos.load_rx('psri_ref_scan.in');
t = macos.trace(G.R.iCHK);  w = t.rmsWFE;                  % WaveUnits = mm on these decks
t = macos.trace(G.R.iRC);   w_rc = t.rmsWFE;
t = macos.trace(G.R.iPIN);  w_pin = t.rmsWFE;
end

function p = ref_peak(base, trim, kc1, kc2)
% mask-plane peak / sum of the reference arm at the pinhole seat
G = macos.design.psri_bench(base{:}, 'REF_TRIM', trim, 'LR1_Kc', kc1, 'LR2_Kc', kc2);
G.br.emit('psri_ref_scan.in');  macos.load_rx('psri_ref_scan.in');
I = abs(macos.complex_field(G.R.iPIN)).^2;  p = max(I(:))/sum(I(:));
end
