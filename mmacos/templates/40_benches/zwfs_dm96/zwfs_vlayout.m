function out = zwfs_vlayout(varargin)
%ZWFS_VLAYOUT  The vector (polarized-dimple) Zernike sensor's layout.
%   out = zwfs_vlayout() builds the record ZWFS test arm (zwfs_params +
%   macos.design.twyman_green), replaces its camera with the vector
%   sensor's split, emits the two channel decks and draws the layout:
%
%     ... FocalMask (the geometric-phase metasurface in the etched plate's
%     seat) -> MaskSphereOut -> field lens -> quarter-wave plate (fast axis
%     45 deg between the cube's s and p) -> polarizing beam-splitter cube
%     (cemented MacNeille, macos.design.pbs_macneille) -> camera A on the
%     transmitted port, camera B on the reflected port, both at the pupil
%     image (the cube's glass path lengthens the image distance by
%     a (1 - 1/n); both legs get it).
%
%   The engine does not split rays, so the two channels are two decks:
%   zwfs_v_camA.in (transmit) and zwfs_v_camB.in (reflect).  The figure
%   zwfs_vlayout.png (top: the whole train; bottom: the tail from the
%   mask to the two cameras, both channels' ray bundles overlaid) is drawn
%   from the engine's own traces (macos.view_rx).
%
%   Options (name/value): 'cube_side' (mm, default 12.7), 'qwp_gap' (field
%   lens to plate, default 2), 'cube_gap' (plate to cube face, default 2),
%   'MODEL' (engine size for the drawing, default 512), 'NGRID' (65),
%   'draw' (true).  Returns the two Bench objects, the indices and the
%   leg lengths.
%
%   Why a cube and two cameras rather than a Wollaston and one: the pupil
%   image here is 9.4 mm across 32 mm behind the field lens, so two images
%   side by side on one camera would need a 17-deg split from a prism at
%   the lens -- a calcite Wollaston's limit -- while the cube's two ports
%   place identical legs by construction (macos.design.Bench.pbs_cube),
%   and its diagonal is a real coating whose leakage between the channels
%   can be priced with the same engine (the next item after V3).
%
%   See also zwfs_run (stage 'bench'), macos.design.twyman_green,
%   macos.design.Bench.pbs_cube, macos.view_rx.
o = struct('cube_side', 12.7, 'qwp_gap', 2, 'cube_gap', 2, 'MODEL', 512, 'NGRID', 65, 'draw', true);
for i = 1:2:numel(varargin), o.(varargin{i}) = varargin{i+1}; end
exdir = fileparts(mfilename('fullpath'));  if isempty(exdir), exdir = pwd; end
if isempty(which('macos.init')), run(fullfile(exdir, '..', '..', '..', 'mmacos_setup.m')); end
cd(exdir);
P = zwfs_params();
% the drawing deck: the record bench at a grid the drawing model holds
gridn = 256;  griddx = P.grid.DX_G * P.grid.N_G / gridn;
if ~isfile(P.grid.flat_file), macos.write_grid_file(P.grid.flat_file, zeros(P.grid.N_G)); end

out = struct();
modes = {'transmit', 'reflect'};  cams = {'A', 'B'};
for c = 1:2
    bn = fieldnames(P.bench);  bp = rmfield(P.bench, bn(strncmp(bn, 'coat_', 5)));   % runner-level knobs, not twyman_green's (zwfs_run bench_args_)
    % The seat may be the string 'scan' (zwfs_params since 2026-09-17): the
    % RUNNER re-finds the focus, a stage a layout drawing does not have, and
    % the builder needs a scalar.  Draw with the solved lens-rig seat; a
    % millimetre of seat is invisible at layout scale.
    if ~isnumeric(bp.MASK_TRIM)
        if c == 1, fprintf(['zwfs_vlayout: bench.MASK_TRIM is ''%s'' (the runner re-scans it); ' ...
                            'drawing with the solved lens-rig seat 1.231759 mm.\n'], bp.MASK_TRIM); end
        bp.MASK_TRIM = 1.231759;
    end
    bf = fieldnames(bp);  bargs = cell(1, 2*numel(bf));
    for i = 1:numel(bf), bargs{2*i-1} = bf{i};  bargs{2*i} = bp.(bf{i}); end
    G = macos.design.twyman_green(bargs{:}, 'ngridpts', o.NGRID, ...
        'to_grid_file', P.grid.flat_file, 'to_grid_n', gridn, 'to_grid_dx', griddx);
    bt = G.bt;  bt.wavelen = P.LAM;
    iDET = G.T.iDET;  iFL = iDET - 1;
    assert(strcmp(bt.E(iDET).element, 'FocalPlane') && strcmp(bt.E(iFL).element, 'Refractor'), ...
        'zwfs_vlayout: expected the field lens then the detector at the end of the test arm');
    det_leg = G.det_leg;                                  % field lens exit -> pupil image, in air
    % rewind the bench to the field lens' exit face and build the split
    bt.E(iDET) = [];
    bt.pos = bt.E(iFL).vpt;  bt.dir = bt.E(iFL).psi;  bt.path_len = bt.E(iFL).s;
    zhat = [0; 0; 1];                                     % the fold plane's normal: the cube's s axis
    phat = cross(bt.dir, zhat);  phat = phat / norm(phat);   % in the fold plane: the cube's p axis
    out_dir = phat;                                       % the reflected port turns in the fold plane
    iQWP = bt.add_waveplate(o.qwp_gap, (zhat + phat)/sqrt(2), 0.25, 'name', 'QWP');
    PBS = macos.design.pbs_macneille();
    tok = bt.pbs_cube(o.cube_gap + o.cube_side/2, out_dir, 'side', o.cube_side, ...
        'n', PBS.n_glass, 'coat', PBS.layers, 'name', 'PBS');
    idx = bt.add_pbs_pass(tok, 'mode', modes{c}, 'tag', cams{c});
    n = PBS.n_glass;
    d_rest = det_leg - (o.qwp_gap + o.cube_gap + o.cube_side) + o.cube_side*(1 - 1/n);
    assert(d_rest > 3, 'zwfs_vlayout: no room for the camera behind the cube (%.1f mm)', d_rest);
    iCAM = bt.add_detector(d_rest, sprintf('Camera%s', cams{c}));
    deck = sprintf('zwfs_v_cam%s.in', cams{c});
    bt.emit(deck);
    out.(cams{c}) = struct('bench', bt, 'deck', deck, 'iMASK', G.T.iMASK, 'iFL', iFL, 'iQWP', iQWP, ...
        'iPBS', idx, 'iCAM', iCAM, 'det_leg', det_leg, 'd_rest', d_rest, 'cube_side', o.cube_side, 'n', n);
    fprintf('channel %s (%s): deck %s, %d elements; field lens -> pupil image %.2f mm in air; QWP at %.1f, cube face at %.1f, cube %.1f mm (n %.4f), camera %.2f mm behind the cube\n', ...
        cams{c}, modes{c}, deck, numel(bt.E), det_leg, o.qwp_gap, o.qwp_gap + o.cube_gap, o.cube_side, n, d_rest);
end
if ~o.draw, return; end

% ---- the figure: the engine's own traces of both channel decks ---------
macos.init(o.MODEL);
purple = [124 58 237]/255;  orange = [237 161 0]/255;  ink = [11 11 11]/255;
f = figure('Color', 'w', 'Position', [40 40 1800 1010], 'Visible', 'off');
tl = tiledlayout(f, 4, 1, 'Padding', 'compact', 'TileSpacing', 'compact');
ax1 = nexttile(tl, [1 1]);  ax2 = nexttile(tl, [3 1]);
E = out.A.bench.E;  Eb = out.B.bench.E;
passive = find(strcmp({E.element}, 'Reference'));          % the mask spheres, the mask, the recomb plane: not hardware
for c = 1:2
    macos.load_rx(out.(cams{c}).deck);
    col = purple;  if c == 2, col = orange; end
    macos.view_rx('ax', ax1, 'ray_color', col, 'title', '', 'labels', false, 'hide', passive);
    macos.view_rx('ax', ax2, 'ray_color', col, 'title', '', 'labels', false, 'hide', passive);
end
% the cube body: a square about the diagonal's centre, sides along the chief and the reflected port
cc = E(out.A.iPBS(2)).vpt;  dA = E(out.A.iFL).psi;  dB = Eb(out.B.iCAM).psi;  h = o.cube_side/2;
sq = [cc + h*(-dA - dB), cc + h*(dA - dB), cc + h*(dA + dB), cc + h*(-dA + dB), cc + h*(-dA - dB)];
plot3(ax2, sq(1,:), sq(2,:), sq(3,:) + 0.1, '-', 'Color', [82 81 78]/255, 'LineWidth', 1.2);
% top: the whole train, from above the fold plane
axes(ax1);  axis(ax1, 'equal');  view(ax1, 0, 90);  axis(ax1, 'off');
title(ax1, 'The vector Zernike sensor: the interferometer''s test arm, the metasurface in the mask seat, and the split behind the field lens (purple: channel A, transmitted; orange: channel B, reflected)', ...
    'Color', ink, 'FontWeight', 'normal', 'FontSize', 13);
% bottom: the tail, mask to cameras, from above; element labels
axes(ax2);  axis(ax2, 'equal');  view(ax2, 0, 90);
pm = E(out.A.iMASK).vpt;  pa = E(out.A.iCAM).vpt;  pb = Eb(out.B.iCAM).vpt;
allp = [pm pa pb E(out.A.iFL).vpt];
xlim(ax2, [min(allp(1,:)) - 4, max(allp(1,:)) + 10]);  ylim(ax2, [min(allp(2,:)) - 9, max(allp(2,:)) + 9]);
% labels off the beam, with leader lines
lab = {E(out.A.iMASK).vpt,   [14 -7], 'metasurface in the mask seat (the focus)'; ...
       E(out.A.iFL-1).vpt,   [-7 -13], 'field lens'; ...
       E(out.A.iQWP).vpt,    [-6 9],  'quarter-wave plate'; ...
       cc,                   [8 11],  sprintf('PBS cube, %.1f mm (coated diagonal)', o.cube_side); ...
       pa,                   [0 6],   'camera A: the +phi pupil image'; ...
       pb,                   [4 -5],  'camera B: the -phi pupil image'};
for k = 1:size(lab, 1)
    p = lab{k,1};  d = lab{k,2};
    plot3(ax2, [p(1) p(1)+d(1)], [p(2) p(2)+d(2)], [0.2 0.2], '-', 'Color', [137 135 129]/255, 'LineWidth', 1.0);
    text(ax2, p(1)+d(1), p(2)+d(2), 0.3, lab{k,3}, 'Color', ink, 'FontSize', 17, ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'BackgroundColor', 'w', 'Margin', 1);  %#ok<*NASGU>
end
xlabel(ax2, 'bench x, mm', 'Color', ink, 'FontSize', 14);  ylabel(ax2, 'bench y, mm', 'Color', ink, 'FontSize', 14);  set(ax2, 'FontSize', 13);
title(ax2, sprintf('The tail: focus -> field lens -> quarter-wave plate -> %.1f mm cube -> two cameras at the pupil image, %.1f mm behind the cube on each port', ...
    o.cube_side, out.A.d_rest), 'Color', ink, 'FontWeight', 'normal', 'FontSize', 15);
grid(ax2, 'on');  set(ax2, 'GridColor', [225 224 217]/255, 'Color', 'w');
print(f, 'zwfs_vlayout.png', '-dpng', '-r130');  close(f);
fprintf('wrote zwfs_vlayout.png\n');
macos.unload();
end
