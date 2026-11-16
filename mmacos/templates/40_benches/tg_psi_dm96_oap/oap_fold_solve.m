function out = oap_fold_solve(varargin)
%OAP_FOLD_SOLVE  The reflective front end's fold angles, FROM the clearance.
%   The brief's rule (BRIEF_to_reflective section 1): "the OAPs' off-axis
%   distance and fold angle come out of [the clearance solve], not the other
%   way round".  This sweeps the two fold angles (and, optionally, the two
%   fold SIDES), builds the rig at each point, MEASURES every part against
%   every beam it is not in with dmg_bench_clearance, and reports the worst
%   clearance and the part that binds.
%
%   Why a sweep and not a rule.  tg96_run's solve_fold_ asks only that the
%   source / camera BODY clear the collimated beam laterally,
%   F*sin(2*AOI) >= beam + body + margin, which 5 / 9 deg satisfy.  It never
%   asks what the folded leg CROSSES on its way, and on this bench the folded
%   legs cross the node: at 5 deg the chief turns 170 deg, so the source sits
%   587 mm PAST the splitter and its diverging beam travels back along the
%   whole node to reach the pole (oap22: input polarizer -102.3 mm, splitter
%   -49.6), and at 9 deg the converging tail runs back across the analyzer
%   and the output quarter-wave plate (-93.9, -89.4).  Only a measurement of
%   the whole train sees that.
%
%   The off-axis distance is not a separate variable: for a parabola fed at
%   conjugate distance r, off-axis = r*sin(2*AOI) and the parent focal length
%   is r*cos^2(AOI).  Picking the fold angle picks both.
%
%   Usage
%     out = oap_fold_solve                      % the default coarse sweep
%     out = oap_fold_solve('A1',5:5:45, 'A2',5:5:45, 'sides',[1 1; 1 -1])
%     out = oap_fold_solve('tag','foldfine', 'A1',18:2:30, 'A2',10:2:26)
%
%   Options (name/value)
%     'A1','A2'   fold AOI grids, deg (default 5:5:45)
%     'sides'     N x 2 of [OAP1_SIDE OAP2_SIDE] (default [1 1; 1 -1; -1 1; -1 -1])
%     'MARGIN'    clearance spec, mm (default from tg96_params, 25)
%     'BODY'      struct part-stem -> physical body radius (default: the
%                 Stage-A half-widths -- source and camera 50, DM 90,
%                 reference flat 60; see dmg_bench_clearance)
%     'MODEL','NGRID'  engine resolution for the geometry (512 / 65)
%     'tag'       run tag; writes runs/<tag>/ (default 'fold')
%
%   Writes <tag>_fold.mat (the full table), <tag>_fold.png (the worst-
%   clearance map per side pair) and <tag>_fold.txt (the printed table).

exdir = fileparts(mfilename('fullpath'));  if isempty(exdir), exdir = pwd; end
if isempty(which('macos.init'))
    run(fullfile(exdir, '..', '..', '..', 'mmacos_setup.m'));
end
addpath(fullfile(exdir, '..', 'dm_gauge_lib'));

% POL_IN 'source' moves the input polarizer into the diverging leg.
% D_RC_L2 (default: the param file's) is the output optics -> OAP2 standoff.
% It is a CLEARANCE variable on a reflective rig, not just a packaging one: an
% off-axis section's surface spans a range of SAG along its own axis, +-50 mm
% on OAP2 at a 40 deg fold, so a plate closer than that to the pole has mirror
% BEHIND it and the rays that land there can never reach the plate (measured:
% 16 and 169 surface misses at elt 17 = L2, A2 40 and 45 deg, runs/loss_a2).
% SRC_AT_FOCUS feeds the collimator at its true focus (see twyman_green).
o = struct('A1', 5:5:45, 'A2', 5:5:45, ...
           'sides', [1 1; 1 -1; -1 1; -1 -1], ...
           'MARGIN', [], 'BODY', [], 'MODEL', 512, 'NGRID', 65, 'tag', 'fold', ...
           'POL_IN', 'collimated', 'D_RC_L2', [], 'SRC_AT_FOCUS', false);
for i = 1:2:numel(varargin), o.(varargin{i}) = varargin{i+1}; end
if isempty(o.BODY)
    o.BODY = struct('Baffle', 50, 'Detector', 50, 'TestOptic', 90, 'PZT', 60);
end

cd(exdir);
P = tg96_params();
if isempty(o.MARGIN), o.MARGIN = P.clear.MARGIN; end
outdir = fullfile(exdir, 'runs', o.tag);
if ~exist(outdir, 'dir'), mkdir(outdir); end
rep = fopen(fullfile(outdir, [o.tag '_fold.txt']), 'w');
cleaner = onCleanup(@() fclose(rep));
say = @(varargin) say_(rep, varargin{:});

s = P.dm(1).nact / 56;  b = P.bench;
gf = fullfile(outdir, 'fold_flat.txt');
macos.init(o.MODEL);
macos.write_grid_file(gf, zeros(256));

say('=== OAP fold solve: the fold angles from the MEASURED clearance ===\n');
say('bench 22.5 deg node (BS_AOI %g), DM leg %s, beam radius %.1f mm, mount +%g, spec >= %g mm\n', ...
    b.BS_AOI, num2str(dm_leg_(P, s)), s*b.R_TO_AP, P.clear.MOUNT, o.MARGIN);
bn = fieldnames(o.BODY);
say('physical bodies (radius before the mount, mm): ');
for i = 1:numel(bn), say('%s %g  ', bn{i}, o.BODY.(bn{i})); end
drc0 = o.D_RC_L2;  if isempty(drc0), drc0 = b.D_RC_L2; end
say('\ninput polarizer: %s leg; output optics -> OAP2 standoff D_RC_L2 = %g mm\n', ...
    o.POL_IN, drc0);
if o.SRC_AT_FOCUS, say('collimator fed at its TRUE focus (SRC_AT_FOCUS)\n'); end
say('model %d, %d rays; %d x %d angles x %d side pairs = %d builds\n\n', ...
    o.MODEL, o.NGRID, numel(o.A1), numel(o.A2), size(o.sides,1), ...
    numel(o.A1)*numel(o.A2)*size(o.sides,1));

W = nan(numel(o.A1), numel(o.A2), size(o.sides,1));   % worst clearance
BIND = cell(size(W));                                  % the binding part
ROWS = cell(size(W));                                  % the full table
LOST = nan(size(W));            % rays lost on the test arm: a fold angle can
                                %  clear on paper and vignette on the glass
t0 = tic;  n = 0;
for q = 1:size(o.sides,1)
  say('--- sides OAP1 %+d / OAP2 %+d ---\n', o.sides(q,1), o.sides(q,2));
  say('%6s', 'A1\A2');  say('%8d', o.A2);  say('   binding part at the best A2\n');
  for i = 1:numel(o.A1)
    say('%6d', o.A1(i));
    for j = 1:numel(o.A2)
      n = n + 1;
      try
        G = build_(P, s, b, gf, o, o.A1(i), o.A2(j), o.sides(q,:));
        cl = dmg_bench_clearance('G', G, 'MODEL', o.MODEL, 'NGRID', o.NGRID, ...
                                 'BODY', o.BODY, 'quiet', true);
        v = cell2mat(cl.rows(:,6));  v(isnan(v)) = inf;
        [W(i,j,q), k] = min(v);
        BIND{i,j,q} = cl.rows{k,1};
        ROWS{i,j,q} = cl.rows;
        macos.load_rx(cl.decks{1});  tr = macos.trace();
        ri = macos.get_ray_info(tr.nRays);
        LOST(i,j,q) = nnz(~(ri.ok_trace(:) & ri.ok_pass(:)));
      catch e
        W(i,j,q) = NaN;  BIND{i,j,q} = ['ERR:' e.identifier];
      end
      if isnan(W(i,j,q)), say('%8s','--');
      elseif LOST(i,j,q) > 0, say('%7.0f*', W(i,j,q));   % * = rays lost
      else, say('%8.0f', W(i,j,q)); end
    end
    [~, jb] = max(W(i,:,q));
    say('   %s\n', BIND{i,jb,q});
  end
  say('\n');
end
say('%d builds in %.1f min (%.1f s each)\n\n', n, toc(t0)/60, toc(t0)/n);

% ---- the answer: the smallest angles that clear, near-normal preferred ----
best = struct('ok', false);
for q = 1:size(o.sides,1)
  for i = 1:numel(o.A1)
    for j = 1:numel(o.A2)
      if ~(W(i,j,q) >= o.MARGIN), continue; end
      if ~(LOST(i,j,q) == 0), continue; end       % must also trace clean
      c = o.A1(i)^2 + o.A2(j)^2;      % near-normal preferred: least total fold
      if ~best.ok || c < best.c
        best = struct('ok', true, 'c', c, 'a1', o.A1(i), 'a2', o.A2(j), ...
                      'sides', o.sides(q,:), 'worst', W(i,j,q), ...
                      'bind', BIND{i,j,q}, 'rows', {ROWS{i,j,q}}, 'i', i, 'j', j, 'q', q);
      end
    end
  end
end
if best.ok
  say('SOLVED: OAP1 %d deg / OAP2 %d deg, sides %+d/%+d -- worst %+.1f mm (%s)\n', ...
      best.a1, best.a2, best.sides(1), best.sides(2), best.worst, best.bind);
  say('  OAP1 off-axis %.1f mm, parent f %.1f mm; OAP2 off-axis %.1f mm, parent f %.1f mm\n', ...
      s*b.F1*sind(2*best.a1), s*b.F1*cosd(best.a1)^2, ...
      s*b.F2*sind(2*best.a2), s*b.F2*cosd(best.a2)^2);
  say('\n%-22s %-12s %-5s %7s %7s %9s  %s\n', 'element','type','arm','a+mnt','r_beam','clear mm','against');
  for i = 1:size(best.rows,1)
    r = best.rows(i,:);
    say('%-22s %-12s %-5s %7.1f %7.1f %9.1f  %s\n', r{1},r{2},r{3},r{4},r{5},r{6},r{7});
  end
else
  say('NO (A1,A2) IN THIS GRID CLEARS %g mm.  Worst-case best: ', o.MARGIN);
  [wm, k] = max(W(:));  [i,j,q] = ind2sub(size(W), k);
  say('OAP1 %d / OAP2 %d, sides %+d/%+d -> %+.1f mm (%s)\n', ...
      o.A1(i), o.A2(j), o.sides(q,1), o.sides(q,2), wm, BIND{i,j,q});
  say('  the parts still negative there:\n');
  R = ROWS{i,j,q};
  for r = 1:size(R,1)
    if ~isnan(R{r,6}) && R{r,6} < o.MARGIN
      say('  %-22s %+8.1f  against %s\n', R{r,1}, R{r,6}, R{r,7});
    end
  end
  best.a1 = o.A1(i);  best.a2 = o.A2(j);  best.sides = o.sides(q,:);
  best.worst = wm;  best.bind = BIND{i,j,q};  best.rows = R;
end

draw_(W, o, outdir);
say('* = the trace loses rays at that fold (vignetting / surface miss); such\n');
say('  a point is never SOLVED however well it clears.\n');
out = struct('W', W, 'BIND', {BIND}, 'ROWS', {ROWS}, 'LOST', LOST, 'best', best, 'o', o, 's', s);
save(fullfile(outdir, [o.tag '_fold.mat']), 'out');
say('\nwrote %s_fold.{txt,mat,png} in %s\n', o.tag, outdir);
end

% =====================================================================
function G = build_(P, s, b, gf, o, a1, a2, sides)
%BUILD_  the OAP rig at these folds -- the same call tg96_run Stage B makes,
%   with the tuned tail set (geometry only; the tail does not move the node).
drc_ = o.D_RC_L2;  if isempty(drc_), drc_ = b.D_RC_L2; end
G = macos.design.twyman_green('polarizing',b.polarizing, 'ngridpts',o.NGRID, ...
    'optics','oap', 'SRC_AT_FOCUS',o.SRC_AT_FOCUS, 'OAP1_AOI',a1, 'OAP2_AOI',a2, 'POL_IN',o.POL_IN, ...
    'OAP1_SIDE',sides(1), 'OAP2_SIDE',sides(2), 'BS_AOI',b.BS_AOI, ...
    'F1',s*b.F1, 'F2',s*b.F2, 'D_LENS',s*b.D_LENS, 'R_BAFFLE',s*b.R_BAFFLE, ...
    'D_SB',s*b.D_SB, 'BS_T',s*b.BS_T, 'D_L1_BS',s*b.D_L1_BS, ...
    'D_BS_TO',dm_leg_(P, s), 'D_BS_CMP',s*b.D_BS_CMP, 'R_TO_AP',s*b.R_TO_AP, ...
    'L1_Kr',s*b.L1_Kr, 'L1_Kc',b.L1_Kc, 'L2_Kr',-s*abs(b.L2_Kr), 'L2_Kc',b.L2_Kc, ...
    'to_grid_file',gf, 'to_grid_n',256, 'to_grid_dx',s*0.28*384/256, ...
    'qwp_ret',b.qwp_ret, 'pol_in_deg',b.pol_in_deg, 'qwp_test_deg',b.qwp_test_deg, ...
    'qwp_ref_deg',b.qwp_ref_deg, 'out_qwp_deg',b.out_qwp_deg, ...
    'analyzer_deg',b.analyzer_deg, 'tail_arch',b.tail_arch, ...
    'FL_F',s*b.FL_F, 'FL_Kc',b.FL_Kc, 'FL_D',s*b.FL_D, ...
    'D_MASK_FL',s*b.D_MASK_FL, 'DET_TRIM',s*b.DET_TRIM, ...
    'D_RECOMB',b.D_RECOMB, 'D_RC_L2',drc_);
end

function d = dm_leg_(P, s)
%DM_LEG_  tg96_run Stage A's chosen DM leg (the end-body solve), so the node
%   this sweep measures is the node the runner builds.
d = P.bench.D_BS_TO;
if isempty(d)
    beam_r = P.clear.beam_r;  if isempty(beam_r), beam_r = s*P.bench.R_TO_AP; end
    need = beam_r + P.clear.HW_DM + P.clear.MARGIN;
    AOI = P.bench.BS_AOI;
    d = ceil(max(need/sind(2*AOI), s*250)/50)*50;
end
end

function draw_(W, o, outdir)
np = size(W,3);
f = figure('Color','w','Position',[40 40 480*np 420],'Visible','off');
tl = tiledlayout(f, 1, np, 'Padding','compact','TileSpacing','compact');
cl = [-150 60];
for q = 1:np
    ax = nexttile(tl);
    imagesc(ax, o.A2, o.A1, W(:,:,q), cl);  set(ax,'YDir','normal');
    hold(ax,'on');
    contour(ax, o.A2, o.A1, W(:,:,q), [o.MARGIN o.MARGIN], 'k-', 'LineWidth', 2);
    contour(ax, o.A2, o.A1, W(:,:,q), [0 0], 'k--', 'LineWidth', 1);
    xlabel(ax,'OAP2 fold AOI, deg');  ylabel(ax,'OAP1 fold AOI, deg');
    title(ax, sprintf('sides %+d / %+d', o.sides(q,1), o.sides(q,2)), 'FontWeight','normal');
    axis(ax,'square');
end
cb = colorbar(nexttile(tl,np));  cb.Label.String = 'worst part clearance, mm';
sgtitle(f, sprintf(['Worst clearance over every part vs the two fold angles ' ...
    '(solid = the %g mm spec, dashed = touching)'], o.MARGIN), 'FontWeight','normal');
print(f, fullfile(outdir, [o.tag '_fold.png']), '-dpng', '-r140');  close(f);
end

function say_(fid, varargin)
fprintf(varargin{:});  fprintf(fid, varargin{:});
end
