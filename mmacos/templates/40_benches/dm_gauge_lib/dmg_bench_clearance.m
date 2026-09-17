function out = dmg_bench_clearance(varargin)
%DMG_BENCH_CLEARANCE  Does every physical part clear every beam it is not in?
%   out = dmg_bench_clearance('BS_AOI', 7, 'D_BS_CMP', 171, ...) builds the TG96
%   interferometer (both arms, polarizing) with zwfs_params' bench block plus
%   overrides, traces each arm (model 512, 65 rays) for the beam footprint at
%   every element, and for every physical element (lens, plate, polarizer,
%   plate, mirror, camera) finds the smallest clearance to any beam segment
%   it is not an endpoint of: (distance from the element's center to the
%   segment's chief line) - (beam radius there) - (element radius + mount).
%   Negative = the element sits in that beam.  Prints a table, worst first.
%   Records of one physical part (a plate's two faces, a plate's two passes)
%   are grouped by name stem and never tested against their own beam; a
%   beam counts only where it CROSSES the element's plane.  Every position
%   -- the part's centre and both endpoints of every beam segment -- is the
%   element's POLE (macos.design.Bench.station), which is the vertex for an
%   ordinary element and the beam footprint's centre for an off-axis
%   parabola section, whose parent vertex lies far off the beam.  The beam radius
%   is the DM aperture (the traced footprint is the outermost ray, scaled).
%   Options: 'MOUNT' (mm beyond the aperture radius, 8), 'MODEL' (512),
%   'NGRID' (65), 'BODY' (struct part-stem -> physical body radius in mm: the
%   part is scored at max(aperture, body) and is scored even if its element
%   type is not an optic -- this is how the SOURCE head, whose builder element
%   is an Obscuring baffle, and the camera package get into the table at all;
%   default empty = today's behaviour), 'quiet' (false), 'draw' ('' | a PNG path: the train from
%   above, both arms, parts named, plus the node panel), 'G' (an already-built
%   twyman_green result: measure THAT rig instead of rebuilding one from
%   zwfs_params -- what tg96_run's clearance stage passes, so the table
%   describes the bench the run actually used), 'LAM' (mm).  Any other
%   name/value pair overrides zwfs_params' bench block (twyman_green
%   options: BS_AOI, D_L1_BS, D_BS_CMP, D_RECOMB, D_RC_L2, D_POL, D_QWP ...).
%   Returns rows {part, type, arm, a+mount, r_beam, clearance, against}, G.
%   Written 2026-09-15 after Dave found the 7-deg record bench unbuildable
%   (the node parts sat in each other's beams); the Stage-A solve in
%   tg96_run cleared only the three end bodies.
o = struct('MOUNT', 8, 'MODEL', 512, 'NGRID', 65, 'quiet', false, 'draw', '', 'G', [], 'LAM', [], 'BODY', struct());
ov = struct();
for i = 1:2:numel(varargin)
    if isfield(o, varargin{i}), o.(varargin{i}) = varargin{i+1}; else, ov.(varargin{i}) = varargin{i+1}; end
end
LAM = o.LAM;
if isempty(o.G)
    zd = fullfile(fileparts(mfilename('fullpath')), '..', 'zwfs_dm96');
    addpath(zd);
    P = zwfs_params();
    if ~isfile(P.grid.flat_file), P.grid.flat_file = fullfile(zd, P.grid.flat_file); end
    if isempty(LAM), LAM = P.LAM; end
    bn = fieldnames(P.bench);  bp = rmfield(P.bench, bn(strncmp(bn, 'coat_', 5)));
    bp.polarizing = true;  bp.pol_in_deg = 45;  bp.qwp_test_deg = 0;  bp.qwp_ref_deg = 45;  bp.out_qwp_deg = 0;  bp.analyzer_deg = 0;  bp.qwp_ret = 0.25;
    f = fieldnames(ov);  for i = 1:numel(f), bp.(f{i}) = ov.(f{i}); end
    bf = fieldnames(bp);  bargs = cell(1, 2*numel(bf));
    for i = 1:numel(bf), bargs{2*i-1} = bf{i};  bargs{2*i} = bp.(bf{i}); end
    if ~isfile(P.grid.flat_file), macos.write_grid_file(P.grid.flat_file, zeros(P.grid.N_G)); end
    G = macos.design.twyman_green(bargs{:}, 'ngridpts', o.NGRID, 'to_grid_file', P.grid.flat_file, 'to_grid_n', 256, 'to_grid_dx', P.grid.DX_G*P.grid.N_G/256);
else
    % an already-built rig (tg96_run's clearance stage hands its own G): measure
    % THAT bench, not a rebuild of it from another param file.  Its resolved
    % options are G.P, which is where BS_AOI and the scaled R_TO_AP come from.
    assert(isempty(fieldnames(ov)), 'dmg_bench_clearance: overrides are meaningless with ''G'' -- set them on the rig you build');
    G = o.G;  bp = G.P;
    if isempty(LAM)
        LAM = 6.328e-4;
        try
            if ~isempty(G.bt.wavelen), LAM = G.bt.wavelen; end
        catch
        end
    end
end
arms = {G.bt, G.br};  tag = {'test', 'ref'};  decks = cell(1, 2);
recs = struct('name', {}, 'element', {}, 'arm', {}, 'k', {}, 'vpt', {}, 'psi', {}, 'aprad', {}, 'rbeam', {}, 'part', {});
macos.init(o.MODEL);
for a = 1:2
    bt = arms{a};  bt.wavelen = LAM;
    dk = [tempname, sprintf('_dmg_clr_%s.in', tag{a})];  bt.emit(dk);  macos.load_rx(dk);  decks{a} = dk;   % unique per call: two sessions may run this at once
    n = numel(bt.E);
    for k = 1:n
        e = bt.E(k);  rb = NaN;
        try
            sp = macos.spot(k, 'ref', 'telt', 'at', 'elt');
            rb = max(hypot(sp.pts(:,1) - mean(sp.pts(:,1)), sp.pts(:,2) - mean(sp.pts(:,2))));
        catch
        end
        % 'vpt' here is the element's position ON THE BEAM -- its POLE.
        % For every ordinary element the pole IS the vertex; for an
        % off-axis parabola section (add_oap) the vertex is the PARENT
        % conic's vertex, 133-149 mm off the beam on the TG96 reflective
        % rig.  Reading e.vpt there both mislocated the mirror and, worse,
        % moved the ENDPOINTS of the beam segments below -- the source leg
        % ended 149 mm off OAP1 and a 150 mm phantom leg ran from that
        % vertex to the input polarizer, and three node parts were scored
        % against the phantom (measured 2026-09-15).
        recs(end+1) = struct('name', e.name, 'element', e.element, 'arm', tag{a}, 'k', k, 'vpt', macos.design.Bench.station(e), 'psi', e.psi(:), 'aprad', e.aprad, 'rbeam', rb, 'part', ''); %#ok<AGROW>
    end
end
% the traced footprint is the outermost RAY (39 mm at 65 rays); the beam is the
% DM's full aperture (R_TO_AP): scale every footprint up by that ratio
sc = bp.R_TO_AP / max([recs.rbeam]);  for i = 1:numel(recs), recs(i).rbeam = recs(i).rbeam*sc; end
% fill missing beam radii by neighbors
for i = 1:numel(recs), if isnan(recs(i).rbeam), j = find(~isnan([recs.rbeam]), 1); recs(i).rbeam = recs(j).rbeam; end, end
% beam segments: consecutive records within each arm
segs = struct('p1', {}, 'p2', {}, 'r1', {}, 'r2', {}, 'arm', {}, 'lab', {}, 'parts', {});
for a = 1:2
    ia = find(strcmp({recs.arm}, tag{a}));
    for q = 1:numel(ia)-1
        r1 = recs(ia(q));  r2 = recs(ia(q+1));
        if norm(r2.vpt - r1.vpt) < 1e-6, continue; end
        segs(end+1) = struct('p1', r1.vpt, 'p2', r2.vpt, 'r1', r1.rbeam, 'r2', r2.rbeam, 'arm', tag{a}, 'lab', sprintf('%s: %s -> %s', tag{a}, r1.name, r2.name), 'parts', {{}}); %#ok<AGROW>
    end
end
phys = {'Refractor', 'Reflector', 'TrPolarizer', 'WavePlate', 'FocalPlane', 'NSRefractor'};
% 'BODY': a part's PHYSICAL body is not its clear aperture.  The source head
% and the camera package are the cases that matter -- and the source is not
% even scored by default, because the builder's source-side element is an
% Obscuring baffle, which is not an optic.  BODY is a struct part-stem ->
% body radius (mm, before MOUNT): a named part is scored whatever its element
% type, with radius max(aperture, body).  Default empty => today's behaviour
% exactly, so the lens rig's recorded table does not move.  The TG96 rule's
% own half-widths are HW_CAM 50 (source, camera), HW_DM 90, HW_REF 60.
bodynm = fieldnames(o.BODY);
% Group a physical part's RECORDS by name stem, so a plate is never scored
% against its own beam.  The suffixes are the builders' face / pass tags.  The
% four-character forms must precede the three-character ones, since the first
% alternative that matches at a position wins: 'Comptxfd' has to lose 'txfd',
% not 'txf'.  The P/SRI tags (txft / txbt / txfr / txbr on the Mach-Zehnder
% plates, bare txf / txb on its lens-glass compensator) were missing, so
% BS2's transmitted face and BS2's reflection read as two different parts and
% each was scored against the other's beam -- -114.9 mm of pure bookkeeping
% (2026-09-15).  No TG96 name ends in a bare txf / txb, so the TG96 tables do
% not move; checked against the recorded lens table.
part_ = @(nm) regexprep(nm, ['(pow|flat|txff|txbf|txfo|txbo|txft|txbt|txfr|txbr|' ...
                             'txfd|txbd|txfu|txbu|crefr|refl|binr|boutr|txf|txb|In|Out)$'], '');
for i = 1:numel(recs), recs(i).part = part_(recs(i).name); end
% A SUBSTRATE FACE belongs to the element it brackets (2026-09-17).  The
% builder names those faces neutrally -- 'Sub<k>f' / 'Sub<k>b' -- and that is
% deliberate: every arm descriptor in this lane picks the wave plates out with
% contains(name,'QWP'), so a face called 'QWPtestInf' would be handed to
% macos.waveplate as a plate.  The stem rule above therefore cannot see whose
% substrate they are, and the two PASSES of one plate get different numbers
% besides (Sub2f/Sub2b outbound, Sub3f/Sub3b on the way back through the same
% glass).  Left alone, each face is scored against the beam that goes through
% its own element: five rows at -94 mm on the decided substrate set, pure
% bookkeeping, on a bench whose parts all clear.  The deck ORDER says whose
% they are -- an 'f' face is followed by its element, a 'b' face preceded by
% it -- and inheriting the element's stem also inherits its In/Out pass
% grouping, which is what merges the two passes.  Decks with no such faces are
% untouched by construction.
issub_ = @(nm) ~isempty(regexp(nm, '^(Mask)?Sub\d*[fb]$', 'once'));
for a = 1:2
    ia = find(strcmp({recs.arm}, tag{a}));
    for q = 1:numel(ia)
        nm = recs(ia(q)).name;
        if ~issub_(nm), continue; end
        if nm(end) == 'f' && q < numel(ia) && ~issub_(recs(ia(q+1)).name)
            recs(ia(q)).part = recs(ia(q+1)).part;
        elseif nm(end) == 'b' && q > 1 && ~issub_(recs(ia(q-1)).name)
            recs(ia(q)).part = recs(ia(q-1)).part;
        end
    end
end
% the segments' endpoints must be read through the SAME map, or a segment
% labelled '... Sub3b -> Comptxfu' still reports the raw stem and the part it
% belongs to is scored against it anyway
for s = 1:numel(segs)
    a1 = regexprep(segs(s).lab, '^(.*): .* -> .*$', '$1');
    n1 = regexprep(segs(s).lab, '^.*: (.*) -> (.*)$', '$1');
    n2 = regexprep(segs(s).lab, '^.*: (.*) -> (.*)$', '$2');
    segs(s).parts = {stem_of_(recs, a1, n1, part_), stem_of_(recs, a1, n2, part_)};
end
rows = {};  done = {};
for i = 1:numel(recs)
    e = recs(i);
    ib = find(strcmp(bodynm, e.part), 1);
    if ~any(strcmp(e.element, phys)) && isempty(ib), continue; end
    if any(strcmp(done, e.part)), continue; end
    done{end+1} = e.part; %#ok<AGROW>
    aelt = e.aprad;  if aelt <= 0, aelt = e.rbeam + 5; end   % plates: the beam + 5 mm
    if ~isempty(ib), aelt = max(aelt, o.BODY.(bodynm{ib})); end   % physical body
    aelt = aelt + o.MOUNT;
    n = e.psi/norm(e.psi);
    worst = inf;  wlab = '';
    for s = 1:numel(segs)
        sg = segs(s);
        if any(strcmp(sg.parts, e.part)), continue; end      % the part's own beam (either pass, either face)
        h1 = dot(sg.p1 - e.vpt, n);  h2 = dot(sg.p2 - e.vpt, n);
        if h1*h2 > 0, continue; end                           % does not cross the element's plane
        t = h1/(h1 - h2);  q = sg.p1 + t*(sg.p2 - sg.p1);     % the crossing point
        lat = norm(q - e.vpt);
        rb = sg.r1 + (sg.r2 - sg.r1)*t;
        c = lat - rb - aelt;
        if c < worst, worst = c;  wlab = sg.lab; end
    end
    if isinf(worst), worst = NaN; wlab = '(no other beam crosses its plane)'; end
    rows(end+1, :) = {e.part, e.element, e.arm, aelt, e.rbeam, worst, wlab}; %#ok<AGROW>
end
v = cell2mat(rows(:,6)); v(isnan(v)) = inf; [~, ord] = sort(v);  rows = rows(ord, :);
out = struct('rows', {rows}, 'G', G, 'recs', recs, 'segs', segs, 'decks', {decks}, 'bp', bp);
if ~o.quiet
    fprintf('%-22s %-12s %-5s %7s %7s %9s  %s\n', 'element', 'type', 'arm', 'a+mnt', 'r_beam', 'clear mm', 'against');
    for i = 1:size(rows, 1)
        fprintf('%-22s %-12s %-5s %7.1f %7.1f %9.1f  %s\n', rows{i,1}, rows{i,2}, rows{i,3}, rows{i,4}, rows{i,5}, rows{i,6}, rows{i,7});
    end
end

if ~isempty(o.draw)
    blue = [30 90 190]/255;  orange = [214 96 24]/255;  ink = [11 11 11]/255;
    f = figure('Color', 'w', 'Position', [40 40 1800 1100], 'Visible', 'off');
    tl = tiledlayout(f, 5, 1, 'Padding', 'compact', 'TileSpacing', 'compact');
    ax1 = nexttile(tl, [2 1]);  ax2 = nexttile(tl, [3 1]);
    Et = G.bt.E;  Er = G.br.E;
    for a = 1:2
        macos.load_rx(decks{a});
        E = arms{a}.E;  passive = find(strcmp({E.element}, 'Reference'));
        col = blue;  if a == 2, col = orange; end
        macos.view_rx('ax', ax1, 'ray_color', col, 'title', '', 'labels', false, 'hide', passive);
        macos.view_rx('ax', ax2, 'ray_color', col, 'title', '', 'labels', false, 'hide', passive);
    end
    % a name the rig does not carry (the OAP variant renames its collimator,
    % another rig may have no compensator) yields [] and its label is skipped,
    % so the picture degrades rather than erroring.
    nm = {Et.name};  vp = @(n) vpt_(Et, nm, n);
    nr = {Er.name};  vr = @(n) vpt_(Er, nr, n);
    for ax = [ax1 ax2]
        axes(ax);  axis(ax, 'equal');  view(ax, 0, 90);  grid(ax, 'on');  set(ax, 'GridColor', [225 224 217]/255, 'FontSize', 12);
        xlabel(ax, 'bench x, mm', 'FontSize', 13);  ylabel(ax, 'bench y, mm', 'FontSize', 13);
    end
    pbs = vp('BSrefl');
    lab1 = {vp('L1pow'), [-60 110], 'collimator L1'; pbs, [120 120], sprintf('plate splitter, %g deg', bp.BS_AOI); vp('Comptxfd'), [-60 -110], 'compensator'; ...
            vp('TestOptic'), [0 -80], '96 mm DM'; vr('PZT'), [0 -80], 'reference flat + PZT'; vp('L2pow'), [0 90], 'focuser L2'; vp('Detector'), [0 60], 'camera'};
    for k = 1:size(lab1, 1)
        p = lab1{k,1};  d = lab1{k,2};
        if isempty(p), continue; end
        plot3(ax1, [p(1) p(1)+d(1)], [p(2) p(2)+d(2)], [0.2 0.2], '-', 'Color', [137 135 129]/255, 'LineWidth', 1.0);
        text(ax1, p(1)+d(1), p(2)+d(2), 0.3, lab1{k,3}, 'Color', ink, 'FontSize', 15, 'HorizontalAlignment', 'center', 'BackgroundColor', 'w', 'Margin', 1);
    end
    title(ax1, sprintf('The bench from above: splitter at %g deg, test arm blue, reference arm orange', bp.BS_AOI), 'FontWeight', 'normal', 'FontSize', 15);
    % node panel: +-320 mm about the splitter
    xlim(ax2, [pbs(1)-340, pbs(1)+340]);  ylim(ax2, [pbs(2)-260, pbs(2)+260]);
    lab2 = {vp('L1pow'), [0 75], 'L1'; vp('PolIn'), [10 -80], 'input polarizer'; pbs, [-60 150], 'splitter'; vp('Comptxfd'), [-90 -60], 'compensator'; ...
            vp('OutQWP'), [110 -40], 'output QWP'; vp('Analyzer'), [130 -80], 'analyzer'; vp('L2pow'), [-120 60], 'focuser L2'};
    % (the arm quarter-wave plates sit D_QWP before the DM and the flat since 2026-09-15; they are outside the node panel)
    for k = 1:size(lab2, 1)
        p = lab2{k,1};  d = lab2{k,2};
        if isempty(p), continue; end
        plot3(ax2, [p(1) p(1)+d(1)], [p(2) p(2)+d(2)], [0.2 0.2], '-', 'Color', [137 135 129]/255, 'LineWidth', 1.0);
        text(ax2, p(1)+d(1), p(2)+d(2), 0.3, lab2{k,3}, 'Color', ink, 'FontSize', 14, 'HorizontalAlignment', 'center', 'BackgroundColor', 'w', 'Margin', 1);
    end
    v = cell2mat(rows(:,6));  nbad = nnz(v < 0);
    title(ax2, sprintf('The node at %g deg, +-320 mm about the splitter: %d of %d parts sit in another beam (worst %.0f mm); beams %.0f mm, mounts +%g mm', bp.BS_AOI, nbad, nnz(~isnan(v)), min(v), 2*bp.R_TO_AP, o.MOUNT), 'FontWeight', 'normal', 'FontSize', 15);
    print(f, o.draw, '-dpng', '-r130');  close(f);
    fprintf('wrote %s\n', o.draw);
end

end

function v = vpt_(E, nm, n)
%VPT_  the vertex of the element named n, or [] when this rig has no such part.
    i = find(strcmp(nm, n), 1);
    if isempty(i), v = []; else, v = macos.design.Bench.station(E(i)); end
end

function st = stem_of_(recs, arm, nm, part_)
%STEM_OF_  the part stem this deck record carries, after the substrate-face
%   remapping; falls back to the plain suffix rule for a name not in recs.
i = find(strcmp({recs.arm}, arm) & strcmp({recs.name}, nm), 1);
if isempty(i), st = part_(nm); else, st = recs(i).part; end
end
