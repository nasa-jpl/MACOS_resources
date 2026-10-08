function OUT = tma_longslit_run(stages, over)
%TMA_LONGSLIT_RUN  The staged runner of the long-slit TMA template.
%
%   OUT = TMA_LONGSLIT_RUN(STAGES) runs the named stages in order at the
%   default parameters; OUT = TMA_LONGSLIT_RUN(STAGES, OVER) on top of
%   TMA_LONGSLIT_PARAMS(OVER).  STAGES is a char or a cell:
%     'first_order'  the paraxial solve (TLS_FIRST_ORDER) for the stop at
%                    P.stop, with the other stop places alongside for the
%                    trade: mirror powers, the local radii each section must
%                    carry (R_t, R_s), the intermediate focus (if any), the
%                    pupils, footprints, the cone.  No engine.
%     'section'      the 3-D section (TLS_SECTION): emit <tag>_section.in,
%                    measure it in the ENGINE (TLS_MEASURE): per strip field
%                    the chief angle at the slit, the plate scale, the cone
%                    F/# in both directions, the admitted fraction, spots as
%                    placed and at best focus, the footprints and the
%                    clearance table.
%     'figure'       the figure ladder (TLS_FIGURE) on the section: R0 =
%                    the paraboloid seed, then P.ladder (conic -> asph ->
%                    geom by default), each rung warm from the last, emitted
%                    as <tag>_R<n>_<name>.in and measured in the engine:
%                    per field FWHM both axes, F/# both axes, chief angle,
%                    plate error, bow, admitted, footprints, clearance.
%     'e2e'          the best rung that CLEARS with every ray inside the cone
%                    bound, joined to the dyson5 3k Dyson of record
%                    (TLS_E2E -> dyson5_t5f), both rolls, the joined deck's
%                    clearance (TLS_CLEARANCE_JOINED); one table against
%                    Joe's spec and the paper's five.
%   Each stage prints its table, writes <tag>_<stage>.txt and saves
%   <tag>_<stage>.mat in P.outdir; a later stage reloads an earlier one's
%   .mat when it is not run in the same call.
%
%   Examples:
%       OUT = tma_longslit_run({'first_order','section'});
%       OUT = tma_longslit_run('section', struct('aoi_deg',[28 37 15]));
%
%   See also TMA_LONGSLIT_PARAMS, TMA_LONGSLIT.
if nargin < 1 || isempty(stages), stages = {'first_order', 'section'}; end
if ischar(stages) || isstring(stages), stages = cellstr(stages); end
if nargin < 2, over = struct(); end
P = tma_longslit_params(over);
OUT = struct('P', P);
for s = stages(:)'
    switch s{1}
        case 'first_order', OUT.first_order = stage_first_order_(P);
        case 'section'
            if ~isfield(OUT, 'first_order'), OUT.first_order = stage_first_order_(P); end
            OUT.section = stage_section_(P, OUT.first_order);
        case 'figure'
            if ~isfield(OUT, 'first_order'), OUT.first_order = load_or_run_(P, 'first_order', @() stage_first_order_(P)); end
            OUT.figure = stage_figure_(P, OUT.first_order);
        case 'e2e', OUT.e2e = stage_e2e_(P);
        otherwise, error('tma_longslit_run: unknown stage ''%s''', s{1});
    end
end
end

% ===========================================================================
function S = stage_first_order_(P)
[fid, pr] = open_(P, 'first_order');
pr('tma_longslit first_order -- the paraxial zig-zag TMA (%s)\n', datestr(now, 'yyyy-mm-dd HH:MM'));
pr('SPEC: f %.1f mm, D %.1f mm (F/%.2f), strip +-%.2f deg (image +-%.2f mm; slit %.1f mm), telecentric, cone F/%.2f..%.2f\n', ...
   P.f_m*1e3, P.D_m*1e3, P.f_m/P.D_m, P.strip_half_deg, P.f_m*tand(P.strip_half_deg)*1e3, P.slit_m*1e3, P.cone_fnum);
pr('SEED: legs %s mm at f %.0f -> %s mm at f %.0f; chief AOI %s deg; turn %s\n', mat2str(P.seed_legs_m*1e3, 4), P.seed_f_m*1e3, ...
   mat2str(round(P.legs_m*1e4)/10), P.f_m*1e3, mat2str(P.aoi_deg), mat2str(P.turn));
pr('MODEL: unfolded thin mirrors along the chief; the same legs in the fold (t) and strip (s) sections -> one power per mirror;\n');
pr('  a tilted mirror gives phi_t = 2/(R_t cos i), phi_s = 2 cos i/R_s, so each section must carry R_t/R_s = 1/cos^2 i at the chief.\n\n');
stops = {P.stop};
for o = {'M2', 'M1', 0.5}, if ~isequal(o{1}, P.stop), stops{end+1} = o{1}; end, end %#ok<AGROW>
S = struct('P', P, 'rows', []);
for j = 1:numel(stops)
    st = stops{j};
    try
        FO = tls_first_order(P, st);
    catch e
        pr('STOP %s: no solution (%s)\n\n', stopname_(st), e.message);  continue
    end
    if j == 1, S.FO = FO; end
    S.rows = [S.rows, FO];
    pr('STOP %s%s\n', stopname_(st), tern_(j == 1, '  (the record)', ''));
    pr('  %-4s %10s %10s %10s %10s %10s %9s %9s\n', '', 'f (mm)', 'R unf', 'R_t', 'R_s', 'beam', 'foot x', 'foot y*');
    for k = 1:3
        pr('  M%-3d %+10.1f %+10.1f %+10.1f %+10.1f %10.1f %9.1f %9.1f\n', k, FO.f_k(k)*1e3, FO.R_unfolded(k)*1e3, ...
           FO.Rt(k)*1e3, FO.Rs(k)*1e3, FO.beam(k)*1e3, FO.foot_x(k)*1e3, FO.foot_y_surf(k)*1e3);
    end
    ifs = find(~isnan(FO.int_focus));
    if isempty(ifs), ift = 'none (no real intermediate image)';
    else, ift = strjoin(arrayfun(@(k) sprintf('leg %d at %.1f mm', k, FO.int_focus(k)*1e3), ifs, 'uni', 0), ', '); end
    pr('  EFL %.2f mm, back focus %.2f mm, F/%.3f; chief exit slope %.2e per rad (0 = telecentric); intermediate focus: %s\n', ...
       FO.efl*1e3, FO.bfd*1e3, FO.fnum, FO.exit_slope, ift);
    pr('  entrance pupil %+.1f mm from M1 along the input chief (+ = downstream: virtual); field at M2 x%.3f; image half %.2f mm\n', ...
       FO.ep_from_m1*1e3, FO.mag_field_at_m2, FO.image_half*1e3);
    pr('  (beam = on-axis width; foot x = across the strip at the edge field; foot y* = on the surface in the fold plane)\n\n');
end
pr('READING: with the stop at M2 the telecentric condition is f3 = M2->M3; the paraxial chief then exits M3 parallel.\n');
fclose(fid);
save(fullfile(P.outdir, [P.tag '_first_order.mat']), 'S');
end

% ===========================================================================
function S = stage_section_(P, FOS)
FO = FOS.FO;
deck = fullfile(P.outdir, [P.tag '_section.in']);
G = tls_section(P, FO, deck);
[fid, pr] = open_(P, 'section');
pr('tma_longslit section -- the zig-zag as three off-axis conic sections, measured in the engine (%s)\n', datestr(now, 'yyyy-mm-dd HH:MM'));
pr('DECK %s (model %d, %d-pt grid, %.0f nm)\n', deck, P.model, P.ngridpts, P.lambda_m*1e9);
pr('  %-3s %8s %9s %9s %9s %8s %8s %9s  %s\n', '', 'AOI', 'R_t mm', 'R_s mm', 'R par mm', 'K', 'theta', 'h mm', 'pole (mm)');
for k = 1:3
    m = G.m(k);
    pr('  %-3s %8.2f %+9.1f %+9.1f %9.1f %8.4f %8.2f %9.1f  %s\n', m.name, P.aoi_deg(k), m.Rt*1e3, m.Rs*1e3, m.R*1e3, m.K, m.theta, m.h*1e3, ...
       mat2str(round(m.pole'*1e4)/10));
end
pr('  slit at %s mm, normal %s; stop = element %d (%s), ApStop offset [%.2f %.2f] mm\n\n', mat2str(round(G.slit.point'*1e4)/10), ...
   mat2str(round(G.slit.normal'*1e4)/1e4), G.stop_elt, G.m(G.stop_elt).name, G.stop_offset*1e3);
macos.init(P.model);
M = tls_measure(P, G, deck);
pr('PER FIELD (strip along x; chief aimed through the M2 pole for each field):\n');
pr('  %7s | %6s | %8s %8s | %9s %9s %8s | %7s %7s | %8s %8s | %8s\n', 'field', 'pass', 'chief', 'spread', 'x mm', 'y mm', 'bow um', ...
   'F/# x', 'F/# y', 'rms um', 'bf um', 'stop um');
for q = 1:numel(M.fields_deg)
    pr('  %+7.3f | %6.3f | %8.4f %8.4f | %+9.3f %+9.3f %+8.1f | %7.3f %7.3f | %8.1f %8.1f | %8.2g\n', M.fields_deg(q), M.pass(q), ...
       M.chief_deg(q), M.spread_deg(q), M.x_m(q)*1e3, M.y_m(q)*1e3, M.bow_um(q), M.fno_x(q), M.fno_y(q), M.rms_um(q), M.bf_um(q), M.stop_miss_m(q)*1e6);
end
pr('  (chief = angle to the slit normal (chx along the slit = cross-track, chy across it = along-track); spread = to the centre chief; F/# = 1/(2 sin u) of the passing rays; rms = as placed;\n');
pr('   bf = at the field''s own best focus; stop = chief miss of the M2 pole)\n');
pr('PLATE SCALE: local %.2f mm, to the edge %.2f mm (spec %.1f); image of the strip %.2f mm (slit %.1f)\n', M.plate_local*1e3, ...
   M.plate_edge*1e3, P.f_m*1e3, (max(M.x_m) - min(M.x_m))*1e3, P.slit_m*1e3);
pr('FOOTPRINTS (passing rays, all fields, section frame):\n');
for k = 1:numel(M.foot)
    pr('  %-3s x %.1f mm (strip) x y %.1f mm (fold), enclosing radius %.1f mm about the pole\n', M.foot(k).name, M.foot(k).x_m*1e3, ...
       M.foot(k).y_m*1e3, M.foot(k).r_m*1e3);
end
pr('CLEARANCE (footprint + %.0f mm mount, every leg vs every body it does not touch; worst first):\n', M.clear.mount_m*1e3);
for i = 1:size(M.clear.table, 1)
    pr('  %-14s vs %-9s %+8.1f mm\n', M.clear.table{i, :});
end
pr('  -> %s, min %+.1f mm\n', tern_(M.clear.pass, 'CLEAR', 'CONFLICT'), M.clear.min_mm);
fclose(fid);
M = rmfield(M, 'paths');
S = struct('P', P, 'G', G, 'M', M, 'deck', deck);
save(fullfile(P.outdir, [P.tag '_section.mat']), 'S');
end

% ===========================================================================
function S = stage_figure_(P, FOS)
%STAGE_FIGURE_  The figure ladder (TLS_FIGURE) from the first-order section: R0 = the paraboloid SEED scored as is, then
%   each rung of P.ladder warm-started from the previous one; every rung emitted as <tag>_R<n>_<name>.in and measured in
%   the engine over the full strip (TLS_MEASURE, both signs of the field: the symmetry check).
[fid, pr] = open_(P, 'figure');
pr('tma_longslit figure -- the ladder on the zig-zag section (%s)\n', datestr(now, 'yyyy-mm-dd HH:MM'));
pr('ROWS (TLS_FIGURE, um): SPOT per ray about the as-placed centroid on the slit x sqrt(254/N); PLATE x %.3g; BOW (strict chief\n', P.w_plate);
pr('  intercept across the slit vs the centre field) x %.3g; TELE %.3g um/rad; CONE hinge outside F/%.2f..%.2f, %.3g um per F/#;\n', P.w_bow, P.w_tel, P.cone_fnum, P.w_cone);
pr('  WD hinge below %.0f mm.  FWHM = spectrometer_score_fwhm (rays (x) 1 px (x) Airy at %.2f um, F/%.2f), px of %.0f um.\n', ...
   P.work_dist_m*1e3, P.score_lambda_m*1e6, P.f_m/P.D_m, P.pixel_m*1e6);
pr('  The seed (theta = AOI, K = -1 on all three) is a SEED: the solve decides the conics.\n\n');
macos.init(P.model);
X = tls_design(P, FOS.FO);
rungs = struct('name', {}, 'X', {}, 'R', {}, 'M', {}, 'G', {}, 'deck', {});
fr0 = fullfile(P.outdir, [P.tag '_figure.mat']);
if P.resume_upto >= 0
    Z = load(fr0);  rungs = Z.S.rungs(1:P.resume_upto + 1);
    for i = 0:P.resume_upto
        if i > 0   % (assert evaluates its message arguments eagerly: P.ladder(0) would be indexed)
            assert(strcmp(rungs(i + 1).name, P.ladder(i).name), 'resume: rung R%d is %s, the ladder says %s', i, rungs(i + 1).name, P.ladder(i).name);
        end
        pr('R%d %s RESUMED from %s (cost %.4e)\n', i, rungs(i + 1).name, fr0, rungs(i + 1).R.cost);
        rung_table_(pr, P, sprintf('R%d %s (resumed)', i, rungs(i + 1).name), rungs(i + 1).X, rungs(i + 1).G, rungs(i + 1).M);
    end
end
for i = (numel(rungs)):numel(P.ladder)
    if i == 0
        nm = 'seed';  Rr = struct('dofs', {{}}, 'cost0', NaN, 'cost', NaN, 'exitflag', NaN, 'iterations', 0, 'evaluations', 0, 'seconds', 0);
    else
        nm = P.ladder(i).name;
        fr = i - 1;  if isfield(P.ladder, 'from') && ~isempty(P.ladder(i).from), fr = P.ladder(i).from; end
        Xw = rungs(fr + 1).X;                                    % rungs(1) = R0, the seed
        pr('RUNG R%d %s: DOFs %s, warm from R%d %s\n', i, nm, strjoin(P.ladder(i).dofs, ' + '), fr, rungs(fr + 1).name);
        mf = P.maxfev;  if isfield(P.ladder, 'maxfev') && ~isempty(P.ladder(i).maxfev), mf = P.ladder(i).maxfev; end
        Pr = P;                                                  % per-rung row weights
        if isfield(P.ladder, 'w_spot_x') && ~isempty(P.ladder(i).w_spot_x), Pr.w_spot_x = P.ladder(i).w_spot_x; end
        if isfield(P.ladder, 'field_wt') && ~isempty(P.ladder(i).field_wt), Pr.solve_field_wt = P.ladder(i).field_wt; end
        if isfield(P.ladder, 'w_tel') && ~isempty(P.ladder(i).w_tel), Pr.w_tel = P.ladder(i).w_tel; end
        if isfield(P.ladder, 'w_off') && ~isempty(P.ladder(i).w_off), Pr.w_off = P.ladder(i).w_off; end
        pr('  row weights: along-slit spot x%g, across-slit x%g, solve fields %s x %s, TELE %g um/rad\n', Pr.w_spot_x, Pr.w_spot_y, ...
           mat2str(round(linspace(0, P.strip_half_deg, 5)*1000)/1000), mat2str(Pr.solve_field_wt), Pr.w_tel);
        [X, Rr] = tls_figure(Pr, Xw, P.ladder(i).dofs, 'maxfev', mf);
        pr('  lsqnonlin LM: exitflag %d, %d iterations, %d evaluations, %.0f s, cost %.4e -> %.4e\n', Rr.exitflag, Rr.iterations, ...
           Rr.evaluations, Rr.seconds, Rr.cost0, Rr.cost);
        cb = 0;  if isfield(Rr.rows, 'cbow'), cb = Rr.rows.cbow; end
        if isfield(Rr.rows, 'off'), pr('  OFF rows (chief - centroid across the slit): cost %.3e (w_off %g)\n', Rr.rows.off, Pr.w_off); end
        pr('  cost by family at the solution: spot %.3e, plate %.3e, plate_y %.3e, bow %.3e, cbow %.3e, tele %.3e, cone %.3e, wd %.3e\n', Rr.rows.spot, ...
           Rr.rows.plate, Rr.rows.plate_y, Rr.rows.bow, cb, Rr.rows.tele, Rr.rows.cone, Rr.rows.wd);
    end
    deck = fullfile(P.outdir, sprintf('%s_R%d_%s.in', P.tag, i, nm));
    G = tls_section(P, X, deck);
    M = tls_measure(P, G, deck);
    rung_table_(pr, P, sprintf('R%d %s', i, nm), X, G, M);
    M = rmfield(M, 'paths');
    rungs(end+1) = struct('name', nm, 'X', X, 'R', Rr, 'M', M, 'G', G, 'deck', deck);   %#ok<AGROW>
    S = struct('P', P, 'rungs', rungs);  save(fullfile(P.outdir, [P.tag '_figure.mat']), 'S');
end
fclose(fid);
end

function rung_table_(pr, P, label, X, G, M)
if ~isfield(M, 'cbow_um'), M.cbow_um = nan(size(M.fields_deg)); end   % records measured before the centroid bow was kept
if ~isfield(M, 'chief_x_mrad'), M.chief_x_mrad = nan(size(M.fields_deg));  M.chief_y_mrad = M.chief_x_mrad; end
pr('  %s\n', label);
pr('    %-3s %9s %9s %7s %9s %8s %9s %9s\n', '', 'R_t mm', 'R_s mm', 'theta', 'R par mm', 'K', 'a4 um', 'a6 um');
for k = 1:3
    m = G.m(k);
    pr('    %-3s %+9.1f %+9.1f %7.2f %9.1f %8.4f %+9.1f %+9.1f\n', m.name, m.Rt*1e3, m.Rs*1e3, m.theta, m.R*1e3, m.K, X.asph(k, 1), X.asph(k, 2));
end
pr('    layout: AOI %s deg, legs %s mm, slit dz %+.3f mm\n', mat2str(round(X.aoi*100)/100), mat2str(round(X.legs*1e4)/10), X.slit_dz*1e3);
pr('    %7s | %5s | %7s %8s %8s | %6s %6s | %6s %6s %5s | %7s | %8s %8s %8s\n', 'field', 'pass', 'chief', 'chx mrad', 'chy mrad', ...
   'F/# x', 'F/# y', 'FWHMx', 'FWHMy', 'EiP', 'rms um', 'x err um', 'bow um', 'cbow um');
for q = 1:numel(M.fields_deg)
    pr('    %+7.3f | %5.3f | %7.4f %+8.4f %+8.4f | %6.3f %6.3f | %6.2f %6.2f %5.2f | %7.2f | %+8.2f %+8.2f %+8.2f\n', M.fields_deg(q), M.pass(q), ...
       M.chief_deg(q), M.chief_x_mrad(q), M.chief_y_mrad(q), M.fno_x(q), M.fno_y(q), M.fwhm_x_px(q), M.fwhm_y_px(q), M.eip(q), M.rms_um(q), ...
       (M.x_m(q) - P.f_m*tand(M.fields_deg(q)))*1e6, (M.y_m(q) - M.y_m(M.fields_deg == 0))*1e6, M.cbow_um(q));
end
cone_ok = all(M.fno_x >= P.cone_fnum(1) & M.fno_y >= P.cone_fnum(1));
pr('    along-track (fold-plane) focal length centre / edge %.2f / %.2f mm\n', M.plate_y_centre*1e3, M.plate_y_edge*1e3);
pr('    plate local / edge %.2f / %.2f mm; worst chief %.3f deg; F/# range x %.3f..%.3f y %.3f..%.3f (%s); M2 footprint %.1f x %.1f mm\n', ...
   M.plate_local*1e3, M.plate_edge*1e3, max(M.chief_deg), min(M.fno_x), max(M.fno_x), min(M.fno_y), max(M.fno_y), ...
   tern_(cone_ok, 'no ray below F/1.7', 'RAYS BELOW the cone bound: FAILED RUNG'), M.foot(2).x_m*1e3, M.foot(2).y_m*1e3);
pr('    footprints M1 %.1f x %.1f, M3 %.1f x %.1f mm; clearance %s min %+.1f mm (%s vs %s); working distance %.1f mm\n\n', ...
   M.foot(1).x_m*1e3, M.foot(1).y_m*1e3, M.foot(3).x_m*1e3, M.foot(3).y_m*1e3, tern_(M.clear.pass, 'CLEAR', 'CONFLICT'), ...
   M.clear.min_mm, M.clear.table{1, 1}, M.clear.table{1, 2}, X.legs(3)*1e3);
end

% ===========================================================================
function S = stage_e2e_(P)
%STAGE_E2E_  End to end on the best rung of the figure record (or P.e2e_rung by name): the rung that CLEARS, keeps every
%   ray inside the cone bound and has the smallest worst-field FWHM, joined to the spectrometer of P.e2e by TLS_E2E, the
%   joined deck's clearance by TLS_CLEARANCE_JOINED.  Two launches (CC 2026-10-07): 'centroid' -- each field's bundle
%   centroid on the slit line, the SLIT-FILLED proxy and the spec's smile -- and 'chief' -- the record's point-source
%   launch, whose smile is the telescope's chief-minus-centroid offset across the slit (reported as the point-source
%   across-slit shift).  Every metric of the table is the slit-filled (centroid-launch) one; the shift sits beside it.
Z = load(fullfile(P.outdir, [P.tag '_figure.mat']));  R = Z.S.rungs;
ok = arrayfun(@(r) r.M.clear.pass && min([r.M.fno_x r.M.fno_y]) >= P.cone_fnum(1), R);
w = arrayfun(@(r) max([r.M.fwhm_x_px r.M.fwhm_y_px]), R);
if ~isempty(P.e2e_rung), j = find(strcmp({R.name}, P.e2e_rung), 1);
else, c = find(ok);  [~, i] = min(w(c));  j = c(i); end
r = R(j);
[fid, pr] = open_(P, 'e2e');
pr('tma_longslit e2e -- rung R%d %s joined to the dyson5 3k Dyson of record (CaF2 240, size:F:240) (%s)\n', j - 1, r.name, datestr(now, 'yyyy-mm-dd HH:MM'));
pr('  rung: clears (%+.1f mm), no ray below F/%.1f, worst telescope FWHM %.2f px\n', r.M.clear.min_mm, P.cone_fnum(1), w(j));
pr('  TWO LAUNCHES: smile (slit-filled) = each field''s CENTROID on the slit line (the spec''s convention); the point-source\n');
pr('  across-slit shift = the smile of the record''s CHIEF-on-the-slit-line launch (the telescope''s chief - centroid spread)\n');
L = struct();
for lm = {'centroid', 'chief'}
    macos.init(P.model);
    L.(lm{1}) = tls_e2e(P, r.deck, 'roll_deg', P.e2e_roll_deg, 'suffix', ['_' P.tag], 'launch', lm{1});
end
E = L.centroid;
pr('  join deck %s (header ApStop = the engine entrance pupil %s mm, two-chief miss %.2g m)\n', E.join_deck, mat2str(round(E.ap_stop'*1e4)/10), E.ep_miss);
for q = 1:numel(E.rows)
    RE = E.rows(q).S.e2e;  RC = L.chief.rows(q).S.e2e;
    macos.init(P.model);
    Cj = tls_clearance_joined(P, fullfile(fileparts(E.join_deck), E.rows(q).S.file));
    E.rows(q).clear = Cj;
    pr('\n  ROLL %d deg: e2e decks %s (centroid) / %s (chief)\n', E.rows(q).roll_deg, E.rows(q).S.file, L.chief.rows(q).S.file);
    um = P.pixel_m*1e6;  sp = P.spec_paper;  sj = P.spec_joe;
    pr('    %-34s %8s %9s | %9s %11s\n', '', 'px', 'um', 'Joe (px)', 'paper (um)');
    row = @(nm, v, joe, pap) pr('    %-34s %8.3f %9.2f | %9s %11s %s\n', nm, v, v*um, lim_(joe), lim_(pap), pfu_(v, joe, v*um, pap));
    row('smile (slit-filled)', RE.smile_max, sj.smile_px, sp.smile_um);
    pr('    %-34s %8.3f %9.2f |   (stated beside the smile; not a requirement)\n', 'point-source across-slit shift', RC.smile_max, RC.smile_max*um);
    row('keystone', RE.keystone_max, sj.keystone_px, sp.keystone_um);
    row('CRF (worst)', RE.crf_max, sj.crf_px, sp.crf_um);
    row('SRF (worst)', RE.srf_max, sj.srf_px(2), sp.srf_um);
    row('ARF (telescope FWHM y)', max(r.M.fwhm_y_px), Inf, sp.arf_um);
    pr('    (SRF floor: rect(%d-px slit) (x) rect(1 px) = %.3f px for a PERFECT spectrometer -- Joe''s 1.5-2.0 is a slit width)\n', ...
       P.e2e_slit_px, spectrometer_score_fwhm(zeros(1, 1000), P.e2e_slit_px, 0));
    pr('    %-34s %8.3f     | > %.2f\n', 'energy in a pixel (min)', RE.ee_min, P.spec_joe.eip);
    pr('    %-34s %8.3f\n', 'grating admits (min)', min(RE.pass_frac(:)));
    pr('    %-34s %+8.1f mm  (%s vs %s)  %s\n', 'clearance, joined', Cj.min_mm, Cj.table{1, 1}, Cj.table{1, 2}, tern_(Cj.pass, 'CLEAR', 'CONFLICT'));
    pr('    (chief launch, for the record: smile %.3f / keystone %.3f / CRF %.3f / SRF %.3f px)\n', RC.smile_max, RC.keystone_max, RC.crf_max, RC.srf_max);
end
fclose(fid);
S = struct('P', P, 'rung', j - 1, 'rung_name', r.name, 'deck', r.deck, 'E', E, 'E_chief', L.chief);
save(fullfile(P.outdir, [P.tag '_e2e.mat']), 'S');
end

function s = pf_(v, joe, paper)
s = sprintf('(Joe %s, paper %s)', tern_(v < joe, 'PASS', 'fail'), tern_(v < paper, 'PASS', 'fail'));
end

function s = pfu_(vpx, joe_px, vum, paper_um)
% Joe in our pixels, the paper in micrometres
if isinf(joe_px), sj = '--'; else, sj = tern_(vpx <= joe_px, 'PASS', 'fail'); end
s = sprintf('(Joe %s, paper %s)', sj, tern_(vum < paper_um, 'PASS', 'fail'));
end

function s = lim_(v)
if isinf(v), s = '--'; else, s = sprintf('< %.4g', v); end
end

function S = load_or_run_(P, stage, fn)
f = fullfile(P.outdir, sprintf('%s_%s.mat', P.tag, stage));
if exist(f, 'file'), Z = load(f);  S = Z.S; else, S = fn(); end
end

% ===========================================================================
function [fid, pr] = open_(P, stage)
fid = fopen(fullfile(P.outdir, sprintf('%s_%s.txt', P.tag, stage)), 'w');
pr = @(varargin) dual_(fid, varargin{:});
end
function dual_(fid, varargin), fprintf(varargin{:});  fprintf(fid, varargin{:}); end
function s = stopname_(st)
if ischar(st) || isstring(st), s = char(st); else, s = sprintf('%.2f of M1->M2', st); end
end
function s = tern_(c, a, b), if c, s = a; else, s = b; end, end
