%S8_CORONAGRAPH  e2e stage 8: the CTB coronagraph behind the telescope, folded into the shroud.
%
%   Consumes s2_instrument.in (the telescope with its imaging relay) and
%   the CTB template's parameter block (templates/30_instruments/bench_ctb/
%   example_ctb.m, read at run time, the template unchanged).  Produces the
%   coronagraph channel the mmacos manual's chapter 2 drew unfolded ("case
%   C": OAP1 matched to the telescope's f/#, DM1 at the telescope stop's
%   image) and then FOLDS it into the telescope's own launch envelope under
%   the design drivers (PLAN_CONSOLIDATION 3a, Dave 2026-10-28):
%
%     - every fold at under 15 deg of incidence (the coronagraph AOI rule);
%     - a numeric clearance from the shroud, and from every telescope body
%       and beam (design/src/deck_clearance, the exact signed leg-to-body
%       measure; margins are NUMBERS);
%     - a small relay (the relay scale is the LAST knob tried, 1 preferred).
%
%   [1] the telescope's exit state and its own hardware union (shroud_deck):
%       the gate is that the coronagraph adds NOTHING to the diameter the
%       telescope already needs -- the instrument lives inside the envelope.
%   [2] case C, unfolded, measured: rays, wavefront, the DM walk at the
%       dark-zone edge and at 10", AOI, and how far outside the envelope it
%       reaches (the chapter's statement, now a number).
%   [3] the finder: one added near-normal fold on the converging feed (its
%       station before the focus and its out-of-plane tilt), the accordion
%       in the vertical plane through the feed, both turn senses, and the
%       relay scale; every candidate scored on the centre field for shroud
%       margin, clearance and height; the pick = the first relay scale at
%       which a candidate clears, then the largest shroud margin.
%   [4] the pick re-measured in full (centre + the 10" ring, every leg);
%       the gates, each a number: rays and wavefront equal to the unfolded
%       train (a flat fold is an isometry), the walk equal, every AOI
%       under the rule, shroud margin >= 0, clearance >= P.coro.clear_min_m,
%       the CTB no lower than the telescope's bench.
%   [5] products: s8_C_ctb.in / s8_C_full.in (unfolded bench / spliced),
%       s8_fold_ctb.in / s8_fold_full.in (folded), s8_views.png (the
%       folded train), s8_bench.png (the folded bench alone), s8_shroud.png
%       (end-on), s8_candidates.csv, s8_report.txt, s8_run.mat.
%
%   What this stage does NOT do: the dark zone.  The CTB's diffraction
%   decks (ctb_dcr.in) are the bench's own; a coupled telescope+CTB
%   diffraction model is the next beat (ruled AFTER the fold, 2026-10-28),
%   and a relay scale other than 1 changes the physical lambda/D at the
%   masks (reported here as lamD_fpm_um) which that model must carry.
%
%   See also E2E_PARAMS, CTB_FOLD_BUILD, CTB_FOLD_MEASURE, DECK_CLEARANCE, SHROUD_DECK.

exdir = fileparts(mfilename('fullpath'));
run(fullfile(exdir, '..', '..', '..', 'mmacos_setup.m'));
addpath(fullfile(exdir, '..', '..', '..', 'design', 'src'));
P = e2e_params();  Q = P.coro;
cd(exdir);
t0 = tic;  L = {};
say = @(varargin) fprintf('%s\n', sprintf(varargin{:}));

tel = fullfile(exdir, 's2_instrument.in');
assert(isfile(tel), 's8_coronagraph: run s2_instrument first (%s missing)', tel);
Pc = ctb_template_params();                                   % the CTB's own parameter block (mm), read from its file
mm = 1e-3;                                                     %#ok<NASGU>

%% -- [1] the telescope's exit state and its own envelope -----------------
say('\n==================== e2e s8 -- the CTB coronagraph folded into the shroud');
Pd = struct('shroud_D_m', 0, 'model', P.model_size);
sh = shroud_deck(tel, Pd, 'model', P.model_size);
Rgate = sh.D/2 + Q.shroud_grow_m;
say('[1] telescope alone: hardware union %.4f m (%.3f x D), train length %.3f m -> gate radius %.4f m', ...
    sh.D, sh.D/P.D_m, sh.len, Rgate);
holes = struct('M1', max(P.hole_min_r_m, 0.232));

%% -- [2] case C, unfolded ------------------------------------------------
if isnan(Q.dm1_m)
    % the telescope stop's image after OAP1 (the chapter's measure): where the
    % chief rays of a 3" ring leaving OAP1 cross the centre field's chief
    Rm = ctb_fold_build(tel, fullfile(tempdir, 's8_match'), Pc, struct('model', P.model_size));
    Q.dm1_m = round(ctb_stop_image(Rm, 3), 4);
    delete(Rm.full_in);  delete(Rm.bench_in);
    say('    the telescope stop images %.4f m after OAP1 (measured on a 3" ring): DM1 goes there', Q.dm1_m);
end
Rc = ctb_fold_build(tel, 's8_C', Pc, struct('dm1_m', Q.dm1_m, 'model', P.model_size));
Mc = ctb_fold_measure(Rc, 'shroud_D_m', 2*Rgate, 'holes', holes, 'leg_from', 3, ...
                      'ring_as', Q.ring_as, 'clear_as', Q.clear_as, 'quiet', true);
Me = ctb_fold_measure(Rc, 'shroud_D_m', 2*Rgate, 'holes', holes, 'leg_from', 3, ...
                      'ring_as', Q.edge_as, 'clear_as', Q.clear_as, 'quiet', true);
say('[2] case C unfolded: f/%.2f, OAP1 f %.4f m, DM1 %.4f m after it; %d of %d rays at the FPA, %.4f waves', ...
    Rc.fno, Rc.fOAP1, Rc.K.dm1_m, Mc.rays(end), Mc.nrays, Mc.wfe_waves);
say('    CTB bodies reach r = %.3f m: %.3f m OUTSIDE the gate (the chapter''s "folds into the shroud", as a number)', ...
    Mc.shroud.ctb_max_r, Mc.shroud.ctb_max_r - Rgate);
say('    walk at %.3f" (the %g lambda/D edge): DM1 %.3f mm, DM2 %.3f mm (%.3f r_pup); at %g": DM2 %.3f mm (%.3f r_pup)', ...
    Q.edge_as, Q.edge_lamD, Me.walk.dm1_m*1e3, Me.walk.dm2_m*1e3, Me.walk.dm2_rpup, Q.ring_as, Mc.walk.dm2_m*1e3, Mc.walk.dm2_rpup);

%% -- [3] the finder -------------------------------------------------------
say('[3] finder: fold station a (m before the focus) x tilt (deg, AOI = tilt/2) x turn sense, relay scale %s', mat2str(Q.scales));
rows = {};  n = 0;
for sc = Q.scales
  for a = Q.fold_m
    for tdeg = Q.tilt_deg
      for sg = Q.signs
       for fl = Q.flip_at
        n = n + 1;
        K = struct('fold0_m', a, 'fold0_tilt_deg', tdeg, 'plane_deg', Q.plane_deg, 'sign0', sg, 'flip_at', fl, ...
                   'relay_scale', sc, 'dm1_m', Q.dm1_m, 'model', P.model_size);
        R = ctb_fold_build(tel, fullfile(tempdir, sprintf('s8c%03d', n)), Pc, K);
        nF = macos.load_rx(R.full_in);  s = macos.trace(nF);  ri = macos.get_ray_info(s.nRays);
        okF = nnz(ri.ok_pass & ri.ok_trace);
        C = deck_clearance(R.full_in, 'model', P.model_size, 'holes', holes, 'elts', 3:nF, 'addon_from', R.n_base, 'quiet', true);
        ctb = [C.bodies.elt] > R.n_base;
        rmax = max([C.bodies(ctb).ext]);  zmin = min([C.bodies(ctb).z]);  zmax = max([C.bodies(ctb).z]);
        rows(end+1,:) = {n, sc, a, tdeg, sg, fl, okF, rmax, Rgate - rmax, C.min, C.min_addon, zmin, zmax, ...
                         C.worst.leg_name, C.worst.body_name};                               %#ok<AGROW>
        delete(R.full_in);  delete(R.bench_in);
       end
      end
    end
  end
  Tc = cell2table(rows, 'VariableNames', {'n','scale','fold_m','tilt_deg','sign','flip_at','rays','rmax','shroud_margin','clear_m','clear_addon_m','zmin','zmax','worst_leg','worst_body'});
  ok = Tc.scale == sc & Tc.rays >= Mc.rays(end) & Tc.shroud_margin >= Q.shroud_min_m & Tc.clear_m >= Q.clear_min_m ...
       & Tc.clear_addon_m >= Q.clear_min_m & Tc.zmin >= Q.z_floor_m - 1e-3;      % a millimetre: a footprint's mean sits microns under its plane
  if any(ok), break, end
end
writetable(Tc, fullfile(exdir, 's8_candidates.csv'));                 % every candidate, for the record and for a widened scan
if ~any(ok)
    f1 = Tc.rays >= Mc.rays(end);  f2 = Tc.shroud_margin >= Q.shroud_min_m;  f3 = Tc.clear_m >= Q.clear_min_m;
    f4 = Tc.clear_addon_m >= Q.clear_min_m;  f5 = Tc.zmin >= Q.z_floor_m - 1e-3;
    say('    no candidate clears: rays %d, shroud %d, clearance %d, own %d, floor %d of %d pass each gate', nnz(f1), nnz(f2), nnz(f3), nnz(f4), nnz(f5), height(Tc));
    error('s8_coronagraph: no candidate clears at any relay scale -- widen P.coro (see s8_candidates.csv)');
end
% the pick: among the clearing candidates at this relay scale, the most
% compact aft (the smallest z extent), ties to the larger shroud margin
cand = Tc(ok, :);  [~, ib] = sortrows([cand.zmax - cand.zmin, -cand.shroud_margin]);  pick = cand(ib(1), :);
say('    %d candidates; %d clear at relay scale %.2f; pick #%d: fold %.2f m before the focus, tilt %g deg, sense %+d, pattern inverted after turn %d', ...
    height(Tc), nnz(ok), pick.scale, pick.n, pick.fold_m, pick.tilt_deg, pick.sign, pick.flip_at);
rp = Q.pick;
same = pick.fold_m == rp.fold_m && pick.tilt_deg == rp.tilt_deg && pick.sign == rp.sign && pick.flip_at == rp.flip_at && pick.scale == rp.scale;
say('    the recorded pick in e2e_params (fold %.2f, tilt %g, sense %+d, inversion %d, scale %.2f) is %s', ...
    rp.fold_m, rp.tilt_deg, rp.sign, rp.flip_at, rp.scale, tern_(same, 'the same', 'DIFFERENT -- update P.coro.pick and the test'));

%% -- [4] the pick, in full --------------------------------------------------
Kp = struct('fold0_m', pick.fold_m, 'fold0_tilt_deg', pick.tilt_deg, 'plane_deg', Q.plane_deg, ...
            'sign0', pick.sign, 'flip_at', pick.flip_at, 'relay_scale', pick.scale, 'dm1_m', Q.dm1_m, 'model', P.model_size);
Rf = ctb_fold_build(tel, 's8_fold', Pc, Kp);
Mf = ctb_fold_measure(Rf, 'shroud_D_m', 2*Rgate, 'holes', holes, 'leg_from', 3, 'ring_as', Q.ring_as, 'clear_as', Q.clear_as, 'quiet', true);
Mfe = ctb_fold_measure(Rf, 'shroud_D_m', 2*Rgate, 'holes', holes, 'leg_from', 3, 'ring_as', Q.edge_as, 'clear_as', Q.clear_as, 'quiet', true);
aoi_max = max([Mf.aoi.aoi_deg]);
G = struct();
G.rays   = struct('v', Mf.rays(end) - Mc.rays(end), 'ok', Mf.rays(end) == Mc.rays(end), 'say', sprintf('rays at the FPA %d (unfolded %d)', Mf.rays(end), Mc.rays(end)));
G.wfe    = struct('v', abs(Mf.wfe_waves - Mc.wfe_waves), 'ok', abs(Mf.wfe_waves - Mc.wfe_waves) <= Q.wfe_tol_waves, 'say', sprintf('FPA wavefront %.5f waves (unfolded %.5f)', Mf.wfe_waves, Mc.wfe_waves));
G.walk   = struct('v', abs(Mf.walk.dm2_m - Mc.walk.dm2_m), 'ok', abs(Mf.walk.dm2_m - Mc.walk.dm2_m) <= Q.walk_tol_m, 'say', sprintf('DM2 walk at %g" %.3f mm (unfolded %.3f)', Q.ring_as, Mf.walk.dm2_m*1e3, Mc.walk.dm2_m*1e3));
G.aoi    = struct('v', aoi_max, 'ok', aoi_max < Q.aoi_max_deg, 'say', sprintf('largest AOI %.2f deg (rule < %g)', aoi_max, Q.aoi_max_deg));
G.shroud = struct('v', Rgate - Mf.shroud.ctb_max_r, 'ok', Rgate - Mf.shroud.ctb_max_r >= Q.shroud_min_m, 'say', sprintf('shroud margin %+.3f m (CTB to r = %.3f, gate %.3f, floor %.2f; the train''s union %.3f m)', Rgate - Mf.shroud.ctb_max_r, Mf.shroud.ctb_max_r, Rgate, Q.shroud_min_m, Mf.shroud.D));
G.clear  = struct('v', Mf.clear.min, 'ok', Mf.clear.min >= Q.clear_min_m, 'say', sprintf('clearance CTB vs telescope %+.1f mm (leg from %s vs %s; gate %.0f mm)', Mf.clear.min*1e3, Mf.clear.worst.leg_name, Mf.clear.worst.body_name, Q.clear_min_m*1e3));
G.own    = struct('v', Mf.clear.min_addon, 'ok', Mf.clear.min_addon >= Q.clear_min_m, 'say', sprintf('clearance within the CTB %+.1f mm (leg from %s vs %s; bodies and legs over the %g" field); the telescope''s own worst pair %+.1f mm (%s vs %s), as built', Mf.clear.min_addon*1e3, Mf.clear.worst_addon.leg_name, Mf.clear.worst_addon.body_name, Q.clear_as, Mf.clear.min_base*1e3, Mf.clear.worst_base.leg_name, Mf.clear.worst_base.body_name));
G.zfloor = struct('v', Mf.zmin_ctb - Q.z_floor_m, 'ok', Mf.zmin_ctb >= Q.z_floor_m - 1e-3, 'say', sprintf('CTB lowest body z %.3f m (floor %.3f, the bench; keep-out %.3f behind M1: margin %+.3f)', Mf.zmin_ctb, Q.z_floor_m, Q.keepout_m, Mf.zmin_ctb - Q.keepout_m));
gn = fieldnames(G);  allok = true;
say('[4] the folded train:');
for k = 1:numel(gn), say('    [%s] %s', tern_(G.(gn{k}).ok, 'PASS', 'FAIL'), G.(gn{k}).say);  allok = allok && G.(gn{k}).ok; end
say('    added fold F0: %.2f m before the focus, AOI %.1f deg; the accordion in the vertical plane; relay scale %.2f (lambda/D at the FPM %.1f um, case C %.1f)', ...
    pick.fold_m, pick.tilt_deg/2, pick.scale, Rf.lamD_fpm_um, Rc.lamD_fpm_um);
say('    CTB z from %.3f to %.3f m behind M1; the train''s length %.3f m (telescope alone %.3f)', ...
    Mf.zmin_ctb, max([Mf.clear.bodies([Mf.clear.bodies.elt] > Rf.n_base).z]), Mf.shroud.len, sh.len);

%% -- [5] products -------------------------------------------------------------
macos.init(P.model_size);  macos.load_rx(Rf.full_in);
try
    fv = macos.view_std('title', sprintf('e2e s8: the CTB folded behind the telescope (relay x%.2f)', pick.scale), 'visible', false);
    print(fv, fullfile(exdir, 's8_views.png'), '-dpng', '-r150');  close(fv);
catch ME, say('    (views: %s)', ME.message);
end
try
    macos.load_rx(Rf.bench_in);                                % the folded bench alone, fed at the telescope's focus
    fb = macos.view_std('title', 'e2e s8: the folded CTB alone (F0 at the feed, the accordion in the vertical plane)', 'visible', false);
    print(fb, fullfile(exdir, 's8_bench.png'), '-dpng', '-r150');  close(fb);
    macos.load_rx(Rf.full_in);
catch ME, say('    (bench view: %s)', ME.message);
end
try
    shroud_deck(Rf.full_in, struct('shroud_D_m', 2*Rgate, 'model', P.model_size), 'model', P.model_size, ...
                'png', fullfile(exdir, 's8_shroud.png'), 'labels', {'telescope + CTB folded'});
catch ME, say('    (shroud figure: %s)', ME.message);
end
rep = {};
rep{end+1} = sprintf('e2e s8 -- the CTB coronagraph behind the telescope, folded into the shroud (%s)', datestr(now, 'yyyy-mm-dd'));
rep{end+1} = sprintf('telescope: %s; hardware union %.4f m = %.3f x D, train %.3f m; gate radius %.4f m (+%.3f m growth allowed)', 's2_instrument.in', sh.D, sh.D/P.D_m, sh.len, Rgate, Q.shroud_grow_m);
rep{end+1} = sprintf('case C unfolded: f/%.2f, OAP1 f %.4f m, DM1 %.4f m; %d/%d rays, %.4f waves; CTB to r = %.3f m (%+.3f m vs the gate)', Rc.fno, Rc.fOAP1, Rc.K.dm1_m, Mc.rays(end), Mc.nrays, Mc.wfe_waves, Mc.shroud.ctb_max_r, Rgate - Mc.shroud.ctb_max_r);
rep{end+1} = sprintf('   walk: %.3f" DM1 %.4f mm DM2 %.4f mm (%.4f r_pup); %g" DM1 %.4f mm DM2 %.4f mm (%.4f r_pup); rays DM2 %d..%d FPA %d..%d', Q.edge_as, Me.walk.dm1_m*1e3, Me.walk.dm2_m*1e3, Me.walk.dm2_rpup, Q.ring_as, Mc.walk.dm1_m*1e3, Mc.walk.dm2_m*1e3, Mc.walk.dm2_rpup, Mc.walk.rays_dm2, Mc.walk.rays_fpa);
rep{end+1} = sprintf('finder: %d candidates (scales %s, fold %s m, tilt %s deg, senses %s, pattern inversion after turn %s); %d clear at scale %.2f', height(Tc), mat2str(Q.scales), mat2str(Q.fold_m), mat2str(Q.tilt_deg), mat2str(Q.signs), mat2str(Q.flip_at), nnz(ok), pick.scale);
rep{end+1} = sprintf('pick #%d: F0 %.2f m before the focus, tilt %g deg (AOI %.1f), sense %+d, inversion after turn %d, relay scale %.2f; lambda/D at the FPM %.2f um (case C %.2f)', pick.n, pick.fold_m, pick.tilt_deg, pick.tilt_deg/2, pick.sign, pick.flip_at, pick.scale, Rf.lamD_fpm_um, Rc.lamD_fpm_um);
for k = 1:numel(gn), rep{end+1} = sprintf('   [%s] %s', tern_(G.(gn{k}).ok, 'PASS', 'FAIL'), G.(gn{k}).say); end
rep{end+1} = sprintf('   walk folded: %.3f" DM2 %.4f mm (%.4f r_pup); %g" DM2 %.4f mm (%.4f r_pup)', Q.edge_as, Mfe.walk.dm2_m*1e3, Mfe.walk.dm2_rpup, Q.ring_as, Mf.walk.dm2_m*1e3, Mf.walk.dm2_rpup);
rep{end+1} = '   AOI per CTB element (deg):';
for k = 1:numel(Mf.aoi), rep{end+1} = sprintf('      %-10s %.3f', Mf.aoi(k).name, Mf.aoi(k).aoi_deg); end
rep{end+1} = '   clearance, worst twelve (mm, signed; leg from -> body):';
for k = 1:min(12, numel(Mf.clear.table)), rep{end+1} = sprintf('      %-10s -> %-10s %+9.1f', Mf.clear.table(k).leg_name, Mf.clear.table(k).body_name, Mf.clear.table(k).d*1e3); end
rep{end+1} = '   the candidates:';
rep{end+1} = '      n  scale  fold_m  tilt  sign  flip  rays   rmax   margin   clear_mm  own_mm   zmin   zmax';
for k = 1:height(Tc)
    rep{end+1} = sprintf('   %4d  %5.2f  %6.2f  %4g  %+d   %2d   %4d  %6.3f  %+7.3f  %+9.1f %+8.1f  %6.3f %6.3f', Tc.n(k), Tc.scale(k), Tc.fold_m(k), Tc.tilt_deg(k), Tc.sign(k), Tc.flip_at(k), Tc.rays(k), Tc.rmax(k), Tc.shroud_margin(k), Tc.clear_m(k)*1e3, Tc.clear_addon_m(k)*1e3, Tc.zmin(k), Tc.zmax(k));
end
rep{end+1} = sprintf('products: s8_C_ctb.in s8_C_full.in (unfolded), s8_fold_ctb.in s8_fold_full.in (folded), s8_views.png, s8_bench.png, s8_shroud.png, s8_candidates.csv, s8_run.mat; %.0f s', toc(t0));
rep{end+1} = 'not measured here: the dark zone (the coupled diffraction model is the next beat; a flat fold is an isometry, the geometric train above is the gate)';
fid = fopen(fullfile(exdir, 's8_report.txt'), 'w');  fprintf(fid, '%s\n', rep{:});  fclose(fid);
save(fullfile(exdir, 's8_run.mat'), 'Rc', 'Mc', 'Me', 'Rf', 'Mf', 'Mfe', 'Tc', 'pick', 'G', 'sh', 'Rgate', 'Q');
say('[5] report s8_report.txt, figures s8_views.png s8_shroud.png, s8_run.mat; %s; %.0f s', tern_(allok, 'ALL GATES PASS', 'A GATE FAILED'), toc(t0));

function s = tern_(c, a, b)
    if c, s = a; else, s = b; end
end
