function OUT = offset_imager(over)
%OFFSET_IMAGER  Wide-field imager with an offset field -- the template.
%
%   OUT = OFFSET_IMAGER() runs the five-stage design flow at the default
%   parameter set (the rodgers3 challenge instance).  OUT =
%   OFFSET_IMAGER(OVER) applies parameter overrides first (see
%   OFFSET_IMAGER_PARAMS) -- that is how a different instrument is run.
%
%   THE STORY (each stage emits a layout figure, a dense WFE-vs-field
%   map, and a report section; the running ladder is the product):
%
%     S1  Coaxial three-mirror imager solved AT THE ON-AXIS BOX:
%         first-order seed (EFL exact, Petzval = 0), then symmetric
%         conics + aspheres by damped Gauss-Newton on the strict WFE
%         over the solve grid.
%     S2  THE SAME MIRRORS evaluated at the OFFSET box -- FPA tilt/focus
%         refit only.  The "disaster map" is deliberate pedagogy: this
%         is what pushing the field off axis costs.
%     S3  Re-solve the symmetric surfaces AT the offset field (the bias
%         doctrine: solve at the used field; expect oblate-class conic
%         flips).
%     S4  Open mirror tilts/decenters (+ radii stay open, EFL held by
%         the closure) under the constraint set: exit-beam direction and
%         beam/mirror clearances become PASS/FAIL gates.
%     S5  Open Zernike departures (BornWolf, P.zern_modes per mirror,
%         REPLACING the aspheres; lMon frozen at the traced footprints;
%         power pinned to the radii, tilt to the pointing -- the solve
%         doctrine).  Seeded by least-squares refit of the S4 asphere
%         sag onto the S5 basis.
%
%   Metric, stated once and quoted with every number: strict RMS WFE
%   (design/src strict_refs), reference sphere centred on the SPOT
%   CENTROID on the stage's own FROZEN focal plane, anchored at the
%   exit pupil, piston-only removal; the headline statistic per stage is
%   the MAXIMUM of the dense map over the box (Mike Rodgers' statistic,
%   decoded in challenges/rodgers3).  Solve set != scoring set.
%
%   Artifacts (in P.outdir, default = this directory): per-stage decks
%   <tag>_s*.in, figures <tag>_s*_{layout,map}.png, the assembled report
%   <tag>_REPORT.md, and <tag>_run.mat with every stage's design struct,
%   map, gates and history.
%
%   See also OFFSET_IMAGER_PARAMS, OI_SEED, OI_SOLVE, OI_SCORE, OI_GATES.

    arguments
        over struct = struct()
    end
    here = fileparts(mfilename('fullpath'));
    run(fullfile(here,'..','..','..','mmacos_setup.m'));
    addpath(here);

    P = offset_imager_params(over);
    if isempty(P.outdir), P.outdir = here; end
    if ~exist(P.outdir,'dir'), mkdir(P.outdir); end
    tag = fullfile(P.outdir, P.tag);

    macos.init(P.model);

    rpt = fopen([tag '_REPORT.md'], 'w');
    cr = onCleanup(@() fclose(rpt));
    hdr_(rpt, P);

    OUT = struct('P',P, 'when',datestr(now,31)); %#ok<TNOW1,DATST>
    ladder = struct('stage',{},'map_max_nm',{},'map_avg_nm',{},'map_std_nm',{});

    % ================= S1: coaxial solve, on-axis box ======================
    if any(P.stages == 1)
        stage_banner_('S1  coaxial symmetric-asphere solve, on-axis box');
        X = oi_seed(P);
        X.eliminate = 'R2R3';        % S1: EFL + Petzval=0 identities
        [X, hist1] = oi_solve(X, P, 'S1', 'offset', 0);
        [X, G, fo] = close_refit_(X, P, 0);
        [OUT, ladder] = stage_out_(OUT, ladder, rpt, P, tag, 's1', X, G, fo, ...
            0, 'S1 coaxial, on-axis box', hist1, []);
    end

    % ================= S2: same mirrors at the offset box ===================
    if any(P.stages == 2)
        stage_banner_('S2  the SAME mirrors at the offset box -- FPA refit only');
        X.fpa_refit = [0 0];
        X = pose_stop_once_(X, P);   % construct at the offset, then freeze
        [X, hist2] = oi_solve(X, P, 'fpa');
        [X, G, fo] = close_refit_(X, P, P.offset_deg);
        [OUT, ladder] = stage_out_(OUT, ladder, rpt, P, tag, 's2', X, G, fo, ...
            P.offset_deg, 'S2 offset box, FPA tilt/focus refit only', hist2, []);
        cost = ladder(end).map_max_nm / max(ladder(1).map_max_nm, eps);
        fprintf(rpt, ['\n**The cost of the offset:** map max grows %.0fx ' ...
                '(%.0f -> %.0f nm) when the box moves %g%c off axis with ' ...
                'nothing but the FPA allowed to follow.\n'], ...
                cost, ladder(1).map_max_nm, ladder(end).map_max_nm, ...
                P.offset_deg, char(176));
        OUT.s2.cost_of_offset = cost;
    end

    % ================= S3: re-solve symmetric surfaces at the offset ========
    if any(P.stages == 3)
        stage_banner_('S3  re-solve symmetric aspheres/conics AT the offset field');
        K0 = X.K;
        % TWO-CANDIDATE START.  At a large offset the on-axis-optimized
        % aspheres are poison (22 deg: the S1-shape start scores 142 um
        % and the solve crawls; fresh Petzval-flat spheres score 25.6 um
        % and reach single-digit um in 8 iterations) -- but at a mild
        % offset the carry is the better basin.  Score both at the
        % offset and start from the winner; re-solving at the used
        % field means solving THERE, from the better of the two starts.
        Xc = X;  Xc.fpa_refit = [0 0];  Xc.eliminate = 'R2R3';
        Xf = Xc;
        Xs = oi_seed(P);
        Xf.K = Xs.K;  Xf.asph = Xs.asph;  Xf.R = Xs.R;
        qc = start_qmean_(Xc, P);
        qf = start_qmean_(Xf, P);
        fprintf('  S3 start candidates: carry %.1f nm, fresh seed %.1f nm -> %s\n', ...
                qc, qf, tern_(qf < qc, 'fresh', 'carry'));
        if qf < qc, X = Xf; else, X = Xc; end
        X = pose_stop_once_(X, P);   % re-construct at S3 entry, then free
        [X, hist3] = oi_solve(X, P, 'S3');
        [X, G, fo] = close_refit_(X, P, P.offset_deg);
        [OUT, ladder] = stage_out_(OUT, ladder, rpt, P, tag, 's3', X, G, fo, ...
            P.offset_deg, 'S3 symmetric surfaces re-solved at the offset box', hist3, []);
        fprintf(rpt, ['\nConic migration under the bias doctrine ' ...
                '(solve at the used field): K = [%.4g %.4g %.4g] -> ' ...
                '[%.4g %.4g %.4g].\n'], K0, X.K);
    end

    % ================= S4: tilts/decenters + radii, constraints ==============
    if any(P.stages == 4)
        stage_banner_('S4  open tilt/decenter + radii under the constraint set');
        X.eliminate = 'R3';          % S4+: Petzval released, EFL still exact
        walls = @(Xc, Gc) wall_check_(Xc, Gc, P);
        [X, hist4] = oi_solve(X, P, 'S4', 'walls', walls, 'clear', true);
        [X, G, fo] = close_refit_(X, P, P.offset_deg);
        gt4 = oi_gates(X, G, P, P.offset_deg);
        [OUT, ladder] = stage_out_(OUT, ladder, rpt, P, tag, 's4', X, G, fo, ...
            P.offset_deg, 'S4 + mirror tilt/decenter + radii', hist4, gt4);
    end

    % ================= S5: Zernike departures ================================
    if any(P.stages == 5)
        stage_banner_('S5  open Zernike departures (BornWolf, lMon frozen)');
        X = oi_zern_seed(X, P);
        walls = @(Xc, Gc) wall_check_(Xc, Gc, P);
        [X, hist5] = oi_solve(X, P, 'S5', 'walls', walls, 'clear', true);
        [X, G, fo] = close_refit_(X, P, P.offset_deg);
        gt5 = oi_gates(X, G, P, P.offset_deg);
        [OUT, ladder] = stage_out_(OUT, ladder, rpt, P, tag, 's5', X, G, fo, ...
            P.offset_deg, 'S5 + Zernike departures (aspheres replaced)', hist5, gt5);
    end

    % ================= the ladder ============================================
    fprintf(rpt, '\n## The ladder\n\n');
    fprintf(rpt, '| stage | map max (nm) | map avg | map std |\n|---|---|---|---|\n');
    fprintf('\n  LADDER (dense %dx%d map max / avg / std, strict centroid nm)\n', ...
            P.map_n, P.map_n);
    for i = 1:numel(ladder)
        fprintf(rpt, '| %s | %.1f | %.1f | %.1f |\n', ladder(i).stage, ...
                ladder(i).map_max_nm, ladder(i).map_avg_nm, ladder(i).map_std_nm);
        fprintf('   %-4s %10.1f %10.1f %10.1f\n', ladder(i).stage, ...
                ladder(i).map_max_nm, ladder(i).map_avg_nm, ladder(i).map_std_nm);
    end
    OUT.ladder = ladder;
    save([tag '_run.mat'], 'OUT');
    fprintf('\nsaved %s_run.mat + %s_REPORT.md\n', tag, tag);
end

% =========================================================================
function [X, G, fo] = close_refit_(X, P, offset)
%CLOSE_REFIT_  Closure + the stage's FPA refit deltas -> final pose.
    [X, G, fo] = oi_close(X, P, 'offset_deg', offset);
    X.fpa = oi_apply_fpa(X);
    G.fpa = X.fpa;
end

function q = start_qmean_(X, P)
%START_QMEAN_  Quadratic-mean strict WFE of a candidate S3 start over
%   the solve set at the offset (fast path).
    try
        [Xc, G] = oi_close(X, P, 'offset_deg', P.offset_deg);
        Xc.fpa = oi_apply_fpa(Xc);  G.fpa = Xc.fpa;
        D = fill_(Xc, P);
        if isfield(P,'solve_sampling') && ~isempty(P.solve_sampling)
            D.sampling = P.solve_sampling;
        end
        sc = oi_score(oi_deck(D), G, oi_fieldset(P, P.offset_deg, P.nsolve), ...
                      'anchor','center');
        w = sc.wfe_cen_nm;
        q = sqrt(mean(w(isfinite(w)).^2));
        if ~isfinite(q), q = 1e9; end
    catch
        q = 1e9;
    end
end

function s = tern_(c,a,b), if c, s=a; else, s=b; end, end

function X = pose_stop_once_(X, P)
%POSE_STOP_ONCE_  Construct the offset stop pose ONCE at stage entry
%   (EP construction + the exit-pointing secant when P.exit_dir is
%   pinned), then FREEZE it: from here on the stop decenter is an
%   explicit solve variable driven by the exit residual row.
%
%   BOUNDED (2026-08-21): the EP construction intersects two probe
%   chiefs, and for some surface states that crossing is
%   ill-conditioned (t4-wide, fresh spheres at 20 deg: the construction
%   put the stop 33 m off axis and every ray then missed M1, so S3's
%   first residual was the 1e9 sentinel and the stage froze).  If the
%   constructed pose moves more than 2x the envelope span from the pose
%   the state arrived with, keep the ARRIVED pose -- it traced the
%   previous stage, and the pose is packaging, not physics.
    old = [];
    if isfield(X,'stopC') && ~isempty(X.stopC), old = X.stopC; end
    X.stop_fixed = false;
    [X, ~] = oi_close(X, P, 'offset_deg', P.offset_deg);
    X.stop_fixed = true;
    span = max(1e-3, abs(X.spacings(1)) + abs(X.spacings(3)));
    if ~isempty(old) && norm(X.stopC - old) > 2*span
        fprintf(['  pose_stop_once_: constructed stop pose %.3g m from ' ...
                 'the carried one (span %.3g m) -- degenerate EP ' ...
                 'construction, keeping the carried pose\n'], ...
                norm(X.stopC - old), span);
        X.stopC = old;
    end
end

function bad = wall_check_(Xc, Gc, P) %#ok<INUSD>
%WALL_CHECK_  Reserved for genuine packaging WALLS.  The exit-beam
%   direction is NOT here: it is an EQUALITY constraint carried as a
%   weighted residual row inside OI_SOLVE (a boolean wall freezes any
%   stage that starts outside tolerance -- the base Jacobian becomes all
%   wall rows and no step can be evaluated).  Clearances are report
%   gates (OI_GATES); promoting one to a wall goes here.
    bad = false;
end

function [OUT, ladder] = stage_out_(OUT, ladder, rpt, P, tag, sid, X, G, fo, ...
                                    offset, lbl, hist, gt)
%STAGE_OUT_  Score, draw, gate and report one stage.
    % dense map + figures
    [png_m, mp] = oi_map_fig(X, G, P, offset, lbl, [tag '_' sid '_map.png']);
    png_l = oi_layout_fig(X, G, P, offset, lbl, [tag '_' sid '_layout.png']);
    % deck
    txt = oi_deck(fill_(X, P));
    fdeck = [tag '_' sid '.in'];
    fid = fopen(fdeck,'w');  fprintf(fid,'%s',txt);  fclose(fid);
    % gates
    if isempty(gt), gt = oi_gates(X, G, P, offset); end

    % report section
    fprintf(rpt, '\n## %s\n\n', lbl);
    fprintf(rpt, ['Metric: strict RMS WFE, centroid reference on the frozen ' ...
        'stage FPA, exit-pupil anchor, piston-only removal; dense %dx%d map ' ...
        'over the %gx%g%c box at YAN %+g%c; solve set %dx%d (solve set != ' ...
        'scoring set).\n\n'], P.map_n, P.map_n, P.box_deg, char(176), ...
        offset, char(176), P.nsolve, P.nsolve);
    fprintf(rpt, '| quantity | value |\n|---|---|\n');
    fprintf(rpt, '| EFL (identity) | %.6f m = EPD %.0f mm x F/%.4g |\n', ...
            fo.EFL_m, P.EPD_m*1e3, P.Fno);
    fprintf(rpt, '| paraxial BFD | %.4f m |\n', fo.BFD_m);
    fprintf(rpt, '| petzval c1-c2+c3 | %.3e 1/m |\n', fo.petzval);
    fprintf(rpt, '| plate scale | %.2f um/arcmin |\n', fo.plate_um_per_amin);
    fprintf(rpt, '| stop semi-diameter (traced) | %.1f mm |\n', fo.stop_semi_m*1e3);
    fprintf(rpt, '| radii R1..R3 | %.5f / %.5f / %.5f m |\n', X.R);
    fprintf(rpt, '| conics K1..K3 | %.5g / %.5g / %.5g |\n', X.K);
    if any(X.yde ~= 0) || any(X.ade ~= 0)
        fprintf(rpt, '| YDE (mm) | %+.3f / %+.3f / %+.3f |\n', X.yde*1e3);
        fprintf(rpt, '| ADE (deg) | %+.4f / %+.4f / %+.4f |\n', X.ade);
    end
    fprintf(rpt, '| solve | %s: %.1f -> %.1f nm (qmean over solve set), %d iters |\n', ...
            sid, hist.rms0, hist.rms, hist.iters);
    fprintf(rpt, '| **map max** | **%.1f nm** at XAN %+.1f YAN %+.1f |\n', ...
            mp.max_nm, mp.max_at);
    fprintf(rpt, '| map avg / std / min | %.1f / %.1f / %.1f nm |\n', ...
            mp.avg_nm, mp.std_nm, mp.min_nm);
    fprintf(rpt, '| exit chief | %.3f%c in Y-Z%s |\n', gt.exit_ang_deg, char(176), ...
            exit_txt_(gt, P));
    fprintf(rpt, '| clearance floor | %.1f mm (%s; gate >= %.0f mm%s) |\n', ...
            gt.clear_min_m*1e3, pf_(gt.clear_pass), min(P.clear_m)*1e3, ...
            warn_txt_(gt, P));
    fprintf(rpt, '\nFigures: `%s`, `%s`.  Deck: `%s`.\n', ...
            base_(png_l), base_(png_m), base_(fdeck));

    st = struct('X',X,'G',G,'fo',fo,'map',mp,'gates',gt,'hist',hist, ...
                'deck',fdeck,'fig_layout',png_l,'fig_map',png_m);
    OUT.(sid) = st;
    ladder(end+1) = struct('stage',sid, 'map_max_nm',mp.max_nm, ...
                           'map_avg_nm',mp.avg_nm, 'map_std_nm',mp.std_nm);
    fprintf('  %s: map max %.1f nm  avg %.1f  (gates: exit %s, clear %s)\n', ...
            sid, mp.max_nm, mp.avg_nm, pf_(gt.exit_pass), pf_(gt.clear_pass));
end

function hdr_(rpt, P)
    fprintf(rpt, '# %s -- offset_imager run\n\n', P.name);
    fprintf(rpt, ['%s.  EPD %.0f mm, F/%.4g (EFL %.3f m held as an identity), ' ...
        'lambda %.2f um, box %gx%g%c offset %+g%c, spacings [%g %g %g] m, ' ...
        'model %d, nGridpts %d.\n'], datestr(now,31), P.EPD_m*1e3, P.Fno, ...
        P.EFL_m, P.lambda_m*1e6, P.box_deg, char(176), P.offset_deg, ...
        char(176), P.spacings_m, P.model, P.sampling); %#ok<TNOW1,DATST>
    fprintf(rpt, ['\nEvery WFE number below: strict RMS WFE, sphere centred on ' ...
        'the spot centroid on the stage''s frozen FPA, anchored at the exit ' ...
        'pupil, piston-only removal (design/src strict kernel); headline = ' ...
        'dense-map MAXIMUM over the box.\n']);
end

function stage_banner_(s)
    fprintf('\n=================================================================\n');
    fprintf(' %s\n', s);
    fprintf('=================================================================\n');
end

function D = fill_(X, P)
    D = X;
    D.EPD_m = P.EPD_m;  D.WL_m = P.lambda_m;
    D.sampling = P.sampling;  D.name = P.name;
end

function t = exit_txt_(gt, P)
    if isempty(P.exit_dir), t = ' (report-only; no pin)';
    else, t = sprintf('; err %.3f%c vs pin -> %s', gt.exit_err_deg, char(176), ...
                      pf_(gt.exit_pass));
    end
end

function t = warn_txt_(gt, P)
    if gt.clear_warn, t = sprintf('; WARN < %.0f mm', max(P.clear_m)*1e3);
    else, t = ''; end
end

function s = pf_(p), if p, s = 'PASS'; else, s = 'FAIL'; end, end
function b = base_(p), [~,n,e] = fileparts(p);  b = [n e]; end
