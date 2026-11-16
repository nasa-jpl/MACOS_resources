function OUT = cf2_efc(over)
%CF2_EFC  Coronagraph-family campaign S2: EFC floors per family.
%
%   For each S1 family, on the DM-augmented deck (r1_seg_dm.in) at
%   N = P.dj.model (512, one grid for G / EFC / scoring):
%
%     static -> fixed-G floor -> relinearized floor
%
%   G is ENGINE-MEASURED per mask config (1760 forward pokes through the
%   full masked chain), cached per config tag with a chain_opts stamp
%   (verified on load by ctb_jac_check -- the file NAME is a hint, the
%   stamp is the authority) and a committed .fp.json fingerprint.  The
%   loop is the ctb_efc idiom: REAL-STACKED solve ([Re G; Im G] da =
%   -[Re e; Im e] -- the mex silently drops Im(da), trap 1), per-
%   iteration Tikhonov line search against MEASURED contrast, monotone
%   accept, and the accepted state RE-APPLIED before any re-measure
%   (trap 2).  Relinearization: G re-measured about the dug commands,
%   loop warm-started from them.
%
%   ATTRIBUTION per family (the CTB reading): fixed-vs-relin gain
%   (control-limited), floor vs the numerical class (physics-limited),
%   and the dark-zone SYMMETRIC-field fraction with the Talbot number
%   z/z_T at the outer working angle (amplitude/Talbot-limited -- the
%   0.15 m DM spacing gives ~0.4% Talbot authority at 15 lambda/D, the
%   R4/S3 knob).
%
%   Resumable: a family whose cf2_<tag>_run.mat exists is SKIPPED
%   ('force' re-runs); Jacobian caches are reused when stamped correctly.
%
%   OUT = CF2_EFC()                 all six families (~4 h cold)
%   OUT = CF2_EFC(struct('cf2', struct('families', {{'aplc','v4'}})))
%
%   See also CF_CHAIN, CF1_FAMILIES, ctb_efc, ctb_dm_jacobian,
%   ctb_jac_check, jac_fingerprint.

    arguments
        over struct = struct()
    end
    here = fileparts(mfilename('fullpath'));
    run(fullfile(here,'..','..','..','mmacos_setup.m'));
    ov = over;  cf2 = struct();
    if isfield(ov, 'cf2'), cf2 = ov.cf2;  ov = rmfield(ov, 'cf2'); end
    P = e2e6m_r2_params(ov);
    if ~isfield(cf2, 'families')
        cf2.families = {'hard','apl','aplc','blc','v4','v6'};
    end
    if ~isfield(cf2, 'force'),      cf2.force = false;  end
    if ~isfield(cf2, 'niter'),      cf2.niter = 15;     end
    if ~isfield(cf2, 'niter_r1'),   cf2.niter_r1 = 10;  end
    if ~isfield(cf2, 'alphas'),     cf2.alphas = logspace(-6, -2, 5); end
    addpath(fullfile(here,'..','..','30_instruments','bench_ctb'));
    lib = cf_efc_lib();

    rx = fullfile(P.outdir, 'r1_seg_dm.in');
    assert(isfile(rx), 'cf2_efc: %s not found -- run r1_dm first', rx);
    A = load(fullfile(P.outdir,'r1_dm_run.mat'));
    aug = A.OUT.aug;
    beam_d = 2 * 0.023771;                  % measured pupil at the DMs (R1)

    FCFG = family_cfgs_(P);   % from cf1_run.mat -- the configs S1 RAN
    keys = cf2.families;

    L = {};  t0 = tic;
    L = say_(L, '==================== e2e6m CF2 -- EFC floors per family');
    L = say_(L, 'deck %s, model %d, poke %g nm surface, %d+%d iters, annulus %g-%g lambda/D', ...
             rx, P.dj.model, P.dj.h*1e9, cf2.niter, cf2.niter_r1, ...
             P.co.inner_lamD, P.co.outer_lamD);

    R = struct();
    for f = 1:numel(keys)
        key = keys{f};
        assert(isfield(FCFG, key), 'cf2_efc: unknown family "%s"', key);
        fam = FCFG.(key);
        state = fullfile(P.outdir, sprintf('cf2_%s_run.mat', key));
        if isfile(state) && ~cf2.force
            S = load(state);
            R.(key) = S.res;
            L = say_(L, '\n---- %s: SKIP (state exists: %s) ----', fam.name, state);
            L = say_(L, '    static %.3e -> fixed-G %.3e -> relin %.3e', ...
                     S.res.c_static, S.res.c_fixed, S.res.c_relin);
            continue
        end
        L = say_(L, '\n---- %s ----', fam.name);
        ch = cf_chain('rx', rx, 'model_size', P.dj.model, ...
                      'prolate_iter', P.co.prolate_iter, ...
                      'circ_stop_frac', P.cf.circ_stop_frac, fam.cfg{:});
        L = say_(L, '    tag %s | lambda/D %.3f px | peak_bare %.4e | thru %.3f | stop r %.1f px (area %.3f)', ...
                 ch.tag, ch.lamD_px, ch.peak_bare, ch.thru, ch.r_stop_px, ch.area_factor);

        dm = cell(1, numel(aug.ielt));
        for k = 1:numel(dm)
            dm{k} = ctb_dm('ielt', aug.ielt(k), 'ng', aug.ng, ...
                           'gdx_mm', aug.gdx_mm(k), 'nact', P.dj.nact, ...
                           'beam_d_mm', beam_d, 'pitch_mm', beam_d/P.dj.nact, ...
                           'coupling', P.dj.coupling);
            dm{k}.clear();
        end
        dz_idx = find(ch.dz_mask(P.co.inner_lamD, P.co.outer_lamD));

        % ---- round 0: G about flat, fixed-G loop ------------------------
        a0 = cellfun(@(d) zeros(d.nact^2,1), dm, 'UniformOutput', false);
        [G0, jmeta0] = lib.jacobian(ch, dm, a0, dz_idx, P, ...
            fullfile(P.outdir, sprintf('cf2_G_%s.mat', ch.tag)));
        r1cache = fullfile(P.outdir, sprintf('cf2_G_%s_r1.mat', ch.tag));
        resumed = false;
        if isfile(r1cache) && ~cf2.force
            % RESTART RESUME (2026-08-26): a family whose _r1 cache exists
            % but whose run state does not died between the relin-G
            % measure and the final save.  The _r1 cache stores the a0 it
            % was measured about = the fixed-G dug commands, and the line
            % search is NOT bit-deterministic across a restart (a replay
            % fails the cache's own 1e-15 a0 assert), so the cache is the
            % AUTHORITY: adopt its commands, re-measure the two endpoints
            % the lost process printed, skip the replay.  The round-1
            % jacobian load then passes its a0 assert BY CONSTRUCTION.
            Jr = load(r1cache, 'a0');
            afix = Jr.a0;
            lib.seta(dm, a0);    E = ch.run();
            c_static = mean(abs(E(dz_idx)).^2) / ch.peak_bare;
            lib.seta(dm, afix);  E = ch.run();
            c_fixed  = mean(abs(E(dz_idx)).^2) / ch.peak_bare;
            con_fix  = [c_static c_fixed];   alph_fix = NaN;
            resumed  = true;
            L = say_(L, '    fixed-G: RESUMED from the r1 cache (endpoints re-measured: %.3e -> %.3e)', ...
                     c_static, c_fixed);
        else
            [afix, con_fix, alph_fix] = lib.efc(ch, dm, G0, a0, dz_idx, ...
                                                cf2.niter, cf2.alphas);
            c_static = con_fix(1);
            c_fixed  = con_fix(end);
            L = say_(L, '    fixed-G: %.3e -> %.3e in %d iters', ...
                     c_static, c_fixed, numel(con_fix)-1);
        end

        % ---- round 1: relinearize about the dug state -------------------
        [G1, jmeta1] = lib.jacobian(ch, dm, afix, dz_idx, P, ...
            fullfile(P.outdir, sprintf('cf2_G_%s_r1.mat', ch.tag)));
        [arel, con_rel, alph_rel] = lib.efc(ch, dm, G1, afix, dz_idx, ...
                                            cf2.niter_r1, cf2.alphas);
        c_relin = con_rel(end);
        L = say_(L, '    relin:   %.3e -> %.3e in %d iters', ...
                 con_rel(1), c_relin, numel(con_rel)-1);

        % ---- linear-achievable floors (MEASURED attribution) ------------
        % Two readings of the rank curve: at the stroke BOUND (what more
        % stroke could buy IF the linear model held there -- at 50 nm it
        % does not: 2 nm pokes extrapolated to 1.3 rad of phase), and at
        % the ACHIEVED stroke (the honest control-vs-substrate question:
        % did the loop reach what its own G says is possible at the
        % strokes it actually used -- the CTB "4.5e-9 at 11 nm" pattern).
        la0 = lib.linfloor(G0, P.cf.stroke_bound_nm);
        la1 = lib.linfloor(G1, P.cf.stroke_bound_nm);
        ach_nm = 1e9 * rms_(cell2mat(cellfun(@(x) x(x~=0), arel(:).', ...
                                             'UniformOutput', false).'));
        la1_ach = floor_at_(la1, ach_nm);
        L = say_(L, '    linear-achievable: G1 %.3e at the ACHIEVED %.1f nm (rank %d)', ...
                 la1_ach.floor, ach_nm, la1_ach.rank);
        L = say_(L, '    (at the %g nm bound: G0 %.3e @ %.1f nm | G1 %.3e @ %.1f nm -- linear model', ...
                 P.cf.stroke_bound_nm, la0.floor, la0.stroke_nm, la1.floor, la1.stroke_nm);
        L = say_(L, '     NOT valid at those strokes; bound values are the model''s claim, not physics)');

        % ---- attribution measurements -----------------------------------
        % dark-zone symmetric-field fraction (about the star): amplitude-
        % type (gap) speckle is even under 180-deg rotation, phase-type
        % odd -- the amplitude-dominance diagnostic.
        lib.seta(dm, arel);
        E = ch.run();
        sym_after = lib.sym_frac(E, ch.center_px, dz_idx);
        lib.seta(dm, a0);
        E = ch.run();
        sym_before = lib.sym_frac(E, ch.center_px, dz_idx);
        % Talbot authority at the outer working angle
        D_beam = beam_d;                                     % m
        p_min  = D_beam / P.co.outer_lamD;                   % speckle period
        z_T    = 2 * p_min^2 / P.lambda_m;
        talbot = P.b2.d_dm2 / z_T;
        strokes = cellfun(@(x) 1e9 * rms_(x(x~=0)), arel);
        L = say_(L, '    strokes rms [%s] nm | DZ symmetric fraction %.2f -> %.2f | z/z_T(%g lamD) %.2e', ...
                 num2str(strokes, '%.2f '), sym_before, sym_after, ...
                 P.co.outer_lamD, talbot);

        lib.seta(dm, a0);                      % leave the engine flat
        res = struct('name', fam.name, 'tag', ch.tag, 'config', {ch.config}, ...
            'c_static', c_static, 'c_fixed', c_fixed, 'c_relin', c_relin, ...
            'con_fixed', con_fix, 'con_relin', con_rel, ...
            'alpha_fixed', alph_fix, 'alpha_relin', alph_rel, ...
            'a', {arel}, 'strokes_nm', strokes, ...
            'sym_before', sym_before, 'sym_after', sym_after, ...
            'talbot_zzT', talbot, 'thru', ch.thru, ...
            'la0', la0, 'la1', la1, 'la1_ach', la1_ach, 'ach_nm', ach_nm, ...
            'circ_stop_frac', ch.circ_stop_frac, 'area_factor', ch.area_factor, ...
            'lamD_px', ch.lamD_px, 'peak_bare', ch.peak_bare, ...
            'jac0', jmeta0, 'jac1', jmeta1, 'resumed', resumed, ...
            'N', P.dj.model);
        save(state, 'res');
        R.(key) = res;
    end

    % ---- the closed-loop table ------------------------------------------
    L = say_(L, '\n==== the S2 table (N=%d; static/floors CLOSED-LOOP annulus %g-%g lambda/D) ====', ...
             P.dj.model, P.co.inner_lamD, P.co.outer_lamD);
    L = say_(L, '  %-20s | %-10s | %-10s | %-10s | %-10s | %-9s | %s', ...
             'family', 'static', 'fixed-G', 'relin', 'lin-ach', 'stroke nm', 'attribution');
    L = say_(L, '  %s', repmat('-', 1, 112));
    for f = 1:numel(keys)
        r = R.(keys{f});
        L = say_(L, '  %-20s | %.3e | %.3e | %.3e | %.3e | %4.1f/%4.1f | %s', ...
                 r.name, r.c_static, r.c_fixed, r.c_relin, r.la1_ach.floor, ...
                 r.strokes_nm(1), r.strokes_nm(2), attrib_(r));
    end
    L = say_(L, '  %s', repmat('-', 1, 112));
    L = say_(L, '  lin-ach = linear-achievable floor of the RELIN G at the ACHIEVED');
    L = say_(L, '  stroke -- the measured attribution: a floor within ~2x of lin-ach');
    L = say_(L, '  is the SUBSTRATE speaking, not the controller.');
    L = say_(L, '  z/z_T at %g lambda/D = %.2e (DM spacing %g m): amplitude authority is', ...
             P.co.outer_lamD, R.(keys{1}).talbot_zzT, P.b2.d_dm2);
    L = say_(L, '  Talbot-weak on this train -- the S3 spacing trade measures the knob.');

    png = fullfile(P.outdir, 'cf2_floors.png');
    fig_(R, keys, P, png);
    L = say_(L, '\n  figure: %s', png);

    L = say_(L, '\nCF2 DONE in %.1f min', toc(t0)/60);
    txt = strjoin(L, newline);
    fid = fopen(fullfile(P.outdir,'cf2_report.txt'),'w');
    fprintf(fid,'%s\n',txt);  fclose(fid);
    OUT = struct('P',P, 'R',R, 'keys',{keys}, 'text',txt, 'figure',png, ...
                 'when',datestr(now,31)); %#ok<TNOW1,DATST>
    save(fullfile(P.outdir,'cf2_run.mat'),'OUT');
end

% =========================================================================
function F = family_cfgs_(P)
%FAMILY_CFGS_  The family configs, read from the S1 artifact (cf1_run.mat)
%   so S2 closes the loop on EXACTLY the configurations S1 scored -- no
%   second copy of the family table to drift.
    c1 = fullfile(P.outdir, 'cf1_run.mat');
    assert(isfile(c1), 'cf2_efc: %s not found -- run cf1_families first', c1);
    S = load(c1);
    F = struct();
    for k = 1:numel(S.OUT.F)
        F.(S.OUT.F(k).key) = struct('name', S.OUT.F(k).name, ...
                                    'cfg', {S.OUT.F(k).cfg});
    end
end

function a = attrib_(r)
%ATTRIB_  MEASURED attribution: the achieved relin floor against the relin
%   G's linear-achievable floor at the stroke bound.  Within ~2x = the
%   substrate's number (then the symmetric fraction says amplitude/Talbot
%   vs phase); above = control-limited (the loop under-runs its own G).
    gain_relin = r.c_fixed / max(r.c_relin, realmin);
    ratio_la = r.c_relin / max(r.la1_ach.floor, realmin);
    if r.c_relin < 1e-13
        a = 'numerical floor (nothing uncontrollable)';
    elseif ratio_la <= 2
        if r.sym_after > 0.6
            a = sprintf('linear-optimal; amplitude/Talbot-limited (la %.1fx, sym %.2f)', ...
                        ratio_la, r.sym_after);
        else
            a = sprintf('linear-optimal; substrate-limited (la %.1fx, sym %.2f)', ...
                        ratio_la, r.sym_after);
        end
    else
        a = sprintf('control-limited (%.1fx above lin-ach; relin gained %.1fx)', ...
                    ratio_la, gain_relin);
    end
end

function fig_(R, keys, P, png)
    f = figure('Visible','off','Color','w','Position',[60 60 1400 620]);
    tl = tiledlayout(f, 1, 2, 'TileSpacing','compact', 'Padding','compact');
    title(tl, sprintf(['e2e6m families -- EFC floors (N=%d, %d actuators, ' ...
        'engine-measured G, annulus %g-%g \\lambda/D)'], ...
        R.(keys{1}).N, 1760, P.co.inner_lamD, P.co.outer_lamD), ...
        'FontWeight','bold', 'Interpreter','tex');
    ax = nexttile(tl); hold(ax,'on'); set(ax,'YScale','log');
    cols = lines(numel(keys));
    h = gobjects(1, numel(keys));
    for f2 = 1:numel(keys)
        r = R.(keys{f2});
        cc = [r.con_fixed, r.con_relin(2:end)];
        h(f2) = semilogy(ax, 0:numel(cc)-1, cc, 'o-', 'Color', cols(f2,:), ...
                         'LineWidth', 1.5, 'MarkerSize', 4);
        xline(ax, numel(r.con_fixed)-1, ':', 'Color', cols(f2,:), ...
              'HandleVisibility','off');
    end
    grid(ax,'on'); box(ax,'on');
    xlabel(ax,'EFC iteration (relin joins at the dotted line)');
    ylabel(ax,'dark-zone mean contrast');
    legend(ax, h, cellfun(@(k) R.(k).name, keys, 'UniformOutput', false), ...
           'Location','northeast');
    title(ax,'fixed-G then one relinearization');
    ax = nexttile(tl); hold(ax,'on'); set(ax,'YScale','log');
    x = 1:numel(keys);
    st = cellfun(@(k) R.(k).c_static, keys);
    fx = cellfun(@(k) R.(k).c_fixed,  keys);
    rl = cellfun(@(k) R.(k).c_relin,  keys);
    plot(ax, x, st, 'ks', 'MarkerFaceColor',[0.7 0.7 0.7], 'MarkerSize',9);
    plot(ax, x, fx, 'b^', 'MarkerFaceColor',[0.4 0.6 0.9], 'MarkerSize',9);
    plot(ax, x, rl, 'ro', 'MarkerFaceColor',[0.9 0.5 0.4], 'MarkerSize',9);
    set(ax,'XTick',x,'XTickLabel',cellfun(@(k) R.(k).name, keys, ...
        'UniformOutput',false),'XTickLabelRotation',25);
    grid(ax,'on'); box(ax,'on');
    ylabel(ax,'dark-zone mean contrast');
    legend(ax, {'static (pre-control)','fixed-G floor','relin floor'}, ...
           'Location','northeast');
    title(ax,'the closed-loop column');
    exportgraphics(f, png, 'Resolution', 150);
    close(f);
end

function fa = floor_at_(la, stroke_nm)
%FLOOR_AT_  Read the linear-achievable rank curve at a queried stroke.
    ok = la.curve_stroke_nm <= stroke_nm;
    if ~any(ok), rk = 1; else, rk = find(ok, 1, 'last'); end
    fa = struct('floor', la.curve_con(rk), 'rank', rk, ...
                'stroke_nm', la.curve_stroke_nm(rk));
end

function r = rms_(v), v = v(:); if isempty(v), r = 0; else, r = sqrt(mean(v.^2)); end, end
function L = say_(L, varargin)
    s = sprintf(varargin{:});  L{end+1} = s;  fprintf('%s\n', s);
end
