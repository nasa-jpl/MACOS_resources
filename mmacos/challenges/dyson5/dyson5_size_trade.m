function OUT = dyson5_size_trade(over)
%DYSON5_SIZE_TRADE  How small can the Dyson block be at R4 (and at SPEC)?
%
%   OUT = DYSON5_SIZE_TRADE() answers Dave's size question
%   (BRIEF_ccmac_dyson_size, rounds 1 and 2) with engine scores, new files only.
%   Each point is one ladder solve (rung 3 = R3, the de-concentred block with its
%   face conic + h^4/h^6 and NO meniscus; rung 5 = R4, + the meniscus corrector),
%   warm-started by CONTINUATION from the previous point's solved design, the deck
%   emitted with apertures and ENGINE-scored on the 7x7 grid.
%
%   Families:
%     A  R4 silica 54 mm, 220 -> 150 mm   (round 1; also the solver check)
%     B  R4 CaF2   54 mm, 220 -> 100 mm   (round 1)
%     C  R4 silica AND CaF2 27 mm, 220 -> 60 mm   (round 1; two 1500-px modules)
%     D  R3 silica 27 mm, 220 -> 60 mm    (round 2: no meniscus)
%     E  R3 CaF2   27 mm, 220 -> 60 mm    (round 2: no meniscus)
%     F  R3 CaF2   54 mm, 300 -> 180 mm   (round 2: a one-module no-meniscus Dyson at any size?)
%     G  R4 silica 54 mm, 220 mm          (round 2: the THICK-meniscus basins of the
%                                          global search, starts 12 and 7, engine-scored)
%
%   TWO verdicts per point: MATCHES R4 (smile/keystone < 0.1, CRF <= 1.33,
%   SRF <= 2.05, EE >= 0.76 at R4's precision, no bound) and MEETS SPEC
%   (smile/keystone < 0.1, CRF < 1.5, SRF < 2.1, no bound; EE reported, not gated).
%   A THROUGHPUT column: the air-glass crossings on the slit->detector path and the
%   UNCOATED Fresnel product at 1 um, normal incidence (Jim's point about the
%   meniscus's four extra uncoated surfaces).
%
%   OUT = DYSON5_SIZE_TRADE(OVER) overrides options:
%     .which   families, subset of {'A'..'G'}            default all
%     .max_iter  lsqnonlin iterations per point           default 30 (beat 4e's)
%     .rung      force the rung for every family (3 or 5)  default [] (per family)
%     .bound_scale_runs  re-solve an on-bound point with the meniscus-thickness
%                        bound scaled by the block radius, reported beside it  default true
%     .radiiA/B/C, .radiiDE, .radiiF   the radius walks (mm)
%     .quiet   suppress the per-point console line          default false
%
%   Seeds (read, never written): A/B/C from the R4 rung of dyson5_s3.mat the way
%   stage_s4env_ reads it; D/E/F from the R3 rung; G from dyson5_s3_r4global.mat
%   (Gs.res(12).P and Gs.res(7).P).  All spec fields are overridden on a copy of
%   dyson5_params(), never through dyson5_params(over).  Writes dyson5_size.txt /
%   .mat; decks dyson5_size_<family>_r<mm>.in; dyson5_size_fig draws the figure.
%
%   NEW FILE (BRIEF_ccmac_dyson_size).  Does NOT touch dyson5_run.m,
%   dyson5_params.m, dyson5_envelope.m or dyson_ladder.m.
    arguments
        over struct = struct()
    end
    here = fileparts(mfilename('fullpath'));
    P0  = dyson5_params();
    tag = 'dyson5_size';

    opt = struct('which', {{'A', 'B', 'C', 'D', 'E', 'F', 'G'}}, 'max_iter', P0.env_max_iter, ...
                 'quiet', false, 'bound_scale_runs', true, 'rung', [], ...
                 'radiiA', [220 200 180 165 150], ...
                 'radiiB', [220 200 180 160 140 120 100], ...
                 'radiiC', [220 190 160 130 100 80 60], ...
                 'radiiDE', [220 190 160 130 100 80 60], ...
                 'radiiF',  [300 270 240 210 190 180]);
    fn = fieldnames(over);
    for k = 1:numel(fn)
        if ~isfield(opt, fn{k}), error('dyson5_size_trade:unknown', 'unknown option "%s"', fn{k}); end
        opt.(fn{k}) = over.(fn{k});
    end

    % ---- seeds (read the saved ladder the stage_s4env_ way; read, do not call)
    fn3 = fullfile(here, 'dyson5_s3.mat');
    assert(isfile(fn3), 'dyson5_size_trade needs the ladder record %s (stage s3)', fn3);
    S3 = load(fn3);  L3 = S3.S;
    r4seed = rung_P_(L3, 'R4 ', fn3);
    r3seed = rung_P_(L3, 'R3 ', fn3);

    % ---- family (walk) definitions
    DEF = {'men_ca', 0.05, 8; 'men_cb', 0.05, 8};         % round-1 R4 widening (beat 4e)
    GLB = {'men_ca', -8, 8; 'men_cb', -8, 8; 'men_t', 0.004, 0.040};  % the global search's R4 bounds
    walks = struct('key', {}, 'rung', {}, 'glass', {}, 'slit_m', {}, 'radii', {}, 'force_full', {}, ...
                   'seed', {}, 'slit_bridge', {}, 'bounds', {}, 'max_iter', {});
    add = @(varargin) struct('key', varargin{1}, 'rung', varargin{2}, 'glass', varargin{3}, ...
                             'slit_m', varargin{4}, 'radii', varargin{5}, 'force_full', varargin{6}, ...
                             'seed', varargin{7}, 'slit_bridge', varargin{8}, 'bounds', {varargin{9}}, 'max_iter', varargin{10});
    rf = opt.rung;                                        % [] = per-family rung; else forced
    for ww = 1:numel(opt.which)                           % build in the ORDER requested (so 'D' can report first)
        switch opt.which{ww}
            case 'A', walks(end+1) = add('A', defrung_(rf,5), 'Silica', [],    opt.radiiA,  true,  r4seed, [],         DEF, []);
            case 'B', walks(end+1) = add('B', defrung_(rf,5), 'CaF2',   [],    opt.radiiB,  false, r4seed, [],         DEF, []);
            case 'C'
                walks(end+1) = add('C-silica', defrung_(rf,5), 'Silica', 27e-3, opt.radiiC, false, r4seed, [],         DEF, []);
                walks(end+1) = add('C-CaF2',   defrung_(rf,5), 'CaF2',   27e-3, opt.radiiC, false, r4seed, [],         DEF, []);
            case 'D', walks(end+1) = add('D', defrung_(rf,3), 'Silica', 27e-3, opt.radiiDE, false, r3seed, [54e-3 40e-3], {}, []);
            case 'E', walks(end+1) = add('E', defrung_(rf,3), 'CaF2',   27e-3, opt.radiiDE, false, r3seed, [54e-3 40e-3], {}, []);
            case 'F', walks(end+1) = add('F', defrung_(rf,3), 'CaF2',   [],    opt.radiiF,  false, r3seed, [],            {}, []);
            case 'G'
                [s12, s7] = global_starts_(here);
                walks(end+1) = add('G-s12', defrung_(rf,5), 'Silica', [], 220, true, s12, [], GLB, 0);   % score the thick basin as-is
                walks(end+1) = add('G-s7',  defrung_(rf,5), 'Silica', [], 220, true, s7,  [], GLB, 0);
            otherwise, error('dyson5_size_trade:which', 'unknown family "%s"', opt.which{ww});
        end
    end

    macos.init(P0.model);
    txt = fullfile(here, [tag '.txt']);  fid = fopen(txt, 'w');  hdr_(fid, opt);
    pr = @(varargin) dualprint_(fid, opt.quiet, varargin{:});

    rows = empty_rows_();
    for w = 1:numel(walks)
        W = walks(w);
        pr('\n--- family %s: rung R%d, %s, %g mm slit ---\n', W.key, W.rung, W.glass, slit_len_(P0, W)*1e3);
        seed = W.seed;  mi = tern_(isempty(W.max_iter), opt.max_iter, W.max_iter);

        % slit bridge: step the slit down at the first radius to warm the seed (not recorded)
        for sb = W.slit_bridge
            Qb = qfor_(P0, W, W.radii(1), sb);
            Rb = solve_point_(Qb, sprintf('%s_%s_bridge%d', tag, W.key, round(sb*1e3)), ...
                              fullfile(here, sprintf('%s_%s_bridge%d.in', tag, W.key, round(sb*1e3))), ...
                              seed, mi, {}, W.rung, W.bounds);
            seed = Rb.P;
            pr('    (slit bridge %g mm: CRF %.3f EE %.3f -> seed)\n', sb*1e3, Rb.engine.crf_max, Rb.engine.ee_min);
        end

        last_ok = 0;
        for i = 1:numel(W.radii)
            r_mm = W.radii(i);
            Q = qfor_(P0, W, r_mm, []);
            tagp = sprintf('%s_%s_r%d', tag, W.key, round(r_mm));
            deck = fullfile(here, [tagp '.in']);
            R = solve_point_(Q, tagp, deck, seed, mi, {}, W.rung, W.bounds);
            row = row_(W, r_mm, 'solve', R);
            rows(end+1) = row;  print_row_(pr, row);   %#ok<AGROW>
            seed = R.P;

            if ~isempty(R.on_bounds) && opt.bound_scale_runs
                eb = scaled_bounds_(R.on_bounds, r_mm, W.rung);
                if ~isempty(eb)
                    R2 = solve_point_(Q, [tagp '_bs'], fullfile(here, [tagp '_bs.in']), seed, mi, eb, W.rung, W.bounds);
                    rows(end+1) = row_(W, r_mm, 'bound-scaled', R2);  print_row_(pr, rows(end));   %#ok<AGROW>
                end
            end

            if row.matches_R4 || row.meets_spec, last_ok = i; end
            if ~W.force_full
                if last_ok >= 1 && (i - last_ok) >= 2, break; end     % two points past the last that passed either rule
                if last_ok == 0 && i >= 3, break; end
            end
        end
    end

    T = struct2table(rmfield(rows, 'P'), 'AsArray', true);
    OUT = struct('table', T, 'rows', rows, 'walks', walks, 'r4seed', r4seed, 'r3seed', r3seed, 'opt', opt, 'P0', P0);
    OUT.record = struct('CRF', 1.327, 'EE', 0.759, 'SRF', 2.032, 'smile', 0.0051, 'keystone', 0.0026, ...
                        'r_mm', 220, 'blockT_mm', 221.2, 'blockD_mm', 146.6, 'n_cross', 8);
    [OUT.record.edged_L, OUT.record.edged_kg] = edged_(146.6, 221.2, 'Silica');
    fprintf(fid, '\n');  foot_(fid, OUT);  fclose(fid);
    save(fullfile(here, [tag '.mat']), 'OUT');
    if ~opt.quiet, fprintf('dyson5_size_trade: wrote %s.txt and %s.mat (%d rows)\n', tag, tag, height(T)); end
end

% ======================================================================
function v = defrung_(forced, dflt)
    if isempty(forced), v = dflt; else, v = forced; end
end

function P = rung_P_(L3, name, fn)
    k = find(strncmp({L3.rung.name}, name, numel(name)), 1);
    assert(~isempty(k), 'dyson5_size_trade: rung %s is not in %s', strtrim(name), fn);
    P = L3.rung(k).P;
end

function [s12, s7] = global_starts_(here)
    fg = fullfile(here, 'dyson5_s3_r4global.mat');
    assert(isfile(fg), 'dyson5_size_trade: family G needs %s (the global search)', fg);
    G = load(fg);  res = G.Gs.res;
    s12 = res(12).P;  s7 = res(7).P;                      % the two thick-meniscus basins (~29 / 31 mm)
end

function L = slit_len_(P0, W)
    if isempty(W.slit_m), L = P0.npix(1)*P0.pixel_m; else, L = W.slit_m; end
end

function Q = qfor_(P0, W, r_mm, slit_override)
    Q = P0;  Q.glass = W.glass;  Q.block_r_m = r_mm*1e-3;
    sl = W.slit_m;  if nargin >= 4 && ~isempty(slit_override), sl = slit_override; end
    if ~isempty(sl)
        Q.npix   = [round(sl/Q.pixel_m), Q.npix(2)];
        Q.slit_m = sl;
    end
end

function R = solve_point_(Q, tagp, deck, seed, max_iter, extra_bounds, rung, base_bounds)
    bounds = base_bounds;  if isempty(bounds), bounds = cell(0,3); end
    if ~isempty(extra_bounds), bounds = [bounds; extra_bounds]; end
  try
    L = dyson_ladder(Q, tagp, 'rungs', rung, 'seed', seed, 'deck', deck, ...
                     'nx', Q.ladder_nx, 'nlam', Q.ladder_nlam, ...
                     'w_dist', Q.ladder_w_dist, 'w_blur', Q.ladder_w_blur, ...
                     'clear_m', Q.ladder_clear_m, 'max_iter', max_iter, ...
                     'quiet', true, 'bounds', bounds);
    r  = L.rung(1);
    Re = r.engine;
    G  = spectrometer_geom('dyson', r.P);
    g  = geom_cols_(G);
    Cl = spectrometer_clearance(G, Q, 'quiet', true);

    % throughput: air-glass crossings on the slit->detector path, uncoated Fresnel at 1 um
    n_cross = sum(strcmp({G.surf.act}, 'refract'));
    nidx = G.n(1e-6);  Rf = ((nidx - 1)/(nidx + 1))^2;  thru = (1 - Rf)^n_cross;

    % uniformity ESTIMATES: double-pass on-axis glass path x {dn=1e-6 ; 0.1 K}
    L_dp = 2 * g.block_thick_mm * 1e-3;  lam = 1e-6;
    unif_inhom = 1e-6 * L_dp / lam;  unif_therm = 1e-5 * 0.1 * L_dp / lam;
    [edL, edkg] = edged_(g.block_diam_mm, g.block_thick_mm, Q.glass);

    R = struct('engine', Re, 'P', r.P, 'on_bounds', {r.on_bounds}, 'G', G, 'Cl', Cl, ...
               'blockT_mm', g.block_thick_mm, 'blockD_mm', g.block_diam_mm, ...
               'gratR_mm', G.Rg*1e3, 'gratD_mm', g.grating_diam_mm, 'length_mm', g.length_mm, ...
               'edged_L', edL, 'edged_kg', edkg, 'unif_inhom_waves', unif_inhom, 'unif_therm_waves', unif_therm, ...
               'clear_min_mm', Cl.min_mm, 'clear_pair', '', 'n_cross', n_cross, 'thru', thru, ...
               'fails', {{}}, 'matches_R4', false, 'meets_spec', false);
    if ~isempty(Cl.table) && height(Cl.table) >= 1
        R.clear_pair = sprintf('%s vs %s', Cl.table.leg{1}, Cl.table.body{1});
    end
    onb = isempty(r.on_bounds);
    sk  = Re.smile_max < 0.10 && Re.keystone_max < 0.10;
    R.matches_R4 = sk && round(Re.crf_max,2) <= 1.33 && round(Re.srf_max,2) <= 2.05 && round(Re.ee_min,2) >= 0.76 && onb;
    R.meets_spec = sk && Re.crf_max < 1.5 && Re.srf_max < 2.1 && onb;
    fails = {};
    if ~sk, fails{end+1} = 'smile/keyst'; end
    if ~(Re.crf_max < 1.5),  fails{end+1} = 'CRF';  end
    if ~(Re.srf_max < 2.1),  fails{end+1} = 'SRF';  end
    if ~onb,                 fails{end+1} = 'on-bound'; end
    R.fails = fails;
  catch e
    R = failed_point_(Q, seed, e.message);
  end
end

function R = failed_point_(Q, seed, msg)
    nanE = struct('smile_max', NaN, 'keystone_max', NaN, 'crf_max', NaN, 'srf_max', NaN, 'ee_min', NaN);
    R = struct('engine', nanE, 'P', seed, 'on_bounds', {{}}, 'G', [], 'Cl', [], ...
               'blockT_mm', NaN, 'blockD_mm', NaN, 'gratR_mm', NaN, 'gratD_mm', NaN, 'length_mm', NaN, ...
               'edged_L', NaN, 'edged_kg', NaN, 'unif_inhom_waves', NaN, 'unif_therm_waves', NaN, ...
               'clear_min_mm', NaN, 'clear_pair', '', 'n_cross', NaN, 'thru', NaN, ...
               'fails', {{['ERROR:' regexprep(msg, '\s+', ' ')]}}, 'matches_R4', false, 'meets_spec', false);
end

function rows = empty_rows_()
    rows = struct('family', {}, 'glass', {}, 'slit_mm', {}, 'r_mm', {}, 'rung', {}, 'variant', {}, ...
                  'smile', {}, 'keystone', {}, 'CRF', {}, 'SRF', {}, 'EE', {}, 'matches_R4', {}, 'meets_spec', {}, ...
                  'fails', {}, 'on_bounds', {}, 'blockT_mm', {}, 'blockD_mm', {}, 'edged_L', {}, 'edged_kg', {}, ...
                  'gratR_mm', {}, 'gratD_mm', {}, 'length_mm', {}, 'clear_min_mm', {}, 'clear_pair', {}, ...
                  'n_cross', {}, 'thru', {}, 'unif_inhom_waves', {}, 'unif_therm_waves', {}, 'P', {});
end

function row = row_(W, r_mm, variant, R)
    Re = R.engine;  sl = slit_len_(struct('npix', [3000 500], 'pixel_m', 18e-6), W);
    row = struct('family', W.key, 'glass', W.glass, 'slit_mm', round(sl*1e3), 'r_mm', r_mm, 'rung', W.rung, ...
                 'variant', variant, 'smile', Re.smile_max, 'keystone', Re.keystone_max, 'CRF', Re.crf_max, ...
                 'SRF', Re.srf_max, 'EE', Re.ee_min, 'matches_R4', R.matches_R4, 'meets_spec', R.meets_spec, ...
                 'fails', {strjoin(R.fails, ',')}, 'on_bounds', {strjoin(R.on_bounds, ',')}, ...
                 'blockT_mm', R.blockT_mm, 'blockD_mm', R.blockD_mm, 'edged_L', R.edged_L, 'edged_kg', R.edged_kg, ...
                 'gratR_mm', R.gratR_mm, 'gratD_mm', R.gratD_mm, 'length_mm', R.length_mm, ...
                 'clear_min_mm', R.clear_min_mm, 'clear_pair', {R.clear_pair}, 'n_cross', R.n_cross, 'thru', R.thru, ...
                 'unif_inhom_waves', R.unif_inhom_waves, 'unif_therm_waves', R.unif_therm_waves, 'P', R.P);
end

function eb = scaled_bounds_(on_bounds, r_mm, rung)
    eb = {};
    if rung >= 4 && any(strcmp(on_bounds, 'men_t'))        % the 4 mm meniscus-thickness bound, scaled to the radius
        eb = [eb; {'men_t', 0.004*(r_mm/220), 0.040}];
    end
end

function g = geom_cols_(G)
    g.length_mm = (G.surf(G.iG).vpt(3) - G.slit(3))*1e3;
    ib  = find(strcmp({G.surf.name}, 'BlockSphereOut'), 1);
    ifc = find(strcmp({G.surf.name}, 'BlockFaceIn'), 1);
    F   = G.footprints();  m = 5e-3;
    g.block_diam_mm  = 2*(F(ib).radius + m)*1e3;
    g.block_thick_mm = (G.surf(ib).vpt(3) - G.surf(ifc).C(3))*1e3;
    g.grating_diam_mm = 2*(F(G.iG).radius + m)*1e3;
end

function [vol_L, mass_kg] = edged_(diam_mm, thick_mm, glass)
    rho = 2.20;  if strcmpi(glass, 'CaF2'), rho = 3.18; end
    vol_cm3 = pi*(diam_mm/2/10)^2 * (thick_mm/10);
    vol_L = vol_cm3/1000;  mass_kg = vol_cm3*rho/1000;
end

function hdr_(fid, opt)
    fprintf(fid, 'dyson5 block-size trade (%s) -- CCMac, BRIEF_ccmac_dyson_size rounds 1+2\n', datestr(now, 'yyyy-mm-dd HH:MM'));
    fprintf(fid, 'CONVENTIONS: each point one ladder solve (rung 3 = R3 no-meniscus / rung 5 = R4 + meniscus; %d iterations, merit\n', opt.max_iter);
    fprintf(fid, '  w_dist=10 / w_blur=1 of record), warm-started by CONTINUATION; deck emitted with apertures and ENGINE-scored\n');
    fprintf(fid, '  on the 7x7 grid.  R4 (matches R4 of record): smile/keyst < 0.1, CRF <= 1.33, SRF <= 2.05, EE >= 0.76 (rounded\n');
    fprintf(fid, '  to R4''s 2 dp), no bound.  SPEC: smile/keyst < 0.1, CRF < 1.5, SRF < 2.1, no bound (EE reported, not gated).\n');
    fprintf(fid, '  thru = uncoated Fresnel product at 1 um over nX air-glass crossings (normal incidence), the 1st radiometric\n');
    fprintf(fid, '  entry.  edged = circumscribing rod (clear diam x on-axis thickness).  unif (ESTIMATES) = double-pass glass\n');
    fprintf(fid, '  path x {dn=1e-6 ; 0.1 K at |dn/dT|=1e-5/K}, waves at 1 um.  R3 seeds enter via a 54->40->27 mm slit bridge.\n');
    fprintf(fid, '%-9s %-7s %4s %4s %2s %-12s %7s %7s %6s %6s %6s %-3s %-4s %-11s %-8s %6s %6s %5s %5s %4s %5s %6s %5s %5s\n', ...
        'family', 'glass', 'slit', 'r', 'Ru', 'variant', 'smile', 'keyst', 'CRF', 'SRF', 'EE', 'R4', 'spec', 'fails', 'on_bnd', ...
        'blkT', 'blkD', 'edgL', 'edgkg', 'nX', 'thru', 'clear', 'uInh', 'uThm');
end

function print_row_(pr, r)
    pr('%-9s %-7s %4d %4d R%d %-12s %7.4f %7.4f %6.3f %6.3f %6.3f %-3s %-4s %-11s %-8s %6.1f %6.1f %5.2f %5.1f %4d %5.3f %6.2f %5.2f %5.2f\n', ...
       r.family, r.glass, r.slit_mm, r.r_mm, r.rung, r.variant, r.smile, r.keystone, r.CRF, r.SRF, r.EE, ...
       tern_(r.matches_R4, 'yes', '-'), tern_(r.meets_spec, 'yes', '-'), empty_(r.fails), empty_(r.on_bounds), ...
       r.blockT_mm, r.blockD_mm, r.edged_L, r.edged_kg, r.n_cross, r.thru, r.clear_min_mm, ...
       r.unif_inhom_waves, r.unif_therm_waves);
end

function foot_(fid, OUT)
    fprintf(fid, 'record (R4, silica 220 mm): CRF %.3f EE %.3f SRF %.3f; 8 crossings, thru %.3f; edged rod %.1f L / %.1f kg.\n', ...
        OUT.record.CRF, OUT.record.EE, OUT.record.SRF, (1-((1.450417-1)/(1.450417+1))^2)^8, OUT.record.edged_L, OUT.record.edged_kg);
    T = OUT.table;  fams = unique(T.family, 'stable');
    for i = 1:numel(fams)
        for rule = {'matches_R4', 'meets_spec'}
            sel = strcmp(T.family, fams{i}) & T.(rule{1}) & strcmp(T.variant, 'solve');
            if any(sel)
                idx = find(sel);  [rmin, j] = min(T.r_mm(idx));  jj = idx(j);
                fprintf(fid, 'family %-8s smallest %-10s r %3d mm: blockT %.0f mm, edged %.2f L / %.1f kg, thru %.3f, CRF %.3f EE %.3f\n', ...
                    fams{i}, rule{1}, rmin, T.blockT_mm(jj), T.edged_L(jj), T.edged_kg(jj), T.thru(jj), T.CRF(jj), T.EE(jj));
            else
                fprintf(fid, 'family %-8s NO %-10s point in the walk\n', fams{i}, rule{1});
            end
        end
    end
end

function dualprint_(fid, quiet, varargin)
    fprintf(fid, varargin{:});  if ~quiet, fprintf(varargin{:}); end
end
function s = tern_(c, a, b), if c, s = a; else, s = b; end, end
function s = empty_(x), if isempty(x), s = '-'; else, s = x; end, end
