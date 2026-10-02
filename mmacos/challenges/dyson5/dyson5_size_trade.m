function OUT = dyson5_size_trade(over)
%DYSON5_SIZE_TRADE  How small can the Dyson block be at R4-of-record performance?
%
%   OUT = DYSON5_SIZE_TRADE() answers Dave's size question (BRIEF_ccmac_dyson_size):
%   the 220 mm silica block of R4 of record is heavy, thick and non-uniform; how
%   much can it shrink while still MATCHING R4 (smile and keystone < 0.1 px,
%   CRF <= 1.33, SRF <= 2.05, EE >= 0.76, no variable on a bound)?  Each point is
%   one R4 solve (the ladder's rung 5) warm-started by CONTINUATION from the
%   PREVIOUS point's solved design, stepping the block radius down ~10 % at a
%   time, then emitted and ENGINE-scored.  Three families walk downward:
%
%       A  silica, 54 mm slit, 220 -> 150 mm  (also the solver check vs beat 4e)
%       B  CaF2,   54 mm slit, 220 -> 100 mm
%       C  silica AND CaF2, 27 mm slit (1500 px: two modules share the swath),
%          220 -> 60 mm
%
%   A point CLOSES only when it matches R4 at the precision the spec is quoted
%   (CRF, EE, SRF compared ROUNDED to R4's own 2 / 2 / 2 decimals; R4 of record
%   -- CRF 1.327, EE 0.759, SRF 2.032 -- therefore counts as closing) and NO
%   variable sits on a bound.  A walk stops two points after it last closes.
%
%   OUT = DYSON5_SIZE_TRADE(OVER) overrides options:
%     .which   families to run, subset of {'A','B','C'}   default all three
%     .max_iter  lsqnonlin iterations per point           default 30 (beat 4e's)
%     .bound_scale_runs  re-solve an on-bound point with the meniscus-thickness
%                        bound scaled by the block radius, reported beside it
%                        (BRIEF: a bound written for 220 mm, scaled, is a real
%                        statement; a point on a bound is NOT closed)   default true
%     .radiiA/.radiiB/.radiiC  the radius walks (mm)
%     .quiet   suppress the per-point console line          default false
%
%   The R4 SEED is read from dyson5_s3.mat the way stage_s4env_ reads it (NOT by
%   calling the stage, which rewrites the envelope's record files).  All knobs
%   come from dyson5_params(); family fields are overridden on a copy of P, never
%   through dyson5_params(over), so no field is added behind the params file.
%
%   Writes dyson5_size.txt (the table) and dyson5_size.mat (OUT).  Decks land as
%   dyson5_size_<family>_r<mm>.in.  dyson5_size_fig draws the figure.
%
%   NEW FILE (BRIEF_ccmac_dyson_size).  Does NOT touch dyson5_run.m,
%   dyson5_params.m, dyson5_envelope.m or dyson_ladder.m.
    arguments
        over struct = struct()
    end
    here = fileparts(mfilename('fullpath'));
    P0  = dyson5_params();
    tag = 'dyson5_size';

    opt = struct('which', {{'A', 'B', 'C'}}, 'max_iter', P0.env_max_iter, 'quiet', false, ...
                 'bound_scale_runs', true, ...
                 'radiiA', [220 200 180 165 150], ...
                 'radiiB', [220 200 180 160 140 120 100], ...
                 'radiiC', [220 190 160 130 100 80 60]);
    fn = fieldnames(over);
    for k = 1:numel(fn)
        if ~isfield(opt, fn{k}), error('dyson5_size_trade:unknown', 'unknown option "%s"', fn{k}); end
        opt.(fn{k}) = over.(fn{k});
    end

    % ---- the R4 seed, read the stage_s4env_ way (read it; do not call the stage)
    fn3 = fullfile(here, 'dyson5_s3.mat');
    assert(isfile(fn3), 'dyson5_size_trade needs the ladder record %s (stage s3)', fn3);
    S3 = load(fn3);  L3 = S3.S;
    kr = find(strncmp({L3.rung.name}, 'R4 ', 3), 1);
    assert(~isempty(kr), 'dyson5_size_trade: the R4 rung is not in %s', fn3);
    r4seed = L3.rung(kr).P;

    % ---- family (walk) definitions: name, glass, slit length (m; [] = default 54 mm), radii (mm), force every rung
    walks = struct('key', {}, 'glass', {}, 'slit_m', {}, 'radii', {}, 'force_full', {});
    if any(strcmp(opt.which, 'A'))
        walks(end+1) = mkwalk_('A', 'Silica', [], opt.radiiA, true);     % the solver check runs its full 220..150 list
    end
    if any(strcmp(opt.which, 'B'))
        walks(end+1) = mkwalk_('B', 'CaF2', [], opt.radiiB, false);
    end
    if any(strcmp(opt.which, 'C'))
        walks(end+1) = mkwalk_('C-silica', 'Silica', 27e-3, opt.radiiC, false);
        walks(end+1) = mkwalk_('C-CaF2',  'CaF2',   27e-3, opt.radiiC, false);
    end

    macos.init(P0.model);

    txt = fullfile(here, [tag '.txt']);
    fid = fopen(txt, 'w');
    hdr_(fid, opt);
    pr = @(varargin) dualprint_(fid, opt.quiet, varargin{:});

    rows = struct('family', {}, 'glass', {}, 'slit_mm', {}, 'r_mm', {}, 'variant', {}, ...
                  'smile', {}, 'keystone', {}, 'CRF', {}, 'SRF', {}, 'EE', {}, 'closes', {}, ...
                  'fails', {}, 'on_bounds', {}, 'blockT_mm', {}, 'blockD_mm', {}, ...
                  'edged_L', {}, 'edged_kg', {}, 'gratR_mm', {}, 'gratD_mm', {}, 'length_mm', {}, ...
                  'clear_min_mm', {}, 'clear_pair', {}, 'unif_inhom_waves', {}, 'unif_therm_waves', {}, ...
                  'P', {});

    for w = 1:numel(walks)
        W = walks(w);
        pr('\n--- family %s: %s, %g mm slit ---\n', W.key, W.glass, slit_len_(P0, W)*1e3);
        seed = r4seed;                       % the record's design knobs; the first point re-solves the envelope point
        last_close = 0;
        for i = 1:numel(W.radii)
            r_mm = W.radii(i);
            Q = qfor_(P0, W, r_mm);
            tagp = sprintf('%s_%s_r%d', tag, W.key, round(r_mm));
            deck = fullfile(here, [tagp '.in']);
            R = solve_point_(Q, tagp, deck, seed, opt.max_iter, {});
            row = row_(W, r_mm, 'solve', R);
            rows(end+1) = row;   %#ok<AGROW>
            print_row_(pr, row);
            seed = R.P;                      % CONTINUATION: next point warm-starts from this solved design

            % a point on a bound is not closed; re-solve with the meniscus-thickness
            % bound scaled by the radius and report it BESIDE (do not widen silently)
            if ~isempty(R.on_bounds) && opt.bound_scale_runs
                eb = scaled_bounds_(R.on_bounds, r_mm);
                if ~isempty(eb)
                    tagp2 = [tagp '_bs'];
                    deck2 = fullfile(here, [tagp2 '.in']);
                    R2 = solve_point_(Q, tagp2, deck2, seed, opt.max_iter, eb);
                    row2 = row_(W, r_mm, 'bound-scaled', R2);
                    rows(end+1) = row2;   %#ok<AGROW>
                    print_row_(pr, row2);
                end
            end

            if row.closes, last_close = i; end
            if ~W.force_full
                if last_close >= 1 && (i - last_close) >= 2, break; end   % two points past the last that closed
                if last_close == 0 && i >= 3, break; end                  % never closed (should not happen at 220)
            end
        end
    end

    T = struct2table(rmfield(rows, 'P'), 'AsArray', true);   % P (a nested struct) stays in OUT.rows only
    OUT = struct('table', T, 'rows', rows, 'walks', walks, 'r4seed', r4seed, 'opt', opt, 'P0', P0);
    % record of record, for the figure's reference lines
    OUT.record = struct('CRF', 1.327, 'EE', 0.759, 'SRF', 2.032, 'smile', 0.0051, 'keystone', 0.0026, ...
                        'r_mm', 220, 'blockT_mm', 221.2, 'blockD_mm', 146.6, 'edged_L', NaN, 'edged_kg', NaN);
    [OUT.record.edged_L, OUT.record.edged_kg] = edged_(146.6, 221.2, 'Silica');

    fprintf(fid, '\n');
    foot_(fid, OUT);
    fclose(fid);
    save(fullfile(here, [tag '.mat']), 'OUT');
    if ~opt.quiet, fprintf('dyson5_size_trade: wrote %s.txt and %s.mat (%d rows)\n', tag, tag, height(T)); end
end

% ======================================================================
function W = mkwalk_(key, glass, slit_m, radii, force_full)
    W = struct('key', key, 'glass', glass, 'slit_m', slit_m, 'radii', radii, 'force_full', force_full);
end

function L = slit_len_(P0, W)
    if isempty(W.slit_m), L = P0.npix(1)*P0.pixel_m; else, L = W.slit_m; end
end

function Q = qfor_(P0, W, r_mm)
%QFOR_  A copy of the params with this family's spec overrides (never via dyson5_params(over)).
    Q = P0;
    Q.glass     = W.glass;
    Q.block_r_m = r_mm*1e-3;
    if ~isempty(W.slit_m)                              % mirror dyson5_envelope's slit axis: npix follows the slit
        Q.npix   = [round(W.slit_m/Q.pixel_m), Q.npix(2)];
        Q.slit_m = W.slit_m;                           % harmless extra field (the envelope sets it too)
    end
end

function R = solve_point_(Q, tagp, deck, seed, max_iter, extra_bounds)
%SOLVE_POINT_  One R4 solve (ladder rung 5) with point_'s options and bounds.
    bounds = {'men_ca', 0.05, 8; 'men_cb', 0.05, 8};   % beat 4e's envelope widening (R4 of record sits on 0.5)
    if ~isempty(extra_bounds), bounds = [bounds; extra_bounds]; end
  try
    L = dyson_ladder(Q, tagp, 'rungs', 5, 'seed', seed, 'deck', deck, ...
                     'nx', Q.ladder_nx, 'nlam', Q.ladder_nlam, ...
                     'w_dist', Q.ladder_w_dist, 'w_blur', Q.ladder_w_blur, ...
                     'clear_m', Q.ladder_clear_m, 'max_iter', max_iter, ...
                     'quiet', true, 'bounds', bounds);
    r  = L.rung(1);
    Re = r.engine;
    G  = spectrometer_geom('dyson', r.P);
    g  = geom_cols_(G);
    Cl = spectrometer_clearance(G, Q, 'quiet', true);  % the gate as stage s3 runs it

    % uniformity, order of magnitude (ESTIMATES): double-pass on-axis glass path
    L_dp = 2 * g.block_thick_mm * 1e-3;                % m (down the block and back)
    lam  = 1e-6;                                       % waves quoted at 1 um
    unif_inhom = 1e-6 * L_dp / lam;                    % index inhomogeneity dn = 1e-6
    dndT = 1e-5;                                        % |dn/dT| ~ 1e-5 /K (silica +, CaF2 -)
    unif_therm = dndT * 0.1 * L_dp / lam;              % a 0.1 K gradient

    [edL, edkg] = edged_(g.block_diam_mm, g.block_thick_mm, Q.glass);

    R = struct();
    R.engine = Re;  R.P = r.P;  R.on_bounds = r.on_bounds;  R.G = G;  R.Cl = Cl;
    R.blockT_mm = g.block_thick_mm;  R.blockD_mm = g.block_diam_mm;
    R.gratR_mm = G.Rg*1e3;  R.gratD_mm = g.grating_diam_mm;  R.length_mm = g.length_mm;
    R.edged_L = edL;  R.edged_kg = edkg;
    R.unif_inhom_waves = unif_inhom;  R.unif_therm_waves = unif_therm;
    R.clear_min_mm = Cl.min_mm;
    if ~isempty(Cl.table) && height(Cl.table) >= 1
        R.clear_pair = sprintf('%s vs %s', Cl.table.leg{1}, Cl.table.body{1});
    else
        R.clear_pair = '';
    end
    % closure: match R4 of record at the precision the spec is quoted
    fails = {};
    if ~(Re.smile_max   < 0.10),            fails{end+1} = 'smile';    end
    if ~(Re.keystone_max< 0.10),            fails{end+1} = 'keystone'; end
    if ~(round(Re.crf_max, 2) <= 1.33),     fails{end+1} = 'CRF';      end
    if ~(round(Re.srf_max, 2) <= 2.05),     fails{end+1} = 'SRF';      end
    if ~(round(Re.ee_min,  2) >= 0.76),     fails{end+1} = 'EE';       end
    R.fails  = fails;
    R.closes = isempty(fails) && isempty(r.on_bounds);
  catch e
    R = failed_point_(Q, seed, e.message);
  end
end

function R = failed_point_(Q, seed, msg)
%FAILED_POINT_  A point whose solve or score threw (the chain lost the beam at a
%   small radius): a NaN row flagged 'solve', so the walk continues and the
%   stopping rule still bites.
    nanE = struct('smile_max', NaN, 'keystone_max', NaN, 'crf_max', NaN, 'srf_max', NaN, 'ee_min', NaN);
    R = struct('engine', nanE, 'P', seed, 'on_bounds', {{}}, 'G', [], 'Cl', [], ...
               'blockT_mm', NaN, 'blockD_mm', NaN, 'gratR_mm', NaN, 'gratD_mm', NaN, 'length_mm', NaN, ...
               'edged_L', NaN, 'edged_kg', NaN, 'unif_inhom_waves', NaN, 'unif_therm_waves', NaN, ...
               'clear_min_mm', NaN, 'clear_pair', '', 'fails', {{['ERROR:' regexprep(msg, '\s+', ' ')]}}, 'closes', false);
end

function row = row_(W, r_mm, variant, R)
    Re = R.engine;
    row = struct('family', W.key, 'glass', W.glass, 'slit_mm', round(slit_len_v_(W, r_mm)*1e3), ...
                 'r_mm', r_mm, 'variant', variant, ...
                 'smile', Re.smile_max, 'keystone', Re.keystone_max, 'CRF', Re.crf_max, ...
                 'SRF', Re.srf_max, 'EE', Re.ee_min, 'closes', R.closes, ...
                 'fails', {strjoin(R.fails, ',')}, 'on_bounds', {strjoin(R.on_bounds, ',')}, ...
                 'blockT_mm', R.blockT_mm, 'blockD_mm', R.blockD_mm, ...
                 'edged_L', R.edged_L, 'edged_kg', R.edged_kg, ...
                 'gratR_mm', R.gratR_mm, 'gratD_mm', R.gratD_mm, 'length_mm', R.length_mm, ...
                 'clear_min_mm', R.clear_min_mm, 'clear_pair', {R.clear_pair}, ...
                 'unif_inhom_waves', R.unif_inhom_waves, 'unif_therm_waves', R.unif_therm_waves, ...
                 'P', R.P);
end

function L = slit_len_v_(W, ~)
    if isempty(W.slit_m), L = 54e-3; else, L = W.slit_m; end
end

function eb = scaled_bounds_(on_bounds, r_mm)
%SCALED_BOUNDS_  The meniscus-thickness lower bound (4 mm, written for 220 mm)
%   scaled by the block radius; the vertex lower bound already follows the block
%   in dyson_ladder.  Only the documented thickness bound is scaled here.
    eb = {};
    if any(strcmp(on_bounds, 'men_t'))
        lb = 0.004 * (r_mm/220);                       % 4 mm at 220, proportional below
        eb = [eb; {'men_t', lb, 0.040}];
    end
end

function g = geom_cols_(G)
%GEOM_COLS_  Element sizes from the chain geometry (the formulas of dyson5_trade's
%   geom_, replicated here so this new file touches nothing).
    g.length_mm = (G.surf(G.iG).vpt(3) - G.slit(3))*1e3;
    ib  = find(strcmp({G.surf.name}, 'BlockSphereOut'), 1);
    ifc = find(strcmp({G.surf.name}, 'BlockFaceIn'), 1);
    F   = G.footprints();  m = 5e-3;                   % declared aperture = footprint + 5 mm margin
    g.block_diam_mm  = 2*(F(ib).radius + m)*1e3;
    g.block_thick_mm = (G.surf(ib).vpt(3) - G.surf(ifc).C(3))*1e3;
    g.grating_diam_mm = 2*(F(G.iG).radius + m)*1e3;
    im = find(strcmp({G.surf.name}, 'MenA_out'), 1);
    if isempty(im), g.men_diam_mm = 0; else, g.men_diam_mm = 2*(F(im).radius + m)*1e3; end
end

function [vol_L, mass_kg] = edged_(diam_mm, thick_mm, glass)
%EDGED_  The EDGED block as the circumscribing rod (Dave's framing: a rod of the
%   clear diameter and the on-axis thickness), litres and kg.  Silica 2.20,
%   CaF2 3.18 g/cm^3.  (The plano-convex optic is a touch lighter -- the shallow
%   cap -- but the rod is the number the brief quotes: 147 x 221 mm = 3.7 L / 8 kg.)
    rho = 2.20;  if strcmpi(glass, 'CaF2'), rho = 3.18; end
    vol_cm3 = pi*(diam_mm/2/10)^2 * (thick_mm/10);     % mm -> cm
    vol_L   = vol_cm3/1000;
    mass_kg = vol_cm3*rho/1000;
end

function hdr_(fid, opt)
    fprintf(fid, 'dyson5 block-size trade (%s) -- CCMac, BRIEF_ccmac_dyson_size\n', datestr(now, 'yyyy-mm-dd HH:MM'));
    fprintf(fid, 'CONVENTIONS: each point one R4 solve (ladder rung 5, %d iterations on the exact chain, merit weights w_dist=10 /\n', opt.max_iter);
    fprintf(fid, '  w_blur=1 of record), warm-started by CONTINUATION from the previous radius''s solved design; the deck emitted\n');
    fprintf(fid, '  with apertures and ENGINE-scored on the 7x7 grid.  CLOSES = matches R4 of record: smile and keystone < 0.1 px,\n');
    fprintf(fid, '  CRF <= 1.33, SRF <= 2.05, EE >= 0.76 (CRF/SRF/EE compared ROUNDED to R4''s 2 dp, so R4 itself -- 1.327 / 2.032 /\n');
    fprintf(fid, '  0.759 -- closes) AND no variable on a bound.  meniscus curvature bounds [0.05, 8] /m (beat 4e); vertex lower\n');
    fprintf(fid, '  bound follows the block (+20 mm).  edged = circumscribing rod (clear diameter x on-axis thickness).  unif =\n');
    fprintf(fid, '  ESTIMATES: double-pass on-axis glass path x {dn=1e-6 ; 0.1 K at |dn/dT|=1e-5/K}, in waves at 1 um.\n');
    fprintf(fid, '%-9s %-7s %5s %5s %-12s %7s %7s %6s %6s %6s %-5s %-10s %-10s %7s %7s %6s %6s %7s %7s %7s %8s %6s %6s\n', ...
        'family', 'glass', 'slit', 'r', 'variant', 'smile', 'keyst', 'CRF', 'SRF', 'EE', 'close', 'fails', 'on_bounds', ...
        'blockT', 'blockD', 'edgeL', 'edgekg', 'gratR', 'gratD', 'length', 'clear', 'uInh', 'uThm');
    fprintf(fid, '%-9s %-7s %5s %5s %-12s %7s %7s %6s %6s %6s %-5s %-10s %-10s %7s %7s %6s %6s %7s %7s %7s %8s %6s %6s\n', ...
        '', '', 'mm', 'mm', '', 'px', 'px', 'px', 'px', '1px', '', '', '', 'mm', 'mm', 'L', 'kg', 'mm', 'mm', 'mm', 'mm/pair', 'wav', 'wav');
end

function print_row_(pr, r)
    pr('%-9s %-7s %5d %5d %-12s %7.4f %7.4f %6.3f %6.3f %6.3f %-5s %-10s %-10s %7.1f %7.1f %6.2f %6.1f %7.1f %7.1f %7.1f %8.2f %6.2f %6.2f\n', ...
       r.family, r.glass, r.slit_mm, r.r_mm, r.variant, r.smile, r.keystone, r.CRF, r.SRF, r.EE, ...
       tern_(r.closes, 'yes', 'NO'), empty_(r.fails), empty_(r.on_bounds), r.blockT_mm, r.blockD_mm, ...
       r.edged_L, r.edged_kg, r.gratR_mm, r.gratD_mm, r.length_mm, r.clear_min_mm, ...
       r.unif_inhom_waves, r.unif_therm_waves);
end

function foot_(fid, OUT)
    fprintf(fid, 'record (R4, silica 220 mm): CRF %.3f EE %.3f SRF %.3f; edged rod %.1f L / %.1f kg (147 x 221 mm).\n', ...
        OUT.record.CRF, OUT.record.EE, OUT.record.SRF, OUT.record.edged_L, OUT.record.edged_kg);
    % smallest closing radius per family
    T = OUT.table;  fams = unique(T.family, 'stable');
    for i = 1:numel(fams)
        sel = strcmp(T.family, fams{i}) & T.closes & strcmp(T.variant, 'solve');
        if any(sel)
            rr = T.r_mm(sel);  [rmin, j] = min(rr);  idx = find(sel);  jj = idx(j);
            fprintf(fid, 'family %-8s smallest closing radius %3d mm: blockT %.0f mm, edged %.2f L / %.1f kg, CRF %.3f EE %.3f\n', ...
                fams{i}, rmin, T.blockT_mm(jj), T.edged_L(jj), T.edged_kg(jj), T.CRF(jj), T.EE(jj));
        else
            fprintf(fid, 'family %-8s NO closing radius in the walk\n', fams{i});
        end
    end
end

function dualprint_(fid, quiet, varargin)
    fprintf(fid, varargin{:});
    if ~quiet, fprintf(varargin{:}); end
end
function s = tern_(c, a, b), if c, s = a; else, s = b; end, end
function s = empty_(x), if isempty(x), s = '-'; else, s = x; end, end
