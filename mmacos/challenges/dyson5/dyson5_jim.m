function OUT = dyson5_jim(part, over)
%DYSON5_JIM  Jim's comparison (BRIEF_ccmac_dyson_size round 3), spectrometer side.
%
%   DYSON5_JIM('3a') re-cuts the round-2 trade (dyson5_size.mat) into Jim's
%   two-vs-four question: 2 x (3k px, 54 mm slit, CaF2, no meniscus) against
%   4 x (1.5k px, 27 mm slit, silica, no meniscus), with the silica-3k-with-
%   meniscus record and the CaF2-1.5k small end as references.  Per-module and
%   per-SYSTEM glass (module count x edged volume/mass), the single-crystal CaF2
%   CARVE volume (circumscribing rod + a stated grinding margin), the engine
%   image scores, the air-glass crossings and uncoated throughput, and the
%   grating/length geometry.  Writes dyson5_jim_3a.txt.  No engine run -- a re-cut.
%
%   DYSON5_JIM('3b', OVER) runs the three throughput routes (added later).
%
%   NEW FILE, round 3.  CaF2 is NOT priced -- the carve volume is reported for
%   Jim to price.  Margin for the carve: MARGIN_MM all round (radial + axial),
%   stated in the record.
    arguments
        part (1,:) char = '3a'
        over struct = struct()
    end
    here = fileparts(mfilename('fullpath'));
    switch part
        case '3a', OUT = table_3a_(here);
        case '3b', OUT = routes_3b_(here, over);
        otherwise, error('dyson5_jim: part "%s" not implemented yet', part);
    end
end

% ======================================================================
function OUT = routes_3b_(here, over)
%ROUTES_3B_  "Maybe you and AI can do better" -- three measured throughput routes.
    opt = struct('max_iter', 30, 'standoffs_mm', [0.5 1 1.5 2 2.5 3]);
    fn = fieldnames(over);  for k=1:numel(fn), opt.(fn{k}) = over.(fn{k}); end
    P0 = dyson5_params();
    S  = load(fullfile(here, 'dyson5_size.mat'));
    seedD = rowP_(S.OUT, 'D', 130);                 % the 130 mm silica 27-mm-slit R3 design of record (round 2)
    macos.init(P0.model);
    fid = fopen(fullfile(here, 'dyson5_jim_3b.txt'), 'w');  pr = @(varargin) dp_(fid, varargin{:});
    pr('dyson5 round 3b -- throughput routes (%s); 130 mm silica block, 27 mm slit, R3 (no meniscus)\n', datestr(now,'yyyy-mm-dd HH:MM'));
    pr('CONVENTIONS: engine scores on the 7x7 grid; clearance as stage s3 runs it; throughput = uncoated Fresnel at 1 um\n');
    pr('  unless a coating is named.  Route 1 counts a deposited slit / cemented window as INDEX-MATCHED (not air-glass).\n');
    OUT = struct();

    % ---------- Route 1: deposited slit + cemented window -> the two flat-face crossings go index-matched
    % The ray geometry is UNCHANGED (same block, same small standoff): depositing the slit
    % on the glass face and cementing the detector window turns the two flat-face air-glass
    % crossings into glass-glass (index-matched), leaving only the convex pair.  A literal
    % zero standoff is the degenerate concentric case (the m=0 image lands on the slit), so
    % the standoff stays physical; the win is purely the crossing count.
    pr('\nROUTE 1 -- deposited slit + cemented detector window: the two flat-face crossings go index-matched (4 -> 2)\n');
    nSi = sellmeier_local_('Silica', 1e-6);  Tsurf = 1 - frac_(1, nSi);
    r_std = score_r3_(P0, seedD, 130, [], opt.max_iter);                 % the D-130 design of record (standoff free)
    pr('  %-26s face_off %5.2f mm  CRF %6.3f  EE %6.3f  clear %+6.2f mm\n', '130 mm silica R3 (D)', r_std.face_mm, r_std.CRF, r_std.EE, r_std.clear);
    pr('  air gap (slit/window in air)      : 4 air-glass crossings, uncoated throughput %.3f\n', Tsurf^4);
    pr('  deposited slit + cemented window  : 2 air-glass crossings (flat face index-matched), uncoated throughput %.3f\n', Tsurf^2);
    pr('  -> same image and clearance; removing the two flat-face crossings lifts uncoated throughput %.3f -> %.3f (+%.0f%%)\n', ...
       Tsurf^4, Tsurf^2, 100*(Tsurf^2/Tsurf^4 - 1));
    OUT.route1 = struct('design', r_std, 'thru4', Tsurf^4, 'thru2', Tsurf^2);

    % ---------- Route 3: working-distance scan
    pr('\nROUTE 3 -- working-distance scan: slit AND detector standoff equal, R3 re-solved at each (seed chained)\n');
    pr('  %-10s %7s %7s %8s   (block 130 mm silica, 27 mm slit)\n', 'standoff', 'CRF', 'EE', 'clear');
    seed = seedD;  sc = struct('mm',{},'CRF',{},'EE',{},'clear',{});
    for fo = opt.standoffs_mm
        rr = score_r3_(P0, setfo_(seed, fo*1e-3), 130, fo*1e-3, opt.max_iter);
        pr('  %6.1f mm %7.3f %7.3f %+7.2f mm\n', fo, rr.CRF, rr.EE, rr.clear);
        sc(end+1) = struct('mm',fo,'CRF',rr.CRF,'EE',rr.EE,'clear',rr.clear);   %#ok<AGROW>
        seed = rr.P;
    end
    OUT.route3 = sc;

    % ---------- Route 2: broadband AR coating 400-2500 nm (thinfilm_rt, no engine trace)
    pr('\nROUTE 2 -- broadband AR 400-2500 nm (macos.design.thinfilm_rt, Abeles; normal incidence)\n');
    pr('  indices (refractiveindex.info, representative, dispersion over the band neglected -- stated): MgF2 1.384,\n');
    pr('  SiO2 1.46, Al2O3 1.63, Ta2O5 2.10; substrates silica 1.450, CaF2 1.429 at 1 um.\n');
    lams = (400:50:2500)'*1e-9;
    OUT.route2 = struct('glass',{},'uncoated_T4',{},'mgf2_T4',{},'two_T4',{},'four_T4',{});
    for gl = {'Silica','CaF2'}
        nsub = sellmeier_local_(gl{1}, 1e-6);
        Tbare = mean(1 - arrayfun(@(l) R_stack_(zeros(0,2), nsub, l), lams));
        % single quarter-wave MgF2 at 900 nm
        d_mgf2 = 900e-9/(4*1.384);
        Tmgf2 = mean(1 - arrayfun(@(l) R_stack_([1.384 d_mgf2], nsub, l), lams));
        % optimise a 2-layer (MgF2 outer / Al2O3) and a 4-layer (SiO2/Ta2O5 x2) for min mean R
        [T2, L2] = opt_ar_([1.384 1.63], nsub, lams);
        [T4, L4] = opt_ar_([1.46 2.10 1.46 2.10], nsub, lams);
        pr('  %-7s uncoated T/surf %.3f (T^4 %.3f); MgF2 1/4-wave %.3f (%.3f); 2-layer %.3f (%.3f); 4-layer %.3f (%.3f)\n', ...
           gl{1}, Tbare, Tbare^4, Tmgf2, Tmgf2^4, T2, T2^4, T4, T4^4);
        OUT.route2(end+1) = struct('glass',gl{1},'uncoated_T4',Tbare^4,'mgf2_T4',Tmgf2^4,'two_T4',T2^4,'four_T4',T4^4);  %#ok<AGROW>
    end
    pr('  (single-layer MgF2 helps little over a 6:1 band -- the index mismatch to a 1.45 substrate leaves R high at the ends.)\n');
    fclose(fid);
    fprintf('dyson5_jim: wrote dyson5_jim_3b.txt\n');
end

function r = score_r3_(P0, seed, r_mm, fo_fixed, max_iter)
%SCORE_R3_  Solve R3 (no meniscus) at r_mm silica 27-mm-slit; optionally pin face_offset.
    Q = P0;  Q.glass = 'Silica';  Q.block_r_m = r_mm*1e-3;
    Q.npix = [round(27e-3/Q.pixel_m), Q.npix(2)];  Q.slit_m = 27e-3;
    bounds = cell(0,3);
    if ~isempty(fo_fixed), bounds = {'face_offset', fo_fixed, fo_fixed}; end   % lb==ub fixes it
    deck = tempname;  deck = [deck '.in'];
  try
    L = dyson_ladder(Q, 'jim', 'rungs', 3, 'seed', seed, 'deck', deck, ...
                     'nx', Q.ladder_nx, 'nlam', Q.ladder_nlam, 'w_dist', Q.ladder_w_dist, ...
                     'w_blur', Q.ladder_w_blur, 'clear_m', Q.ladder_clear_m, 'max_iter', max_iter, 'quiet', true, 'bounds', bounds);
    rr = L.rung(1);  Re = rr.engine;  G = spectrometer_geom('dyson', rr.P);
    Cl = spectrometer_clearance(G, Q, 'quiet', true);
    r = struct('CRF', Re.crf_max, 'SRF', Re.srf_max, 'EE', Re.ee_min, 'clear', Cl.min_mm, ...
               'face_mm', rr.P.face_offset*1e3, 'P', rr.P);
  catch e
    r = struct('CRF', NaN, 'SRF', NaN, 'EE', NaN, 'clear', NaN, 'face_mm', NaN, 'P', seed);
    fprintf('  (score_r3_ failed at r=%g fo=%s: %s)\n', r_mm, mat2str(fo_fixed), regexprep(e.message,'\s+',' '));
  end
end

function P = setfo_(P, fo), P.face_offset = fo; end

function f = frac_(n_inc, n_sub), f = ((n_sub - n_inc)/(n_sub + n_inc))^2; end

function R = R_stack_(layers2, nsub, lam)
%R_STACK_  Normal-incidence power reflectance of a stack on nsub (layers2 = [n thick; ...]).
    o = macos.design.thinfilm_rt(layers2, 1.0, nsub, 0, lam);
    R = o.Rs;                                        % s=p at normal incidence
end

function [Tmean, L] = opt_ar_(nlayers, nsub, lams)
%OPT_AR_  Optimise layer thicknesses (quarter-wave-ish) for min mean R over the band.
    n = nlayers(:);
    d0 = 1000e-9 ./ (4*n);                           % quarter-wave at 1 um seed
    f = @(d) mean(arrayfun(@(l) R_stack_([n max(d(:),1e-9)], nsub, l), lams));
    o = optimset('Display','off','MaxFunEvals',2000,'MaxIter',2000);
    d = fminsearch(f, d0, o);  d = max(d(:),1e-9);
    Tmean = 1 - f(d);  L = [n d];
end

function n = sellmeier_local_(glass, lam)
%SELLMEIER_LOCAL_  Mirror of spectrometer_geom's sellmeier_ (silica/CaF2), for the AR substrate index.
    l2 = (lam*1e6)^2;
    switch glass
        case 'Silica', B=[0.6961663 0.4079426 0.8974794]; C=[0.0684043 0.1162414 9.896161].^2;
        case 'CaF2',   B=[0.5675888 0.4710914 3.8484723]; C=[0.050263605 0.1003909 34.649040].^2;
        otherwise, error('sellmeier_local_: %s', glass);
    end
    n = sqrt(1 + sum(B.*l2./(l2 - C)));
end

% ======================================================================
function OUT = table_3a_(here)
    MARGIN_MM = 10;                                   % grinding/mount margin, all round, for the single-crystal carve
    S = load(fullfile(here, 'dyson5_size.mat'));  T = S.OUT.table;
    g = @(fam, r) one_(T, fam, r);

    % the two systems Jim is comparing, plus the two reference rows
    rowdefs = {
        '2 x CaF2  3k  54mm (F, no men)',   g('F', 240), 2, 'CaF2',   true
        '4 x SiO2  1.5k 27mm (D, no men)',  g('D', 130), 4, 'Silica', false
        '  D small end (100 mm)',           g('D', 100), 4, 'Silica', false
        '  F comfortable EE (300 mm)',      g('F', 300), 2, 'CaF2',   true
        'ref: SiO2 3k 54mm 220 (R4, 4mm meniscus, 8x)', g('A', 220), 2, 'Silica', false
        'ref: CaF2 1.5k 27mm 80 (small)',   g('E', 80),  4, 'CaF2',   true };

    fid = fopen(fullfile(here, 'dyson5_jim_3a.txt'), 'w');
    pr = @(varargin) dp_(fid, varargin{:});
    pr('dyson5 round 3a -- Jim''s two-vs-four comparison (%s), engine scores re-cut from dyson5_size.mat\n', datestr(now,'yyyy-mm-dd HH:MM'));
    pr('CONVENTIONS: block radius = thickness (concentric Dyson); edged = circumscribing rod (clear diam x thickness); carve =\n');
    pr('  single-crystal CaF2 cylinder (clear diam + %d mm) x (thickness + %d mm), the blank a lens is ground from (n/a for\n', 2*MARGIN_MM, 2*MARGIN_MM);
    pr('  fused silica -- a melt, any size); throughput = uncoated Fresnel at 1 um over nX air-glass crossings.  CaF2 NOT priced:\n');
    pr('  the carve VOLUME is stated for Jim to price (price grows faster than volume; a boule is several crystals).\n\n');
    pr('%-46s %4s %6s %6s %6s %7s %7s %4s %5s %8s %8s %9s %10s %9s %9s %7s %7s\n', ...
       'configuration', 'r_mm', 'CRF', 'SRF', 'EE', 'smile', 'keyst', 'nX', 'thru', 'edgL/mod', 'edgkg/md', 'sysGlassL', 'sysGlasskg', 'carveL/md', 'sysCarveL', 'gratD', 'length');

    OUT.rows = struct('label',{},'r_mm',{},'CRF',{},'SRF',{},'EE',{},'smile',{},'keystone',{}, ...
                      'n_cross',{},'thru',{},'edged_L',{},'edged_kg',{},'nmod',{},'sys_glass_L',{},'sys_glass_kg',{}, ...
                      'carve_L',{},'sys_carve_L',{},'gratD_mm',{},'length_mm',{});
    for i = 1:size(rowdefs,1)
        lab = rowdefs{i,1};  r = rowdefs{i,2};  nmod = rowdefs{i,3};  glass = rowdefs{i,4}; isCaF2 = rowdefs{i,5};
        rho = 2.20;  if isCaF2, rho = 3.18; end
        sysL = nmod*r.edged_L;  syskg = nmod*r.edged_kg;
        if isCaF2
            carveL  = pi*((r.blockD_mm+2*MARGIN_MM)/2/10)^2*((r.blockT_mm+2*MARGIN_MM)/10)/1000;
            sysCarve = nmod*carveL;
        else
            carveL = NaN;  sysCarve = NaN;                 % fused silica: not a carve
        end
        pr('%-46s %4d %6.3f %6.3f %6.3f %7.4f %7.4f %4d %5.3f %7.2f %7.1f %8.2f %8.1f %8s %8s %8.1f %7.1f\n', ...
           lab, round(r.r_mm), r.CRF, r.SRF, r.EE, r.smile, r.keystone, r.n_cross, r.thru, ...
           r.edged_L, r.edged_kg, sysL, syskg, numornan_(carveL), numornan_(sysCarve), r.gratD_mm, r.length_mm);
        OUT.rows(end+1) = struct('label',lab,'r_mm',r.r_mm,'CRF',r.CRF,'SRF',r.SRF,'EE',r.EE,'smile',r.smile, ...
            'keystone',r.keystone,'n_cross',r.n_cross,'thru',r.thru,'edged_L',r.edged_L,'edged_kg',r.edged_kg, ...
            'nmod',nmod,'sys_glass_L',sysL,'sys_glass_kg',syskg,'carve_L',carveL,'sys_carve_L',sysCarve, ...
            'gratD_mm',r.gratD_mm,'length_mm',r.length_mm);
    end
    pr('\nSYSTEM TOTALS Jim asked for:\n');
    pr('  2 x CaF2 240 mm (3k):  grating/detector/slit count 2 / 2 x 3k;  glass %.1f L / %.1f kg edged, carve %.1f L single-crystal CaF2\n', ...
       2*g('F',240).edged_L, 2*g('F',240).edged_kg, 2*pi*((g('F',240).blockD_mm+2*MARGIN_MM)/2/10)^2*((g('F',240).blockT_mm+2*MARGIN_MM)/10)/1000);
    pr('  4 x SiO2 130 mm (1.5k): grating/detector/slit count 4 / 4 x 1.5k; glass %.1f L / %.1f kg edged fused silica (no carve)\n', ...
       4*g('D',130).edged_L, 4*g('D',130).edged_kg);
    pr('  4 x SiO2 100 mm (1.5k): glass %.1f L / %.1f kg edged (the smallest D that matches R4)\n', 4*g('D',100).edged_L, 4*g('D',100).edged_kg);
    fclose(fid);
    OUT.margin_mm = MARGIN_MM;  OUT.table = T;
    fprintf('dyson5_jim: wrote dyson5_jim_3a.txt\n');
end

function P = rowP_(OUT, fam, rad)
%ROWP_  The solved parameter struct for a family/radius row (the continuation seed).
    rows = OUT.rows;
    for i = 1:numel(rows)
        if strcmp(rows(i).family, fam) && rows(i).r_mm == rad && strcmp(rows(i).variant, 'solve')
            P = rows(i).P;  return
        end
    end
    error('dyson5_jim: no solved row %s r=%d in dyson5_size.mat', fam, rad);
end

function r = one_(T, fam, rad)
    sel = strcmp(T.family, fam) & T.r_mm == rad & strcmp(T.variant, 'solve');
    assert(any(sel), 'dyson5_jim: row %s r=%d not in dyson5_size.mat', fam, rad);
    i = find(sel, 1);
    r = struct('r_mm', T.r_mm(i), 'CRF', T.CRF(i), 'SRF', T.SRF(i), 'EE', T.EE(i), ...
               'smile', T.smile(i), 'keystone', T.keystone(i), 'n_cross', T.n_cross(i), 'thru', T.thru(i), ...
               'edged_L', T.edged_L(i), 'edged_kg', T.edged_kg(i), 'blockD_mm', T.blockD_mm(i), ...
               'blockT_mm', T.blockT_mm(i), 'gratD_mm', T.gratD_mm(i), 'length_mm', T.length_mm(i));
end

function s = numornan_(x), if isnan(x), s = 'n/a'; else, s = sprintf('%.2f', x); end, end
function dp_(fid, varargin), fprintf(fid, varargin{:}); fprintf(varargin{:}); end
