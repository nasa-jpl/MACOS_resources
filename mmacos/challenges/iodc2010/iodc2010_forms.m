function [C, info] = iodc2010_forms(nElt, p, opts)
%IODC2010_FORMS  Enumerate green-lens forms: sign patterns, seed spacings, stop places.
%   C = iodc2010_forms(nElt, p) returns a struct array of SEED lenses with
%   nElt elements, each already scaled to |EFL| = p.efl_mm, for the outer
%   design loop (iodc2010_outer) to pre-screen and solve.
%
%   The enumeration (the paper's own description of the space, Sec. 4):
%   * eight lens forms per element -- equi-convex [+ -], equi-concave [- +],
%     plano-convex [+ 0] / [0 -], plano-concave [- 0] / [0 +], meniscus
%     [+ +] / [- -] (plano-plano has no power and is left out);
%   * PETZVAL BALANCE: with one glass and one |R| every curved surface has
%     the same |Petzval contribution|, and 36 of the 37 entries had equal
%     numbers of converging and diverging surfaces (Sec. 5, Olczak the
%     exception).  With focus NOT removed from the RMS, field curvature is
%     the first thing that ends the field, so the balanced patterns are the
%     ones enumerated (opts.petzval = 'zero'; 'any' keeps them all);
%   * the air spaces from a grid in units of the radius (opts.air_grid; the
%     net power of a balanced lens comes from its spacings, which is why the
%     entries are long -- Fig. 15, R ~ 4 OAL^0.6), every glass thickness
%     opts.glass_frac of the radius;
%   * the stop at the middle of each internal air space, ahead of the first
%     surface, or behind the last ('front' / 'air j' / 'back').
%   Each combination is traced paraxially (iodc2010_paraxial): kept when the
%   focal length is finite (either sign: an intermediate image is allowed),
%   the image is real and beyond the last surface, the pupils are finite,
%   and the scaled radius lies in opts.R_range.  The stop semi-diameter is
%   set so the paraxial entrance pupil is opts.D_target mm.
%
%   C(i): .L (the seed, iodc2010_lens, scaled), .pattern (1x2nElt signs),
%   .pattern_id, .gaps_R (air spaces in R), .stop_place, .efl, .R, .oal,
%   .D, .intermediate (the marginal ray crosses the axis inside the lens).
%   info: counts at each stage of the pruning.
arguments
    nElt (1,1) double {mustBeInteger, mustBePositive}
    p struct
    opts.air_grid (1,:) double = [0.1 0.25 0.5 1 2]
    opts.glass_frac (1,1) double = 0.05
    opts.R0 (1,1) double = 100
    opts.D_target (1,1) double = 10
    opts.petzval (1,:) char = 'zero'
    opts.R_range (1,2) double = [5 5000]
    opts.stop_places (1,:) cell = {'front', 'air', 'back'}
end
forms = {[1 -1], [-1 1], [1 0], [0 -1], [-1 0], [0 1], [1 1], [-1 -1]};
fname = {'bicx', 'bicv', 'pcx', 'xcp', 'pcv', 'vcp', 'men+', 'men-'};
% converging surfaces: air->glass with +R, glass->air with -R
conv_of = @(f) (f(1) == 1) + (f(2) == -1);
div_of  = @(f) (f(1) == -1) + (f(2) == 1);

nF = numel(forms);
idx = cell(1, nElt);  [idx{:}] = ndgrid(1:nF);
combos = reshape(cat(nElt + 1, idx{:}), [], nElt);     % every nElt-tuple of forms
info.n_patterns_all = size(combos, 1);
keep = true(size(combos, 1), 1);
for i = 1:size(combos, 1)
    cv = 0;  dv = 0;
    for e = 1:nElt, cv = cv + conv_of(forms{combos(i, e)});  dv = dv + div_of(forms{combos(i, e)}); end
    if strcmp(opts.petzval, 'zero'), keep(i) = (cv == dv); end
end
combos = combos(keep, :);
info.n_patterns_balanced = size(combos, 1);

nG = numel(opts.air_grid);
if nElt > 1
    gi = cell(1, nElt - 1);  [gi{:}] = ndgrid(1:nG);
    gaps = reshape(cat(nElt, gi{:}), [], nElt - 1);
else
    gaps = zeros(1, 0);
end
places = {};
if any(strcmp(opts.stop_places, 'front')), places{end+1} = 0; end
if any(strcmp(opts.stop_places, 'air')), for j = 1:nElt-1, places{end+1} = j; end, end %#ok<AGROW>
if any(strcmp(opts.stop_places, 'back')), places{end+1} = nElt; end

C = struct('L', {}, 'pattern', {}, 'pattern_id', {}, 'gaps_R', {}, 'stop_place', {}, ...
    'efl', {}, 'R', {}, 'oal', {}, 'D', {}, 'intermediate', {});
n_tried = 0;  n_parax = 0;
R0 = opts.R0;  tg = opts.glass_frac * R0;
glass = repmat({'N-BK7', 'air'}, 1, nElt);
for i = 1:size(combos, 1)
    sgn = zeros(1, 2 * nElt);
    pid = '';
    for e = 1:nElt, sgn(2*e-1:2*e) = forms{combos(i, e)};  pid = [pid fname{combos(i, e)} '_']; end %#ok<AGROW>
    pid = pid(1:end-1);
    for g = 1:size(gaps, 1)
        t = zeros(1, 2 * nElt);
        t(1:2:end) = tg;
        for j = 1:nElt-1, t(2*j) = opts.air_grid(gaps(g, j)) * R0; end
        t(end) = NaN;
        for s = 1:numel(places)
            n_tried = n_tried + 1;
            pl = places{s};
            z = [0, cumsum(t(1:end-1))];
            if pl == 0,        zs = -0.1 * R0;                       sp = 'front';
            elseif pl < nElt,  zs = z(2*pl) + 0.5 * t(2*pl);          sp = sprintf('air%d', pl);
            else,              zs = z(end) + 0.1 * R0;                sp = 'back';
            end
            name = sprintf('f%d_%s_g%d_%s', nElt, pid, g, sp);
            try
                L = iodc2010_lens('R', R0, 'sgn', sgn, 't', t, 'glass', glass, 'stop', [zs 1], 'name', name);
                P = iodc2010_paraxial(L, p.lambda_nm);
            catch
                continue
            end
            if ~isfinite(P.efl) || P.efl == 0 || ~isfinite(P.z_ep) || ~isfinite(P.D), continue; end
            if P.z_img <= L.z(end), continue; end                      % the image must be real, in air
            if pl == nElt && zs >= P.z_img, continue; end              % a rear stop must sit before the image
            sc = p.efl_mm / abs(P.efl);
            Rs = R0 * sc;
            if Rs < opts.R_range(1) || Rs > opts.R_range(2), continue; end
            n_parax = n_parax + 1;
            Ls = L;  Ls.R = Rs;  Ls.t = L.t * sc;  Ls.z = L.z * sc;
            Ls.stop = [zs * sc, 1];
            Ps = iodc2010_paraxial(Ls, p.lambda_nm);                   % D for a = 1 -> a for D_target
            Ls.stop(2) = opts.D_target / Ps.D;
            Ps = iodc2010_paraxial(Ls, p.lambda_nm);
            inter = any(sign(Ps.h_marg(2:end)) ~= sign(Ps.h_marg(1)));
            C(end+1) = struct('L', Ls, 'pattern', sgn, 'pattern_id', pid, 'gaps_R', opts.air_grid(gaps(g, :)), ...
                'stop_place', sp, 'efl', Ps.efl, 'R', Rs, 'oal', Ls.z(end), 'D', Ps.D, 'intermediate', inter); %#ok<AGROW>
        end
    end
end
info.n_tried = n_tried;
info.n_seeds = numel(C);
end
