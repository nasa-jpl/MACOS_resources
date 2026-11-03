function [C, info] = iodc2010_chain(p, opts)
%IODC2010_CHAIN  Structured seeds for four and more elements: a chain of zero-Petzval pair units.
%   C = iodc2010_chain(p) returns seed lenses (the struct array iodc2010_forms
%   returns) built from UNITS, each an equi-convex and an equi-concave element
%   of the common radius separated by an air space u (units of R):
%
%       [+ -] --u-- [- +]   ('pn')      or      [- +] --u-- [+ -]   ('np')
%
%   A unit has a zero Petzval sum (two converging surfaces, two diverging)
%   and a net power that comes from its spacing alone,
%
%       phi_unit = phi1 + phi2 - u R phi1 phi2 = u R phi^2,   phi = 2 (n - 1) / R,
%
%   positive in either order: it is the WEAK POSITIVE ELEMENT the one-radius
%   problem otherwise lacks -- its power is dialled by u, not by R (the
%   paper's Fig. 15, R ~ 4 OAL^0.6: the entries' power came from their
%   separations).  Its focal length is R / (u (2 (n - 1))^2): 0.93 R at
%   u = 1, 3.1 R at u = 0.3 (N-BK7 at 532 nm).  Units are chained with link
%   gaps from opts.link_gap; a link longer than the unit's focal length puts a
%   real intermediate image inside it (C(i).intermediate), the next unit
%   re-images it -- the relay form of the long entries.  Every chain is
%   Petzval-balanced by construction, so the forms family's balance filter
%   has nothing to prune and the count is the grids' product:
%   opts.n_units units (2 or 3 -> 4 or 6 elements), each unit's order from
%   {'pn', 'np'} and its u from opts.unit_gap, every link from opts.link_gap;
%   the stop ahead of the lens, at the middle of any unit gap or link, or
%   behind.  opts.symmetric = true keeps, for an even number of units, only
%   the chains mirrored about the central link (a unit's mirror is the other
%   order with the same u) with the stop at that link's middle: the
%   symmetric form, whose distortion and coma cancel at unit magnification.
%
%   Names carry the gap VALUES, not indices (ch4_pn0p3_L4_np1_s_l1: unit 1
%   'pn' with u = 0.3 R, a 4 R link, unit 2 'np' with u = 1 R, the stop in
%   link 1), so a table and a grid cannot disagree (the 2026-10-31 record
%   defect).  The same paraxial screen as iodc2010_forms (a finite focal
%   length of either sign, a real image beyond the last surface, the stop
%   before the image, the scaled R in opts.R_range), the stop sized for
%   opts.D_target.  C(i).gaps_R = [u1 L1 u2 L2 ... u_m].
arguments
    p struct
    opts.n_units (1,:) double = 2
    opts.unit_gap (1,:) double = [0.1 0.3 1 3]
    opts.link_gap (1,:) double = [0.5 1.5 4 10]
    opts.symmetric (1,1) logical = false
    opts.stop_places (1,:) cell = {'front', 'unit', 'link', 'back'}
    opts.glass_frac (1,1) double = 0.04
    opts.R0 (1,1) double = 100
    opts.D_target (1,1) double = 10
    opts.R_range (1,2) double = [5 5000]
end
C = struct('L', {}, 'pattern', {}, 'pattern_id', {}, 'gaps_R', {}, 'stop_place', {}, ...
    'efl', {}, 'R', {}, 'oal', {}, 'D', {}, 'intermediate', {});
R0 = opts.R0;  tg = opts.glass_frac * R0;
orders = {[1 -1; -1 1], [-1 1; 1 -1]};  oname = {'pn', 'np'};
nu = numel(opts.unit_gap);  nl = numel(opts.link_gap);
n_tried = 0;  n_chains = 0;
for m = opts.n_units
    % every (order, u) per unit and every link: the index tuples
    oi = tuples_(2, m);  ui = tuples_(nu, m);  li = tuples_(nl, max(m - 1, 1));
    if m == 1, li = ones(1, 1); end
    for a = 1:size(oi, 1)
        for b = 1:size(ui, 1)
            for c = 1:size(li, 1)
                ord = oi(a, :);  u = opts.unit_gap(ui(b, :));  lk = opts.link_gap(li(c, 1:max(m - 1, 0)));
                if opts.symmetric
                    if mod(m, 2) ~= 0, continue; end
                    if any(ord ~= 3 - fliplr(ord)) || any(u ~= fliplr(u)) || any(lk ~= fliplr(lk)), continue; end
                end
                n_chains = n_chains + 1;
                % the surfaces: unit k = element 2k-1 (first of the pair), air u_k, element 2k, air link_k
                nE = 2 * m;  sgn = zeros(1, 2 * nE);  t = zeros(1, 2 * nE);
                parts = cell(1, 2 * m - 1);
                for k = 1:m
                    sh = orders{ord(k)};
                    e1 = 2 * k - 1;  e2 = 2 * k;
                    sgn(2*e1-1:2*e1) = sh(1, :);  sgn(2*e2-1:2*e2) = sh(2, :);
                    t(2*e1-1) = tg;  t(2*e1) = u(k) * R0;  t(2*e2-1) = tg;
                    if k < m, t(2*e2) = lk(k) * R0; else, t(2*e2) = NaN; end
                    parts{2*k-1} = sprintf('%s%g', oname{ord(k)}, u(k));
                    if k < m, parts{2*k} = sprintf('L%g', lk(k)); end
                end
                z = [0, cumsum(t(1:end-1))];
                base = strrep(['ch' num2str(nE) '_' strjoin(parts, '_')], '.', 'p');
                % the stop places
                stops = {};
                if any(strcmp(opts.stop_places, 'front')), stops{end+1} = {-0.1 * R0, 'front'}; end %#ok<AGROW>
                for k = 1:m
                    if any(strcmp(opts.stop_places, 'unit'))
                        stops{end+1} = {z(2*(2*k-1)) + 0.5 * u(k) * R0, sprintf('u%d', k)}; %#ok<AGROW>
                    end
                    if k < m && any(strcmp(opts.stop_places, 'link'))
                        stops{end+1} = {z(2*(2*k)) + 0.5 * lk(k) * R0, sprintf('l%d', k)}; %#ok<AGROW>
                    end
                end
                if any(strcmp(opts.stop_places, 'back')), stops{end+1} = {z(end) + 0.1 * R0, 'back'}; end %#ok<AGROW>
                if opts.symmetric, stops = stops(strcmp(cellfun(@(s) s{2}, stops, 'UniformOutput', false), sprintf('l%d', m / 2))); end
                glass = repmat({'N-BK7', 'air'}, 1, nE);
                for s = 1:numel(stops)
                    n_tried = n_tried + 1;
                    zs = stops{s}{1};  sp = stops{s}{2};
                    name = [base '_s_' sp];
                    try
                        L = iodc2010_lens('R', R0, 'sgn', sgn, 't', t, 'glass', glass, 'stop', [zs 1], 'name', name);
                        P = iodc2010_paraxial(L, p.lambda_nm);
                    catch
                        continue
                    end
                    if ~isfinite(P.efl) || P.efl == 0 || ~isfinite(P.z_ep) || ~isfinite(P.D), continue; end
                    if P.z_img <= L.z(end), continue; end                 % the image must be real, in air
                    if zs >= P.z_img, continue; end                       % the stop sits before the image
                    sc = p.efl_mm / abs(P.efl);  Rs = R0 * sc;
                    if Rs < opts.R_range(1) || Rs > opts.R_range(2), continue; end
                    Ls = L;  Ls.R = Rs;  Ls.t = L.t * sc;  Ls.z = L.z * sc;  Ls.stop = [zs * sc, 1];
                    Ps = iodc2010_paraxial(Ls, p.lambda_nm);
                    Ls.stop(2) = opts.D_target / Ps.D;
                    Ps = iodc2010_paraxial(Ls, p.lambda_nm);
                    inter = any(sign(Ps.h_marg(2:end)) ~= sign(Ps.h_marg(1)));
                    gaps = zeros(1, 2 * m - 1);  gaps(1:2:end) = u;  gaps(2:2:end) = lk;
                    C(end+1) = struct('L', Ls, 'pattern', sgn, 'pattern_id', base, 'gaps_R', gaps, ...
                        'stop_place', sp, 'efl', Ps.efl, 'R', Rs, 'oal', Ls.z(end), 'D', Ps.D, 'intermediate', inter); %#ok<AGROW>
                end
            end
        end
    end
end
info.n_chains = n_chains;  info.n_tried = n_tried;  info.n_seeds = numel(C);
info.n_intermediate = nnz([C.intermediate]);
end

function T = tuples_(n, k)
% every k-tuple over 1..n, one per row (n^k rows)
if k == 0, T = zeros(1, 0); return; end
if k == 1, T = (1:n).'; return; end          % ndgrid(x) with one output is the 2-D grid, not the vector
idx = cell(1, k);  [idx{:}] = ndgrid(1:n);
T = reshape(cat(k + 1, idx{:}), [], k);
end
