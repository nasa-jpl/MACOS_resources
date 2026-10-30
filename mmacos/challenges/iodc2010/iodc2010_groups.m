function [C, info] = iodc2010_groups(p, opts)
%IODC2010_GROUPS  Structured seeds: groups of equal elements whose power comes from their separation.
%   C = iodc2010_groups(p) returns seed lenses (the struct array iodc2010_forms
%   returns) of the form
%
%       [a x equi-convex]  --g1--  [b x equi-concave]  --g2--  [c x equi-convex]
%
%   with b = a + c (Petzval balance: every equi-convex element brings two
%   converging surfaces, every equi-concave two diverging).  Each group's
%   elements are close-packed (opts.pack in units of R); the gaps g1, g2 come
%   from opts.gap_grid (units of R).  c = 0 is the telephoto (power from the
%   separation: phi = g phi1 |phi2|, which is why the entries are long and
%   R ~ 4 OAL^0.6), a = 0 the retrofocus (long back focus, wide field), and
%   a long g1 with c > 0 puts a real intermediate image between the groups
%   (the winner's form).  The stop goes ahead of the lens, in the middle of
%   g1, or in the middle of g2.  The same paraxial screen as iodc2010_forms,
%   the stop sized for opts.D_target.
arguments
    p struct
    opts.a_range (1,:) double = 0:2
    opts.c_range (1,:) double = 0:2
    opts.b_max (1,1) double = 4
    opts.gap_grid (1,:) double = [0.3 1 3 10]
    opts.pack (1,1) double = 0.02
    opts.glass_frac (1,1) double = 0.04
    opts.R0 (1,1) double = 100
    opts.D_target (1,1) double = 10
    opts.R_range (1,2) double = [5 5000]
end
C = struct('L', {}, 'pattern', {}, 'pattern_id', {}, 'gaps_R', {}, 'stop_place', {}, ...
    'efl', {}, 'R', {}, 'oal', {}, 'D', {}, 'intermediate', {});
R0 = opts.R0;  tg = opts.glass_frac * R0;  tp = opts.pack * R0;
n_tried = 0;
for a = opts.a_range
    for c = opts.c_range
        b = a + c;
        if b < 1 || b > opts.b_max, continue; end
        for g1 = opts.gap_grid
            for g2 = opts.gap_grid
                if c == 0 && g2 ~= opts.gap_grid(1), continue; end     % no third group: g2 is moot
                % the element list: shapes and the air space AFTER each element
                shapes = [repmat({[1 -1]}, 1, a), repmat({[-1 1]}, 1, b), repmat({[1 -1]}, 1, c)];
                nE = numel(shapes);
                air = tp * ones(1, nE);                       % packed inside a group
                if a > 0, air(a) = g1 * R0; end
                if c > 0, air(a + b) = g2 * R0; end
                sgn = zeros(1, 2 * nE);  t = zeros(1, 2 * nE);
                for e = 1:nE, sgn(2*e-1:2*e) = shapes{e};  t(2*e-1) = tg;  t(2*e) = air(e); end
                t(end) = NaN;
                z = [0, cumsum(t(1:end-1))];
                stops = {{-0.1 * R0, 'front'}};
                if a > 0, stops{end+1} = {z(2*a) + 0.5 * g1 * R0, 'gap1'}; end %#ok<AGROW>
                if c > 0, stops{end+1} = {z(2*(a+b)) + 0.5 * g2 * R0, 'gap2'}; end %#ok<AGROW>
                if c == 0, stops{end+1} = {z(end) + 0.1 * R0, 'back'}; end %#ok<AGROW>
                for s = 1:numel(stops)
                    n_tried = n_tried + 1;
                    zs = stops{s}{1};  sp = stops{s}{2};
                    pid = sprintf('a%db%dc%d', a, b, c);
                    name = sprintf('grp_%s_g%g_%g_%s', pid, g1, g2, sp);
                    name = strrep(name, '.', 'p');
                    glass = repmat({'N-BK7', 'air'}, 1, nE);
                    try
                        L = iodc2010_lens('R', R0, 'sgn', sgn, 't', t, 'glass', glass, 'stop', [zs 1], 'name', name);
                        P = iodc2010_paraxial(L, p.lambda_nm);
                    catch
                        continue
                    end
                    if ~isfinite(P.efl) || P.efl == 0 || ~isfinite(P.z_ep) || ~isfinite(P.D), continue; end
                    if P.z_img <= L.z(end), continue; end
                    if zs >= P.z_img, continue; end
                    sc = p.efl_mm / abs(P.efl);  Rs = R0 * sc;
                    if Rs < opts.R_range(1) || Rs > opts.R_range(2), continue; end
                    Ls = L;  Ls.R = Rs;  Ls.t = L.t * sc;  Ls.z = L.z * sc;  Ls.stop = [zs * sc, 1];
                    Ps = iodc2010_paraxial(Ls, p.lambda_nm);
                    Ls.stop(2) = opts.D_target / Ps.D;
                    Ps = iodc2010_paraxial(Ls, p.lambda_nm);
                    inter = any(sign(Ps.h_marg(2:end)) ~= sign(Ps.h_marg(1)));
                    C(end+1) = struct('L', Ls, 'pattern', sgn, 'pattern_id', pid, 'gaps_R', [g1 g2], ...
                        'stop_place', sp, 'efl', Ps.efl, 'R', Rs, 'oal', Ls.z(end), 'D', Ps.D, 'intermediate', inter); %#ok<AGROW>
                end
            end
        end
    end
end
info.n_tried = n_tried;  info.n_seeds = numel(C);
end

function out = tern_(c, a, b)
if c, out = a; else, out = b; end
end
