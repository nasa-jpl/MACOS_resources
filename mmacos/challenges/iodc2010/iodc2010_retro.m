function [C, info] = iodc2010_retro(p, opts)
%IODC2010_RETRO  Seeds of the retrofocus form the long IODC entries share.
%   C = iodc2010_retro(p) returns seed lenses (the struct array the other
%   families return) of the form the paper's Figures 6-11 show: a weak
%   DIVERGING front (opts.n_front equi-concave elements), a long air gap
%   (opts.gap_R, units of R) over which the beam expands, and a thick
%   CONVERGING rear train -- opts.n_men menisci bent toward the object
%   ([+ +]) and the n_front equi-convex elements that balance the front's
%   Petzval, the equi-convex first or last in the train (opts.where).  Its
%   focal length is |f_front| f_rear / gap: both groups stay weak while the
%   system reaches EFL 100, which is why those entries are metres long.
%   The stop sits just ahead of the rear group ('rear') or ahead of the
%   lens ('front'); for either the exit pupil lies beyond the image plane
%   (iodc2010_eval scores it on the mirror sphere).  Glass thicknesses
%   opts.tau_R, rear air gaps opts.g_R, every seed scaled to |EFL| = p.efl_mm
%   with the stop sized for opts.D_target; a seed whose marginal ray crosses
%   the axis inside the lens is dropped (opts.internal = false).
%
%   Names carry the values: retro_nf1_G3_m8_t0p3_g0p05_last_rear.
%   C(i).gaps_R = [gap_R, g_R]; .intermediate is false by construction.
arguments
    p struct
    opts.n_front (1,:) double = [1 2]
    opts.gap_R (1,:) double = [3 10]
    opts.n_men (1,:) double = [2 4 8]
    opts.tau_R (1,:) double = [0.3 0.6]
    opts.g_R (1,:) double = [0.05 0.3]
    opts.tau_front_R (1,1) double = 0.3
    opts.where (1,:) cell = {'first', 'last'}
    opts.stop_places (1,:) cell = {'rear'}
    opts.internal (1,1) logical = false
    opts.R0 (1,1) double = 100
    opts.D_target (1,1) double = 10
    opts.R_range (1,2) double = [5 20000]
end
C = struct('L', {}, 'pattern', {}, 'pattern_id', {}, 'gaps_R', {}, 'stop_place', {}, ...
    'efl', {}, 'R', {}, 'oal', {}, 'D', {}, 'intermediate', {});
R0 = opts.R0;  n_tried = 0;
for nf = opts.n_front
 for G = opts.gap_R
  for m = opts.n_men
   for tau = opts.tau_R
    for g = opts.g_R
     for wh = opts.where
      for sp = opts.stop_places
        n_tried = n_tried + 1;
        sgn = [];  t = [];  glass = {};
        for k = 1:nf
            sgn = [sgn, -1, 1];  t = [t, opts.tau_front_R * R0, 0.2 * R0];  glass = [glass, {'N-BK7', 'air'}]; %#ok<AGROW>
        end
        t(end) = G * R0;
        rear = {};
        for k = 1:nf, rear{end+1} = [1 -1]; end %#ok<AGROW>
        men = repmat({[1 1]}, 1, m);
        if strcmp(wh{1}, 'first'), rear = [rear, men]; else, rear = [men, rear]; end
        for k = 1:numel(rear)
            sgn = [sgn, rear{k}];  t = [t, tau * R0, g * R0];  glass = [glass, {'N-BK7', 'air'}]; %#ok<AGROW>
        end
        t(end) = NaN;
        z = [0, cumsum(t(1:end-1))];
        if strcmp(sp{1}, 'front'), zs = -0.1 * R0; else, zs = z(2 * nf + 1) - 0.5 * g * R0 - 0.01 * R0; end
        name = strrep(sprintf('retro_nf%d_G%g_m%d_t%g_g%g_%s_%s', nf, G, m, tau, g, wh{1}, sp{1}), '.', 'p');
        try
            L = iodc2010_lens('R', R0, 'sgn', sgn, 't', t, 'glass', glass, 'stop', [zs 1], 'name', name);
            P = iodc2010_paraxial(L, p.lambda_nm);
        catch
            continue
        end
        if ~isfinite(P.efl) || P.efl == 0 || ~isfinite(P.z_ep) || ~isfinite(P.D), continue; end
        if P.z_img <= L.z(end), continue; end
        inter = any(sign(P.h_marg(2:end)) ~= sign(P.h_marg(1)));
        if inter && ~opts.internal, continue; end
        sc = p.efl_mm / abs(P.efl);  Rs = R0 * sc;
        if Rs < opts.R_range(1) || Rs > opts.R_range(2), continue; end
        Ls = L;  Ls.R = Rs;  Ls.t = L.t * sc;  Ls.z = L.z * sc;  Ls.stop = [zs * sc, 1];
        Ps = iodc2010_paraxial(Ls, p.lambda_nm);
        Ls.stop(2) = opts.D_target / Ps.D;
        Ps = iodc2010_paraxial(Ls, p.lambda_nm);
        C(end+1) = struct('L', Ls, 'pattern', sgn, 'pattern_id', sprintf('nf%d_m%d_%s', nf, m, wh{1}), 'gaps_R', [G g], ...
            'stop_place', sp{1}, 'efl', Ps.efl, 'R', Rs, 'oal', Ls.z(end), 'D', Ps.D, 'intermediate', inter); %#ok<AGROW>
      end
     end
    end
   end
  end
 end
end
info.n_tried = n_tried;  info.n_seeds = numel(C);
end
