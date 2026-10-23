function [a, cvec, info] = ctb_push(ch, dm, J, a_in, dz_idx, targets_nm)
%CTB_PUSH  Stroke-released EFC step for the CTB chain (port of the e2e6m
%   cf_efc_lib.push_).  The per-iteration monotone line search in ctb_efc
%   never accepts the large steps the deep-mode solution needs -- a single
%   big Tikhonov step always loses one measured shot -- so on the
%   charge-6/Lyot-0.80 vortex it plateaus ~3.5e-9 far above the
%   linear-achievable floor (ctb_linfloor ~1e-17).  This walks the
%   TRUNCATED-SVD solution to a TARGET command stroke in ~25 nm sub-steps,
%   re-tracing between (NON-monotone along the walk), and accept-rejects
%   the best visited state against the start.  Each target in targets_nm
%   (rms over all actuators) is an independent probe from a_in.
%
%   Commands are in mm (ctb_dm convention); targets_nm in nm.  Returns the
%   best state, the visited-contrast trace, and info (c0, c1, target,
%   stroke reached).  Column->DM mapping matches ctb_efc (J.col_dm order).
%
%   See also: ctb_efc, ctb_dm_jacobian, ctb_linfloor, cf_efc_lib (twin).
    G  = double(J.G);
    Gr = [real(G); imag(G)];
    [U, S, V] = svd(Gr, 'econ');
    s  = diag(S);
    udm  = unique(J.col_dm, 'stable');
    dmof = arrayfun(@(c) find(udm == c, 1), J.col_dm);
    ndm  = numel(dm);
    for k = 1:ndm, dm{k}.apply(a_in{k}); end
    E  = ch.run();  pb = ch.peak_bare;
    c0 = mean(abs(E(dz_idx)).^2) / pb;
    e  = double(E(dz_idx));
    coef = (U' * [real(e); imag(e)]) ./ s;
    stroke_r = 1e6 * sqrt(cumsum(coef.^2) / size(Gr, 2));   % mm -> nm
    best = struct('c', c0, 'a', {a_in}, 'target', 0, 'stroke', NaN);
    cvec = c0;
    for T = targets_nm(:).'
        r = find(stroke_r <= T, 1, 'last');
        if isempty(r) || r < 2, continue; end
        da = -V(:, 1:r) * coef(1:r);
        nsub = max(4, ceil(T / 25));
        at = a_in;
        for ss = 1:nsub
            for k = 1:ndm
                sel = dmof == k;
                at{k}(J.col_act(sel)) = at{k}(J.col_act(sel)) + da(sel) / nsub;
            end
            for k = 1:ndm, dm{k}.apply(at{k}); end
            Et = ch.run();
            c  = mean(abs(Et(dz_idx)).^2) / pb;
            cvec(end+1) = c; %#ok<AGROW>
            if c < best.c
                allc = cell2mat(cellfun(@(x) x(x~=0), at(:), 'UniformOutput', false));
                best = struct('c', c, 'a', {at}, 'target', T, ...
                              'stroke', 1e6 * rms(allc));
            end
        end
        fprintf('    [push] target %.0f nm: best %.3e (stroke %.1f nm)\n', ...
                T, best.c, best.stroke);
    end
    a = best.a;
    for k = 1:ndm, dm{k}.apply(a{k}); end
    ch.run();                                  % engine = accepted state
    info = struct('c0', c0, 'c1', best.c, 'target', best.target, ...
                  'stroke', best.stroke);
end
