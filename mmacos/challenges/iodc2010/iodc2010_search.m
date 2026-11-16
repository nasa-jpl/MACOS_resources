function B = iodc2010_search(L0, p, opts)
%IODC2010_SEARCH  Spacing solve with CALIB inside a scan over R and the stop, scored by the metric.
%   B = iodc2010_search(L0, p) takes a seed green lens L0 (its sign pattern,
%   thicknesses, media and stop are the start) and, for every radius in
%   opts.R_list and every stop semi-diameter in opts.a_list:
%     1. emits the deck with a CALIB block (iodc2010_emit's opt argument):
%        WFE target at the exit-pupil Return, OptFEX on, the design fields
%        [0, opts.design_field_deg/2, opts.design_field_deg], every lens
%        surface's axial position free (PIST) and the image plane's too;
%     2. runs macos.calib and reads the surfaces back (macos.get_elt_vpt);
%     3. SCALES the whole lens to |EFL| = 100 (radius, thicknesses, stop
%        position and semi-diameter together: the merit D*theta scales with
%        D and the common-radius rule survives a scale);
%     4. scores it with iodc2010_score (the metric of the README).
%   Candidates with a non-positive thickness or air space after the solve
%   are recorded and rejected (the problem's glass constraint).
%
%   B.table   one row per candidate: R, a, EFL before scaling, CALIB iterations
%             and stop reason, WFE before/after at the design fields, D,
%             theta_lim, merit, limit reason, ok
%   B.best    the best candidate's lens (scaled) and score
%
%   Knobs (opts): R_list (mm), a_list (mm), design_field_deg, max_iters, var_fp.
%   This is the inner solve of BRIEF_to_iodc_challenge step 3-4; the sign
%   pattern and the element count are the outer loop a driver writes.
arguments
    L0 struct
    p struct
    opts.R_list (1,:) double = L0.R
    opts.a_list (1,:) double = L0.stop(2)
    opts.design_field_deg (1,1) double = 3
    opts.max_iters (1,1) double = 30
    opts.var_fp (1,1) logical = true
    opts.quiet (1,1) logical = false
end
macos.init(p.model_size);
deck = fullfile(p.outdir, [L0.name '_calib.in']);
rows = {};  best = [];  bestMerit = -inf;
for a = opts.a_list
    for R = opts.R_list
        % the seed SCALED to this radius (a scaled lens is the same lens: the
        % spacings follow R, so every start is as sound as the seed), the
        % stop's semi-diameter then set to a
        sc0 = R / L0.R;
        L = L0;  L.R = R;  L.t = L0.t * sc0;  L.z = L0.z * sc0;  L.stop = [L0.stop(1) * sc0, a];  L.semidia = [];
        P = iodc2010_paraxial(L, p.lambda_nm);
        opt = struct('fields_deg', [0 opts.design_field_deg/2 opts.design_field_deg], ...
            'max_iters', opts.max_iters, 'var_surf', 1:L.N, 'var_fp', opts.var_fp);
        Lw = L;  Lw.semidia = inf(1, L.N);           % only the stop clips during the solve
        E = iodc2010_emit(Lw, P, p, deck, opt);
        macos.load_rx(E.path);
        try
            res = macos.calib();
        catch err                                   % a failed initial evaluation (rays lost): record, move on
            res = struct('n_iter', -1, 'stop_reason', ['calib error: ' err.message], 'old_wfe', NaN, 'new_wfe', NaN);
            rows{end+1} = struct('R', R, 'a', a, 'efl_solved', NaN, 'n_iter', -1, 'stop_reason', res.stop_reason, ...
                'wfe_before', NaN, 'wfe_after', NaN, 'D', NaN, 'theta', NaN, 'merit', 0, 'limit', 'calib-failed', ...
                'ok', false, 'scale', NaN); %#ok<AGROW>
            if ~opts.quiet, fprintf('  R %7.2f a %5.2f: CALIB failed (%s)\n', R, a, err.message); end
            macos.modify();
            continue
        end
        z = zeros(1, L.N);
        for k = 1:L.N, v = macos.get_elt_vpt(E.iSurf(k));  z(k) = v(3); end
        vf = macos.get_elt_vpt(E.iFP);  z_fp = vf(3);
        t = [diff(z), z_fp - z(end)];
        Ls = L;  Ls.t = t - 0;  Ls.z = z - z(1);  Ls.stop(1) = L.stop(1) - z(1);  Ls.t(end) = z_fp - z(end);
        Ls.z = [0, cumsum(Ls.t(1:end-1))];
        ok_geom = all(Ls.t > 0);
        row = struct('R', R, 'a', a, 'efl_solved', NaN, 'n_iter', res.n_iter, 'stop_reason', res.stop_reason, ...
            'wfe_before', res.old_wfe(:).', 'wfe_after', res.new_wfe(:).', 'D', NaN, 'theta', NaN, ...
            'merit', 0, 'limit', 'geometry', 'ok', ok_geom, 'scale', NaN);
        if ok_geom
            Ps = iodc2010_paraxial(Ls, p.lambda_nm);
            row.efl_solved = Ps.efl;
            sc = p.efl_mm / abs(Ps.efl);
            Ls.R = Ls.R * sc;  Ls.t = Ls.t * sc;  Ls.z = Ls.z * sc;  Ls.stop = Ls.stop * sc;
            Ls.name = sprintf('%s_R%g_a%g', L0.name, R, a);
            S = iodc2010_score(Ls, p, 'quiet', true, 'deck', fullfile(p.outdir, [Ls.name '.in']));
            row.D = S.D_mm;  row.theta = S.theta_lim_deg;  row.merit = S.merit;  row.limit = S.limit_reason;
            row.scale = sc;  row.ok = S.efl_ok && S.rules.ok && S.image_real;
            if row.ok && S.merit > bestMerit, bestMerit = S.merit;  best = struct('lens', Ls, 'score', S); end
        end
        rows{end+1} = row; %#ok<AGROW>
        if ~opts.quiet
            fprintf('  R %7.2f a %5.2f: calib %2d it (%s) wfe %s -> %s mm | EFL %.3f x%.3f | D %.3f theta %.2f merit %.1f %s%s\n', ...
                R, a, res.n_iter, res.stop_reason, mat2str(row.wfe_before, 3), mat2str(row.wfe_after, 3), ...
                row.efl_solved, row.scale, row.D, row.theta, row.merit, row.limit, tern_(row.ok, '', ' REJECTED'));
        end
    end
end
B = struct('table', struct2table([rows{:}], 'AsArray', true), 'best', best, 'seed', L0, 'opts', opts);
end

function out = tern_(c, a, b)
if c, out = a; else, out = b; end
end
