function OUT = cf3a_lyot(over)
%CF3A_LYOT  Coronagraph-family campaign S3a: the Lyot-fraction trade (STATIC).
%
%   Sweeps r_lyot_frac for the vortex legs (v4, v6) and the prolate leg
%   (apl -- the prolate family's STOPPED representative per CF1b: the
%   aperture-matched stopped prolate is solver-limited and stands
%   flagged, so the R1 prolate carries the family here) on the S1 deck
%   (r1_seg_prop.in, N = P.co.model), under the S0b circular stop.
%   CTB analog: ctb_vortex_lyot_sweep -- contrast AND throughput per
%   point, the S1 family operating points overlaid on the trade curve.
%
%   The apl prolate is designed ONCE at its S1 configuration and
%   SUPPLIED across the sweep (the apodizer is a pupil-plane object; it
%   does not depend on the downstream Lyot).  Vortex masks rebuild per
%   chain (cheap, 8x-binned).
%
%   STATIC by design: the closed-loop rebalance at the chosen operating
%   point is S4's job (band+pol at the throughput-rebalanced Lyot).
%
%   OUT = CF3A_LYOT()
%   OUT = CF3A_LYOT(struct('cf3', struct('lyots', 0.5:0.1:0.9)))
%
%   See also CF_CHAIN, CF1_FAMILIES, CF2_EFC, ctb_vortex_lyot_sweep.

    arguments
        over struct = struct()
    end
    here = fileparts(mfilename('fullpath'));
    run(fullfile(here,'..','..','..','mmacos_setup.m'));
    ov = over;  cf3 = struct();
    if isfield(ov,'cf3'), cf3 = ov.cf3;  ov = rmfield(ov,'cf3'); end
    P = e2e6m_r2_params(ov);
    if ~isfield(cf3,'lyots'), cf3.lyots = [0.50 0.60 0.70 0.80 0.90 0.95 0.98]; end
    if ~isfield(cf3,'legs'),  cf3.legs  = {'v4','v6','apl'}; end
    addpath(fullfile(here,'..','..','30_instruments','bench_ctb'));

    rx = fullfile(P.outdir, 'r1_seg_prop.in');
    assert(isfile(rx), 'cf3a_lyot: %s not found', rx);
    C1 = load(fullfile(P.outdir,'cf1_run.mat'));
    FC = struct();
    for k = 1:numel(C1.OUT.F), FC.(C1.OUT.F(k).key) = C1.OUT.F(k); end

    L = {};  t0 = tic;
    L = say_(L, '==== e2e6m CF3a -- Lyot-fraction trade (STATIC, stop c=%g, N=%d, annulus %g-%g lambda/D)', ...
             P.cf.circ_stop_frac, P.co.model, P.co.inner_lamD, P.co.outer_lamD);
    L = say_(L, 'deck %s | legs %s | lyots %s', rx, strjoin(cf3.legs, ' '), mat2str(cf3.lyots));

    apl_A = [];                       % the once-designed prolate (supplied after)
    R = struct();
    for g = 1:numel(cf3.legs)
        key  = cf3.legs{g};
        assert(isfield(FC, key), 'cf3a_lyot: unknown leg "%s"', key);
        cfg0 = FC.(key).cfg;
        con = nan(1, numel(cf3.lyots));  thr = con;
        L = say_(L, '\n-- %s (%s)', key, FC.(key).name);
        for j = 1:numel(cf3.lyots)
            cfg = set_nv_(cfg0, 'r_lyot_frac', cf3.lyots(j));
            if strcmp(key, 'apl') && ~isempty(apl_A)
                cfg = set_nv_(cfg, 'apod_kind', 'supplied');
                cfg = [cfg, {'apod_mask', apl_A}];             %#ok<AGROW>
            end
            ch = cf_chain('rx', rx, 'model_size', P.co.model, ...
                          'prolate_iter', P.co.prolate_iter, ...
                          'circ_stop_frac', P.cf.circ_stop_frac, cfg{:});
            if strcmp(key, 'apl') && isempty(apl_A), apl_A = ch.masks.A; end
            E  = ch.run();
            I  = abs(E).^2;
            dz = macos.dark_zone_metrics(I, ch.peak_bare, ch.lamD_px, ...
                                         P.co.inner_lamD, P.co.outer_lamD);
            con(j) = dz.mean;  thr(j) = ch.thru;
            L = say_(L, '   L=%.2f: contrast %.3e | thru %.3f | tag %s', ...
                     cf3.lyots(j), con(j), thr(j), ch.tag);
        end
        R.(key) = struct('name', FC.(key).name, 'lyots', cf3.lyots, ...
                         'con', con, 'thru', thr);
    end

    % S1 family operating points (all six), for the trade overlay
    S1 = struct('key', {}, 'name', {}, 'con', {}, 'thru', {});
    for k = 1:numel(C1.OUT.F)
        f = C1.OUT.F(k);
        S1(end+1) = struct('key', f.key, 'name', f.name, ...
            'con', f.res.dz.mean, 'thru', thru_of_(f));        %#ok<AGROW>
    end

    png = fullfile(P.outdir, 'cf3a_lyot.png');
    fig_(R, cf3.legs, S1, P, png);
    L = say_(L, '\n  figure: %s', png);
    L = say_(L, 'CF3a DONE in %.1f min', toc(t0)/60);

    txt = strjoin(L, newline);
    fid = fopen(fullfile(P.outdir,'cf3a_report.txt'),'w');
    fprintf(fid,'%s\n',txt);  fclose(fid);
    OUT = struct('P',P, 'R',R, 'legs',{cf3.legs}, 'S1',S1, 'text',txt, ...
                 'figure',png, 'when',datestr(now,31)); %#ok<TNOW1,DATST>
    save(fullfile(P.outdir,'cf3a_run.mat'),'OUT');
end

% =========================================================================
function cfg = set_nv_(cfg, key, val)
%SET_NV_  Replace (or append) one name-value pair in a config cell.
    for i = 1:2:numel(cfg)-1
        if strcmp(cfg{i}, key), cfg{i+1} = val;  return; end
    end
    cfg = [cfg, {key, val}];
end

function t = thru_of_(f)
%THRU_OF_  The S1 record's throughput, wherever that revision stored it.
    if isfield(f.res, 'thru'), t = f.res.thru;
    elseif isfield(f, 'thru'),  t = f.thru;
    else, t = NaN;
    end
end

function fig_(R, legs, S1, P, png)
    f = figure('Visible','off','Color','w','Position',[60 60 1400 560]);
    tl = tiledlayout(f, 1, 2, 'TileSpacing','compact', 'Padding','compact');
    title(tl, sprintf(['e2e6m families -- the Lyot trade (STATIC, N=%d, ' ...
        'circular stop, annulus %g-%g \\lambda/D)'], ...
        P.co.model, P.co.inner_lamD, P.co.outer_lamD), ...
        'FontWeight','bold', 'Interpreter','tex');
    cols = lines(numel(legs));
    ax = nexttile(tl); hold(ax,'on'); set(ax,'YScale','log'); grid(ax,'on'); box(ax,'on');
    for g = 1:numel(legs)
        r = R.(legs{g});
        semilogy(ax, r.lyots, r.con, 'o-', 'Color', cols(g,:), ...
                 'LineWidth', 1.6, 'DisplayName', r.name);
    end
    xlabel(ax, 'Lyot fraction (of the circular geometric pupil)');
    ylabel(ax, 'dark-zone mean contrast (PRE-CONTROL)');
    legend(ax, 'Location', 'northwest');
    title(ax, 'contrast vs Lyot fraction');
    ax = nexttile(tl); hold(ax,'on'); set(ax,'YScale','log'); grid(ax,'on'); box(ax,'on');
    for g = 1:numel(legs)
        r = R.(legs{g});
        semilogy(ax, r.thru, r.con, 'o-', 'Color', cols(g,:), ...
                 'LineWidth', 1.6, 'DisplayName', [r.name ' (sweep)']);
    end
    for k = 1:numel(S1)
        if ~isfinite(S1(k).thru), continue; end
        semilogy(ax, S1(k).thru, S1(k).con, 'kp', 'MarkerSize', 11, ...
                 'MarkerFaceColor', [0.85 0.85 0.85], 'HandleVisibility', 'off');
        text(ax, S1(k).thru, S1(k).con*1.5, S1(k).key, 'FontSize', 8, ...
             'HorizontalAlignment', 'center');
    end
    xlabel(ax, 'throughput (collecting-area x apodizer x Lyot)');
    ylabel(ax, 'dark-zone mean contrast (PRE-CONTROL)');
    legend(ax, 'Location', 'northeast');
    title(ax, 'the trade curve (stars: the S1 family operating points)');
    exportgraphics(f, png, 'Resolution', 150);
    close(f);
end

function L = say_(L, varargin)
    s = sprintf(varargin{:});  L{end+1} = s;  fprintf('%s\n', s);
end
