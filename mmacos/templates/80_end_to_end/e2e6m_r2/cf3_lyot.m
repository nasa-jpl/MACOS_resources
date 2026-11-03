function OUT = cf3_lyot(over)
%CF3_LYOT  Coronagraph-family campaign S3a: the Lyot-fraction trade.
%
%   The ctb_vortex_lyot_sweep replay on the segmented train: a dense
%   Lyot-fraction grid for the vortex (charge 4 AND 6) and both prolate
%   legs (clear-disc 'apl' and aperture-matched 'aplc'), PRE-CONTROL,
%   N = P.co.model, against the S1 fixed points -- contrast AND
%   throughput, same grid/annulus/normalisation.  The chain is built
%   once per family; only the Lyot stop is re-sized per point
%   (cf_chain.set_lyot -- pupil-plane, wavelength-invariant).
%
%   CONSISTENCY PIN: at each family's S1 operating fraction the sweep
%   must reproduce the S1 table number bit-consistently (same chain,
%   same mask build).
%
%   OUT = CF3_LYOT()      defaults (fracs 0.50:0.05:0.95)
%   OUT = CF3_LYOT(OVER)  with e2e6m_r2_params overrides
%
%   See also CF_CHAIN, CF1_FAMILIES, ctb_vortex_lyot_sweep.

    arguments
        over struct = struct()
    end
    here = fileparts(mfilename('fullpath'));
    run(fullfile(here,'..','..','..','mmacos_setup.m'));
    P = e2e6m_r2_params(over);
    addpath(fullfile(here,'..','..','30_instruments','bench_ctb'));
    rx = fullfile(P.outdir, 'r1_seg_prop.in');
    fracs = 0.50:0.05:0.95;

    C1 = load(fullfile(P.outdir, 'cf1_run.mat'));
    F1 = C1.OUT.F;
    fam_keys = {'v4','v6','apl','aplc'};

    L = {};  t0 = tic;
    L = say_(L, '==================== e2e6m CF3a -- the Lyot-fraction trade (PRE-CONTROL)');
    L = say_(L, 'deck %s, model %d, annulus %g-%g lambda/D, fracs %s', ...
             rx, P.co.model, P.co.inner_lamD, P.co.outer_lamD, mat2str(fracs));

    S = struct();
    for f = 1:numel(fam_keys)
        key = fam_keys{f};
        k1 = find(strcmp({F1.key}, key), 1);
        assert(~isempty(k1), 'cf3_lyot: family %s not in cf1_run', key);
        fam = F1(k1);
        L = say_(L, '\n---- %s ----', fam.name);
        ch = cf_chain('rx', rx, 'model_size', P.co.model, ...
                      'prolate_iter', P.co.prolate_iter, ...
                      'circ_stop_frac', P.cf.circ_stop_frac, fam.cfg{:});
        s1_frac = cfgval_(fam.cfg, 'r_lyot_frac');
        dzm = nan(1, numel(fracs));  dzmed = nan(1, numel(fracs));
        thr = nan(1, numel(fracs));
        for q = 1:numel(fracs)
            thr(q) = ch.set_lyot(fracs(q));
            E = ch.run();
            I = abs(E).^2;
            dz = macos.dark_zone_metrics(I, ch.peak_bare, ch.lamD_px, ...
                                         P.co.inner_lamD, P.co.outer_lamD);
            dzm(q) = dz.mean;  dzmed(q) = dz.median;
            L = say_(L, '    frac %.2f: DZ mean %.3e median %.3e thru %.3f', ...
                     fracs(q), dz.mean, dz.median, thr(q));
        end
        % the S1 consistency pin
        qpin = find(abs(fracs - s1_frac) < 1e-9, 1);
        if ~isempty(qpin)
            relpin = abs(dzm(qpin) - fam.res.dz.mean) / fam.res.dz.mean;
            L = say_(L, '    S1 pin at frac %.2f: sweep %.6e vs S1 %.6e (rel %.3g)  [%s]', ...
                     s1_frac, dzm(qpin), fam.res.dz.mean, relpin, ...
                     gate_(relpin < 1e-12));
        else
            L = say_(L, '    (S1 operating frac %.2f not on the sweep grid -- no pin)', s1_frac);
        end
        S.(key) = struct('name', fam.name, 'fracs', fracs, 'dzm', dzm, ...
                         'dzmed', dzmed, 'thru', thr, 's1_frac', s1_frac, ...
                         's1_dzm', fam.res.dz.mean, 'tag', ch.tag);
    end

    % ---- figure: contrast vs frac + contrast vs throughput --------------
    png = fullfile(P.outdir, 'cf3_lyot.png');
    fig_(S, fam_keys, P, png);
    L = say_(L, '\n  figure: %s', png);

    L = say_(L, '\nCF3a DONE in %.1f min', toc(t0)/60);
    txt = strjoin(L, newline);
    fid = fopen(fullfile(P.outdir,'cf3_lyot_report.txt'),'w');
    fprintf(fid,'%s\n',txt);  fclose(fid);
    OUT = struct('P',P, 'S',S, 'keys',{fam_keys}, 'text',txt, 'figure',png, ...
                 'when',datestr(now,31)); %#ok<TNOW1,DATST>
    save(fullfile(P.outdir,'cf3_lyot_run.mat'),'OUT');
end

% =========================================================================
function v = cfgval_(cfg, key)
    v = [];
    for i = 1:2:numel(cfg)-1
        if strcmp(cfg{i}, key), v = cfg{i+1}; return; end
    end
end

function fig_(S, keys, P, png)
    f = figure('Visible','off','Color','w','Position',[60 60 1500 620]);
    tl = tiledlayout(f, 1, 2, 'TileSpacing','compact', 'Padding','compact');
    title(tl, sprintf(['e2e6m segmented train -- Lyot-fraction trade, PRE-CONTROL ' ...
        '(N=%d, annulus %g-%g \\lambda/D)'], P.co.model, ...
        P.co.inner_lamD, P.co.outer_lamD), 'FontWeight','bold', 'Interpreter','tex');
    cols = lines(numel(keys));
    ax = nexttile(tl); hold(ax,'on'); set(ax,'YScale','log');
    h = gobjects(1, numel(keys));
    for k = 1:numel(keys)
        s = S.(keys{k});
        h(k) = plot(ax, s.fracs, s.dzm, 'o-', 'Color', cols(k,:), 'LineWidth', 1.6);
        plot(ax, s.s1_frac, s.s1_dzm, 'p', 'MarkerSize', 14, ...
             'MarkerFaceColor', cols(k,:), 'MarkerEdgeColor','k', ...
             'HandleVisibility','off');
    end
    grid(ax,'on'); box(ax,'on');
    xlabel(ax,'Lyot fraction of the geometric pupil');
    ylabel(ax,'mean dark-zone contrast (PRE-CONTROL)');
    legend(ax, h, cellfun(@(k) S.(k).name, keys, 'UniformOutput', false), ...
           'Location','best');
    title(ax,'stars = the S1 operating points');
    ax = nexttile(tl); hold(ax,'on'); set(ax,'YScale','log');
    for k = 1:numel(keys)
        s = S.(keys{k});
        plot(ax, 100*s.thru, s.dzm, 'o-', 'Color', cols(k,:), 'LineWidth', 1.6);
    end
    grid(ax,'on'); box(ax,'on');
    xlabel(ax,'off-axis throughput proxy (%)');
    ylabel(ax,'mean dark-zone contrast (PRE-CONTROL)');
    title(ax,'the contrast-throughput frontier per family');
    exportgraphics(f, png, 'Resolution', 150);
    close(f);
end

function L = say_(L, varargin)
    s = sprintf(varargin{:});  L{end+1} = s;  fprintf('%s\n', s);
end
function s = gate_(ok), if ok, s = 'PASS'; else, s = 'FAIL'; end, end
