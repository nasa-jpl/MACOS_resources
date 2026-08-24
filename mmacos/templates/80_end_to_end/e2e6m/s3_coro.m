function OUT = s3_coro(over)
%S3_CORO  e2e6m stage 3b: the APLC coronagraph, and what the gaps cost.
%
%   Builds the back end on BOTH primaries -- the 19-segment one and the
%   monolithic one -- converts each to a diffraction deck
%   (`macos.design.prop_layout`), and scores the SAME apodized pupil Lyot
%   coronagraph on each with the committed `ctb_aplc` chain.
%
%   THE COMPARISON IS THE POINT.  The APLC's prolate apodizer is the
%   dominant eigenfunction of the APLC operator for a CLEAR CIRCULAR
%   pupil -- that is what `ctb_apod_prolate` solves, and the committed
%   2.1e-10 is a clear-pupil number.  A segmented pupil is not that
%   pupil: the gaps are a fixed high-spatial-frequency structure the
%   apodizer never saw, and they scatter light into the dark zone no
%   amount of apodization removes.  Running the SAME mask on both
%   primaries makes the gap cost a measured difference rather than an
%   assertion, and that difference IS the demo content.
%
%   Re-optimizing the apodizer for the segmented aperture (a segmented
%   APLC / SCDA design) is explicitly out of this brief's scope; the
%   number here is what a clear-pupil APLC does on a segmented pupil.
%
%   OUT = S3_CORO()      run at the default parameter set
%   OUT = S3_CORO(OVER)  ... with e2e6m_params overrides
%
%   See also S3_BACKEND, macos.design.prop_layout, ctb_aplc,
%   macos.dark_zone_metrics, macos.radial_contrast.

    arguments
        over struct = struct()
    end
    here = fileparts(mfilename('fullpath'));
    setup_(here);
    P = e2e6m_params(over);
    if isempty(P.outdir), P.outdir = here; end
    addpath(fullfile(here,'..','..','30_instruments','bench_ctb'));

    L = {};  t0 = tic;
    L = say_(L, '==================== e2e6m S3b -- the APLC coronagraph');
    L = say_(L, 'metric: dark-zone contrast, Strehl-normalised to the BARE');
    L = say_(L, '        on-axis peak of the SAME train; annulus %g-%g lambda/D', ...
             P.co.inner_lamD, P.co.outer_lamD);
    L = say_(L, '        at %g nm, at the CORONAGRAPH exit pupil (elt ExitPupil),', ...
             P.lambda_m*1e9);
    L = say_(L, '        occulter %g lambda/D, Lyot %g of the geometric pupil,', ...
             P.co.r_occ_lamD, P.co.r_lyot_frac);
    L = say_(L, '        model %d, nGridpts %d', P.co.model, P.co.ngridpts);

    V = struct('tag',{'seg','mono'}, ...
               'base',{'s2_segmented.in','s1_telescope.in'}, ...
               'label',{'19-segment primary','monolithic primary'}, ...
               'res',{[],[]});

    for v = 1:numel(V)
        L = say_(L, '\n---- %s (%s) ----', V(v).tag, V(v).label);
        bk = P.bk;  bk.tag = V(v).tag;  bk.base_in = V(v).base;
        B = s3_backend(struct('outdir',P.outdir, 'bk',bk));      %#ok<NASGU>
        full_in = fullfile(P.outdir, sprintf('s3_%s_full.in', V(v).tag));
        prop_in = fullfile(P.outdir, sprintf('s3_%s_prop.in', V(v).tag));

        kinds = kinds_from_deck_(full_in);
        info = macos.design.prop_layout(full_in, kinds, 'out', prop_in, ...
                    'model', P.co.model, 'ngridpts', P.co.ngridpts, ...
                    'verify', true);
        L = say_(L, '    diffraction deck %d elements; chief vs geometric %.3g; PSF %s', ...
                 info.nElt, info.chk.chief_max, ...
                 tern_(info.chk.psf_centred,'CENTRED','** OFF-CENTRE **'));
        L = say_(L, '    FEX radii: FPM %.6f m, ExitPupil %.6f m', ...
                 info.R.FPM, info.R.ExitPupil);

        elt = struct('DM1', info.ix.(seedname_(info,1)), ...
                     'DM2', info.ix.(seedname_(info,2)), ...
                     'Apodizer', info.ix.Apodizer, 'FPM', info.ix.FPM, ...
                     'Lyot', info.ix.Lyot, 'ExitPupil', info.ix.ExitPupil, ...
                     'FPA', info.ix.Science);
        r = ctb_aplc('rx', prop_in, 'elt', elt, 'model_size', P.co.model, ...
                     'r_occ_lamD', P.co.r_occ_lamD, ...
                     'r_lyot_frac', P.co.r_lyot_frac, ...
                     'prolate_iter', P.co.prolate_iter, ...
                     'inner_lamD', P.co.inner_lamD, ...
                     'outer_lamD', P.co.outer_lamD, ...
                     'outdir', P.outdir, 'visible', false);
        movefile(r.figure, fullfile(P.outdir, sprintf('s3_%s_aplc.png', V(v).tag)));
        r.figure = fullfile(P.outdir, sprintf('s3_%s_aplc.png', V(v).tag));
        V(v).res = r;  V(v).info = info; %#ok<AGROW>

        L = say_(L, '    lambda/D at the FPA %.3f px | bare on-axis peak %.4e', ...
                 r.lamD_px, r.peak_bare);
        L = say_(L, '    apodizer throughput %.3f (prolate Lambda0 %.4f)', ...
                 r.apodizer_throughput, r.prolate_info.lambda0);
        L = say_(L, '    APLC dark zone %g-%g lambda/D: mean %.3e, median %.3e, floor %.3e', ...
                 P.co.inner_lamD, P.co.outer_lamD, r.dz_aplc.mean, ...
                 r.dz_aplc.median, r.dz_aplc.floor);
        L = say_(L, '    on-axis suppression %.3e | net throughput %.3f', ...
                 r.supp_aplc, r.thru_aplc);
    end

    % ---- the gap cost ----------------------------------------------------
    a = V(1).res.dz_aplc;  b = V(2).res.dz_aplc;
    L = say_(L, '\n---- what the segment gaps cost ----');
    L = say_(L, '    dark-zone mean   segmented %.3e   monolithic %.3e   ratio %.2fx', ...
             a.mean, b.mean, a.mean/max(b.mean,realmin));
    L = say_(L, '    dark-zone median segmented %.3e   monolithic %.3e   ratio %.2fx', ...
             a.median, b.median, a.median/max(b.median,realmin));
    L = say_(L, '    on-axis suppr    segmented %.3e   monolithic %.3e', ...
             V(1).res.supp_aplc, V(2).res.supp_aplc);
    L = say_(L, '    SAME mask on both trains -- the apodizer is the clear-pupil');
    L = say_(L, '    prolate either way, so the ratio is the gaps and nothing else.');

    fig = compare_fig_(V, P, fullfile(P.outdir,'s3_contrast.png'));
    L = say_(L, '\n    contrast comparison figure: %s', fig);

    L = say_(L, '\nS3b DONE in %.1f min', toc(t0)/60);
    txt = strjoin(L, newline);
    fid = fopen(fullfile(P.outdir,'s3_coro_report.txt'),'w');
    fprintf(fid,'%s\n',txt);  fclose(fid);

    OUT = struct('P',P, 'V',V, 'text',txt, 'figure',fig, ...
                 'when',datestr(now,31)); %#ok<TNOW1,DATST>
    save(fullfile(P.outdir,'s3_coro_run.mat'),'OUT');
end

% =========================================================================
function setup_(here)
    run(fullfile(here,'..','..','..','mmacos_setup.m'));
end

function kinds = kinds_from_deck_(rx)
%KINDS_FROM_DECK_  Station labels for prop_layout, read from the deck.
%   The back end names its stations, so the labels come from EltName
%   rather than from an index table that would drift the moment the
%   segment count changes.
    nm = regexp(fileread(rx), '^\s*EltName=\s*(\S+)', 'tokens', 'lineanchors');
    nm = cellfun(@(c) c{1}, nm, 'UniformOutput', false);
    kinds = repmat({'optic'}, 1, numel(nm));
    for k = 1:numel(nm)
        switch nm{k}
            case {'Apodizer','Lyot'}, kinds{k} = 'marker';
            case 'FPM',               kinds{k} = 'focus';
            case 'Science',           kinds{k} = 'image';
        end
    end
end

function f = seedname_(info, which)
%SEEDNAME_  The near-field seed pair's element names.  ctb_aplc walks the
%   chain from two stations it calls DM1/DM2; here they are just the two
%   planes where the field first exists.
    fn = fieldnames(info.ix);
    p = fn(~cellfun('isempty', regexp(fn, '^Prop\d+_(start|end)$', 'once')));
    assert(numel(p) >= 2, 's3_coro: no near-field seed pair in the deck');
    if which == 1
        f = p{~cellfun('isempty', regexp(p,'_start$','once'))};
    else
        f = p{~cellfun('isempty', regexp(p,'_end$','once'))};
    end
end

function png = compare_fig_(V, P, png)
%COMPARE_FIG_  The two radial contrast curves on one axis, with the dark
%   zone shaded.  One figure, one claim: what the gaps cost.
    f = figure('Visible','off','Color','w','Position',[80 80 900 620]);
    ax = axes(f); hold(ax,'on'); set(ax,'YScale','log');
    cols = [0.75 0.15 0.15; 0.15 0.35 0.75];
    h = gobjects(1,numel(V));
    for v = 1:numel(V)
        r = V(v).res;
        [rr, cc] = macos.radial_contrast(r.I_aplc, r.peak_bare, r.lamD_px, ...
                                         P.co.outer_lamD + 3);
        h(v) = plot(ax, rr, max(cc,1e-14), '-', 'Color', cols(v,:), 'LineWidth', 1.8);
    end
    yl = ylim(ax);
    p = patch(ax, [P.co.inner_lamD P.co.outer_lamD P.co.outer_lamD P.co.inner_lamD], ...
              [yl(1) yl(1) yl(2) yl(2)], [0.90 0.90 0.95], ...
              'FaceAlpha',0.45, 'EdgeColor','none', 'HandleVisibility','off');
    uistack(p,'bottom');
    grid(ax,'on'); box(ax,'on');
    xlabel(ax,'separation  [\lambda/D]');  ylabel(ax,'contrast');
    legend(ax, h, arrayfun(@(v) sprintf('%s  (DZ mean %.2e)', V(v).label, ...
           V(v).res.dz_aplc.mean), 1:numel(V), 'UniformOutput', false), ...
           'Location','northeast');
    title(ax, sprintf(['APLC on a 6 m primary: the same clear-pupil prolate ' ...
                       'on both apertures\n%g \\lambda/D occulter, Lyot %g, ' ...
                       '%g nm; dark zone %g-%g \\lambda/D shaded'], ...
          P.co.r_occ_lamD, P.co.r_lyot_frac, P.lambda_m*1e9, ...
          P.co.inner_lamD, P.co.outer_lamD), 'Interpreter','tex');
    exportgraphics(f, png, 'Resolution', 150);
    close(f);
end

function L = say_(L, varargin)
    s = sprintf(varargin{:});
    L{end+1} = s;
    fprintf('%s\n', s);
end

function s = tern_(c,a,b), if c, s = a; else, s = b; end, end
