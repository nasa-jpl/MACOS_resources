function OUT = r1_coro(over)
%R1_CORO  e2e6m round 2: the APLC on the DM-bearing train, both primaries.
%
%   Round 1's s3_coro, re-run on the R1 back end: builds the DM-bearing
%   train on BOTH primaries (19-segment and monolithic), converts each
%   to a diffraction deck with `macos.design.prop_layout`, and scores
%   the SAME clear-pupil APLC on each with the committed `ctb_aplc`
%   chain, so the gap cost stays a measured difference.
%
%   WHAT MOVED vs round 1:
%   * the near-field SEED is the DM1 -> DM2 leg (the CTB convention),
%     so the complex field exists AT the DM planes -- the EFC layer's
%     probe planes -- and ctb_aplc's DM1/DM2 stations are the real
%     planes rather than apodizer-leg stand-ins;
%   * the deck carries FieldStop (quartet) and Backend (pupil marker)
%     stations, traversed transparently by the scoring walk.
%
%   The mask parameters are round 1's exactly (occulter 2.8 lambda/D,
%   Lyot 0.90, prolate on the clear pupil), so the R1-vs-round-1
%   contrast difference measures the TOPOLOGY change alone, and the
%   seg-vs-mono ratio measures the gaps alone.
%
%   OUT = R1_CORO()      defaults
%   OUT = R1_CORO(OVER)  with e2e6m_r2_params overrides
%
%   See also R1_BACKEND, ../e2e6m/s3_coro, ctb_aplc,
%   macos.design.prop_layout.

    arguments
        over struct = struct()
    end
    here = fileparts(mfilename('fullpath'));
    run(fullfile(here,'..','..','..','mmacos_setup.m'));
    P = e2e6m_r2_params(over);
    addpath(fullfile(here,'..','..','30_instruments','bench_ctb'));

    L = {};  t0 = tic;
    L = say_(L, '==================== e2e6m R1 -- APLC on the DM-bearing train');
    L = say_(L, 'metric: dark-zone contrast, Strehl-normalised to the BARE');
    L = say_(L, '        on-axis peak of the SAME train; annulus %g-%g lambda/D', ...
             P.co.inner_lamD, P.co.outer_lamD);
    L = say_(L, '        occulter %g lambda/D, Lyot %g, %g nm, model %d, nGridpts %d', ...
             P.co.r_occ_lamD, P.co.r_lyot_frac, P.lambda_m*1e9, ...
             P.co.model, P.co.ngridpts);

    V = struct('tag',{'seg','mono'}, ...
               'base',{'s2_segmented.in','s1_telescope.in'}, ...
               'label',{'19-segment primary','monolithic primary'}, ...
               'res',{[],[]});

    for v = 1:numel(V)
        L = say_(L, '\n---- %s (%s) ----', V(v).tag, V(v).label);
        b2 = P.b2;  b2.tag = V(v).tag;  b2.base_in = V(v).base;
        B = r1_backend(struct('b2',b2));
        full_in = fullfile(P.outdir, sprintf('r1_%s_full.in', V(v).tag));
        prop_in = fullfile(P.outdir, sprintf('r1_%s_prop.in', V(v).tag));

        kinds = kinds_from_deck_(full_in);
        idm1  = B.stations(strcmp({B.stations.name},'DM1')).ielt;
        info = macos.design.prop_layout(full_in, kinds, 'out', prop_in, ...
                    'model', P.co.model, 'ngridpts', P.co.ngridpts, ...
                    'nf_legs', idm1, 'verify', true);
        L = say_(L, '    diffraction deck %d elements; chief vs geometric %.3g; PSF %s', ...
                 info.nElt, info.chk.chief_max, ...
                 tern_(info.chk.psf_centred,'CENTRED','** OFF-CENTRE **'));
        L = say_(L, '    seed on the DM1->DM2 leg (station %d); FEX radii: FPM %.6f m, ExitPupil %.6f m', ...
                 idm1, info.R.FPM, info.R.ExitPupil);

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
        movefile(r.figure, fullfile(P.outdir, sprintf('r1_%s_aplc.png', V(v).tag)));
        r.figure = fullfile(P.outdir, sprintf('r1_%s_aplc.png', V(v).tag));
        V(v).res = r;  V(v).info = info;  V(v).B = B; %#ok<AGROW>

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

    % ---- the gap cost, and the topology check ---------------------------
    a = V(1).res.dz_aplc;  b = V(2).res.dz_aplc;
    L = say_(L, '\n---- what the segment gaps cost (DM-bearing train) ----');
    L = say_(L, '    dark-zone mean   segmented %.3e   monolithic %.3e   ratio %.2fx', ...
             a.mean, b.mean, a.mean/max(b.mean,realmin));
    L = say_(L, '    dark-zone median segmented %.3e   monolithic %.3e   ratio %.2fx', ...
             a.median, b.median, a.median/max(b.median,realmin));
    L = say_(L, '    on-axis suppr    segmented %.3e   monolithic %.3e', ...
             V(1).res.supp_aplc, V(2).res.supp_aplc);
    ref = round1_ref_(P);
    if ~isempty(ref)
        L = say_(L, '    round-1 (no-DM train) seg dark-zone mean %.3e -> topology cost %.2fx', ...
                 ref, a.mean/max(ref,realmin));
        L = say_(L, '    (open-loop: the DMs are FLAT here; R4 closes the loop)');
    end

    fig = compare_fig_(V, P, fullfile(P.outdir,'r1_contrast.png'));
    L = say_(L, '\n    contrast comparison figure: %s', fig);

    L = say_(L, '\nR1 coro DONE in %.1f min', toc(t0)/60);
    txt = strjoin(L, newline);
    fid = fopen(fullfile(P.outdir,'r1_coro_report.txt'),'w');
    fprintf(fid,'%s\n',txt);  fclose(fid);

    % PRUNE derivable heavy arrays before saving (the >=20 MB derived-
    % binary rule round 1 bent: its s3_coro_run.mat committed 32 MB of
    % intensity maps).  The reports + PNGs carry the content; masks and
    % intensities rebuild deterministically via ctb_aplc.
    pruned = {};
    for v = 1:numel(V)
        fn = fieldnames(V(v).res);
        for q = 1:numel(fn)
            x = V(v).res.(fn{q});
            if isnumeric(x) && numel(x) > 1e6
                V(v).res.(fn{q}) = [];
                pruned{end+1} = sprintf('%s.%s', V(v).tag, fn{q}); %#ok<AGROW>
            end
        end
    end
    OUT = struct('P',P, 'V',V, 'text',txt, 'figure',fig, ...
                 'pruned',{pruned}, 'when',datestr(now,31)); %#ok<TNOW1,DATST>
    save(fullfile(P.outdir,'r1_coro_run.mat'),'OUT');
end

% =========================================================================
function kinds = kinds_from_deck_(rx)
%KINDS_FROM_DECK_  Station labels for prop_layout, read from EltName.
    nm = regexp(fileread(rx), '^\s*EltName=\s*(\S+)', 'tokens', 'lineanchors');
    nm = cellfun(@(c) c{1}, nm, 'UniformOutput', false);
    kinds = repmat({'optic'}, 1, numel(nm));
    for k = 1:numel(nm)
        switch nm{k}
            case {'Apodizer','Lyot','Backend'}, kinds{k} = 'marker';
            case {'FPM','FieldStop'},           kinds{k} = 'focus';
            case 'Science',                     kinds{k} = 'image';
        end
    end
end

function f = seedname_(info, which)
%SEEDNAME_  The near-field seed pair's element names (here: the planes
%   AT DM1 and DM2 -- the seed leg is the DM gap).
    fn = fieldnames(info.ix);
    p = fn(~cellfun('isempty', regexp(fn, '^Prop\d+_(start|end)$', 'once')));
    assert(numel(p) >= 2, 'r1_coro: no near-field seed pair in the deck');
    if which == 1
        f = p{~cellfun('isempty', regexp(p,'_start$','once'))};
    else
        f = p{~cellfun('isempty', regexp(p,'_end$','once'))};
    end
end

function ref = round1_ref_(P)
%ROUND1_REF_  Round 1's segmented dark-zone mean, if its run is present.
    f = fullfile(P.r1dir, 's3_coro_run.mat');
    ref = [];
    if isfile(f)
        S = load(f, 'OUT');
        try  ref = S.OUT.V(1).res.dz_aplc.mean;  catch,  ref = [];  end
    end
end

function png = compare_fig_(V, P, png)
%COMPARE_FIG_  Two radial contrast curves, dark zone shaded.
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
    title(ax, sprintf(['APLC on the DM-bearing train (DMs flat, open loop)\n' ...
                       '%g \\lambda/D occulter, Lyot %g, %g nm; dark zone ' ...
                       '%g-%g \\lambda/D shaded'], ...
          P.co.r_occ_lamD, P.co.r_lyot_frac, P.lambda_m*1e9, ...
          P.co.inner_lamD, P.co.outer_lamD), 'Interpreter','tex');
    exportgraphics(f, png, 'Resolution', 150);
    close(f);
end

function L = say_(L, varargin)
    s = sprintf(varargin{:});  L{end+1} = s;  fprintf('%s\n', s);
end
function s = tern_(c,a,b), if c, s = a; else, s = b; end, end
