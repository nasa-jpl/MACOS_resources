function OUT = cf3_spacing(over)
%CF3_SPACING  Coronagraph-family campaign S3b: the DM-spacing (Talbot) trade.
%
%   THE FLAGGED TRADE (R4 / the brief): the 0.15 m DM1->DM2 spacing at a
%   47 mm beam gives Talbot amplitude authority z/z_T ~ 0.4% at
%   15 lambda/D -- the stated reason the gap-speckle dig needed damping
%   and floored near 1.9e-7.  This runner MEASURES the knob: >= 3
%   spacings from 0.15 m to the CTB-proportional value (CTB: 0.5 m on a
%   21.3 mm beam = z/z_T 6.2%; the same authority on this train's
%   47.5 mm beam needs ~2.49 m), each spacing's deck RE-EMITTED through
%   the generator chain (r1_backend -> prop_layout -> ctb_dm_rx; never
%   hand-edited), each spacing's caches tag-separated (cfd<pct> tags).
%
%   TWO STAGES, TWO PROCESSES (the model-transition heap rule -- one
%   MATLAB process per model size):
%     stage 'decks'  emit every spacing's full/prop/dm decks (engine at
%                    P.model=256 for the backend gates, P.co.model=1024
%                    for prop_layout verify -- r1_coro's own sizes) and
%                    record the SHROUD per spacing (the packaging cost
%                    of Talbot authority).
%     stage 'efc'    at N = P.dj.model (512): per spacing, cf_chain on
%                    the family config + engine-measured G + fixed-G
%                    EFC; relinearization at the LARGEST spacing (the
%                    0.15 m point reuses the S2 result -- same deck,
%                    same chain, same G caches).
%
%   Family: over.cf3.family (default 'apl', the R1-baseline chain the
%   deck's Talbot statement was written about).
%
%   Run:  matlab -batch "cf3_spacing(struct('cf3',struct('stage','decks')))"
%         matlab -batch "cf3_spacing(struct('cf3',struct('stage','efc')))"
%
%   See also CF2_EFC, CF_EFC_LIB, R1_BACKEND, ctb_dm_rx,
%   macos.design.prop_layout.

    arguments
        over struct = struct()
    end
    here = fileparts(mfilename('fullpath'));
    run(fullfile(here,'..','..','..','mmacos_setup.m'));
    ov = over;  cf3 = struct();
    if isfield(ov, 'cf3'), cf3 = ov.cf3;  ov = rmfield(ov, 'cf3'); end
    P = e2e6m_r2_params(ov);
    if ~isfield(cf3, 'stage'),    cf3.stage = 'efc';           end
    if ~isfield(cf3, 'family'),   cf3.family = 'apl';          end
    if ~isfield(cf3, 'spacings'), cf3.spacings = [0.15 0.60 2.49]; end
    if ~isfield(cf3, 'niter'),    cf3.niter = 15;              end
    if ~isfield(cf3, 'niter_r1'), cf3.niter_r1 = 10;           end
    if ~isfield(cf3, 'alphas'),   cf3.alphas = logspace(-6,-2,5); end
    addpath(fullfile(here,'..','..','30_instruments','bench_ctb'));
    addpath(fullfile(here,'..','..','..','design','src'));
    lib = cf_efc_lib();

    d0 = P.b2.d_dm2;                           % the committed spacing
    sw = cf3.spacings(abs(cf3.spacings - d0) > 1e-12);   % to generate

    switch cf3.stage
    % =====================================================================
    case 'decks'
        L = {};  t0 = tic;
        L = say_(L, '==================== e2e6m CF3b -- spacing decks (generator, never hand-edited)');
        for d = sw
            tag = tag_(d);
            L = say_(L, '\n---- spacing %.2f m (%s) ----', d, tag);
            b2 = P.b2;  b2.d_dm2 = d;  b2.tag = tag;
            B = r1_backend(struct('b2', b2));
            L = say_(L, '    full deck %s | shroud %.3f m (gate %.1f)', ...
                     B.full, B.shroud.D, P.shroud_D_m);
            full_in = B.full;
            prop_in = fullfile(P.outdir, sprintf('r1_%s_prop.in', tag));
            kinds = kinds_from_deck_(full_in);
            idm1  = B.stations(strcmp({B.stations.name},'DM1')).ielt;
            info = macos.design.prop_layout(full_in, kinds, 'out', prop_in, ...
                        'model', P.co.model, 'ngridpts', P.co.ngridpts, ...
                        'nf_legs', idm1, 'verify', true);
            L = say_(L, '    prop deck %d elements; chief vs geometric %.3g; PSF %s', ...
                     info.nElt, info.chk.chief_max, ...
                     tern_(info.chk.psf_centred,'CENTRED','** OFF-CENTRE **'));
            dm_in = fullfile(P.outdir, sprintf('r1_%s_dm.in', tag));
            Aug = ctb_dm_rx('rx_in', prop_in, 'rx_out', dm_in, ...
                            'dms', P.dm.names, 'ng', P.dm.ng);
            L = say_(L, '    dm deck %s: DM elts %s, grid dx %s m', ...
                     dm_in, mat2str(Aug.ielt), mat2str(Aug.gdx_mm, 3));
            save(fullfile(P.outdir, sprintf('cf3_decks_%s.mat', tag)), ...
                 'B', 'info', 'Aug', 'd');
        end
        L = say_(L, '\nCF3b decks DONE in %.1f min', toc(t0)/60);
        txt = strjoin(L, newline);
        fid = fopen(fullfile(P.outdir,'cf3_decks_report.txt'),'w');
        fprintf(fid,'%s\n',txt);  fclose(fid);
        OUT = struct('stage','decks', 'spacings',sw, 'text',txt);
        return

    % =====================================================================
    case 'efc'
        key = cf3.family;
        C1 = load(fullfile(P.outdir, 'cf1_run.mat'));
        k1 = find(strcmp({C1.OUT.F.key}, key), 1);
        assert(~isempty(k1), 'cf3_spacing: family %s not in cf1_run', key);
        fam = C1.OUT.F(k1);

        L = {};  t0 = tic;
        L = say_(L, '==================== e2e6m CF3b -- the DM-spacing (Talbot) trade');
        L = say_(L, 'family %s, model %d, spacings %s m, annulus %g-%g lambda/D', ...
                 fam.name, P.dj.model, mat2str(cf3.spacings), ...
                 P.co.inner_lamD, P.co.outer_lamD);

        S = struct('d',{}, 'tag',{}, 'zzT',{}, 'shroud',{}, ...
                   'c_static',{}, 'c_fixed',{}, 'c_relin',{}, ...
                   'strokes_nm',{}, 'sym_after',{});
        dmax = max(cf3.spacings);
        for d = cf3.spacings
            zzT = d / (2 * (2*0.023771 / P.co.outer_lamD)^2 / P.lambda_m);
            if abs(d - d0) < 1e-12
                % the committed spacing = the S2 point, reused verbatim
                s2 = fullfile(P.outdir, sprintf('cf2_%s_run.mat', key));
                assert(isfile(s2), 'cf3_spacing: %s missing -- run cf2_efc first', s2);
                Q = load(s2);
                L = say_(L, '\n---- %.2f m: the committed spacing = the S2 point (reused) ----', d);
                L = say_(L, '    static %.3e -> fixed-G %.3e -> relin %.3e', ...
                         Q.res.c_static, Q.res.c_fixed, Q.res.c_relin);
                B1 = load(fullfile(P.outdir, 'r1_seg_run.mat'));
                S(end+1) = struct('d',d, 'tag','seg(S2)', 'zzT',zzT, ...
                    'shroud', B1.OUT.shroud.D, ...
                    'c_static',Q.res.c_static, 'c_fixed',Q.res.c_fixed, ...
                    'c_relin',Q.res.c_relin, 'strokes_nm',Q.res.strokes_nm, ...
                    'sym_after',Q.res.sym_after); %#ok<AGROW>
                continue
            end
            tag = tag_(d);
            L = say_(L, '\n---- spacing %.2f m (%s), z/z_T(%g lamD) %.3f ----', ...
                     d, tag, P.co.outer_lamD, zzT);
            D = load(fullfile(P.outdir, sprintf('cf3_decks_%s.mat', tag)));
            rx = fullfile(P.outdir, sprintf('r1_%s_dm.in', tag));
            assert(isfile(rx), 'cf3_spacing: %s missing -- run stage ''decks''', rx);

            ch = cf_chain('rx', rx, 'model_size', P.dj.model, ...
                          'prolate_iter', P.co.prolate_iter, ...
                          'circ_stop_frac', P.cf.circ_stop_frac, fam.cfg{:});
            beam_d = 2 * 0.023771;
            dm = cell(1, numel(D.Aug.ielt));
            for k = 1:numel(dm)
                dm{k} = ctb_dm('ielt', D.Aug.ielt(k), 'ng', D.Aug.ng, ...
                               'gdx_mm', D.Aug.gdx_mm(k), 'nact', P.dj.nact, ...
                               'beam_d_mm', beam_d, 'pitch_mm', beam_d/P.dj.nact, ...
                               'coupling', P.dj.coupling);
                dm{k}.clear();
            end
            dz_idx = find(ch.dz_mask(P.co.inner_lamD, P.co.outer_lamD));
            a0 = cellfun(@(x) zeros(x.nact^2,1), dm, 'UniformOutput', false);
            G0 = lib.jacobian(ch, dm, a0, dz_idx, P, ...
                fullfile(P.outdir, sprintf('cf3_G_%s_%s.mat', tag, key)));
            [afix, con_fix] = lib.efc(ch, dm, G0, a0, dz_idx, cf3.niter, cf3.alphas);
            c_relin = NaN;
            aend = afix;
            if abs(d - dmax) < 1e-12
                G1 = lib.jacobian(ch, dm, afix, dz_idx, P, ...
                    fullfile(P.outdir, sprintf('cf3_G_%s_%s_r1.mat', tag, key)));
                [aend, con_rel] = lib.efc(ch, dm, G1, afix, dz_idx, ...
                                          cf3.niter_r1, cf3.alphas);
                c_relin = con_rel(end);
            end
            lib.seta(dm, aend);
            E = ch.run();
            sym_after = lib.sym_frac(E, ch.center_px, dz_idx);
            lib.seta(dm, a0);
            strokes = cellfun(@(x) 1e9 * rms_(x(x~=0)), aend);
            L = say_(L, '    static %.3e -> fixed-G %.3e -> relin %s | strokes [%s] nm | sym %.2f | shroud %.3f m', ...
                     con_fix(1), con_fix(end), fmtn_(c_relin), ...
                     num2str(strokes, '%.2f '), sym_after, D.B.shroud.D);
            S(end+1) = struct('d',d, 'tag',tag, 'zzT',zzT, ...
                'shroud', D.B.shroud.D, ...
                'c_static',con_fix(1), 'c_fixed',con_fix(end), ...
                'c_relin',c_relin, 'strokes_nm',strokes, ...
                'sym_after',sym_after); %#ok<AGROW>
        end
        [~, ord] = sort([S.d]);  S = S(ord);

        L = say_(L, '\n==== the spacing trade (%s; the deck''s Talbot statement, regenerated from measurement) ====', fam.name);
        L = say_(L, '  %-8s | %-8s | %-10s | %-10s | %-10s | %-11s | %-6s | %s', ...
                 'd [m]', 'z/z_T', 'static', 'fixed-G', 'relin', 'strokes nm', 'sym', 'shroud m');
        L = say_(L, '  %s', repmat('-', 1, 96));
        for q = 1:numel(S)
            L = say_(L, '  %-8.2f | %-8.3f | %.3e | %.3e | %-10s | %4.1f/%4.1f | %.2f | %.3f', ...
                     S(q).d, S(q).zzT, S(q).c_static, S(q).c_fixed, ...
                     fmtn_(S(q).c_relin), S(q).strokes_nm(1), S(q).strokes_nm(2), ...
                     S(q).sym_after, S(q).shroud);
        end
        L = say_(L, '  %s', repmat('-', 1, 96));

        png = fullfile(P.outdir, 'cf3_spacing.png');
        fig_(S, P, fam, png);
        L = say_(L, '\n  figure: %s', png);
        L = say_(L, '\nCF3b DONE in %.1f min', toc(t0)/60);
        txt = strjoin(L, newline);
        fid = fopen(fullfile(P.outdir,'cf3_spacing_report.txt'),'w');
        fprintf(fid,'%s\n',txt);  fclose(fid);
        OUT = struct('P',P, 'S',S, 'family',key, 'text',txt, 'figure',png, ...
                     'when',datestr(now,31)); %#ok<TNOW1,DATST>
        save(fullfile(P.outdir,'cf3_spacing_run.mat'),'OUT');

    otherwise
        error('cf3_spacing: unknown stage "%s"', cf3.stage);
    end
end

% =========================================================================
function t = tag_(d)
    t = sprintf('cfd%03d', round(100*d));
end

function kinds = kinds_from_deck_(rx)
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

function fig_(S, P, fam, png)
    f = figure('Visible','off','Color','w','Position',[60 60 1500 600]);
    tl = tiledlayout(f, 1, 2, 'TileSpacing','compact', 'Padding','compact');
    title(tl, sprintf(['e2e6m -- DM-spacing (Talbot) trade, %s chain ' ...
        '(N=%d, annulus %g-%g \\lambda/D)'], fam.name, P.dj.model, ...
        P.co.inner_lamD, P.co.outer_lamD), 'FontWeight','bold', 'Interpreter','tex');
    ax = nexttile(tl); hold(ax,'on'); set(ax,'XScale','log','YScale','log');
    plot(ax, [S.zzT], [S.c_static], 'ks--', 'MarkerFaceColor',[0.7 0.7 0.7], ...
         'LineWidth',1.2, 'MarkerSize',9);
    plot(ax, [S.zzT], [S.c_fixed], 'b^-', 'MarkerFaceColor',[0.4 0.6 0.9], ...
         'LineWidth',1.6, 'MarkerSize',9);
    m = ~isnan([S.c_relin]);
    plot(ax, [S(m).zzT], [S(m).c_relin], 'ro-', 'MarkerFaceColor',[0.9 0.5 0.4], ...
         'LineWidth',1.6, 'MarkerSize',9);
    for q = 1:numel(S)
        text(ax, S(q).zzT*1.06, S(q).c_fixed*1.2, sprintf('%.2f m', S(q).d), ...
             'FontSize', 12);
    end
    grid(ax,'on'); box(ax,'on');
    xlabel(ax, sprintf('Talbot authority z/z_T at %g \\lambda/D', P.co.outer_lamD));
    ylabel(ax,'dark-zone mean contrast');
    legend(ax, {'static (pre-control)','fixed-G floor','relin floor'}, 'Location','best');
    title(ax,'what DM spacing buys against gap speckle');
    ax = nexttile(tl); hold(ax,'on');
    yyaxis(ax,'left');
    plot(ax, [S.d], [S.shroud], 'o-', 'LineWidth',1.6, 'MarkerSize',8);
    ylabel(ax,'shroud diameter [m]');
    yline(ax, P.shroud_D_m, ':', sprintf('%.0f m gate', P.shroud_D_m));
    yyaxis(ax,'right');
    st = arrayfun(@(s) max(s.strokes_nm), S);
    plot(ax, [S.d], st, 's--', 'LineWidth',1.4, 'MarkerSize',8);
    ylabel(ax,'max DM stroke rms [nm]');
    grid(ax,'on'); box(ax,'on');
    xlabel(ax,'DM1 \rightarrow DM2 spacing [m]');
    title(ax,'the packaging + stroke price of Talbot authority');
    exportgraphics(f, png, 'Resolution', 150);
    close(f);
end

function t = fmtn_(v)
    if isnan(v), t = '--'; else, t = sprintf('%.3e', v); end
end
function r = rms_(v), v = v(:); if isempty(v), r = 0; else, r = sqrt(mean(v.^2)); end, end
function L = say_(L, varargin)
    s = sprintf(varargin{:});  L{end+1} = s;  fprintf('%s\n', s);
end
function s = tern_(c,a,b), if c, s = a; else, s = b; end, end
