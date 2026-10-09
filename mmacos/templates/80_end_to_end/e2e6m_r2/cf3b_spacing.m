function OUT = cf3b_spacing(over)
%CF3B_SPACING  S3b: the DM-spacing sweep -- the decisive Talbot knob.
%
%   CF2 measured every family within ~2x of its own Jacobian's
%   linear-achievable floor: the substrate (amplitude gap speckle vs
%   Talbot-weak authority, z/z_T = 3.7e-3 at 15 lambda/D for the
%   0.15 m baseline) owns the floors.  This sweep MOVES the knob: for
%   each DM1->DM2 spacing the back end is RE-EMITTED VIA THE GENERATOR
%   (r1_backend -> prop_layout -> ctb_dm_rx, the r1_coro recipe; decks
%   are never hand-edited), and the CF2 protocol runs on the apodized-
%   Lyot leg (the campaign winner, held at its CF2 operating point
%   L=0.90 so the sweep isolates SPACING) -- Jacobian + fixed-G +
%   relin + lin-ach -- plus vortex c4 at a subset for cross-family
%   shape.
%
%   Spacings (default): 0.15 (baseline, reused from CF2 -- no re-run),
%   0.40, 0.70, 1.10 m (the CTB-proportional value: 0.5 m at a 21.3 mm
%   beam -> ~1.1 m at this 47 mm beam).  Packaging is measured and
%   ANNOTATED per point (the r1_backend shroud number), never a reason
%   to skip the physics.
%
%   Caches are tag-separated (the spacing is in the deck tag AND in
%   chain_opts via the per-spacing rx path).  Expectation to CHECK,
%   not assume: floor tracks the improving amplitude authority until
%   some other term takes over -- the report flags the knee if it is
%   inside the sweep.
%
%   OUT = CF3B_SPACING()
%   OUT = CF3B_SPACING(struct('cf3b', struct('spacings', [0.15 0.7])))
%
%   See also CF2_EFC, R1_BACKEND, R1_CORO, R1_DM, cf_efc_lib.

    arguments
        over struct = struct()
    end
    here = fileparts(mfilename('fullpath'));
    run(fullfile(here,'..','..','..','mmacos_setup.m'));
    ov = over;  c3 = struct();
    if isfield(ov,'cf3b'), c3 = ov.cf3b;  ov = rmfield(ov,'cf3b'); end
    P = e2e6m_r2_params(ov);
    if ~isfield(c3,'spacings'),    c3.spacings = [0.15 0.40 0.70 1.10]; end
    if ~isfield(c3,'v4_spacings'), c3.v4_spacings = [0.15 0.70]; end
    addpath(fullfile(here,'..','..','30_instruments','bench_ctb'));
    lib = cf_efc_lib();

    A0 = load(fullfile(P.outdir,'r1_dm_run.mat'));
    beam_d = 2 * 0.023771;
    C1 = load(fullfile(P.outdir,'cf1_run.mat'));
    FC = struct();
    for k = 1:numel(C1.OUT.F), FC.(C1.OUT.F(k).key) = C1.OUT.F(k); end

    L = {};  t0 = tic;
    L = say_(L, '==== e2e6m CF3b -- the DM-spacing sweep (apl leg %s m; v4 at %s m)', ...
             mat2str(c3.spacings), mat2str(c3.v4_spacings));

    R = struct('d',{}, 'leg',{}, 'tag',{}, 'c_static',{}, 'c_fixed',{}, ...
               'c_relin',{}, 'la_floor',{}, 'ach_nm',{}, 'zzT',{}, ...
               'shroud',{}, 'resumed',{});
    for d = unique([c3.spacings, c3.v4_spacings])
        legs = {};
        if any(abs(c3.spacings - d) < 1e-9),    legs{end+1} = 'apl'; end %#ok<AGROW>
        if any(abs(c3.v4_spacings - d) < 1e-9), legs{end+1} = 'v4';  end %#ok<AGROW>

        if abs(d - P.b2.d_dm2) < 1e-9
            % the baseline: CF2's own artifacts ARE this point
            for g = 1:numel(legs)
                B = load(fullfile(P.outdir, sprintf('cf2_%s_run.mat', legs{g})));
                R(end+1) = point_(d, legs{g}, B.res.tag, B.res.c_static, ...
                    B.res.c_fixed, B.res.c_relin, B.res.la1_ach.floor, ...
                    B.res.ach_nm, zzT_(P, d, beam_d), NaN, 'cf2-baseline'); %#ok<AGROW>
                L = say_(L, 'd=%.2f m %-4s: CF2 baseline reused (%.3e relin)', ...
                         d, legs{g}, B.res.c_relin);
            end
            continue
        end

        % ---- re-emit the train at this spacing (the r1_coro recipe) -----
        vtag = sprintf('seg_d%03d', round(100*d));
        L = say_(L, '\n-- spacing %.2f m: emitting r1_%s_* via the generator', d, vtag);
        b2 = P.b2;  b2.tag = vtag;  b2.d_dm2 = d;
        B2 = r1_backend(struct('b2', b2));
        shroud = B2.shroud;
        full_in = fullfile(P.outdir, sprintf('r1_%s_full.in', vtag));
        prop_in = fullfile(P.outdir, sprintf('r1_%s_prop.in', vtag));
        kinds = kinds_from_deck_(full_in);
        idm1  = B2.stations(strcmp({B2.stations.name},'DM1')).ielt;
        info = macos.design.prop_layout(full_in, kinds, 'out', prop_in, ...
                    'model', P.co.model, 'ngridpts', P.co.ngridpts, ...
                    'nf_legs', idm1, 'verify', true);
        assert(info.chk.psf_centred, 'cf3b: PSF off-centre at d=%.2f', d);
        dmrx = fullfile(P.outdir, sprintf('r1_%s_dm.in', vtag));
        Adm = ctb_dm_rx('rx_in', prop_in, 'rx_out', dmrx, ...
                        'dms', P.dm.names, 'ng', P.dm.ng);
        L = say_(L, '   emitted %s (DM elts %s); shroud %s', dmrx, ...
                 mat2str(Adm.ielt), shroud_str_(shroud));

        for g = 1:numel(legs)
            key = legs{g};
            ch = cf_chain('rx', dmrx, 'model_size', P.dj.model, ...
                          'prolate_iter', P.co.prolate_iter, ...
                          'circ_stop_frac', P.cf.circ_stop_frac, FC.(key).cfg{:});
            tag = sprintf('%s_%s', vtag, ch.tag);
            dm = cell(1, numel(Adm.ielt));
            for k = 1:numel(dm)
                dm{k} = ctb_dm('ielt', Adm.ielt(k), 'ng', Adm.ng, ...
                    'gdx_mm', Adm.gdx_mm(k), 'nact', P.dj.nact, ...
                    'beam_d_mm', beam_d, 'pitch_mm', beam_d/P.dj.nact, ...
                    'coupling', P.dj.coupling);
                dm{k}.clear();
            end
            dz_idx = find(ch.dz_mask(P.co.inner_lamD, P.co.outer_lamD));
            a0 = cellfun(@(dd) zeros(dd.nact^2,1), dm, 'UniformOutput', false);
            [G0, ~] = lib.jacobian(ch, dm, a0, dz_idx, P, ...
                fullfile(P.outdir, sprintf('cf2_G_%s.mat', tag)));
            [afix, cf, ~] = lib.efc(ch, dm, G0, a0, dz_idx, 15, logspace(-6,-2,5));
            [G1, ~] = lib.jacobian(ch, dm, afix, dz_idx, P, ...
                fullfile(P.outdir, sprintf('cf2_G_%s_r1.mat', tag)));
            [arel, cr, ~] = lib.efc(ch, dm, G1, afix, dz_idx, 10, logspace(-6,-2,5));
            la1 = lib.linfloor(G1, P.cf.stroke_bound_nm);
            ach = 1e9 * rms_(cell2mat(cellfun(@(x) x(x~=0), arel(:).', ...
                                              'UniformOutput', false).'));
            fa = floor_at_(la1, ach);
            R(end+1) = point_(d, key, tag, cf(1), cf(end), cr(end), ...
                fa.floor, ach, zzT_(P, d, beam_d), shroud, 'measured'); %#ok<AGROW>
            L = say_(L, '   %-4s: %.3e -> %.3e -> %.3e | lin-ach %.3e @ %.1f nm | z/z_T %.2e', ...
                     key, cf(1), cf(end), cr(end), fa.floor, ach, R(end).zzT);
            lib.seta(dm, a0);
        end
    end

    % ---- curves + knee check --------------------------------------------
    png = fullfile(P.outdir, 'cf3b_spacing.png');
    fig_(R, P, png);
    apl = R(strcmp({R.leg}, 'apl'));
    [~, si] = sort([apl.d]);  apl = apl(si);
    if numel(apl) >= 3
        f = [apl.c_relin];
        mono = all(diff(f) < 0);
        if mono
            L = say_(L, '\nno knee inside the sweep: the floor still improves at %.2f m -- the', apl(end).d);
            L = say_(L, 'authority term has not yet handed over.');
        else
            [~, kn] = min(f);
            L = say_(L, '\nKNEE at d = %.2f m: floor %.3e, worsening beyond.', apl(kn).d, f(kn));
        end
    end
    L = say_(L, '\n  figure: %s', png);
    L = say_(L, 'CF3b DONE in %.1f min', toc(t0)/60);
    txt = strjoin(L, newline);
    fid = fopen(fullfile(P.outdir,'cf3b_report.txt'),'w');
    fprintf(fid,'%s\n',txt);  fclose(fid);
    OUT = struct('R',R, 'text',txt, 'figure',png, 'when',datestr(now,31)); %#ok<TNOW1,DATST>
    save(fullfile(P.outdir,'cf3b_run.mat'),'OUT');
end

% =========================================================================
function r = point_(d, leg, tag, cs, cfx, crl, la, ach, zzT, shroud, how)
    r = struct('d',d, 'leg',leg, 'tag',tag, 'c_static',cs, 'c_fixed',cfx, ...
               'c_relin',crl, 'la_floor',la, 'ach_nm',ach, 'zzT',zzT, ...
               'shroud',shroud, 'resumed',how);
end

function z = zzT_(P, d, beam_d)
%ZZT_  z/z_T at the outer working angle for spacing d.
    p_min = beam_d / P.co.outer_lamD;
    z = d / (2 * p_min^2 / P.lambda_m);
end

function s = shroud_str_(sh)
    if isstruct(sh) && isfield(sh, 'D'),  s = sprintf('%.3f m', sh.D);
    elseif isnumeric(sh) && isscalar(sh), s = sprintf('%.3f m', sh);
    else,                                   s = 'see r1_backend report';
    end
end

function kinds = kinds_from_deck_(rx)
%KINDS_FROM_DECK_  Station labels for prop_layout (r1_coro's rule).
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

function fa = floor_at_(la, stroke_nm)
    ok = la.curve_stroke_nm <= stroke_nm;
    if ~any(ok), rk = 1; else, rk = find(ok, 1, 'last'); end
    fa = struct('floor', la.curve_con(rk), 'rank', rk, ...
                'stroke_nm', la.curve_stroke_nm(rk));
end

function fig_(R, P, png)
    f = figure('Visible','off','Color','w','Position',[60 60 760 540]);
    ax = axes(f); hold(ax,'on'); grid(ax,'on'); box(ax,'on');
    set(ax,'YScale','log','XScale','log');
    legs = unique({R.leg});
    cols = lines(numel(legs));  h = gobjects(1, 2*numel(legs));  n = 0;
    for g = 1:numel(legs)
        r = R(strcmp({R.leg}, legs{g}));
        [~, si] = sort([r.d]);  r = r(si);
        n = n + 1;
        h(n) = loglog(ax, [r.d], [r.c_relin], 'o-', 'Color', cols(g,:), ...
            'LineWidth', 1.8, 'DisplayName', [legs{g} ' relin floor']);
        n = n + 1;
        h(n) = loglog(ax, [r.d], [r.la_floor], 's--', 'Color', cols(g,:), ...
            'LineWidth', 1.1, 'DisplayName', [legs{g} ' lin-achievable']);
    end
    xlabel(ax, 'DM1 \rightarrow DM2 spacing  [m]');
    ylabel(ax, 'dark-zone mean contrast');
    title(ax, {'The Talbot knob: closed-loop floor vs DM spacing', ...
        sprintf('z/z_T at %g \\lambda/D spans %.1e..%.1e over the sweep', ...
        P.co.outer_lamD, min([R.zzT]), max([R.zzT]))}, 'FontWeight','bold');
    legend(ax, h(1:n), 'Location', 'southwest');
    exportgraphics(f, png, 'Resolution', 150);
    close(f);
end

function r = rms_(v), v = v(:); if isempty(v), r = 0; else, r = sqrt(mean(v.^2)); end, end
function L = say_(L, varargin)
    s = sprintf(varargin{:});  L{end+1} = s;  fprintf('%s\n', s);
end
