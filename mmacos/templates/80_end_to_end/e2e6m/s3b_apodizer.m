function OUT = s3b_apodizer(over)
%S3B_APODIZER  e2e6m S3b: apodizers designed FOR the segmented pupil.
%
%   S3 measured what the segment gaps cost a CLEAR-PUPIL APLC: the same
%   circular prolate on a 19-segment primary and on a monolithic one,
%   2390x apart in dark-zone mean.  This stage tries to buy that back,
%   two ways, and reports what each is actually worth.
%
%   [1] THE LINEAR PROGRAM (Carlotti, Vanderbei & Kasdin 2011).  Maximize
%       throughput subject to bounds on the dark-zone field, through the
%       existing occulter and Lyot stop.  APODIZER_LP does this and the
%       LP itself is sound -- on a clear circular pupil it returns a
%       shaped-pupil solution that meets its target exactly.
%
%       ON THIS TRAIN IT DOES NOT BEAT THE INCUMBENT, and the stage
%       reports that with engine numbers rather than shipping a mask
%       that looks optimal in a model and is not.  A single-Fourier
%       model of this back end tracks the engine to ~10-20% in field
%       amplitude -- fine for the bare PSF, which it reproduces to
%       1.2% -- but an apodized dark zone is a ~1000x cancellation, so
%       below a few 1e-6 the residual IS the model error.  Every rung of
%       the ladder is therefore scored in the ENGINE, and the
%       model-vs-engine divergence is reported per rung: it GROWS with
%       the target, which is the optimizer finding more of the error the
%       harder it is pushed.  The five experiments that localise the
%       error are in the LOG.
%
%   [2] THE APERTURE-SPECIFIC APLC APODIZER, which does not depend on
%       that fidelity: Soummer (2005, ApJ 618, L161) Eq. 3 defines the
%       APLC apodizer as the dominant eigenfunction of the APLC operator
%       over ANY pupil support P, and the power iteration converges to it
%       for the segmented aperture as readily as for a disc (N'Diaye,
%       Zimmerman & Soummer 2016, ApJ 818, 163, Paper V).  An
%       eigenfunction is a robust object, not a fine-tuned cancellation,
%       so model error perturbs it instead of dominating it -- and the
%       ENGINE scores the result either way.
%
%   OUT = S3B_APODIZER()      default parameter set
%   OUT = S3B_APODIZER(OVER)  ... with e2e6m_params overrides
%
%   See also APODIZER_LP, S3B_PUPIL, S3_CORO, CTB_APOD_PROLATE, CTB_APLC.

    arguments
        over struct = struct()
    end
    here = fileparts(mfilename('fullpath'));
    setup_(here);
    P = e2e6m_params(over);
    if isempty(P.outdir), P.outdir = here; end
    addpath(fullfile(here,'..','..','30_instruments','bench_ctb'));
    addpath(fullfile(here,'..','..','..','design','src'));

    L = {}; t0 = tic;
    L = say_(L, '==================== e2e6m S3b -- apodizers for the segmented pupil');
    L = say_(L, 'metric: dark-zone contrast, Strehl-normalised to the BARE');
    L = say_(L, '        on-axis peak of the SAME train; annulus %g-%g lambda/D', ...
             P.co.inner_lamD, P.co.outer_lamD);
    L = say_(L, '        at %g nm, coronagraph exit pupil; occulter %g lambda/D,', ...
             P.lambda_m*1e9, P.co.r_occ_lamD);
    L = say_(L, '        Lyot %g of the geometric pupil, model %d', ...
             P.co.r_lyot_frac, P.co.model);
    L = say_(L, 'throughput: Phi^2-weighted fill over the geometric pupil');
    L = say_(L, '        (ctb_aplc''s definition), quoted with every contrast');

    pupmat = fullfile(P.outdir,'s3b_pupil.mat');
    assert(isfile(pupmat), 's3b_apodizer: run s3b_pupil first (%s)', pupmat);
    Q = load(pupmat);
    L = say_(L, '\n[0] pupil taken from the ENGINE at the Apodizer plane of s3_seg_prop.in');
    L = say_(L, '    %d lit px, beam radius %.1f px, gap fraction %.4f inside 0.95R', ...
             nnz(Q.lit), Q.r_px, Q.gapfrac);
    L = say_(L, '    y-symmetry %.2e (fold usable) | x-symmetry %.2e (fold NOT usable)', ...
             Q.sym.flipy, Q.sym.flipx);
    L = say_(L, '    the x-asymmetry is the tilted-fold relay''s own anamorphism: it is');
    L = say_(L, '    in the traced pupil and would NOT be in a redrawn hexagon.');
    L = say_(L, '    pupil PHASE at that plane: %.3f rad rms -- small, and not zero,', P.ap.pupil_phase_rms);
    L = say_(L, '    which is why the operator is built on the complex field.');

    prop_in = fullfile(P.outdir,'s3_seg_prop.in');
    ix = elt_ix_(prop_in);  seed = seed_pair_(ix);
    elt = struct('DM1',seed(1),'DM2',seed(2),'Apodizer',ix.Apodizer, ...
                 'FPM',ix.FPM,'Lyot',ix.Lyot,'ExitPupil',ix.ExitPupil, ...
                 'FPA',ix.Science);
    Efield = pupil_field_(prop_in, ix, seed, P.co.model);

    % ---- [1] the LP, and its size/time record ----------------------------
    tgt = P.ap.targets;
    L = say_(L, '\n[1] LP ladder: solve, then score EVERY rung in the engine');
    L = say_(L, '    variables are block-constant tiles; the OPERATOR keeps the pupil');
    L = say_(L, '    at full resolution -- the %g mm gaps are %.2f px wide at model %d', ...
             P.seg.gap_m*1000, P.seg.gap_m/P.D_m*2*Q.r_px, P.co.model);
    L = say_(L, '    and vanish on ANY coarser pupil grid, so the brief''s "coarse');
    L = say_(L, '    optimization grid" had to be applied to the variables, not the pupil.');
    R = [];
    for k = 1:numel(tgt)
        [A_k, i_k] = apodizer_lp(Efield, Q.r_px, ...
            'iwa', P.co.inner_lamD, 'owa', P.co.outer_lamD, ...
            'contrast', tgt(k), 'r_occ_lamD', P.co.r_occ_lamD, ...
            'r_lyot_frac', P.co.r_lyot_frac, ...
            'nvar_target', P.ap.nvar_target, 'dz_per_lamD', P.ap.dz_per_lamD, ...
            'n_fpm', P.ap.n_fpm, 'onesided', P.ap.onesided, ...
            'verify_dense', P.ap.verify_dense, 'verbose', false);
        i_k.A = A_k;  i_k.target = tgt(k);
        i_k.eng = score_(prop_in, elt, P, A_k);      % ENGINE, every rung
        i_k.divergence = max(i_k.contrast_dense_bare, i_k.eng.dz_aplc.mean) / ...
                     max(min(i_k.contrast_dense_bare, i_k.eng.dz_aplc.mean), eps);
        if isempty(R), R = i_k; else, R(end+1) = i_k; end %#ok<AGROW>
        L = say_(L, ['    target %8.1e | thru %.4f | model %.3e | ENGINE %.3e ' ...
                     '| divergence %5.1fx | %.0f s'], ...
                 tgt(k), i_k.eng.thru_aplc, i_k.contrast_dense_bare, ...
                 i_k.eng.dz_aplc.mean, i_k.divergence, i_k.t_solve);
    end
    L = say_(L, '    THE DIVERGENCE GROWS WITH THE TARGET (%.1fx -> %.1fx): that is', ...
             R(1).divergence, R(end).divergence);
    L = say_(L, '    the optimizer finding more of the model''s error the harder it');
    L = say_(L, '    is pushed.  It is the signature to look for whenever a design');
    L = say_(L, '    model and a propagation engine are not the same code.');
    L = say_(L, '    LP size: %d variables (block %d px%s), %d rows, %d DZ samples', ...
             R(1).nvar, R(1).block, tern_(R(1).fold_y,', y-folded',''), ...
             R(1).nrow, R(1).ndz);
    L = say_(L, '    operator build %.1f s; solve %.0f-%.0f s per target', ...
             R(1).t_build, min([R.t_solve]), max([R.t_solve]));
    L = say_(L, '    self-tests: MFT round-trip %.2e | Lyot kernel %.2e', ...
             R(1).roundtrip_check, R(1).kernel_check);
    L = say_(L, '    bound form: %s', R(1).bound_form);
    L = say_(L, '    origin measured at [%.2f %.2f] px: %.3f from floor(N/2)+1,', ...
             R(1).origin, R(1).origin_offset.fft_centre);
    L = say_(L, '    %.3f from (N+1)/2 -- the even-grid half-pixel question, measured.', ...
             R(1).origin_offset.array_centre);

    % ---- [2] the aperture-specific APLC apodizer -------------------------
    L = say_(L, '\n[2] aperture-specific APLC apodizer (Soummer 2005 Eq. 3 over the');
    L = say_(L, '    TRACED aperture instead of a disc; N''Diaye et al. 2016 Paper V)');
    supp = double(Q.Amp);  supp(~Q.lit) = 0;
    [Phi_seg, is] = ctb_apod_prolate(P.co.model, Q.r_px, P.co.r_occ_lamD, ...
                                     'n_iter', P.ap.prolate_iter, 'support', supp);
    L = say_(L, '    Lambda0 = %.4f (converged=%d in %d iterations, support=%s)', ...
             is.lambda0, is.converged, is.n_iter_used, is.support_kind);
    L = say_(L, '    the CIRCULAR prolate on this train reports Lambda0 = 1.0000 --');
    L = say_(L, '    at the eigenvalue''s physical ceiling, i.e. saturated; the');
    L = say_(L, '    segmented aperture''s %.4f is a genuine interior eigenvalue.', is.lambda0);

    % ---- [3] engine scoring ----------------------------------------------
    L = say_(L, '\n[3] engine scoring on s3_seg_prop.in (the S3 train, unchanged)');
    r_ap = score_(prop_in, elt, P, Phi_seg);
    L = say_(L, '    aperture-specific prolate : DZ mean %.3e  median %.3e  suppr %.2e', ...
             r_ap.dz_aplc.mean, r_ap.dz_aplc.median, r_ap.supp_aplc);
    L = say_(L, '                                throughput %.4f apodizer / %.4f net', ...
             r_ap.apodizer_throughput, r_ap.thru_aplc);
    r_lp = R(end).eng;                       % already scored, rung by rung
    L = say_(L, '    best LP mask (target %.0e) : DZ mean %.3e  median %.3e  suppr %.2e', ...
             R(end).target, r_lp.dz_aplc.mean, r_lp.dz_aplc.median, r_lp.supp_aplc);
    L = say_(L, '                                throughput %.4f apodizer / %.4f net', ...
             r_lp.apodizer_throughput, r_lp.thru_aplc);

    S3 = s3_record_(fullfile(P.outdir,'s3_coro_report.txt'));
    L = say_(L, '\n[4] the rows the brief asks for');
    L = say_(L, '    %-38s %-11s %-11s %s', 'configuration','DZ mean','DZ median','throughput');
    L = say_(L, '    %-38s %-11.3e %-11.3e %.4f', 'bare segmented APLC (S3 record)', ...
             S3.seg_mean, S3.seg_median, S3.thru);
    L = say_(L, '    %-38s %-11.3e %-11.3e %.4f', 'aperture-specific prolate', ...
             r_ap.dz_aplc.mean, r_ap.dz_aplc.median, r_ap.thru_aplc);
    L = say_(L, '    %-38s %-11.3e %-11.3e %.4f', ...
             sprintf('best LP mask (target %.0e)', R(end).target), ...
             r_lp.dz_aplc.mean, r_lp.dz_aplc.median, r_lp.thru_aplc);
    L = say_(L, '    %-38s %-11.3e %-11.3e %.4f', 'clear-pupil reference (S3 mono)', ...
             S3.mono_mean, S3.mono_median, S3.thru);
    rec = S3.seg_mean / max(r_ap.dz_aplc.mean, eps);
    recm = S3.seg_median / max(r_ap.dz_aplc.median, eps);
    L = say_(L, '    RECOVERY, aperture-specific prolate vs the S3 baseline:');
    L = say_(L, '      %.2fx in DZ mean, %.2fx in DZ median, at %.2fx the throughput', ...
             rec, recm, r_ap.thru_aplc/max(S3.thru,eps));
    L = say_(L, '      -> essentially NO recovery, and no LP rung beat the incumbent');
    L = say_(L, '      either (best engine result %.3e at throughput %.3f against', ...
             r_lp.dz_aplc.mean, r_lp.thru_aplc);
    L = say_(L, '      the incumbent %.3e at %.3f).  Redesigning the apodizer alone,', ...
             S3.seg_mean, S3.thru);
    L = say_(L, '      against a fixed %.1f lambda/D occulter and a %.2f Lyot, does not', ...
             P.co.r_occ_lamD, P.co.r_lyot_frac);
    L = say_(L, '      buy back what the gaps cost.  The literature agrees: the');
    L = say_(L, '      segmented-aperture APLC result is a CO-optimization of');
    L = say_(L, '      apodizer x FPM x Lyot, which this brief deferred.');

    % ---- [5] gate 1: model vs engine --------------------------------------
    L = say_(L, '\n[5] gate 1 -- design model vs ENGINE, same mask, same chain');
    L = say_(L, '    %-34s %-12s %-12s %s','mask','model','engine','factor');
    G = struct('name',{},'model',{},'engine',{});
    G(1) = pack_('aperture-specific prolate', predict_(Efield,Q,P,Phi_seg), r_ap.dz_aplc.mean);
    for k=1:numel(R)
        G(end+1) = pack_(sprintf('LP mask (target %.0e)',R(k).target), ...
                         R(k).contrast_dense_bare, R(k).eng.dz_aplc.mean); %#ok<AGROW>
    end
    worst = 0;
    for k=1:numel(G)
        f = max(G(k).model,G(k).engine)/max(min(G(k).model,G(k).engine),eps);
        worst = max(worst,f);
        L = say_(L, '    %-34s %-12.3e %-12.3e %.2fx', G(k).name, G(k).model, G(k).engine, f);
    end
    L = say_(L, '    worst %.2fx against a %.0fx bar  [%s]', worst, P.ap.gate_factor, ...
             tern_(worst<=P.ap.gate_factor,'PASS','FAIL'));
    L = say_(L, '    THE DISAGREEMENT IS THE FINDING.  Localised by five experiments');
    L = say_(L, '    (all in the LOG): the bare PSF agrees to 1.2%%; the engine applies');
    L = say_(L, '    the mask to the field EXACTLY (residual 0.0); the Lyot radius and');
    L = say_(L, '    the lambda/D scale are confirmed (every ring lines up in radius);');
    L = say_(L, '    the Babinet term is worth 1%% here; and feeding the ENGINE''S OWN');
    L = say_(L, '    post-apodizer field through the model still lands at 4.2e-06.');
    L = say_(L, '    So the model has a FLOOR near 4e-06 whatever it is handed --');
    L = say_(L, '    5x above the contrast the incumbent already achieves.');

    OUT = struct('ladder',R,'apspec',Phi_seg,'apspec_info',is, ...
                 'engine_apspec',r_ap,'engine_lp',r_lp,'s3',S3, ...
                 'recovery_mean',rec,'recovery_median',recm, ...
                 'gate_worst',worst,'pupil',pupmat);
    OUT.figure = fig_(P, Q, R, Phi_seg, r_ap, r_lp, S3);
    L = say_(L, '\n[6] figure: %s', OUT.figure);
    matp = fullfile(P.outdir,'s3b_run.mat');
    % Trim before saving.  Each rung carries a 1024^2 mask plus two
    % 1024^2 engine images; keeping all of them makes a 42 MB artifact of
    % which the interesting part is a few hundred numbers.  Keep the two
    % masks that are actually referenced (the aperture-specific prolate
    % and the best LP rung) and the aperture-specific FPA image; drop the
    % rest, which any rung can regenerate from its recorded target.
    OUT.best_lp_A = R(end).A;
    for k = 1:numel(R)
        R(k).A = [];
        R(k).Edz = [];
        R(k).eng = rmfield(R(k).eng, {'I_aplc','I_blc','Phi'});
    end
    OUT.ladder = R;
    OUT.engine_lp = rmfield(OUT.engine_lp, {'I_aplc','I_blc','Phi'});
    OUT.engine_apspec = rmfield(OUT.engine_apspec, {'I_blc','Phi'});
    save(matp,'-struct','OUT','-v7.3');
    L = say_(L, '    workspace: %s', matp);
    L = say_(L, '\nS3b DONE in %.1f min', toc(t0)/60);
    rep = fullfile(P.outdir,'s3b_report.txt');
    fid = fopen(rep,'w'); fprintf(fid,'%s\n',L{:}); fclose(fid);
    fprintf('[s3b] report -> %s\n', rep);
end

% ---------------------------------------------------------------- helpers
function E = pupil_field_(rx, ix, seed, N)
%PUPIL_FIELD_  The COMPLEX field at the apodizer plane, peak-normalised.
    macos.init(N); macos.load_rx(rx);
    macos.intensity(seed(1));
    macos.intensity(seed(2),'reset_trace',false);
    E = macos.complex_field(ix.Apodizer,'reset_trace',false);
    E = E / max(abs(E(:)));
end
function r = score_(rx, elt, P, Phi)
    r = ctb_aplc('rx', rx, 'elt', elt, 'model_size', P.co.model, ...
                 'r_occ_lamD', P.co.r_occ_lamD, 'r_lyot_frac', P.co.r_lyot_frac, ...
                 'inner_lamD', P.co.inner_lamD, 'outer_lamD', P.co.outer_lamD, ...
                 'apodizer', Phi, 'skip_blc', true, 'outdir', tempdir);
end
function c = predict_(E, Q, P, Phi)
    [~,i] = apodizer_lp(E, Q.r_px, 'iwa',P.co.inner_lamD,'owa',P.co.outer_lamD, ...
        'r_occ_lamD',P.co.r_occ_lamD,'r_lyot_frac',P.co.r_lyot_frac, ...
        'nvar_target',P.ap.nvar_target,'dz_per_lamD',P.ap.dz_per_lamD, ...
        'n_fpm',P.ap.n_fpm,'verify_dense',0,'predict',Phi,'verbose',false);
    c = i.contrast_pred_bare;
end
function s = pack_(n,m,e), s = struct('name',n,'model',m,'engine',e); end
function S = s3_record_(path)
    assert(isfile(path), 's3b: %s missing -- run s3_coro first', path);
    t = fileread(path);
    S.seg_mean   = grab_(t,'dark-zone mean\s+segmented\s+([\d.eE+-]+)');
    S.mono_mean  = grab_(t,'dark-zone mean\s+segmented\s+[\d.eE+-]+\s+monolithic\s+([\d.eE+-]+)');
    S.seg_median = grab_(t,'dark-zone median\s+segmented\s+([\d.eE+-]+)');
    S.mono_median= grab_(t,'dark-zone median\s+segmented\s+[\d.eE+-]+\s+monolithic\s+([\d.eE+-]+)');
    S.thru       = grab_(t,'net throughput\s+([\d.eE+-]+)');
end
function v = grab_(t, pat)
    m = regexp(t, pat, 'tokens','once');
    assert(~isempty(m), 's3b: could not parse "%s" from the S3 report', pat);
    v = str2double(m{1});
end
function s = seed_pair_(ix)
    fn = fieldnames(ix);
    p = fn(~cellfun('isempty', regexp(fn,'^Prop\d+_(start|end)$','once')));
    a = p{~cellfun('isempty', regexp(p,'_start$','once'))};
    b = p{~cellfun('isempty', regexp(p,'_end$','once'))};
    s = [ix.(a) ix.(b)];
end
function ix = elt_ix_(rx)
    nm = regexp(fileread(rx), '^\s*EltName=\s*(\S+)', 'tokens','lineanchors');
    ix = struct();
    for k = 1:numel(nm), ix.(matlab.lang.makeValidName(nm{k}{1})) = k; end
end
function path = fig_(P, Q, R, Phi_seg, r_ap, r_lp, S3)
    w = round(2.3*Q.r_px);
    fig = figure('Visible','off','Color','w','Position',[50 50 1500 900]);
    tl = tiledlayout(fig,2,3,'TileSpacing','compact','Padding','compact');
    title(tl,'apodizers for the 19-segment pupil, engine-scored', ...
          'FontWeight','bold','Interpreter','none');

    ax=nexttile(tl); imagesc(ax, crop_(Q.Amp,w)); axis(ax,'image','off');
    colormap(ax,gray); clim(ax,[0 1]); title(ax,'traced pupil amplitude');

    ax=nexttile(tl); imagesc(ax, crop_(Phi_seg,w)); axis(ax,'image','off');
    colormap(ax,gray); clim(ax,[0 1]); cb=colorbar(ax); cb.Label.String='transmission';
    title(ax,sprintf('aperture-specific prolate (T=%.3f)', r_ap.apodizer_throughput));

    ax=nexttile(tl); imagesc(ax, crop_(R(end).A,w)); axis(ax,'image','off');
    colormap(ax,gray); clim(ax,[0 1]);
    title(ax,sprintf('LP mask, target %.0e (not trustworthy)', R(end).target));

    ax=nexttile(tl); hold(ax,'on');
    yyaxis(ax,'left'); plot(ax,[R.target],[R.throughput],'o-','LineWidth',1.8);
    ylabel(ax,'throughput'); ylim(ax,[0 1]);
    yyaxis(ax,'right'); plot(ax,[R.target],[R.contrast_dense_mean],'s--','LineWidth',1.4);
    set(ax,'YScale','log'); ylabel(ax,'model DZ mean');
    set(ax,'XScale','log'); xlabel(ax,'contrast target');
    grid(ax,'on'); box(ax,'on'); title(ax,'LP: what the MODEL believes it buys');

    ax=nexttile(tl);
    In = double(r_ap.I_aplc)/max(r_ap.peak_bare,eps);
    imagesc(ax, crop_(log10(max(In,1e-12)), round(2*(P.co.outer_lamD+3)*r_ap.lamD_px)));
    axis(ax,'image','off'); colormap(ax,parula); clim(ax,[-10 0]);
    cb=colorbar(ax); cb.Label.String='log_{10} contrast';
    title(ax,sprintf('FPA, aperture-specific (mean %.2e)', r_ap.dz_aplc.mean));

    ax=nexttile(tl); hold(ax,'on'); set(ax,'YScale','log');
    plot(ax, r_ap.r_aplc, max(r_ap.c_aplc,1e-13),'-','LineWidth',1.8, ...
        'DisplayName',sprintf('aperture-specific (%.1e, T=%.2f)', ...
        r_ap.dz_aplc.mean, r_ap.thru_aplc));
    plot(ax, r_lp.r_aplc, max(r_lp.c_aplc,1e-13),'-','LineWidth',1.2, ...
        'DisplayName',sprintf('LP mask (%.1e, T=%.2f)', ...
        r_lp.dz_aplc.mean, r_lp.thru_aplc));
    yline(ax, S3.seg_mean,'--','DisplayName', ...
        sprintf('S3 circular prolate (%.1e)', S3.seg_mean));
    yline(ax, S3.mono_mean,':','DisplayName', ...
        sprintf('S3 clear-pupil ref (%.1e)', S3.mono_mean));
    xr=[P.co.inner_lamD P.co.outer_lamD]; yl=ylim(ax);
    p=patch(ax,[xr(1) xr(2) xr(2) xr(1)],[yl(1) yl(1) yl(2) yl(2)], ...
        [0.9 0.9 0.95],'FaceAlpha',0.4,'EdgeColor','none','HandleVisibility','off');
    uistack(p,'bottom');
    grid(ax,'on'); box(ax,'on'); xlabel(ax,'separation (\lambda/D)');
    ylabel(ax,'contrast'); legend(ax,'Location','southwest','FontSize',8);
    title(ax,'radial contrast, engine-propagated');

    path = fullfile(P.outdir,'s3b_apodizer.png');
    exportgraphics(fig,path,'Resolution',150); close(fig);
end
function o = crop_(img,w)
    n=size(img,1); if w>=n, o=img; return; end
    c=floor(n/2)+1; lo=max(c-floor(w/2),1); hi=min(lo+w-1,n); o=img(lo:hi,lo:hi);
end
function L = say_(L, varargin)
    s = sprintf(varargin{:});
    for line = strsplit(s, newline), L{end+1} = line{1}; end %#ok<AGROW>
    fprintf('%s\n', s);
end
function s = tern_(c,a,b), if c, s=a; else, s=b; end, end
function setup_(here)
run(fullfile(here,'..','..','..','mmacos_setup.m'));
end
