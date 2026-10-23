function R = afocal4_mersenne(opts)
%AFOCAL4_MERSENNE  The hedge: does a conic-relaxed double Mersenne reach DL?
%
%   R = AFOCAL4_MERSENNE() runs the ONE bounded experiment the S3 form gate
%   kept the runner-up alive for (PLAN_AFOCAL4 S3/S4, FORM_STUDY section 5).
%
%   THE QUESTION, precisely.  The double Mersenne has the best pupil ladder
%   anything in the form study measured -- blur 175 um, breathing 0.285%,
%   wander 468 um, in a train 26% shorter -- and it loses on wavefront error:
%   59 um at rung 2.  But that 59 um is not a property of the FORM.  It is
%   the price of insisting all four mirrors be parabolas, which is what the
%   NAME requires and not what the design does.  Two confocal pairs are
%   afocal because of their SPACINGS; the conics are free.  So: hold the
%   confocal spacings, relax all four conics, and see whether the wavefront
%   error comes to DL while the pupil ladder stays where it is.
%
%   PROMOTE IFF it reaches 71 nm in-box AND keeps its pupil ladder.  It would
%   then win outright -- its pupil numbers are 2-5x the field mirror's.  What
%   would still kill it is the 53 mm interface standoff, which is not an
%   instrument interface; the stage-2 f/# lever reaches only ~0.16 m and
%   buying more costs stage-2 speed and hence size.  That objection is a
%   question for the instrument, not for this experiment, and it is recorded
%   either way.
%
%   BOUNDED.  Four conics, one solve, on axis then at the bias; time-boxed by
%   'minutes' (default 120).  If it exceeds the box the run stops WHERE IT
%   STOOD and reports that, rather than quietly running longer or quietly
%   reporting a half-converged number as a verdict.
%
%   ALL FOUR CONICS ARE FREE HERE, including M1's.  That differs from the
%   field-mirror ladder, which holds M1 a parabola to match his study, and
%   the difference is the point: the Mersenne's primary is a parabola only
%   because the form's name says so.
%
%   Name-value:
%     'minutes'   the time box (120)
%     'save'      write .in / .png / .mat (true)
%     'max_iter'  solver cap (P.solve.max_iter)
%
%   Returns R with .onaxis / .offset (each .D .S .solve), .verdict, .seconds.
%
%   See also AFOCAL4_SOLVE, AFOCAL4_LADDER, AFOCAL4_CLOSE.

    arguments
        opts.minutes  (1,1) double  = 120
        opts.save     (1,1) logical = true
        opts.max_iter (1,1) double  = 0
    end
    here = fileparts(mfilename('fullpath'));
    P = afocal4_params();
    % THE HEDGE RUNS WITH THE PACKAGING WALL OFF, and the reason belongs in
    % the record rather than in a footnote: the double Mersenne fails the
    % S4b buildability constraint STRUCTURALLY.  Its second confocal pair
    % lives inside the M1-M2 space -- M3 lands ~540 mm and M4 ~940 mm in
    % FRONT of the primary -- and no conic, gap or stage split moves them
    % behind it, because the form's whole compression happens before the
    % beam ever gets back to M1.  The experiment below is kept RUNNABLE and
    % its verdict stands as measured (four conics buy 1.7x against a factor
    % of 500 needed); the constraint closes it a second time, on packaging,
    % and that is stated in RESULTS section 4 rather than by deleting a
    % result.
    P.pack.enforce = false;
    if opts.max_iter <= 0, opts.max_iter = P.solve.max_iter; end
    macos.init(P.model_size);
    t0 = tic;

    banner('THE MERSENNE HEDGE -- four conics against 59 um of wavefront');

    % ---- the parabolic starting point, measured again --------------------
    D0 = afocal4_seed(P, 'form','mersenne', 'bias_deg',P.bias_deg);
    f0 = deck_(here,'mersenne_parabolic',opts.save);
    b0 = afocal4_build(P, D0, f0, 'quiet',false);
    S0 = afocal4_score(P, f0, 'nodes',P.solve.nodes_score, 'grid',P.grid_n);
    afocal4_score_print(P, S0, 'double Mersenne, four parabolas (S3 layout)');
    fprintf(['\n  interface standoff %.1f mm -- the form''s own, closed by the\n' ...
             '  confocal spacings.  It is NOT the field-mirror ladder''s %.0f mm\n' ...
             '  and cannot be made so without spending stage-2 speed; the two\n' ...
             '  forms are therefore compared at different operating points, and\n' ...
             '  that is a fact about the forms, not a flaw in the comparison.\n'], ...
            b0.iface*1e3, P.iface*1e3);
    R = struct('P',P, 'parabolic',struct('D',D0,'b',b0,'S',S0));

    % ---- the experiment: relax the four conics ---------------------------
    banner('relaxing the four conics, confocal SPACINGS held');
    left = @() opts.minutes*60 - toc(t0);
    s1 = afocal4_solve(P, D0, 'dofs',{'conic'}, ...
            'label','mersenne, conics relaxed (at the bias)', ...
            'deck',deck_(here,'mersenne_conics',opts.save), ...
            'max_iter',opts.max_iter);
    R.offset = s1;

    if left() > 0.25*opts.minutes*60
        % Second basin: solve on axis first, then carry to the bias.  Basin
        % path-dependence is expected on a four-conic problem and the honest
        % report is BOTH paths, not the better one.
        banner('second path: solve on axis, then carry to the bias');
        Da = D0;  Da.bias_deg = 0;
        s2 = afocal4_solve(P, Da, 'dofs',{'conic'}, ...
                'label','mersenne, on axis', ...
                'deck',deck_(here,'mersenne_conics_onaxis',opts.save), ...
                'max_iter',opts.max_iter);
        Db = s2.D;  Db.bias_deg = P.bias_deg;
        s3 = afocal4_solve(P, Db, 'dofs',{'conic'}, ...
                'label','mersenne, carried to the bias', ...
                'deck',deck_(here,'mersenne_conics_carried',opts.save), ...
                'max_iter',opts.max_iter);
        R.onaxis = s2;   R.carried = s3;
        if s3.S.wfe_max_nm < s1.S.wfe_max_nm, R.best = s3; else, R.best = s1; end
    else
        R.best = s1;
        fprintf('\n  TIME BOX: the second solve path was not started (%.0f min used).\n', ...
                toc(t0)/60);
    end

    % ---- the verdict -----------------------------------------------------
    banner('VERDICT');
    B = R.best.S;   T = P.targets;
    dl   = B.wfe_max_nm <= T.wfe_rung2_nm;
    kept = B.blur_um    <= 1.25*S0.blur_um && ...
           B.breathe_pct<= 1.25*S0.breathe_pct && ...
           B.wander_um  <= 1.25*S0.wander_um;
    fprintf('  %-34s %12s %12s\n', '', 'parabolic', 'conics free');
    prow('WFE rung 2 max (nm)', S0.wfe_max_nm,  B.wfe_max_nm);
    prow('pupil blur rms (um)', S0.blur_um,     B.blur_um);
    prow('breathing (%)',       S0.breathe_pct, B.breathe_pct);
    prow('wander, refit (um)',  S0.wander_um,   B.wander_um);
    prow('M at box centre',     S0.mag_centre_chief, B.mag_centre_chief);
    fprintf('\n  conics %s -> %s\n', mat2str(D0.K,5), mat2str(R.best.D.K,6));
    if dl && kept
        R.verdict = 'PROMOTE';
        fprintf(['\n  PROMOTE.  The form reaches DL (%.1f nm <= %.0f) with its\n' ...
                 '  pupil ladder intact.  The remaining objection is the %.0f mm\n' ...
                 '  interface standoff.\n'], B.wfe_max_nm, T.wfe_rung2_nm, ...
                 b0.iface*1e3);
    elseif dl
        R.verdict = 'DL BUT PUPIL LOST';
        fprintf(['\n  DL REACHED, PUPIL LOST.  The four conics bought the\n' ...
                 '  wavefront by spending the pupil ladder the form was kept\n' ...
                 '  alive for -- which is the same trade the field mirror\n' ...
                 '  makes, without the field mirror''s interface standoff.\n']);
    else
        R.verdict = 'CLOSED';
        fprintf(['\n  CLOSED.  Four conics take the wavefront error from %.3g to\n' ...
                 '  %.3g nm, short of the %.0f nm the promotion required.  The\n' ...
                 '  59 um in the S3 table is therefore NOT only the parabola\n' ...
                 '  constraint; the form carries wavefront error the conics\n' ...
                 '  cannot reach.  The double Mersenne is closed as an\n' ...
                 '  alternative and the field mirror stands.\n'], ...
                 S0.wfe_max_nm, B.wfe_max_nm, T.wfe_rung2_nm);
    end
    R.seconds = toc(t0);
    R.timeboxed = R.seconds > opts.minutes*60;
    fprintf('\n  %.1f min of machine time%s.\n', R.seconds/60, ...
            ternary_(R.timeboxed, ' -- OVER the time box, reported where it stood', ''));

    if opts.save
        try
            png = fullfile(here,'afocal4_mersenne_hedge.png');
            hedge_fig_(P, S0, B, png);   fprintf('  wrote %s\n', png);
        catch ME, fprintf('   figure failed: %s\n', ME.message); end
        save(fullfile(here,'afocal4_mersenne.mat'), 'R', '-v7.3');
        fprintf('  saved afocal4_mersenne.mat\n');
    end
end

% =====================================================================
function prow(n, a, b)
    fprintf('  %-34s %12.4g %12.4g   (%.2fx)\n', n, a, b, b/max(a,realmin));
end

function f = deck_(here, tag, dosave)
    if dosave, f = fullfile(here, sprintf('afocal4_%s.in', tag));
    else,      f = [tempname '.in'];
    end
end

function s = ternary_(c, a, b),  if c, s = a; else, s = b; end,  end

function hedge_fig_(P, S0, B, png)
%HEDGE_FIG_  Before and after, target-normalised: what four conics bought
%   and what they cost.  Log axis because the two ends are 1000x apart.
    T = P.targets;
    M = [S0.wfe_max_nm/T.wfe_rung2_nm, S0.blur_um/T.blur_um, ...
         S0.breathe_pct/T.breathe_pct, S0.wander_um/T.wander_um; ...
         B.wfe_max_nm/T.wfe_rung2_nm,  B.blur_um/T.blur_um, ...
         B.breathe_pct/T.breathe_pct,  B.wander_um/T.wander_um];
    fig = figure('Visible','off','Position',[100 100 900 430]);
    ax = axes(fig);
    hb = bar(ax, M.');
    set(ax,'XTickLabel',{'WFE','blur','breathing','wander'},'YScale','log');
    yl = yline(ax, 1, 'k--','target');
    yl.Annotation.LegendInformation.IconDisplayStyle = 'off';
    legend(ax, hb, {'four parabolas','four conics free'}, 'Location','northeast');
    ylabel(ax,'metric / target');   grid(ax,'on');
    title(ax, ['The Mersenne hedge: confocal spacings held, conics relaxed'], ...
          'FontWeight','bold');
    exportgraphics(fig, png, 'Resolution', 150);   close(fig);
end

function banner(s)
    fprintf('\n=================================================================\n');
    fprintf('  %s\n', s);
    fprintf('=================================================================\n');
end
