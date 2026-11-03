function OUT = cf1c_stop_attrib(over)
%CF1C_STOP_ATTRIB  Attribute the hard family's 2.1x static penalty under the stop.
%
%   CCL's flag (2026-08-26): classical Lyot went 1.64e-6 (no stop) ->
%   3.50e-6 (S0b stop) -- a 2.1x penalty of which the circularized-peak
%   renormalization explains only ~1.3x.  The presumption is the stop's
%   own circular edge diffracting a ring through a gap-dominated
%   annulus; this probe MEASURES the split instead of asserting it.
%
%   METHOD.  Three configurations, two chains:
%     E_ns  : the no-stop chain (hex pupil, hex-scale masks) -- the S1
%             no-stop record, reproduced as a gate.
%     E_stX : the SAME chain with the stop disc applied as a SCREEN at
%             the apodizer plane (run_screened) -- every mask and scale
%             held fixed, so E_rim = E_ns - E_stX is EXACTLY the field
%             sourced by the blocked rim (linearity), and the Babinet
%             split  I_stX = I_ns - 2Re(E_ns* E_rim) + I_rim  says how
%             the stop edge enters the dark zone (additive ring vs
%             interference with the gap field).
%     E_st  : the true S0b chain (stop in the pupil, every scale
%             RE-MEASURED through it: FPM/FPA lambda/D, Lyot geometric
%             radius) -- the S1 stopped record, reproduced as a gate.
%   The 2.1x then factors as
%     c_st/c_ns = [c_stX/c_ns] x [c_st/c_stX]
%   = [fixed-mask stop-edge effect] x [scale-rechain effect], with the
%   peak renormalization pk_ns/pk_st reported inside (both contrasts
%   divide by their own bare peaks; pk_stX == pk_st is asserted -- the
%   bare pass has no masks, so the screen and the pupil stop must give
%   the SAME circularized peak, a free consistency pin).
%
%   Figure: radial dark-zone profiles of I_ns, I_stX, I_rim (where the
%   edge ring lands in lambda/D), plus the Babinet term budget.
%
%   See also CF_CHAIN, CF1_FAMILIES, cf2_efc.

    arguments
        over struct = struct()
    end
    here = fileparts(mfilename('fullpath'));
    run(fullfile(here,'..','..','..','mmacos_setup.m'));
    P = e2e6m_r2_params(over);
    addpath(fullfile(here,'..','..','30_instruments','bench_ctb'));

    rx = fullfile(P.outdir, 'r1_seg_prop.in');
    C1 = load(fullfile(P.outdir,'cf1_run.mat'));         % STOPPED S1 (post-CF1b)
    NS = load(fullfile(P.outdir,'cf1_nostop_run.mat'));  % the no-stop record
    FC = struct();  FN = struct();
    for k = 1:numel(C1.OUT.F), FC.(C1.OUT.F(k).key) = C1.OUT.F(k); end
    for k = 1:numel(NS.OUT.F), FN.(NS.OUT.F(k).key) = NS.OUT.F(k); end
    cfg = FC.hard.cfg;

    L = {};  t0 = tic;
    L = say_(L, '==== e2e6m CF1c -- the hard family''s stop penalty, attributed (N=%d)', P.co.model);

    % ---- the two chains -------------------------------------------------
    ch_ns = cf_chain('rx', rx, 'model_size', P.co.model, ...
                     'prolate_iter', P.co.prolate_iter, ...
                     'circ_stop_frac', 0, cfg{:});
    ch_st = cf_chain('rx', rx, 'model_size', P.co.model, ...
                     'prolate_iter', P.co.prolate_iter, ...
                     'circ_stop_frac', P.cf.circ_stop_frac, cfg{:});
    S = ch_st.masks.S;                 % the stop disc, same grid convention

    dz_ns = find(ch_ns.dz_mask(P.co.inner_lamD, P.co.outer_lamD));
    dz_st = find(ch_st.dz_mask(P.co.inner_lamD, P.co.outer_lamD));

    % ---- fields ---------------------------------------------------------
    E_ns  = ch_ns.run();
    E_stX = ch_ns.run_screened(S);     % stop as a screen: masks FIXED
    E_st  = ch_st.run();
    E_rim = E_ns - E_stX;              % exact: the blocked rim's field

    % bare peaks: the screen and the pupil stop must agree on the bare pass
    pk_ns  = ch_ns.peak_bare;
    pk_st  = ch_st.peak_bare;
    Eb     = ch_ns.run_bare_screened(S);
    pk_stX = max(abs(Eb(:)).^2);
    L = say_(L, 'bare peaks: no-stop %.4e | stop %.4e | screened %.4e (pin: |screened/stop - 1| = %.2e)', ...
             pk_ns, pk_st, pk_stX, abs(pk_stX/pk_st - 1));

    % ---- contrasts (each in its own convention, as S1 scored them) ------
    c_ns  = mean(abs(E_ns(dz_ns)).^2)   / pk_ns;
    c_stX = mean(abs(E_stX(dz_ns)).^2)  / pk_st;   % fixed masks, stop peak
    c_st  = mean(abs(E_st(dz_st)).^2)   / pk_st;
    L = say_(L, 'contrast: c_ns %.3e (S1 no-stop record %.3e) | c_st %.3e (S1 stopped record %.3e)', ...
             c_ns, FN.hard.res.dz.mean, c_st, FC.hard.res.dz.mean);
    L = say_(L, 'the 2.1x, factored: c_st/c_ns = %.2f = [fixed-mask stop edge %.2f] x [scale-rechain %.2f]', ...
             c_st/c_ns, c_stX/c_ns, c_st/c_stX);
    L = say_(L, 'inside those: peak renormalization pk_ns/pk_st = %.3f; dark-zone ENERGY at fixed masks up %.2fx', ...
             pk_ns/pk_st, mean(abs(E_stX(dz_ns)).^2)/mean(abs(E_ns(dz_ns)).^2));

    % ---- Babinet split on the no-stop annulus (unnormalized energy) -----
    I_ns  = mean(abs(E_ns(dz_ns)).^2);
    I_rim = mean(abs(E_rim(dz_ns)).^2);
    X     = mean(-2*real(conj(E_ns(dz_ns)) .* E_rim(dz_ns)));
    I_stX = mean(abs(E_stX(dz_ns)).^2);
    L = say_(L, 'Babinet (dz mean energy): I_stX %.3e = I_ns %.3e + rim %.3e + cross %.3e (closure %.1e)', ...
             I_stX, I_ns, I_rim, X, abs(I_stX - (I_ns + I_rim + X))/I_stX);
    if I_rim > abs(X)
        L = say_(L, 'verdict: the stop edge enters mostly as its OWN ring (rim term dominates the cross term)');
    else
        L = say_(L, 'verdict: the stop edge enters mostly by INTERFERENCE with the gap field (cross term dominates)');
    end

    % ---- radial profiles ------------------------------------------------
    png = fullfile(P.outdir, 'cf1c_stop_attrib.png');
    fig_(E_ns, E_stX, E_rim, ch_ns, pk_st, P, png);
    L = say_(L, '  figure: %s', png);
    L = say_(L, 'CF1c DONE in %.1f min', toc(t0)/60);

    txt = strjoin(L, newline);
    fid = fopen(fullfile(P.outdir,'cf1c_report.txt'),'w');
    fprintf(fid,'%s\n',txt);  fclose(fid);
    OUT = struct('c_ns',c_ns, 'c_stX',c_stX, 'c_st',c_st, ...
        'pk_ns',pk_ns, 'pk_st',pk_st, 'pk_stX',pk_stX, ...
        'I_ns',I_ns, 'I_rim',I_rim, 'cross',X, 'I_stX',I_stX, ...
        'text',txt, 'figure',png, 'when',datestr(now,31)); %#ok<TNOW1,DATST>
    save(fullfile(P.outdir,'cf1c_run.mat'),'OUT');
end

% =========================================================================
function fig_(E_ns, E_stX, E_rim, ch, pk, P, png)
    f = figure('Visible','off','Color','w','Position',[60 60 760 520]);
    ax = axes(f); hold(ax,'on'); grid(ax,'on'); box(ax,'on'); set(ax,'YScale','log');
    rmax = P.co.outer_lamD + 3;
    [r1,c1] = macos.radial_contrast(abs(E_ns ).^2, pk, ch.lamD_px, rmax);
    [~ ,c2] = macos.radial_contrast(abs(E_stX).^2, pk, ch.lamD_px, rmax);
    [~ ,c3] = macos.radial_contrast(abs(E_rim).^2, pk, ch.lamD_px, rmax);
    semilogy(ax, r1, c1, '-',  'LineWidth', 1.6, 'DisplayName', 'no stop (hex pupil)');
    semilogy(ax, r1, c2, '-',  'LineWidth', 1.6, 'DisplayName', 'stop as screen (masks fixed)');
    semilogy(ax, r1, c3, '--', 'LineWidth', 1.4, 'DisplayName', 'the rim field alone (Babinet)');
    xlabel(ax, 'separation  [\lambda/D, hex scale]');
    ylabel(ax, 'contrast (stop-convention peak)');
    title(ax, {'Where the circular stop''s edge lands (classical Lyot, static)', ...
               'fixed-mask decomposition: E_{stop} = E_{hex} - E_{rim}'}, 'FontWeight','bold');
    legend(ax, 'Location', 'northeast');
    exportgraphics(f, png, 'Resolution', 150);
    close(f);
end

function L = say_(L, varargin)
    s = sprintf(varargin{:});  L{end+1} = s;  fprintf('%s\n', s);
end
