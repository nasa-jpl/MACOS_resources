function OUT = r1_dm(over)
%R1_DM  e2e6m round 2: make the DMs REAL -- grid surfaces + poke gate.
%
%   Rewrites DM1/DM2 in the segmented diffraction deck as GridData
%   surfaces (`ctb_dm_rx` -- the frame is built from each DM element's
%   own frame, the rule that makes pokes localize), then GATES the
%   result the way the e5 corpus taught: an off-center influence-
%   function poke on DM1 must appear in the exit-pupil OPD map as a
%   LOCALIZED, OFF-CENTER bump -- the failure mode (wrong grid frame /
%   null frame) paints a central dot or a piston instead.
%
%   UNITS: this deck's BaseUnits is METRES, so every ctb_dm '_mm'
%   field carries metres here (the fields are named for the CTB deck's
%   mm; the machinery is unit-agnostic -- values are in the DECK's
%   base unit).  Poke amplitudes are surface metres.
%
%   The OPD is evaluated at the deck's ExitPupil (nElt-1) -- the R0
%   lesson: pair wavefront claims with the surface they live on.
%
%   Needs r1_coro to have produced r1_seg_prop.in.
%
%   OUT = R1_DM()      defaults
%   OUT = R1_DM(OVER)  with e2e6m_r2_params overrides
%
%   See also CTB_DM_RX, CTB_DM, R1_CORO, E2E6M_R2_PARAMS.

    arguments
        over struct = struct()
    end
    here = fileparts(mfilename('fullpath'));
    run(fullfile(here,'..','..','..','mmacos_setup.m'));
    P = e2e6m_r2_params(over);
    addpath(fullfile(here,'..','..','30_instruments','bench_ctb'));

    prop = fullfile(P.outdir, 'r1_seg_prop.in');
    assert(isfile(prop), 'r1_dm: %s not found -- run r1_coro first', prop);

    L = {};  t0 = tic;
    L = say_(L, '==================== e2e6m R1 -- DM grid surfaces + poke gate');

    % ---- [1] the augmented deck -----------------------------------------
    dmrx = fullfile(P.outdir, 'r1_seg_dm.in');
    A = ctb_dm_rx('rx_in', prop, 'rx_out', dmrx, ...
                  'dms', P.dm.names, 'ng', P.dm.ng);
    L = say_(L, '\n[1] %s: DM elts %s, ng %d, grid dx %s m', ...
             dmrx, mat2str(A.ielt), A.ng, mat2str(A.gdx_mm, 3));

    % ---- [2] reload gate (rule 3) ---------------------------------------
    macos.init(P.co.model);
    n0 = macos.load_rx(prop);   s0 = macos.trace(n0 - 1);
    n1 = macos.load_rx(dmrx);   s1 = macos.trace(n1 - 1);
    L = say_(L, '\n[2] reload: %d -> %d elements, rays %d -> %d  [%s]', ...
             n0, n1, s0.nRays, s1.nRays, ...
             gate_(n1 == n0 && s1.nRays == s0.nRays));

    % ---- [3] the poke gate ----------------------------------------------
    beam_d = 2 * 0.023771;              % measured pupil at the DMs (r1 gate)
    dm = ctb_dm('ielt', A.ielt(1), 'ng', A.ng, 'gdx_mm', A.gdx_mm(1), ...
                'nact', 32, 'beam_d_mm', beam_d, 'pitch_mm', beam_d/32);
    L = say_(L, '\n[3] DM1 lattice: %d x %d, %d active in the %.1f mm beam', ...
             dm.nact, dm.nact, dm.nact_active, beam_d*1e3);

    % nominal exit-pupil OPD
    W0 = macos.opd();
    m0 = fin_(W0);
    % pupil geometry in pixels, from the nominal mask
    [pi_, pj_] = find(m0);
    c0 = [mean(pi_), mean(pj_)];
    Rpup = sqrt(nnz(m0)/pi);

    % one actuator, HALF a pupil radius off center, 20 nm of surface
    tgt = 0.5 * beam_d/2;
    dd  = hypot(dm.acx - tgt, dm.acy - 0);
    dd(~dm.active) = Inf;
    [~, ia] = min(dd);
    a = zeros(dm.nact^2, 1);  a(ia) = 20e-9;
    dm.apply(a);
    macos.trace(n1 - 1);
    W1 = macos.opd();
    dm.clear();
    b  = m0 & fin_(W1);
    dW = zeros(size(W0));
    dW(b) = (W1(b) - mean(W1(b))) - (W0(b) - mean(W0(b)));

    [~, ip] = max(abs(dW(:)));
    [pi1, pj1] = ind2sub(size(dW), ip);
    off = hypot(pi1 - c0(1), pj1 - c0(2));
    [gi, gj] = ndgrid(1:size(dW,1), 1:size(dW,2));
    near = hypot(gi - pi1, gj - pj1) <= 0.15 * Rpup;
    efrac = sum(dW(near).^2) / max(sum(dW(:).^2), realmin);
    pk = max(abs(dW(:)));

    L = say_(L, '    poke: actuator at (%.1f, %.1f) mm, 20 nm surface', ...
             dm.acx(ia)*1e3, dm.acy(ia)*1e3);
    L = say_(L, '    response: peak |dOPD| %.3g m (2x surface = %.3g expected)', ...
             pk, 2*20e-9);
    L = say_(L, '    peak offset from pupil center %.2f Rpup  [%s]  (fails = central dot)', ...
             off/Rpup, gate_(off > 0.2*Rpup));
    L = say_(L, '    energy within 0.15 Rpup of the peak: %.0f%%  [%s]  (fails = piston/wash)', ...
             100*efrac, gate_(efrac > 0.5));

    pass = (off > 0.2*Rpup) && (efrac > 0.5) && (pk > 1e-8);
    L = say_(L, '\n    DM poke gate: [%s]', gate_(pass));

    png = fullfile(P.outdir, 'r1_dm_poke.png');
    fig_(dW, m0, png);
    L = say_(L, '    figure: %s', png);

    L = say_(L, '\nR1 dm DONE in %.1f min', toc(t0)/60);
    txt = strjoin(L, newline);
    fid = fopen(fullfile(P.outdir,'r1_dm_report.txt'),'w');
    fprintf(fid,'%s\n',txt);  fclose(fid);
    OUT = struct('P',P, 'aug',A, 'pass',pass, 'peak',pk, ...
                 'off_rpup',off/Rpup, 'efrac',efrac, 'figure',png, ...
                 'text',txt, 'when',datestr(now,31)); %#ok<TNOW1,DATST>
    save(fullfile(P.outdir,'r1_dm_run.mat'),'OUT');
end

% =========================================================================
function fig_(dW, m0, png)
    f = figure('Visible','off','Color','w','Position',[80 80 640 560]);
    ax = axes(f);
    D = dW;  D(~m0) = NaN;
    imagesc(ax, D.'*1e9);  axis(ax,'image');  set(ax,'YDir','normal');
    cb = colorbar(ax);  cb.Label.String = '\DeltaOPD  [nm]';
    title(ax, 'DM1 single-actuator poke at the exit pupil (20 nm surface)');
    xlabel(ax,'pupil x  [px]');  ylabel(ax,'pupil y  [px]');
    exportgraphics(f, png, 'Resolution', 150);
    close(f);
end

function m = fin_(W)
    m = isfinite(W) & W ~= 0 & abs(W) < 1e30;
end
function L = say_(L, varargin)
    s = sprintf(varargin{:});  L{end+1} = s;  fprintf('%s\n', s);
end
function s = gate_(ok), if ok, s = 'PASS'; else, s = 'FAIL'; end, end
