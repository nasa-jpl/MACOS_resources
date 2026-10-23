function OUT = cf0_gates(over)
%CF0_GATES  Coronagraph-family campaign S0: the runner, gated against R1.
%
%   Three gates before any family lands (BRIEF_e2e6m_coro_families S0):
%
%   [1] BARE-PSF PIN.  Through cf_chain's bare pass, on both primaries:
%       PSF peak on the FFT DC pixel (floor(N/2)+1), bare on-axis peak and
%       FPA lambda/D equal to the committed R1 values (r1_coro_run.mat),
%       and |E|^2 consistent with the intensity read of the same plane.
%
%   [2] PUPIL SANITY.  The traced apodizer-plane pupil: radius in px, and
%       the support fill ratio against the geometric disc -- the SEGMENTED
%       pupil must show its gaps (fill < 1), the monolithic must not.
%
%   [3] R1 REPRODUCED THROUGH THE NEW RUNNER.  cf_chain configured as the
%       R1 APLC (clear-disc prolate, 2.8 lambda/D occulter, Lyot 0.90)
%       must reproduce the committed R1 dark-zone numbers on BOTH decks --
%       the bit-consistency gate: same walk, same masks, same grid, so the
%       difference should be exactly zero.
%
%   Model 1024 (the R1 scoring grid) -- one process, one model size.
%
%   OUT = CF0_GATES()      defaults
%   OUT = CF0_GATES(OVER)  with e2e6m_r2_params overrides
%
%   See also CF_CHAIN, R1_CORO, ctb_aplc.

    arguments
        over struct = struct()
    end
    here = fileparts(mfilename('fullpath'));
    run(fullfile(here,'..','..','..','mmacos_setup.m'));
    P = e2e6m_r2_params(over);
    addpath(fullfile(here,'..','..','30_instruments','bench_ctb'));

    R1 = load(fullfile(P.outdir,'r1_coro_run.mat'));
    L = {};  t0 = tic;  npass = 0;  nfail = 0;
    L = say_(L, '==================== e2e6m CF0 -- chain runner gates vs R1');
    L = say_(L, 'model %d, annulus %g-%g lambda/D, occulter %g lambda/D, Lyot %g', ...
             P.co.model, P.co.inner_lamD, P.co.outer_lamD, ...
             P.co.r_occ_lamD, P.co.r_lyot_frac);

    V = struct('tag',{'seg','mono'}, 'res',{[],[]});
    for v = 1:numel(V)
        rx = fullfile(P.outdir, sprintf('r1_%s_prop.in', V(v).tag));
        r1 = R1.OUT.V(strcmp({R1.OUT.V.tag}, V(v).tag)).res;
        L = say_(L, '\n---- %s (%s) ----', V(v).tag, rx);

        ch = cf_chain('rx', rx, 'model_size', P.co.model, ...
                      'apod_kind', 'prolate', 'prolate_iter', P.co.prolate_iter, ...
                      'fpm_kind', 'hard', 'r_fpm_lamD', P.co.r_occ_lamD, ...
                      'lyot', true, 'r_lyot_frac', P.co.r_lyot_frac);
        L = say_(L, '    config tag %s | lambda/D %.6f px | peak_bare %.6e', ...
                 ch.tag, ch.lamD_px, ch.peak_bare);

        % [1] bare-PSF pin ------------------------------------------------
        Eb = ch.run_bare();
        Ib = abs(Eb).^2;
        [~, ip] = max(Ib(:));
        [pi_, pj_] = ind2sub(size(Ib), ip);
        ok = pi_ == ch.center_px && pj_ == ch.center_px;
        [npass, nfail] = tally_(npass, nfail, ok);
        L = say_(L, '[1] PSF peak pixel (%d,%d), DC = %d  [%s]', ...
                 pi_, pj_, ch.center_px, gate_(ok));
        Ifpa = macos.intensity(ch.elt.FPA, 'reset_trace', false);
        d_ie = max(abs(Ifpa(:) - Ib(:))) / max(Ib(:));
        ok = d_ie < 1e-12;
        [npass, nfail] = tally_(npass, nfail, ok);
        L = say_(L, '    |E|^2 vs intensity read, max rel %.3g  [%s]', d_ie, gate_(ok));
        dpk = reldiff_(ch.peak_bare, r1.peak_bare);
        dld = reldiff_(ch.lamD_px,   r1.lamD_px);
        ok = dpk < 1e-12 && dld < 1e-12;
        [npass, nfail] = tally_(npass, nfail, ok);
        L = say_(L, '    peak_bare vs R1 rel %.3g | lamD_px vs R1 rel %.3g  [%s]', ...
                 dpk, dld, gate_(ok));

        % [2] pupil sanity ------------------------------------------------
        c = ch.center_px;
        [X, Y] = meshgrid((1:ch.N) - c, (1:ch.N) - c);
        disc = hypot(X, Y) <= ch.r_apod_px;
        fill = nnz(ch.support & disc) / max(nnz(disc), 1);
        % measured pins: the seg support fills ~0.79 of the enclosing disc
        % (25 mm gaps + the hex-tiling corners the disc includes); the
        % monolithic fills ~0.99 (rim pixels under the 2% threshold)
        if strcmp(V(v).tag, 'seg'), ok = fill > 0.70 && fill < 0.90;
        else,                        ok = fill > 0.97; end
        [npass, nfail] = tally_(npass, nfail, ok);
        L = say_(L, '[2] pupil r %.1f px, support fill %.4f (%s)  [%s]', ...
                 ch.r_apod_px, fill, ...
                 tern_(strcmp(V(v).tag,'seg'), 'gaps must show', 'must be full'), ...
                 gate_(ok));

        % [3] R1 reproduced through the runner ----------------------------
        E  = ch.run();
        I  = abs(E).^2;
        dz = macos.dark_zone_metrics(I, ch.peak_bare, ch.lamD_px, ...
                                     P.co.inner_lamD, P.co.outer_lamD);
        dm = reldiff_(dz.mean,   r1.dz_aplc.mean);
        dd = reldiff_(dz.median, r1.dz_aplc.median);
        dt = reldiff_(ch.thru_apod, r1.apodizer_throughput);
        ok = dm < 1e-12 && dd < 1e-12 && dt < 1e-12;
        [npass, nfail] = tally_(npass, nfail, ok);
        L = say_(L, '[3] APLC through cf_chain: DZ mean %.6e (R1 %.6e, rel %.3g)', ...
                 dz.mean, r1.dz_aplc.mean, dm);
        L = say_(L, '    DZ median %.6e (R1 %.6e, rel %.3g) | apod thru %.6f (rel %.3g)  [%s]', ...
                 dz.median, r1.dz_aplc.median, dd, ch.thru_apod, dt, gate_(ok));

        V(v).res = struct('tag', V(v).tag, 'chain_tag', ch.tag, ...
            'config', {ch.config}, 'peak_bare', ch.peak_bare, ...
            'lamD_px', ch.lamD_px, 'r_apod_px', ch.r_apod_px, ...
            'fill', fill, 'dz', dz, 'rel_mean', dm, 'rel_median', dd, ...
            'thru', ch.thru, 'prolate_info', ch.prolate_info);
    end

    L = say_(L, '\nCF0 gates: %d PASS, %d FAIL in %.1f min', npass, nfail, toc(t0)/60);
    txt = strjoin(L, newline);
    fid = fopen(fullfile(P.outdir,'cf0_report.txt'),'w');
    fprintf(fid,'%s\n',txt);  fclose(fid);
    OUT = struct('P',P, 'V',V, 'npass',npass, 'nfail',nfail, 'text',txt, ...
                 'when',datestr(now,31)); %#ok<TNOW1,DATST>
    save(fullfile(P.outdir,'cf0_run.mat'),'OUT');
end

% =========================================================================
function [p, f] = tally_(p, f, ok)
    if ok, p = p + 1; else, f = f + 1; end
end
function d = reldiff_(a, b)
    d = abs(a - b) / max(abs(b), realmin);
end
function L = say_(L, varargin)
    s = sprintf(varargin{:});  L{end+1} = s;  fprintf('%s\n', s);
end
function s = gate_(ok), if ok, s = 'PASS'; else, s = 'FAIL'; end, end
function s = tern_(c,a,b), if c, s = a; else, s = b; end, end
