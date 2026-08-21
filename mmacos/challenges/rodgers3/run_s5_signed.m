function OUT = run_s5_signed()
%RUN_S5_SIGNED  Leg C of the S5 budget probe: the honest re-solve.
%
%   The S5 of record AND budget legs A/B solved their clearance hinge
%   rows against the pre-fix oi_clear (25-sample minimum), which read
%   the binding M3->FP x M2 pair ~5 mm optimistic near the 35 mm knee --
%   every S5-class solve therefore settled at a TRUE floor of ~29-30 mm
%   (re-gated post-hoc under the signed model).  This leg re-solves S5
%   from the same S4-of-record start with the FIXED, SIGNED oi_clear in
%   the rows (fresh MATLAB session = fixed model), and at 25 SOLVE
%   FIELDS: leg B proved the record's gap to the reported 53 was the
%   solve-field count, not iterations (25 fields / 60 iters -> 54.1 nm
%   vs leg A's 150-iteration 110.7 at 9 fields -- 9 fields
%   under-determine the 83 variables and the dense map stalls while the
%   solve set converges).  This leg = leg B's winning shape with real
%   clearance rows: the honest S5.
%
%   Metric unchanged: strict RMS WFE, centroid reference, dense 11x11
%   map MAXIMUM.  Artifacts in s5_budget/ beside legs A/B.
%
%   See also RUN_S5_BUDGET, RUN_T3, OI_CLEAR, PACKET.md.

    here = fileparts(mfilename('fullpath'));
    run(fullfile(here,'..','..','mmacos_setup.m'));
    addpath(here);
    addpath(fullfile(here,'..','..','templates','10_telescopes','offset_imager'));

    outdir = fullfile(here,'s5_budget');
    if ~exist(outdir,'dir'), mkdir(outdir); end

    R = load(fullfile(here,'t3','r3t_run.mat'));
    P = R.OUT.P;
    macos.init(P.model);

    fprintf('S5 of record: %.1f nm map max at TRUE floor 29.97 mm\n', ...
            R.OUT.s5.map.max_nm);

    Pc = P;  Pc.nsolve = 5;          % leg B's winning solve-field count
    X = R.OUT.s4.X;                  % as closed -- the record's S5 seed
    X = oi_zern_seed(X, Pc);
    [X, hist] = oi_solve(X, Pc, 'S5', 'clear', true, 'iters', 60);
    [X, Gc] = oi_close(X, Pc);  X.fpa = oi_apply_fpa(X);  Gc.fpa = X.fpa;
    [~, mp] = oi_map_fig(X, Gc, Pc, Pc.offset_deg, ...
        'S5 budget leg C: signed clearance rows', ...
        fullfile(outdir,'s5b_legC_map.png'));
    gt = oi_gates(X, Gc, Pc, Pc.offset_deg);
    legC = struct('label','C: 25 fields, iters 60, signed rows', ...
                  'X',X, 'map',mp, 'gates',gt, 'hist',hist);
    if exist(fullfile(outdir,'s5b_run.mat'),'file')
        S = load(fullfile(outdir,'s5b_run.mat'));  OUT = S.OUT;
    end
    OUT.legC = legC;
    save(fullfile(outdir,'s5b_run.mat'), 'OUT');
    fprintf(['\n  leg C: map max %.1f nm, clearance %.1f mm (%s), ' ...
             'exit err %.3f deg, %d iters\n'], mp.max_nm, ...
            gt.clear_min_m*1e3, pf_(gt.clear_pass), gt.exit_err_deg, ...
            hist.iters);
end

function s = pf_(p), if p, s = 'PASS'; else, s = 'FAIL'; end, end
