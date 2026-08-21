function OUT = run_s5_budget()
%RUN_S5_BUDGET  The S5 long-budget probe: is the 118-vs-53 gap solver budget?
%
%   The T3 run of record closes S5 at 118.2 nm against the reported 53
%   with all 30 iterations consumed (cap-limited, not plateaued) -- the
%   PACKET's claim is "convergence-limited, not physics-limited".  This
%   run buys the claim its evidence, from the SAME starting state as the
%   S5 of record (t3/r3t_run.mat, OUT.s4.X), same constraint rows:
%
%     leg A  same problem, more iterations: 9 solve fields, iters 150
%            (oi_solve's <0.1% plateau rule stops it earlier if earned)
%     leg B  more solve fields: nsolve 5 (25 fields against 82 vars --
%            the of-record 3x3 gives 9 fields, fewer than the variable
%            count), iters 60
%
%   Every reported number: strict RMS WFE, centroid reference, dense
%   11x11 map MAXIMUM over the box (the record's metric, unchanged).
%   Artifacts in s5_budget/ (leg .mat saved after EACH leg -- a killed
%   run keeps its finished leg).  This run does NOT touch t3/ -- the
%   instance of record stands; PACKET gains an addendum quoting this.
%
%   See also RUN_T3, OFFSET_IMAGER, OI_SOLVE, PACKET.md.

    here = fileparts(mfilename('fullpath'));
    run(fullfile(here,'..','..','mmacos_setup.m'));
    addpath(here);
    addpath(fullfile(here,'..','..','templates','10_telescopes','offset_imager'));

    outdir = fullfile(here,'s5_budget');
    if ~exist(outdir,'dir'), mkdir(outdir); end

    R = load(fullfile(here,'t3','r3t_run.mat'));
    P = R.OUT.P;
    macos.init(P.model);

    base_nm = R.OUT.s5.map.max_nm;      % 118.2 of record
    fprintf('S5 of record: %.1f nm map max (%d iters, cap-limited)\n', ...
            base_nm, R.OUT.s5.hist.iters);

    % ---- leg A: same 9 solve fields, iteration cap lifted ----------------
    OUT.legA = leg_(R.OUT.s4.X, P, 150, 'A: 9 fields, iters 150', ...
                    fullfile(outdir,'s5b_legA'));
    save(fullfile(outdir,'s5b_run.mat'), 'OUT');

    % ---- leg B: 25 solve fields ------------------------------------------
    Pb = P;  Pb.nsolve = 5;
    OUT.legB = leg_(R.OUT.s4.X, Pb, 60, 'B: 25 fields, iters 60', ...
                    fullfile(outdir,'s5b_legB'));
    save(fullfile(outdir,'s5b_run.mat'), 'OUT');

    % ---- verdict ----------------------------------------------------------
    fprintf('\n===== S5 budget verdict (of record: %.1f nm; reported: 53) =====\n', base_nm);
    for L = {'legA','legB'}
        g = OUT.(L{1});
        fprintf('  %-22s map max %.1f nm  (clear %.1f mm %s, exit err %.3f deg, %d iters)\n', ...
                g.label, g.map.max_nm, g.gates.clear_min_m*1e3, ...
                pf_(g.gates.clear_pass), g.gates.exit_err_deg, g.hist.iters);
    end
end

% =========================================================================
function G = leg_(X0, P, iters, label, tag)
    fprintf('\n===== leg %s =====\n', label);
    X = X0;  X.fpa_refit = [0 0];
    X = oi_zern_seed(X, P);
    [X, hist] = oi_solve(X, P, 'S5', 'clear', true, 'iters', iters);
    [X, Gc] = oi_close(X, P);  X.fpa = oi_apply_fpa(X);  Gc.fpa = X.fpa;
    [~, mp] = oi_map_fig(X, Gc, P, P.offset_deg, ...
        ['S5 budget ' label], [tag '_map.png']);
    gt = oi_gates(X, Gc, P, P.offset_deg);
    G = struct('label',label, 'X',X, 'map',mp, 'gates',gt, 'hist',hist);
    fprintf('  leg %s: map max %.1f nm, clearance %.1f mm (%s)\n', ...
            label, mp.max_nm, gt.clear_min_m*1e3, pf_(gt.clear_pass));
end

function s = pf_(p), if p, s = 'PASS'; else, s = 'FAIL'; end, end
