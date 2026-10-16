function OUT = run_t5_walk()
%RUN_T5_WALK  The t5 instance solved by PARAMETER CONTINUATION (the walk).
%
%   The t5 cold start FAILS: the unguided experiment crashed
%   (challenges/rodgers3/t5_unguided_REPORT.md), and even the diagnosed
%   redemption run (run_t5r -> t5_redemption/t5r_REPORT.md, verdict
%   PARTIAL) STALLS at ~595 um map max across S3/S4/S5 -- solving the full
%   15x15 deg box at +22.5 deg cold is outside the convergent basin (S2
%   loses 104/121 fields at documented defaults).
%
%   This run solves the SAME instrument by walking the box width open:
%   an easy narrow box first, then [5 8 11 13 15] deg, carrying the solved
%   design as each step's warm start (oi_walk).  Same envelope/seed as
%   run_t5r (the form-true x1.65 rescale of the rodgers3 W-fold, exit beam
%   pinned horizontal, clearances 40/25 mm), the S1 cap at the reference
%   class depth (159 nm) and the S5 solve grid the Zernike freedom needs
%   (nsolve_s5 = 5).  gn_iters is lifted to 30 (each step is a full-freedom
%   S5 solve; the redemption stalls were budget- and basin-bound).
%
%   The result -- ladder that reaches the target box, or a diagnosed
%   refusal at the traceability radius -- is scored exactly like any
%   oi_story run and written to t5_walk/t5_walk_REPORT.md.  Honest either
%   way; the re-walk deck's F15 sequel (CCL) reads this record.
%
%   See also OI_WALK, RUN_T5R, OI_STORY, OFFSET_IMAGER_PARAMS.

    here = fileparts(mfilename('fullpath'));
    run(fullfile(here,'..','..','..','mmacos_setup.m'));
    addpath(here);

    outdir = fullfile(here,'t5_walk');
    if ~exist(outdir,'dir'), mkdir(outdir); end

    diary(fullfile(outdir,'t5_walk_run.log'));
    cleaner = onCleanup(@() diary('off'));
    t0 = tic;
    fprintf('T5-WALK START %s\n', datestr(now,31)); %#ok<TNOW1,DATST>

    OUT = oi_walk(struct( ...
        'name','t5-walk', 'tag','t5_walk', ...
        'outdir', outdir, ...
        'EPD_m',0.150, 'Fno',3.3, 'box_deg',[15 15], 'offset_deg',22.5, ...
        'z_m1_m',0.6649568*1.65, ...
        'spacings_m',[-0.7228968 0 0.7408280]*1.65, ...
        'seed_R1_m',8.8*1.65, ...
        'clear_m',[0.040 0.025], 'exit_dir',[0 0 -1], ...
        's1_target_nm',159, 'nsolve_s5',5, 'gn_iters',30), ...
        'steps',[5 8 11 13 15], ...
        'baseline_nm',595565.2);   % the redemption S4 stall (t5r_REPORT.md)

    fprintf('T5-WALK DONE  %s  (%.1f min, verdict %s)\n', ...
            datestr(now,31), toc(t0)/60, OUT.verdict); %#ok<TNOW1,DATST>
end
