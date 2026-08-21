function OUT = run_t4()
%RUN_T4  The template proven parameterized: a second, materially
%   different instrument through all five stages.
%
%   Parameter choice (and why): EPD 200 mm, F/2.5 (EFL 0.5 m), 10x10-deg
%   box offset 12 deg, lambda 1 um -- 2.7x the rodgers3 aperture, 1.6x
%   faster, half the box, roughly half the offset; same clearance-list
%   constraint style.  The exit direction is REPORT-ONLY here (the exit
%   angle is a first-order property of EFL + spacings + the stop pose,
%   so pinning it is a PACKAGING choice, not a surface-solve
%   constraint).  The clearance spec is the instance's own: >10 / >5 mm.
%   A fast wide-aperture offset imager is the other common corner of
%   this trade space, and the faster speed moves the asphere/Zernike
%   burden onto different surfaces than the rodgers3 instance -- if
%   anything rodgers3-specific were hardcoded, this run would expose it.
%   T4's gate is STRUCTURAL: the flow runs end-to-end and tells the same
%   story (S2 disaster, S3 recovery, S4/S5 gains); it does not need to
%   hit any particular WFE.
%
%   PACKAGING (retracted and re-chosen 2026-08-21).  The original
%   envelope (z_m1 1.0, spacings [-0.10 0 1.10]) was SELF-BLOCKING for a
%   200 mm beam: M2 sat 100 mm inside the incoming corridor with the FP
%   behind it, and the M3 fan ran ~120 mm DEEP through M1 and M2 --
%   invisible while oi_clear's 25 fixed samples stepped over piercings
%   (its "reachable floor ~10 mm" was an artifact of that blindness, and
%   the first honest re-run proved the fold needed to clear (~65 deg of
%   the 100 mm M1->M2 leg) is not reachable by tilt/decenter from any
%   compliant-adjacent start.  Dave caught it ON THE LAYOUT FIGURE:
%   graphics that show the hardware are themselves a gate.)  The
%   envelope is a designer INPUT, so the fix is a workable one: the
%   rodgers3 W-fold -- which clears its own constraint set at EPD 75 --
%   scaled by the EFL ratio 0.5/0.3, which preserves the focal
%   proportions (legs/EFL) that make the first-order seed and the fold
%   geometry work.  (An aperture-ratio 200/75 scaling was tried first
%   and is WRONG here: t4 is faster than rodgers3, so scaling legs by
%   aperture while EFL scales by focal ratio breaks the form -- the
%   seed landed at R1 = 52 m and S1 stalled at ~1 mm WFE.)  The F/2.5
%   beam is 2.67x fatter inside a 1.67x envelope, so the clearance
%   margins are proportionally tighter than rodgers3's -- that deficit
%   is what S4 must earn back with tilt/decenter under the signed
%   clearance rows.
%
%   Artifacts land in t4_wide/.  The suite smoke of this run (reduced
%   knobs, S1-S3) is tests/tOffsetImager.m.
%
%   See also OFFSET_IMAGER, OFFSET_IMAGER_PARAMS, tests/tOffsetImager.m.

    here = fileparts(mfilename('fullpath'));
    run(fullfile(here,'..','..','..','mmacos_setup.m'));
    addpath(here);

    sc = 0.5/0.3;                     % EFL ratio vs rodgers3 (form-true)
    OUT = offset_imager(struct( ...
        'name','t4-wide', 'tag','t4', ...
        'outdir', fullfile(here,'t4_wide'), ...
        'EPD_m',0.200, 'Fno',2.5, ...
        'box_deg',[10 10], 'offset_deg',12, ...
        'z_m1_m',0.6649568*sc, ...
        'spacings_m',[-0.7228968 0 0.7408280]*sc, ...
        'seed_R1_m',8.8*sc, ...
        'clear_m',[0.010 0.005], ...
        'gn_iters',15));
end
