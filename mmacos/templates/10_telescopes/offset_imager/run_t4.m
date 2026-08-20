function OUT = run_t4()
%RUN_T4  The template proven parameterized: a second, materially
%   different instrument through all five stages.
%
%   Parameter choice (and why): EPD 200 mm, F/2.5 (EFL 0.5 m), 10x10-deg
%   box offset 12 deg, lambda 1 um -- 2.7x the rodgers3 aperture, 1.6x
%   faster, half the box, roughly half the offset; same clearance-list
%   constraint style.  The exit direction is REPORT-ONLY here: at these
%   (deliberately arbitrary) spacings the offset chief exits ~4.7 deg
%   off horizontal for ANY symmetric surfaces -- the exit angle is a
%   first-order property of EFL + spacings + the stop pose, so pinning
%   it is a PACKAGING choice, not a surface-solve constraint.  A fast
%   wide-aperture offset imager is the other common corner of this
%   trade space, and the faster speed moves the asphere/Zernike burden
%   onto different surfaces than the rodgers3 instance -- if anything
%   rodgers3-specific were hardcoded, this run would expose it.  T4's
%   gate is STRUCTURAL: the flow runs end-to-end and tells the same
%   story (S2 disaster, S3 recovery, S4/S5 gains); it does not need to
%   hit any particular WFE.
%
%   Packaging: spacings [-0.10 0 1.10] m, M1 at z = 1.0 m -- the same
%   near-concentric compact form scaled to the 0.5 m EFL.
%
%   Artifacts land in t4_wide/.  The suite smoke of this run (reduced
%   knobs, S1-S3) is tests/tOffsetImager.m.
%
%   See also OFFSET_IMAGER, OFFSET_IMAGER_PARAMS, tests/tOffsetImager.m.

    here = fileparts(mfilename('fullpath'));
    run(fullfile(here,'..','..','..','mmacos_setup.m'));
    addpath(here);

    OUT = offset_imager(struct( ...
        'name','t4-wide', 'tag','t4', ...
        'outdir', fullfile(here,'t4_wide'), ...
        'EPD_m',0.200, 'Fno',2.5, ...
        'box_deg',[10 10], 'offset_deg',12, ...
        'z_m1_m',1.0, 'spacings_m',[-0.10 0 1.10], ...
        'seed_R1_m',15, ...
        'gn_iters',15));
end
