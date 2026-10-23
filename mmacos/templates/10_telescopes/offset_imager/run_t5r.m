function OUT = run_t5r()
%RUN_T5R  The t5 REDEMPTION run: the unguided instance, per the fixed docs.
%
%   The t5 unguided experiment (challenges/rodgers3/t5_unguided_REPORT.md)
%   FAILED: three attempts, three crashes, an uncapped S1 making the
%   offset box untraceable.  This run re-instances the SAME instrument
%   exactly as the post-fix README prescribes -- the "Run it" example
%   verbatim (attempt-2's envelope, the one the docs' form-true rescale
%   names) plus the two knobs the new "Choosing an envelope" section
%   mandates: s1_target_nm at the reference class depth (rodgers3 r1 =
%   159 nm) and nsolve_s5 = 5.  Everything else is documented defaults.
%   The result -- ladder + gates or a diagnosed refusal -- feeds slide
%   F15's sequel in the re-walk deck either way.
%
%   See also OI_STORY, OFFSET_IMAGER_PARAMS, run_t4 (retired instance).

    here = fileparts(mfilename('fullpath'));
    run(fullfile(here,'..','..','..','mmacos_setup.m'));
    addpath(here);

    OUT = oi_story(struct( ...
        'name','t5-redemption', 'tag','t5r', ...
        'outdir', fullfile(here,'t5_redemption'), ...
        'EPD_m',0.150, 'Fno',3.3, 'box_deg',[15 15], 'offset_deg',22.5, ...
        'z_m1_m',0.6649568*1.65, ...
        'spacings_m',[-0.7228968 0 0.7408280]*1.65, ...
        'seed_R1_m',8.8*1.65, ...
        'clear_m',[0.040 0.025], 'exit_dir',[0 0 -1], ...
        's1_target_nm',159, 'nsolve_s5',5));
end
