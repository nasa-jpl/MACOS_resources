classdef tOffsetImager < matlab.unittest.TestCase
%TOFFSETIMAGER  The offset_imager template is a template, not a replay.
%
%   Runs the template's S1-S3 arc at the T4 SECOND parameter set (EPD
%   200 mm, F/2.5, 10x10-deg box offset 12 deg -- materially different
%   from the rodgers3 instance in aperture, speed, box and offset) with
%   reduced knobs, and asserts the STRUCTURE of the story rather than
%   any particular WFE:
%
%     1  the flow RUNS end-to-end at a non-rodgers3 parameter set and
%        emits decks, figures and the report (no hidden hardcoding);
%     2  the EFL identity holds EXACTLY at the new parameter set;
%     3  the S2 "disaster": moving the box off axis with only an FPA
%        refit costs a large factor in map max;
%     4  S3's re-solve at the used field recovers a large part of it.
%
%   The FULL five-stage runs are NOT in any suite (a stage solve is
%   minutes of machine time); they live in challenges/rodgers3/PACKET.md
%   and the template's committed report.  Own class rather than a
%   tRodgers3 method because the ladder there plus this smoke would
%   push one class past 10 minutes.
%
%   Model size 256 group: ./run_mmacos_tests.sh freeform
%   Runtime: ~11 min (S1-S3, 5 GN iterations each, 3x3 solve set + maps,
%   nGridpts 21).  The smoke knobs deliberately UNDER-converge (the F/2.5
%   case needs its full 15 iterations to reach tens of nm -- see the
%   committed t4_wide run); the assertions are structural with margins
%   measured at these knobs (2026-08-20: the S3 solve trajectory at the
%   carry start runs 51.8 -> 33.4 um qmean over the first 3 iterations
%   and the map ratio s3/s2 crosses 0.71 there, hence the 5-iteration
%   budget and the 0.75 bound with margin).
%
%   See also OFFSET_IMAGER, OFFSET_IMAGER_PARAMS, tests/tRodgers3.m.

    properties
        OUT
        outdir
    end

    methods (TestClassSetup)
        function setup(tc)
            h = fileparts(mfilename('fullpath'));
            run(fullfile(fileparts(h),'mmacos_setup.m'));
            addpath(fullfile(fileparts(h),'templates','10_telescopes','offset_imager'));
            macos.init(256);
            tc.outdir = fullfile(tempname);  mkdir(tc.outdir);
            tc.OUT = offset_imager(struct( ...
                'name','t4smoke', 'tag','t4s', ...
                'EPD_m',0.200, 'Fno',2.5, ...
                'box_deg',[10 10], 'offset_deg',12, ...
                'z_m1_m',1.0, 'spacings_m',[-0.10 0 1.10], ...
                'seed_R1_m',15, ...
                'stages',1:3, 'gn_iters',5, 'map_n',3, 'nsolve',3, ...
                'sampling',21, 'outdir',tc.outdir));
        end
    end

    methods (TestClassTeardown)
        function teardown(tc)
            if exist(tc.outdir,'dir'), rmdir(tc.outdir,'s'); end
        end
    end

    methods (Test)

        function test_runs_and_emits_artifacts(tc)
            for f = {'t4s_REPORT.md','t4s_run.mat', ...
                     't4s_s1.in','t4s_s2.in','t4s_s3.in', ...
                     't4s_s1_map.png','t4s_s2_map.png','t4s_s3_map.png', ...
                     't4s_s1_layout.png','t4s_s3_layout.png'}
                tc.verifyTrue(exist(fullfile(tc.outdir,f{1}),'file') == 2, ...
                    sprintf('missing artifact %s', f{1}));
            end
        end

        function test_efl_identity_at_the_new_parameters(tc)
            tc.verifyLessThan(abs(tc.OUT.s1.fo.EFL_m - 0.200*2.5), 1e-9, ...
                'EFL identity broke at the second parameter set');
            tc.verifyLessThan(abs(tc.OUT.s3.fo.EFL_m - 0.200*2.5), 1e-9, ...
                'EFL identity broke after the S3 re-solve');
        end

        function test_s2_offset_is_a_disaster(tc)
            tc.verifyGreaterThan(tc.OUT.s2.cost_of_offset, 2.0, sprintf( ...
                ['moving the box 12 deg off axis with only an FPA refit ' ...
                 'cost %.1fx -- the disaster-map pedagogy has vanished'], ...
                tc.OUT.s2.cost_of_offset));
        end

        function test_s3_resolve_recovers(tc)
            tc.verifyLessThan(tc.OUT.s3.map.max_nm, 0.75*tc.OUT.s2.map.max_nm, ...
                sprintf(['the S3 re-solve at the used field failed to ' ...
                'recover (s3 %.0f vs s2 %.0f nm)'], ...
                tc.OUT.s3.map.max_nm, tc.OUT.s2.map.max_nm));
        end
    end
end
