classdef tE2eRmsWaves < matlab.unittest.TestCase
%TE2ERMSWAVES  e2e stages 1 and 2's WFE helper counts every ray the engine wrote.
%
%   s1_telescope.m scores its bias sweep and its conic solve, and
%   s2_instrument.m its field points, with a local rms_waves(W, lam).
%   Until 2026-10-26 each found the pupil as W ~= 0, which under the
%   chief-ray OPD reference (the engine default since 2026-10-10)
%   drops any valid ray whose path equals the chief's -- the value of an
%   empty pixel.  The pupil is macos.opd_mask() (PLAN_CONSOLIDATION 1.2
%   (uuu) / (vvv), Dave 2026-10-26).
%
%   The gate cuts the helper out of each committed script (it is a local
%   function; nothing else can call it), traces that stage's committed deck,
%   sets one valid pixel to exactly 0 and requires the helper to count it.

    properties
        e2e_
    end

    properties (TestParameter)
        stage = struct('s1', {{'s1_telescope'}}, 's2', {{'s2_instrument'}});
    end

    methods (TestClassSetup)
        function setup(tc)
            h = fileparts(mfilename('fullpath'));
            run(fullfile(fileparts(h), 'mmacos_setup.m'));
            tc.e2e_ = fullfile(fileparts(h), 'templates', '80_end_to_end', 'e2e');
        end
    end

    methods (Test)
        function test_a_zero_valued_ray_is_counted(tc, stage)
            src = fileread(fullfile(tc.e2e_, [stage{1} '.m']));
            fn = regexp(src, 'function w = rms_waves\(W, lam\).*?\nend\n', 'match');
            tc.assertNumElements(fn, 1, [stage{1} '.m: rms_waves did not match exactly once']);
            d = tempname;  mkdir(d);
            fid = fopen(fullfile(d, 'rms_waves.m'), 'w');  fwrite(fid, fn{1});  fclose(fid);
            addpath(d);  tc.addTeardown(@() rmdir(d, 's'));  tc.addTeardown(@() rmpath(d));
            macos.init(256);
            n = macos.load_rx(fullfile(tc.e2e_, [stage{1} '.in']));
            macos.trace(n);
            W = macos.opd();  M = macos.opd_mask();
            tc.assertGreaterThan(nnz(M), 100);
            k = find(M & W ~= 0, 1);            % a valid pixel, set to the chief's value
            W(k) = 0;
            lam = 0.5e-6;
            want = std(W(M & isfinite(W))) / lam;
            tc.verifyEqual(rms_waves(W, lam), want, 'RelTol', 1e-12, ...
                'rms_waves must count a valid ray whose OPD is exactly 0 (macos.opd_mask, not W ~= 0)');
        end
    end
end
