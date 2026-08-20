classdef tRodgers3 < matlab.unittest.TestCase
%TRODGERS3  The rodgers3 challenge: Stage-0 reproduction gates.
%
%   Asserts the five-rung reproduction of Mike Rodgers' offset-field
%   imager ladder (challenges/rodgers3) plus the negative control that
%   proves the decisive gate has teeth.
%
%   METRIC (stated per the challenge rule): strict RMS WFE, sphere
%   centred on the spot centroid on the deck-verbatim .seq FPA, anchored
%   at the exit pupil, piston-only removal; the gate statistic is the
%   MAXIMUM of a dense field map over the 20x20-deg box -- that is what
%   Mike's "RMS WFE <= x" numbers are (decoded from his slides' own EMF
%   metadata, r3_s0_report.txt).
%
%   RUNTIME BOUND: the in-suite dense maps are COARSENED to 5x5 (the
%   committed record is 11x11).  The coarsening is visible in r2 and r5
%   only through the map max sitting slightly under the 11x11 value
%   (the r5 peak at YAN +28 is off the 5x5 lattice); the bands below are
%   derived from the MEASURED 5x5 ladder (2026-08-19, this tree) with
%   the Stage-0 margins on top:
%
%     rung   his nm    5x5 map max   ratio     band asserted
%      r1      159        157.74     0.992     [0.90, 1.10]
%      r2     8810       9664.12     1.097     [0.95, 1.25]  (pupil-model
%             caveat: uniform perp-chief bundle vs CODE V stop-gridded
%             rays inflates r2/r4 slightly -- r3_s0_report.txt)
%      r3      168        166.32     0.990     [0.90, 1.10]
%      r4      117        124.29     1.062     [0.95, 1.20]
%      r5       53         52.53     0.991     [0.85, 1.15]  (the ZRN gate;
%             11x11 gives 54.66 = 1.031 -- its peak at YAN +28 is off the
%             5x5 lattice, hence the slightly lower in-suite value)
%
%   NOTHING IS TUNED TOWARD HIS NUMBERS -- the band is the measurement
%   (rodgers1 rule); a band violation is a CONVENTION regression, which
%   is exactly what this class exists to catch.
%
%   Model size 256 group: ./run_mmacos_tests.sh freeform
%   Runtime: ~6 min (one shared ladder run in the class setup + the
%   9-point negative control).
%
%   See also challenges/rodgers3/rodgers3.m, r5_negctl.m, BUILD_R3.

    properties
        OUT      % the shared 5x5 ladder run
        here
    end

    properties (Constant)
        gates_nm = [159 8810 168 117 53];
        band_lo  = [0.90 0.95 0.90 0.95 0.85];
        band_hi  = [1.10 1.25 1.10 1.20 1.15];
    end

    methods (TestClassSetup)
        function setup(tc)
            h = fileparts(mfilename('fullpath'));
            run(fullfile(fileparts(h),'mmacos_setup.m'));
            tc.here = fullfile(fileparts(h),'challenges','rodgers3');
            addpath(tc.here);
            macos.init(256);
            % decks are committed; rebuild = a byte-identity no-op that
            % also proves the generator runs on this tree
            build_r3();
            tc.OUT = rodgers3('map_n',5,'quiet',true,'save',false);
        end
    end

    methods (Test)

        function test_r1_coaxial_onaxis(tc),  tc.gate_(1);  end
        function test_r2_offset_fpa_refit_only(tc), tc.gate_(2); end
        function test_r3_reasphered_at_offset(tc),  tc.gate_(3); end
        function test_r4_tilt_dec_radii(tc),        tc.gate_(4); end
        function test_r5_zernike_convention(tc),    tc.gate_(5); end

        function test_r5_negative_control_c_offset(tc)
        %  The r5 deck with the SCO C-offset deliberately dropped (modes
        %  = C-index verbatim instead of C-index - 1) must MISS its gate
        %  by a large factor -- the proof that the r5 gate is testing
        %  the Zernike convention, not passing vacuously.  Measured
        %  2026-08-19: correct C-offset 30.89 nm 9-pt max; offset dropped
        %  92309 nm -- factor 2988.6x.  Assert a wildly conservative 100x
        %  (a band violation here means the convention machinery itself
        %  regressed, not a numerical drift).
            NC = r5_negctl();
            tc.verifyGreaterThan(NC.factor, 100.0, sprintf( ...
                ['r5 with the C-offset dropped scored only %.1fx the ' ...
                 'correct deck -- the gate has lost its teeth'], NC.factor));
        end
    end

    methods
        function gate_(tc, k)
            rungs = {'r1','r2','r3','r4','r5'};
            o = tc.OUT.(rungs{k});
            tc.assertTrue(o.ran, sprintf('%s did not run (earlier gate failed)', rungs{k}));
            ratio = o.gate_stat / tc.gates_nm(k);
            tc.verifyGreaterThan(ratio, tc.band_lo(k), sprintf( ...
                '%s: 5x5 map max %.2f nm = %.3fx his %g nm, below band', ...
                rungs{k}, o.gate_stat, ratio, tc.gates_nm(k)));
            tc.verifyLessThan(ratio, tc.band_hi(k), sprintf( ...
                '%s: 5x5 map max %.2f nm = %.3fx his %g nm, above band', ...
                rungs{k}, o.gate_stat, ratio, tc.gates_nm(k)));
        end
    end
end
