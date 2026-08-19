classdef tPolContrastCoro < matlab.unittest.TestCase
%TPOLCONTRASTCORO  Phase 2c on the coronagraph chain (ModelSize 512).
%   PLAN_POLARIZATION.md §2c.  Rx_Coro.in declares nGridpts=511, so it
%   MUST run at model >= 512 (Dave 2026-07-27): below that the engine
%   prints "Too many grid points. Resetting npts" and then intermittently
%   SIGSEGVs in intensity.  This class therefore lives in its own
%   run_mmacos_tests.sh batch.
%
%   Rx_Coro is a coaxial chain: seven Reflectors, all at normal
%   incidence, so its polarization aberration is purely NA-driven.
%   Six of those mirrors sit BETWEEN physical-optics propagation legs,
%   which is outside Phase 3a Tranche 1's validity: the seed-once rule
%   freezes the component planes at the first leg and thereafter applies
%   only a common scalar phase, so a mirror after that leg transforms the
%   RAYS but not the diffraction GRID.
%
%   That is not hidden here, it is MEASURED and pinned.  The floor this
%   fixture reports is a LOWER BOUND, and the two tests at the bottom
%   quantify by how much: the grid carries 0.84 of the ray-level
%   cross-polarized fraction bare, 0.57 with the mirrors coated, and the
%   sign of the coating sensitivity inverts (the grid says the floor
%   FALLS 3% when Al is added, while the ray-level cross fraction RISES
%   59%).  Closing that gap is Tranche 2 (§3a.3).  When Tranche 2 lands,
%   these two tests are the ones that must change.

    properties (Constant)
        ModelSize = 512
        Rx        = 'Rx_Coro.in'
        Pup       = 20    % ExitPupil (Return)
        Det       = 21    % FocalPlane
        Mirrors   = [1 4 7 12 15 17 18]
        nAl       = 1.45  % Al at 632.8 nm
        kAl       = 7.54
        thkAl     = 2.0e-4   % Rx_Coro BaseUnits = mm
        Tranche1  = 'macos:pol_contrast_floor:tranche1'
    end

    methods (TestClassSetup)
        function setupClass(testCase)
            macos.init(testCase.ModelSize);
        end
    end

    methods (TestMethodSetup)
        function loadRx(testCase)
            macos.load_rx(rx_fixture_path(testCase.Rx));
        end
    end

    methods (Static)
        function o = quiet_floor(varargin)
            % The Tranche-1 warning is EXPECTED on this fixture and is
            % asserted explicitly in its own test; silence it elsewhere so
            % it does not drown the run.
            id = tPolContrastCoro.Tranche1;
            w = warning('off', id);
            c = onCleanup(@() warning(w));
            o = macos.pol_contrast_floor(varargin{:});
        end
        function s = al_set()
            s = struct('elt', num2cell(tPolContrastCoro.Mirrors), ...
                'index', tPolContrastCoro.nAl, 'extinc', tPolContrastCoro.kAl, ...
                'thickness', tPolContrastCoro.thkAl, 'label', 'bare Al');
        end
    end

    methods (Test)

        function test_floor_reported_by_component(testCase)
            % Measured 2026-07-27 at model 512, x-polarized input:
            %   co 5.913731e+00  cross 2.832516e-08  long 7.22e-35
            %   cross/co 4.78973e-09,  peak cross contrast 1.27052e-09
            %   20..80 px annulus: co mean 4.164e-05, cross mean 5.787e-13
            o = tPolContrastCoro.quiet_floor(testCase.Pup, testCase.Det, ...
                'input', 'x', 'dark_zone', [20 80]);
            testCase.verifyEqual(o.floor.cross_over_co, 4.78973e-09, 'RelTol', 1e-3);
            testCase.verifyEqual(o.floor.contrast_cross_peak, 1.27052e-09, ...
                'RelTol', 1e-3);
            testCase.verifyEqual(o.floor.dark_zone.cross.mean, 5.787e-13, ...
                'RelTol', 0.01);
            testCase.verifyEqual(o.floor.dark_zone.co.mean, 4.1644e-05, ...
                'RelTol', 0.01);
            % the longitudinal channel is empty at a far-field focus here
            testCase.verifyLessThan(o.floor.long / o.floor.co, 1e-30);
            testCase.verifyGreaterThan(o.floor.dark_zone.n_pix, 10000);
        end

        function test_parseval_and_closure_at_scale(testCase)
            o = tPolContrastCoro.quiet_floor(testCase.Pup, testCase.Det, 'input', 'x');
            testCase.verifyLessThan(o.checks.parseval, 1e-15);
            testCase.verifyLessThan(o.checks.closure,  1e-14);
            testCase.verifyEqual(o.I_total, o.I_co + o.I_cross + o.I_long);
        end

        function test_analyzer_is_fully_polarized_and_axis_aligned(testCase)
            % A coaxial train rotates polarization only through its NA, so
            % the mean output state stays the input state to ~1e-16 -- and
            % the pupil stays essentially fully polarized.
            o = tPolContrastCoro.quiet_floor(testCase.Pup, testCase.Det, 'input', 'x');
            testCase.verifyGreaterThan(o.per_state(1).dop, 0.99999);
            testCase.verifyEqual(abs(o.per_state(1).analyzer' * [1; 0]), 1, ...
                'AbsTol', 1e-12);
        end

        function test_unpolarized_sums_two_runs(testCase)
            ox = tPolContrastCoro.quiet_floor(testCase.Pup, testCase.Det, 'input', 'x');
            oy = tPolContrastCoro.quiet_floor(testCase.Pup, testCase.Det, 'input', 'y');
            ou = tPolContrastCoro.quiet_floor(testCase.Pup, testCase.Det, ...
                                              'input', 'unpolarized');
            testCase.verifyEqual(ou.I_cross, ox.I_cross + oy.I_cross);
            testCase.verifyEqual(ou.floor.co, ox.floor.co + oy.floor.co, ...
                'RelTol', 1e-14);
        end

        % ---- the Tranche-1 gap, measured ---------------------------------
        function test_tranche1_shortfall_is_detected(testCase)
            % Six of the seven mirrors sit after the first physical leg,
            % so the grid under-carries the ray-level cross-polarized
            % fraction and the function must SAY SO rather than quote the
            % number as a floor.  Measured: grid 4.78973e-09 vs ray
            % 5.69403e-09, carried 0.8412.
            o = testCase.verifyWarning(@() macos.pol_contrast_floor( ...
                testCase.Pup, testCase.Det, 'input', 'x'), testCase.Tranche1);
            testCase.verifyFalse(o.scope.full_chain);
            testCase.verifyEqual(o.scope.worst, 0.841184, 'RelTol', 1e-3);
            testCase.verifyEqual(o.scope.grid_cross_frac(1), 4.78973e-09, ...
                'RelTol', 1e-3);
            testCase.verifyEqual(o.scope.ray_cross_frac(1), 5.69403e-09, ...
                'RelTol', 1e-3);
        end

        function test_coating_sensitivity_is_not_trustworthy_here(testCase)
            % The Tranche-1 gap in its most damaging form.  Coating all
            % seven mirrors with Al raises the RAY-level cross-polarized
            % fraction from 5.694e-09 to 9.034e-09 (+59%), but only the
            % first mirror's coating precedes the seed leg, so the GRID
            % reports the floor moving by -3.2% -- the wrong SIGN, not
            % merely the wrong size.  The carried fraction drops from 0.84
            % to 0.57, which is the flag that says do not quote this.
            % Pinned so that Tranche 2 has to come here and change it.
            o = tPolContrastCoro.quiet_floor(testCase.Pup, testCase.Det, ...
                'input', 'x', 'coatings', {tPolContrastCoro.al_set()});
            testCase.verifyEqual(o.sweep(1).d_cross_rel, -3.22333e-02, ...
                'RelTol', 0.02);
            testCase.verifyFalse(o.sweep(1).scope.full_chain);
            testCase.verifyEqual(o.sweep(1).scope.worst, 0.565284, 'RelTol', 1e-3);
            testCase.verifyLessThan(o.sweep(1).scope.worst, o.scope.worst, ...
                'coating the chain did not worsen the carried fraction');

            % ...and the ray-level truth the grid is missing, measured on
            % the same loaded state (the sweep leaves the coating applied).
            macos.polarization('on', 'Ex', [1 0], 'Ey', [0 0]);
            macos.vector_diffraction(true);
            macos.trace(testCase.Pup);
            rf = macos.ray_field(testCase.Pup);
            k = rf.status == 0;
            ray_cross = sum(abs(rf.Ey(k)).^2) / sum(abs(rf.Ex(k)).^2);
            testCase.verifyEqual(ray_cross, 9.0336e-09, 'RelTol', 1e-3);
        end
    end
end
