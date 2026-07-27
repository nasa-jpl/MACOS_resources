classdef tPolContrast < matlab.unittest.TestCase
%TPOLCONTRAST  Phase 2c -- polarization contrast floor, exactness gates.
%   PLAN_POLARIZATION.md §2c.  macos.pol_contrast_floor splits the
%   detector field into co-polarized / cross-polarized / longitudinal
%   channels, projecting on an analyzer derived from the pupil coherency
%   matrix.  This class pins the parts that must hold EXACTLY, on two
%   fixtures at ModelSize 256 (both declare nGridpts=256, so the model
%   size meets the "model >= nGridpts" rule):
%
%     Rx_VecChain.in       collimated, on-axis, flat uncoated planes.
%                          Polarization is a no-op by construction, so
%                          the machinery must report NO floor at all --
%                          this is the gate that the decomposition does
%                          not MANUFACTURE a floor, and it is the
%                          literal "x-pol input reduces to the scalar
%                          contrast curve at round-off" gate.
%     Rx_Cass_FarField.in  two mirrors and then a single far-field hop.
%                          Every polarizing surface precedes the one
%                          physical-optics leg, so the Tranche-1 seed
%                          carries the FULL train and the floor here is
%                          physics rather than a lower bound -- the
%                          fixture for the by-component floor and the
%                          coating sensitivity.
%
%   The coronagraph chain itself is tPolContrastCoro (ModelSize 512).
%
%   WHY THE CIRCULAR INPUT STATE IS LOAD-BEARING.  The coherency matrix
%   is C_ij = sum E_i conj(E_j).  Writing it with MATLAB's ' on the wrong
%   operand builds conj(C) instead, whose dominant eigenvector is the
%   CONJUGATE analyzer.  For any LINEAR input state the analyzer is real
%   and the two are identical, so x / y / 45-degree cases all pass
%   vacuously; for a circular state the wrong one is exactly ORTHOGONAL
%   to the truth and the reported cross/co jumps by 12 orders of
%   magnitude.  That bug was live in the first draft of the function and
%   was caught by exactly this gate.

    properties (Constant)
        ModelSize = 256
        RxChain   = 'Rx_VecChain.in'
        RxCass    = 'Rx_Cass_FarField.in'
        % Rx_VecChain: 1 PupilStop, 2 MidStop, 3 Prop2Start, 4 Detector
        ChainPup  = 2
        ChainDet  = 4
        % Rx_Cass_FarField: 1 SecMirObs, 2 Primary, 3 Secondary,
        %                   4 Return1, 5 ExitPupil, 6 Detector
        CassPrim  = 2
        CassSec   = 3
        CassPup   = 5
        CassDet   = 6
        nAl       = 1.45     % Al at 632.8 nm
        kAl       = 7.54
        thkAl     = 2.0e-7   % Rx_Cass_FarField BaseUnits = m
        nMgF2     = 1.38
        thkMgF2   = 1.1e-7   % ~quarter wave at 632.8 nm in MgF2
    end

    methods (TestClassSetup)
        function setupClass(testCase)
            macos.init(testCase.ModelSize);
        end
    end

    methods (Static)
        function loadChain()
            macos.load_rx(rx_fixture_path(tPolContrast.RxChain));
        end
        function loadCass()
            macos.load_rx(rx_fixture_path(tPolContrast.RxCass));
        end
        function c = radial_mean(I)
            % Peak-normalized azimuthal mean vs radius -- the contrast
            % curve, with no lambda/D convention assumed.
            N = size(I, 1);  ctr = (N - 1)/2;
            [xx, yy] = meshgrid(0:N-1, 0:N-1);
            rr = round(hypot(xx - ctr, yy - ctr)) + 1;
            c = accumarray(rr(:), I(:), [], @mean) / max(I(:));
        end
        function s = al_set(elts)
            s = struct('elt', num2cell(elts), 'index', tPolContrast.nAl, ...
                'extinc', tPolContrast.kAl, 'thickness', tPolContrast.thkAl, ...
                'label', 'bare Al');
        end
        function s = mgf2_al_set(elts)
            s = struct('elt', num2cell(elts), ...
                'index',  [tPolContrast.nMgF2 tPolContrast.nAl], ...
                'extinc', [0 tPolContrast.kAl], ...
                'thickness', [tPolContrast.thkMgF2 tPolContrast.thkAl], ...
                'label', 'MgF2 / Al');
        end
    end

    methods (Test)

        % ---- the reduction gate -----------------------------------------
        function test_reduction_to_scalar_contrast_curve(testCase)
            % On a train where polarization does nothing, the co-polarized
            % channel must BE the scalar run and the cross channel must be
            % empty.  Measured 2026-07-27 at model 256: cross exactly 0,
            % |I_co - scalar|/peak = 1.33e-15, contrast curves 2.36e-15.
            tPolContrast.loadChain();
            o = macos.pol_contrast_floor(testCase.ChainPup, testCase.ChainDet, ...
                                         'input', 'x');
            testCase.verifyEqual(o.floor.cross, 0, 'AbsTol', 1e-15 * o.floor.co, ...
                'the split invented a cross-polarized channel on a polarization-neutral train');
            testCase.verifyEqual(o.floor.long, 0, 'AbsTol', 1e-15 * o.floor.co);

            macos.polarization('off');
            Is = macos.intensity(testCase.ChainDet);
            pk = max(Is(:));
            testCase.verifyLessThan(max(abs(o.I_co(:) - Is(:))) / pk, 1e-13);

            cc = tPolContrast.radial_mean(o.I_co);
            cs = tPolContrast.radial_mean(Is);
            g  = isfinite(cc) & isfinite(cs) & cs > 0;
            testCase.verifyGreaterThan(nnz(g), 100);
            testCase.verifyLessThan(max(abs(cc(g) - cs(g)) ./ cs(g)), 1e-12, ...
                'the co-polarized contrast curve does not reduce to the scalar one');
        end

        % ---- the split is a unitary change of basis ----------------------
        function test_parseval_on_the_split(testCase)
            % co + cross == |Ex|^2 + |Ey|^2 pointwise.  This is exact
            % because [analyzer complement] is unitary by construction;
            % measured 1.3e-24 relative to the detector peak.
            tPolContrast.loadCass();
            o = macos.pol_contrast_floor(testCase.CassPup, testCase.CassDet);
            testCase.verifyLessThan(o.checks.parseval, 1e-18);
        end

        function test_energy_bookkeeping(testCase)
            % co + cross + long == the engine's own intensity, and the
            % channel maps sum to I_total identically.
            tPolContrast.loadCass();
            o = macos.pol_contrast_floor(testCase.CassPup, testCase.CassDet);
            testCase.verifyLessThan(o.checks.closure, 1e-14);
            testCase.verifyEqual(o.I_total, o.I_co + o.I_cross + o.I_long);
            testCase.verifyEqual(o.floor.co + o.floor.cross + o.floor.long, ...
                sum(o.I_total(:)), 'RelTol', 1e-14);
        end

        % ---- the analyzer is derived, and derived correctly --------------
        function test_analyzer_tracks_input_state(testCase)
            % A non-rotating train's mean OUTPUT state equals its input
            % state, so the derived analyzer must track an arbitrary input
            % -- including a CIRCULAR one, which is the only case that can
            % see a conjugated coherency matrix (see the class header).
            tPolContrast.loadCass();
            states = {[1; 0], [1; 1]/sqrt(2), [1; 1i]/sqrt(2)};
            for k = 1:numel(states)
                v = states{k};
                o = macos.pol_contrast_floor(testCase.CassPup, testCase.CassDet, ...
                                             'input', v);
                a = o.per_state(1).analyzer;
                testCase.verifyEqual(abs(a' * v), 1, 'AbsTol', 1e-12, ...
                    sprintf('analyzer lost the input state, case %d', k));
                testCase.verifyGreaterThan(o.per_state(1).dop, 0.999);
                % the floor is a property of the train, not of the state
                testCase.verifyLessThan(o.floor.cross_over_co, 1e-5);
            end
        end

        function test_analyzer_would_be_wrong_if_conjugated(testCase)
            % Non-vacuity for the gate above: projecting the circular
            % output on the CONJUGATE analyzer -- the exact error a
            % transposed coherency matrix makes -- reports essentially all
            % the light as cross-polarized.  The gate is therefore
            % measuring something.
            tPolContrast.loadCass();
            v = [1; 1i]/sqrt(2);
            o = macos.pol_contrast_floor(testCase.CassPup, testCase.CassDet, ...
                                         'input', v);
            a  = o.per_state(1).analyzer;
            ac = conj(a);
            testCase.verifyLessThan(abs(ac' * v), 1e-12, ...
                'the conjugate analyzer is not orthogonal to a circular state');
            % pol_contrast_floor restores the pre-call state, so re-arm the
            % engine before reading component planes directly.
            macos.polarization('on', 'Ex', [real(v(1)) imag(v(1))], ...
                                     'Ey', [real(v(2)) imag(v(2))]);
            macos.vector_diffraction(true);
            D1 = macos.complex_field(testCase.CassDet, 'plane', 1);
            D2 = macos.complex_field(testCase.CassDet, 'plane', 2, 'reset_trace', false);
            % The conjugate analyzer collects essentially NONE of the
            % light, so a run built on it would report the whole beam as
            % cross-polarized: implied cross/co = 7.13e5 against the true
            % 1.40e-6, a factor of 5e11.
            bad_co = abs(conj(ac(1))*D1 + conj(ac(2))*D2).^2;
            trans  = abs(D1).^2 + abs(D2).^2;
            testCase.verifyLessThan(sum(bad_co(:)) / o.floor.co, 1e-5, ...
                'the conjugated analyzer is not measurably wrong');
            implied = (sum(trans(:)) - sum(bad_co(:))) / sum(bad_co(:));
            testCase.verifyGreaterThan(implied, 1e5);
            testCase.verifyLessThan(o.floor.cross_over_co, 1e-5);
        end

        % ---- the floor, by component -------------------------------------
        function test_floor_reported_by_component(testCase)
            % Rx_Cass_FarField, bare mirrors: the geometric (NA-driven)
            % cross-polarized fraction.  7.0612e-07 is the same number the
            % r_p sign-fix probe reports for this fixture, arrived at
            % through an independent path (grid planes vs RayE).
            tPolContrast.loadCass();
            o = macos.pol_contrast_floor(testCase.CassPup, testCase.CassDet, ...
                                         'input', 'x', 'dark_zone', [10 40]);
            testCase.verifyEqual(o.floor.cross_over_co, 7.0612e-07, 'RelTol', 1e-3);
            testCase.verifyGreaterThan(o.floor.contrast_cross_peak, 0);
            % The LONGITUDINAL channel is not negligible on this train and
            % must be reported separately, not folded into either
            % transverse channel: long/co = 2.1109e-3, i.e. 2.1065e-3 of
            % the total.  The Phase 3a Tranche-1 attribution independently
            % measured the out-of-plane fraction on the same prescription
            % as 1 - f = 2.110e-3 (at model 128, from the pupil rather
            % than the detector); the two agree to 0.2%.
            testCase.verifyEqual(o.floor.long / o.floor.co, 2.1109e-3, ...
                'RelTol', 1e-3);
            testCase.verifyEqual(o.floor.long / (o.floor.co + o.floor.cross ...
                + o.floor.long), 2.110e-3, 'RelTol', 5e-3);
            % peak-normalized cross contrast in a 10..40 px annulus
            testCase.verifyEqual(o.floor.dark_zone.cross.mean, 9.59e-12, ...
                'RelTol', 0.02);
            testCase.verifyGreaterThan(o.floor.dark_zone.n_pix, 1000);
            testCase.verifyTrue(all(isfield(o.floor, ...
                {'co', 'cross', 'long', 'cross_over_co'})));
        end

        function test_coating_sensitivity(testCase)
            % The §2c contract: the floor AND how it moves with coating
            % choice.  Measured at model 256, both mirrors coated:
            %   bare      dark-zone cross contrast 9.59e-12
            %   bare Al                            3.34e-10   (d_cross_rel  27.9)
            %   MgF2/Al                            1.99e-09   (d_cross_rel 151.3)
            tPolContrast.loadCass();
            mir = [testCase.CassPrim testCase.CassSec];
            o = macos.pol_contrast_floor(testCase.CassPup, testCase.CassDet, ...
                'input', 'x', 'dark_zone', [10 40], 'coatings', ...
                {tPolContrast.al_set(mir), tPolContrast.mgf2_al_set(mir)});
            testCase.verifyNumElements(o.sweep, 2);
            testCase.verifyEqual(o.sweep(1).label, 'bare Al');
            testCase.verifyEqual(o.sweep(2).label, 'MgF2 / Al');
            testCase.verifyEqual(o.sweep(1).d_cross_rel, 27.898, 'RelTol', 0.02);
            testCase.verifyEqual(o.sweep(2).d_cross_rel, 151.31, 'RelTol', 0.02);
            % coating dominates the bare-metal geometric term by ~1.5 decades
            testCase.verifyGreaterThan(o.sweep(1).floor.dark_zone.cross.mean, ...
                                       10 * o.floor.dark_zone.cross.mean);
            testCase.verifyGreaterThan(o.sweep(2).floor.dark_zone.cross.mean, ...
                                       o.sweep(1).floor.dark_zone.cross.mean);
            % every sweep point on this fixture still carries the full train
            testCase.verifyTrue(o.sweep(1).scope.full_chain);
            testCase.verifyTrue(o.sweep(2).scope.full_chain);
        end

        function test_sweep_rejects_mismatched_element_sets(testCase)
            % A coating cannot be cleared, only overwritten, so sets over
            % different elements would silently accumulate.
            tPolContrast.loadCass();
            a = tPolContrast.al_set(testCase.CassPrim);
            b = tPolContrast.al_set([testCase.CassPrim testCase.CassSec]);
            testCase.verifyError(@() macos.pol_contrast_floor( ...
                testCase.CassPup, testCase.CassDet, 'coatings', {a, b}), ...
                'macos:pol_contrast_floor:sweepElts');
        end

        % ---- honesty rules ------------------------------------------------
        function test_small_denominators_are_nan_not_zero(testCase)
            % frac_cross masks a small denominator with NaN -- zero-filling
            % would seed statistics with a ring of perfect nulls.  Drive
            % floor_tol high enough that the mask actually bites.
            tPolContrast.loadCass();
            o = macos.pol_contrast_floor(testCase.CassPup, testCase.CassDet, ...
                                         'floor_tol', 1e-2);
            masked = ~isfinite(o.frac_cross);
            testCase.verifyGreaterThan(nnz(masked), 0, ...
                'floor_tol did not mask anything -- the gate is vacuous');
            testCase.verifyTrue(all(isnan(o.frac_cross(masked))), ...
                'masked pixels are not NaN');
            % and nothing was zero-filled: every finite entry is the ratio
            raw = o.I_cross ./ o.I_total;
            testCase.verifyEqual(o.frac_cross(~masked), raw(~masked));
        end

        function test_unpolarized_is_two_independent_runs(testCase)
            % The incoherent sum is x-in plus y-in, each traced; the second
            % state is never synthesized from the first.
            tPolContrast.loadCass();
            ox = macos.pol_contrast_floor(testCase.CassPup, testCase.CassDet, 'input', 'x');
            oy = macos.pol_contrast_floor(testCase.CassPup, testCase.CassDet, 'input', 'y');
            ou = macos.pol_contrast_floor(testCase.CassPup, testCase.CassDet, ...
                                          'input', 'unpolarized');
            testCase.verifyNumElements(ou.per_state, 2);
            testCase.verifyEqual(ou.per_state(1).state, [1; 0]);
            testCase.verifyEqual(ou.per_state(2).state, [0; 1]);
            testCase.verifyEqual(ou.I_co,    ox.I_co    + oy.I_co);
            testCase.verifyEqual(ou.I_cross, ox.I_cross + oy.I_cross);
            testCase.verifyEqual(ou.I_long,  ox.I_long  + oy.I_long);
        end

        % ---- scope diagnostic ---------------------------------------------
        function test_scope_reports_full_carry_on_a_single_leg_train(testCase)
            % Every polarizing surface precedes the only physical leg, so
            % the grid carries exactly what the rays do.
            tPolContrast.loadCass();
            o = macos.pol_contrast_floor(testCase.CassPup, testCase.CassDet);
            testCase.verifyTrue(o.scope.full_chain);
            testCase.verifyEqual(o.scope.worst, 1, 'AbsTol', 1e-3);
            testCase.verifyGreaterThan(o.scope.ray_cross_frac(1), 1e-9, ...
                'the carry check is vacuous -- there is no cross-pol to carry');
        end

        function test_scope_is_not_faked_on_a_neutral_train(testCase)
            % Rx_VecChain has no cross-polarized light at all: the ratio
            % would be 0/0, so full carry is declared rather than divided.
            tPolContrast.loadChain();
            o = macos.pol_contrast_floor(testCase.ChainPup, testCase.ChainDet);
            testCase.verifyTrue(o.scope.full_chain);
            testCase.verifyEqual(o.scope.carried, 1);
            testCase.verifyLessThan(o.scope.ray_cross_frac(1), 1e-12);
        end

        % ---- veneer -------------------------------------------------------
        function test_session_method(testCase)
            m = macos.Session(testCase.ModelSize);
            m.load_rx(rx_fixture_path(testCase.RxCass));
            o = m.pol_contrast_floor(testCase.CassPup, testCase.CassDet);
            testCase.verifyEqual(o.floor.cross_over_co, 7.0612e-07, 'RelTol', 1e-3);
        end

        function test_restores_polarization_state(testCase)
            tPolContrast.loadCass();
            macos.polarization('off');
            macos.pol_contrast_floor(testCase.CassPup, testCase.CassDet);
            testCase.verifyFalse(macos.polarization().on);
            macos.polarization('on', 'Ex', [0 0], 'Ey', [1 0]);
            macos.pol_contrast_floor(testCase.CassPup, testCase.CassDet);
            s = macos.polarization();
            testCase.verifyTrue(s.on);
            testCase.verifyEqual(s.Ex, complex(0, 0));
            testCase.verifyEqual(s.Ey, complex(1, 0));
        end
    end
end
