classdef tVecChain < matlab.unittest.TestCase
%TVECCHAIN  Phase 3a Tranche 1 -- vector near-field propagation chain.
%   PLAN_POLARIZATION.md §3a.  Pins the two §3a.1 defect fixes and the
%   §3a.2 per-component leg loops:
%
%     3a.1(1)  the vector far-field leg called PFFPROP, a bare per-component
%              FFT missing the Fresnel-integral output factors scalar FFPROP
%              applies via applyfac2 -- a global scale plus the output-plane
%              curvature that is the NEXT leg's input.  It now runs FFPROP
%              once per component plane, so vector and scalar far-field legs
%              share one kernel.
%     3a.1(2)  the polarized field assembly RELOADED the grid from RayE at
%              every physical leg, erasing the diffraction accumulated by
%              earlier legs, and resurrected rays the ray-side aperture
%              masking had already extinguished (RayE carries no vignetting).
%              It now seeds once and then applies the same incremental
%              geometric-phase update the non-polarized branch uses.
%     3a.2     every near-field / DFT leg, FFObscure, and the ray-side
%              clip/taper sites now cover all three component planes.
%
%   The gate prescription is tests/Rx/Rx_VecChain.in: a collimated on-axis
%   source through flat normal-incidence uncoated planes, so the ray E-field
%   direction is a CONSTANT unit vector and the field factorises as
%   E_k = e_k*u(x,y).  Propagating the three planes separately and summing
%   |E_k|^2 must therefore reproduce the scalar intensity to round-off at
%   every leg, for ANY input polarization state.  That makes the comparison
%   exact rather than "close enough", which matters because on a real
%   off-normal train (Rx_Cass_FarField) vector and scalar differ by ~2.6e-3
%   and no tolerance there would be defensible.
%
%   ATTRIBUTION -- MEASURED, and the first guess was half wrong.  That
%   2.6e-3 was believed to be the off-normal train's out-of-plane content.
%   It could not be checked, because the per-plane contribution to a
%   propagated intensity was not reachable.  macos.complex_field(srf,
%   'plane', k) now makes it reachable, and the difference decomposes into
%   TWO mechanisms: (1) the scalar run seeds from |RayE|, so ALL the power
%   -- including what is physically out-of-plane -- propagates in one
%   plane, while the vector run leaves only the fraction f in Ex (a
%   near-pure rescale); (2) Ey and Ez diffract to their own pattern.  So
%   Iv ~ f*Is + Iy + Iz, which drops the difference from 2.56e-3 to
%   2.90e-4.  The naive expectation (difference == out-of-plane intensity)
%   is wrong by ~2x.  Pinned by
%   test_vector_scalar_difference_decomposition below.  The 2.9e-4 that
%   remains is a shape difference between the scalar field and Ex,
%   consistent with their different seeds, and is NOT further verified.
%
%   NON-VACUITY (checked 2026-07-26 against the pre-fix engine, both
%   compilers): the pre-fix code fails these at 0.21 .. 0.38 relative error
%   and mis-states total power by 4-7%.  The 45-degree and circular states
%   are load-bearing -- with an x-only source ALL the energy sits in
%   component plane 1, which the old single-plane propagator happened to
%   carry correctly, so an x-pol-only gate passes vacuously.

    properties (Constant)
        ModelSize = 128
        RxChain   = 'Rx_VecChain.in'
        RxFF      = 'Rx_Cass_FarField.in'
        Leg1      = 2     % MidStop      -- end of near-field leg 1
        Leg2      = 4     % Detector     -- end of near-field leg 2
        FFDet     = 6     % Rx_Cass_FarField detector (single far-field hop)
        % Round-off budget for an N=128 grid through two FFT-pair legs.
        Tol       = 1e-13
    end

    methods (TestClassSetup)
        function setupClass(testCase)
            macos.init(testCase.ModelSize);
        end
    end

    methods (Static)
        function I = run_case(rxName, mode, elt)
            % Fresh load each time: pol state changes dirty the trace, and
            % a clean load is the cheapest way to keep the cases independent.
            macos.load_rx(rx_fixture_path(rxName));
            switch mode
                case 'scalar'
                    macos.polarization('off');
                case 'polsc'                       % polarized, scalar diffraction
                    macos.polarization('on', 'Ex', [1 0], 'Ey', [0 0]);
                    macos.vector_diffraction(false);
                case 'vec_x'                       % vector, x-polarized
                    macos.polarization('on', 'Ex', [1 0], 'Ey', [0 0]);
                    macos.vector_diffraction(true);
                case 'vec_45'                      % vector, 45-degree linear
                    macos.polarization('on', 'Ex', [1 0], 'Ey', [1 0]);
                    macos.vector_diffraction(true);
                case 'vec_circ'                    % vector, circular
                    macos.polarization('on', 'Ex', [1 0], 'Ey', [0 1]);
                    macos.vector_diffraction(true);
                otherwise
                    error('tVecChain:mode', 'unknown mode %s', mode);
            end
            I = macos.intensity(elt);
        end

        function r = relerr(A, B)
            r = norm(A(:) - B(:)) / max(norm(B(:)), eps);
        end
    end

    methods (Test)
        % ---- 3a.1(2): polarized-scalar must reduce to the scalar path ----
        function test_polarized_scalar_is_bit_identical(testCase)
            % With polarization ON but vector diffraction OFF the engine
            % must reproduce the polarization-OFF run EXACTLY: same seed
            % amplitude, same incremental phase, same vignetting.  Before
            % the seed-once + LRayPass fixes this was wrong by 21% after
            % one leg and 38% after two.
            for elt = [testCase.Leg1, testCase.Leg2]
                Is = tVecChain.run_case(testCase.RxChain, 'scalar', elt);
                Ip = tVecChain.run_case(testCase.RxChain, 'polsc',  elt);
                testCase.verifyEqual(Ip, Is, ...
                    sprintf(['polarization-ON/vector-OFF must be ' ...
                             'bit-identical to polarization-OFF at elt %d'], elt));
            end
        end

        % ---- 3a.2 + 3a.1(2): vector chain == scalar chain, any state ----
        function test_vector_equals_scalar_every_state(testCase)
            states = {'vec_x', 'vec_45', 'vec_circ'};
            for elt = [testCase.Leg1, testCase.Leg2]
                Is = tVecChain.run_case(testCase.RxChain, 'scalar', elt);
                for k = 1:numel(states)
                    Iv = tVecChain.run_case(testCase.RxChain, states{k}, elt);
                    % Ex=Ey=1 carries twice the flux; compare shapes.
                    r = tVecChain.relerr(Iv / sum(Iv(:)), Is / sum(Is(:)));
                    testCase.verifyLessThan(r, testCase.Tol, sprintf( ...
                        ['vector chain (%s) must reproduce the scalar ' ...
                         'intensity at elt %d on a polarization-neutral ' ...
                         'train; got rel=%.3e'], states{k}, elt, r));
                end
            end
        end

        % ---- validation ladder 1: energy conservation per leg ------------
        function test_energy_conserved_per_leg(testCase)
            for elt = [testCase.Leg1, testCase.Leg2]
                Is = tVecChain.run_case(testCase.RxChain, 'scalar', elt);
                Iv = tVecChain.run_case(testCase.RxChain, 'vec_x',  elt);
                testCase.verifyEqual(sum(Iv(:)), sum(Is(:)), ...
                    'RelTol', 1e-14, sprintf( ...
                    'total power must survive the vector leg to elt %d', elt));
            end
        end

        % ---- 3a.2: the mask really is on the vector path -----------------
        function test_mask_throughput_identical_on_vector_path(testCase)
            % MidStop carries a central obscuration, so the second leg
            % starts with ~3% less power than the first ends with.  The
            % vector run must lose the SAME fraction: if the ray-side
            % masking (WFZeroPt) were still single-plane, the stale Ey/Ez
            % planes would keep their share of the blocked power.  This
            % guards the mask change directly, without depending on where
            % the diffracted shadow lands (at Fresnel number 25 the centre
            % of a 1 mm obscuration is an Arago bright spot, not a null).
            Ts = @(m) [sum(sum(tVecChain.run_case(testCase.RxChain, m, testCase.Leg1))), ...
                       sum(sum(tVecChain.run_case(testCase.RxChain, m, testCase.Leg2)))];
            ts = Ts('scalar');
            testCase.assertLessThan(ts(2), 0.99 * ts(1), ...
                'fixture broken: MidStop obscuration removes no power');
            for m = {'vec_x', 'vec_45', 'vec_circ'}
                tv = Ts(m{1});
                testCase.verifyEqual(tv(2) / tv(1), ts(2) / ts(1), ...
                    'RelTol', 1e-14, sprintf( ...
                    'mask throughput differs on the vector path (%s)', m{1}));
            end
        end

        % ---- validation ladder 5: far-field normalization A/B ------------
        function test_far_field_vector_matches_scalar_normalization(testCase)
            % PFFPROP applied only 1/N per component and skipped applyfac2;
            % FFPROP applies 1/(i*lambda*dz)*dx1^2 and the output quadratic
            % phase.  With one shared kernel the vector total power must now
            % equal the scalar total exactly (Parseval: the three component
            % planes partition the scalar norm, and the extra factor is a
            % common scale times a unimodular phase).
            %
            % A/B measured 2026-07-26 on this Rx at model 128:
            %   pre-fix  sum(vector) = 8.937660518e-01
            %   post-fix sum(vector) = 1.815495281e+06 == sum(scalar)
            % i.e. the vector far-field leg was low by 2.031e+06 in
            % intensity (1.425e+03 in amplitude) and is now normalized
            % identically to the scalar leg.
            Is = tVecChain.run_case(testCase.RxFF, 'scalar', testCase.FFDet);
            Iv = tVecChain.run_case(testCase.RxFF, 'vec_x',  testCase.FFDet);
            testCase.verifyEqual(sum(Iv(:)), sum(Is(:)), 'RelTol', 1e-12, ...
                ['vector far-field leg must carry the scalar leg''s ' ...
                 'Fresnel output factors (PFFPROP -> FFPROP x3)']);
            % Sanity bound only: the vector run must neither collapse onto
            % the scalar map nor wander far from it.  Rx_Cass_FarField is an
            % off-normal train (|Ez|/|Ex| ~ 8.8e-2 at the exit pupil by
            % ray_field), and the out-of-plane content is the SUSPECTED
            % source of the difference -- now MEASURED and decomposed; see
            % the ATTRIBUTION note in the class header and
            % test_vector_scalar_difference_decomposition.  These remain
            % empirical brackets on the observed 2.6e-3, not a derived
            % budget: the decomposition explains it, it does not predict it
            % from first principles.
            r = tVecChain.relerr(Iv, Is);
            testCase.verifyGreaterThan(r, 1e-4, ...
                'vector far-field run collapsed onto the scalar result');
            testCase.verifyLessThan(r, 1e-2, ...
                'vector/scalar far-field difference outside its observed bracket');
        end

        % ---- plane-selectable complex field: the planes ARE the sum ----
        function test_component_planes_sum_to_intensity(testCase)
            % macos.intensity sums |Ex|^2+|Ey|^2+|Ez|^2 internally; this
            % pins that the per-plane getter returns those very planes and
            % not, say, a stale buffer or the same plane three times.
            macos.load_rx(rx_fixture_path(testCase.RxFF));
            macos.polarization('on', 'Ex', [1 0], 'Ey', [0 0]);
            macos.vector_diffraction(true);
            I = macos.intensity(testCase.FFDet);
            S = zeros(size(I));
            for k = 1:3
                S = S + abs(macos.complex_field(testCase.FFDet, 'plane', k)).^2;
            end
            testCase.verifyLessThan(tVecChain.relerr(S, I), 1e-14, ...
                'the three component planes must sum to the intensity');
            % plane 0 is the historical behaviour, unchanged
            testCase.verifyEqual( ...
                macos.complex_field(testCase.FFDet, 'plane', 0), ...
                macos.complex_field(testCase.FFDet), ...
                'plane 0 must equal the default (backward compatible)');
        end

        function test_component_plane_rejected_in_scalar_mode(testCase)
            % In scalar mode plane k is an UNRELATED wavefront, not a field
            % component.  Returning it would look plausible and be wrong,
            % so the api must refuse.
            macos.load_rx(rx_fixture_path(testCase.RxFF));
            macos.polarization('off');
            macos.intensity(testCase.FFDet);
            % mexErrMsgTxt raises with an EMPTY identifier, so match on the
            % message rather than an id that does not exist.
            threw = false;  msg = '';
            try
                macos.complex_field(testCase.FFDet, 'plane', 2);
            catch err
                threw = true;  msg = err.message;
            end
            testCase.verifyTrue(threw, ...
                'a component plane must be refused when vector mode is off');
            testCase.verifySubstring(msg, 'vector diffraction', ...
                'the refusal must say why');
        end

        % ---- the vector/scalar difference, DECOMPOSED ------------------
        function test_vector_scalar_difference_decomposition(testCase)
            % This is what the plane getter was built for.  Tranche 1 could
            % only say the vector/scalar difference on an off-normal train
            % was "believed to be out-of-plane content" -- unverifiable,
            % because the per-plane contribution was not reachable.
            %
            % It is now measurable, and the one-line story turns out to be
            % HALF right.  Two mechanisms, both driven by the out-of-plane
            % content, contribute:
            %
            %   1. POWER REDISTRIBUTION (dominant).  The scalar run seeds
            %      from |RayE|, so ALL the power -- including what is
            %      physically out-of-plane -- propagates in one plane.  The
            %      vector run leaves only the fraction f in Ex.  That is a
            %      near-pure rescale of the same map: corr(Ix, Is) > 0.9999.
            %   2. The out-of-plane components DIFFRACT to their own
            %      pattern, contributing Iy + Iz.
            %
            % So the prediction is Iv ~ f*Is + Iy + Iz, and the naive
            % expectation (difference ~ ||Iy+Iz||) is wrong by about 2x.
            macos.load_rx(rx_fixture_path(testCase.RxFF));
            macos.polarization('off');
            Is = macos.intensity(testCase.FFDet);
            macos.load_rx(rx_fixture_path(testCase.RxFF));
            macos.polarization('on', 'Ex', [1 0], 'Ey', [0 0]);
            macos.vector_diffraction(true);
            Iv = macos.intensity(testCase.FFDet);
            Ix = abs(macos.complex_field(testCase.FFDet, 'plane', 1)).^2;
            Iy = abs(macos.complex_field(testCase.FFDet, 'plane', 2)).^2;
            Iz = abs(macos.complex_field(testCase.FFDet, 'plane', 3)).^2;

            f = sum(Ix(:)) / sum(Is(:));            % in-plane power fraction
            testCase.verifyGreaterThan(1 - f, 1e-4, ...
                'fixture must actually carry out-of-plane content');

            r_raw = tVecChain.relerr(Iv, Is);       % the number to explain
            r_dec = tVecChain.relerr(f*Is + Iy + Iz, Iv);
            testCase.verifyLessThan(r_dec, r_raw/5, sprintf( ...
                ['the two-term decomposition must explain most of the ' ...
                 'vector/scalar difference (raw %.3e, residual %.3e)'], ...
                r_raw, r_dec));
            % measured 2026-07-26 at model 128: raw 2.564e-3 -> 2.898e-4,
            % i.e. ~89%% explained.  Bound generously; the point is the
            % decomposition works, not the third digit.
            testCase.verifyLessThan(r_dec, 1e-3, ...
                'decomposition residual outside its measured bracket');
            % mechanism 1 really is a near-pure rescale
            c = corrcoef(Ix(:), Is(:));
            testCase.verifyGreaterThan(c(1,2), 0.9999, ...
                'Ix must be a near-pure rescale of the scalar map');
        end
    end
end
