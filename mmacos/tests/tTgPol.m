classdef tTgPol < matlab.unittest.TestCase
%TTGPOL  Polarization phase-shifting Twyman-Green as a DM surface gauge.
%   Gates templates/90_polarization/tg_psi_dm -- the rotating-analyzer PSI
%   built from the REAL engine polarizing elements (TrPolarizer 15 /
%   WavePlate 18) on the macos.design.twyman_green rig, with a GridData DM
%   as the test optic.
%
%   WHAT IS ALREADY GATED ELSEWHERE, and is deliberately not repeated here:
%   the polarizing elements themselves (tPolElement), vector propagation
%   across a chain (tVecChain), the twyman_green 'polarizing' option
%   emitting BIT-IDENTICALLY when off and loading clean when on
%   (tBench/test_twyman_green_polarizing), and the pol-component error
%   budget (templates/90_polarization/bench_ifo_pol slice 3).
%
%   WHAT THIS CLASS PINS -- the four things that measurement adds:
%
%   (1) THE ANALYZER BASIS.  An ideal analyzer at angle t projects onto
%       a(t) = cos t*u1 + sin t*u2, and everything downstream is a fixed
%       linear map per ray, so the detector field is BILINEAR in a(t):
%           E(t) = c^2*A + c*s*B + s^2*C,  A = E(0), C = E(90),
%                                          B = 2*E(45) - A - C.
%       Three traces per arm therefore span the whole sweep.  The gate
%       checks synthesized-vs-traced at an angle NOT in the basis, and
%       pins the ONE term that breaks it: the engine projects the
%       analyzer's material axis into each ray's transverse plane and
%       renormalizes, which is nonlinear at O(beta^2) for a ray beta off
%       the element normal.  The beta^2 law is checked on a Kr ladder --
%       a collimated-only test cannot tell "exact" from "never exercised".
%
%   (2) THE SWEEP IS A PURE 2-THETA FRINGE.  I(t) is quartic in (c,s), so
%       it may contain DC, 2t and 4t and NOTHING ABOVE.  The 6t bin is a
%       structural check on the whole chain, not a tolerance.
%
%   (3) THE BEAMSPLITTER ROTATES AN ARM.  The design intent is that the
%       double-passed QWPs leave the arms in ORTHOGONAL linear states.
%       They do not: the test arm's 45-degree beamsplitter reflection and
%       its tilted glass transits are diattenuators (t_s /= t_p) and
%       rotate that arm by ~7.5 degrees.  A waveplate is unitary and
%       cannot make a non-orthogonal pair orthogonal, so nothing
%       downstream repairs it and the four-step estimator -- which assumes
%       an orthogonal-circular pair -- acquires a ~12% phase GAIN.  Gated
%       BOTH ways: the aligned rig recovers a known piston at gain 1, and
%       the unaligned rig does not (so the alignment step cannot be
%       deleted silently).
%
%   (4) UNITS AND SIGN.  A uniform GridData piston dz must recover as
%       4*pi*dz/lambda of fringe phase and must agree with translating the
%       whole test optic by dz -- which is what settles that a grid value
%       is the SURFACE height, with the double pass supplying the 2.
%
%   The full closure against an injected checkerboard (empirical per-ray
%   pupil map, 0.36 nm rms residual) lives in the example, which is run by
%   hand; here the shape gate is a single OFF-CENTRE actuator localized
%   through the affine part of that map, which is cheap and still catches
%   an orientation flip.

    properties (Constant)
        ModelSize = 128
        LAM   = 6.328e-4     % mm
        NGRID = 31
        N_G   = 128
        DX_G  = 0.7          % 128 nodes at 0.7 mm = 88.9 mm span
        NACT  = 8
        PITCH = 7.0          % 8 x 7 mm = 56 mm DM
        QWP   = 0.25
        THETAS = [0 45 90 135]
    end

    properties
        G            % the rig (flat DM)
        AT           % test-arm descriptor, aligned
        AR           % reference-arm descriptor, aligned
        AT0          % test-arm descriptor at the DESIGN azimuths (unaligned)
        AR0
        alphaT       % solved test-arm waveplate azimuth (deg)
        oq           % solved output-QWP azimuth (deg)
        azRot        % the beamsplitter-induced arm rotation (deg)
        wdir         % scratch directory (GridFile= resolves against the cwd)
        oldcd
    end

    methods (TestClassSetup)
        function setupClass(testCase)
            macos.init(testCase.ModelSize);
            % the shipped DM-map helper lives with the template it serves
            tdir = fullfile(fileparts(mfilename('fullpath')), '..', 'templates', ...
                            '90_polarization', 'tg_psi_dm');
            testCase.assertTrue(isfolder(tdir), ...
                sprintf('template dir missing: %s', tdir));
            addpath(tdir);
            testCase.addTeardown(@() rmpath(tdir));

            testCase.wdir = tempname;  mkdir(testCase.wdir);
            testCase.oldcd = pwd;  cd(testCase.wdir);
            testCase.addTeardown(@() cd(testCase.oldcd));
            testCase.addTeardown(@() rmdir(testCase.wdir, 's'));

            macos.write_grid_file('flat.txt', zeros(testCase.N_G));
            testCase.G = testCase.mkrig('flat.txt');
            testCase.G.bt.emit('t.in');  testCase.G.br.emit('r.in');
            testCase.AT0 = testCase.armDesc('t.in', testCase.G.bt, testCase.G.T, 0);
            testCase.AR0 = testCase.armDesc('r.in', testCase.G.br, testCase.G.R, 45);

            % align once; every measurement test uses the aligned rig
            [testCase.alphaT, testCase.oq, testCase.azRot] = testCase.solveAlign();
            testCase.AT = testCase.setAlign(testCase.AT0, testCase.alphaT, testCase.oq);
            testCase.AR = testCase.setAlign(testCase.AR0, 45,             testCase.oq);
        end
    end

    methods (Test)

        % -----------------------------------------------------------------
        function test_tranche1_order_and_the_grid_sees_the_analyzer(testCase)
            % Tranche-1: a polarizing element AFTER the first physical-optics
            % leg transforms rays but never reaches the diffraction grid, and
            % the failure is silent (crossed polarizers take the ray power to
            % 1e-33 while the detector stays at full brightness).  Assert the
            % order from the EMITTED deck, then check the grid actually
            % responds -- the part a source read cannot give.
            for f = {'t.in', 'r.in'}
                D = tTgPol.readDeck(f{1});
                ipol = find(ismember(D.element, {'TrPolarizer','WavePlate'}));
                ipo  = find(~strcmp(D.proptype, 'Geometric'));
                testCase.verifyNotEmpty(ipol);
                testCase.verifyTrue(isempty(ipo) || max(ipol) < min(ipo), ...
                    sprintf('%s: polarizing element after a physical-optics leg', f{1}));
            end
            % Run the tripwire on the UNALIGNED arm.  After alignment each arm
            % leaves CIRCULAR, and a circular state projected on any linear
            % axis carries the same power -- so a single-arm power-vs-analyzer
            % test on the aligned rig is invariant by construction and would
            % pass whether or not the grid saw the analyzer at all.  On the
            % unaligned arm the state is linear and the modulation is real.
            I0  = testCase.gridIntensity(testCase.AT0,  0);
            I90 = testCase.gridIntensity(testCase.AT0, 90);
            ratio = sum(I90(:))/sum(I0(:));
            testCase.verifyGreaterThan(abs(ratio-1), 0.1, ...
                'the diffraction grid does not respond to the analyzer');
        end

        % -----------------------------------------------------------------
        function test_detector_registration_between_arms(testCase)
            % A PSI that adds two arms pixel by pixel needs them on the same
            % pixels.  Flux-WEIGHTED centroid, not a thresholded mask centroid:
            % the latter is quantized to whole pixels and would report an
            % exact zero whatever the real misregistration was.
            [dxT, cT, rT] = testCase.detGeom(testCase.AT);
            [dxR, cR, rR] = testCase.detGeom(testCase.AR);
            testCase.verifyLessThan(abs(dxT-dxR)/dxT, 1e-6);
            testCase.verifyLessThan(norm(cT-cR), 0.05);
            testCase.verifyLessThan(abs(rT-rR), 0.05);
        end

        % -----------------------------------------------------------------
        function test_beamsplitter_rotates_an_arm_and_the_solve_nulls_it(testCase)
            % The finding, both halves: the departure is real and large, and
            % turning the arm waveplate restores an orthogonal-circular pair.
            testCase.verifyGreaterThan(abs(testCase.azRot), 1, ...
                'expected a several-degree beamsplitter rotation to correct');
            st = testCase.armState(testCase.AT, testCase.AT.iOQ);
            sr = testCase.armState(testCase.AR, testCase.AR.iOQ);
            testCase.verifyLessThan(abs(st'*sr)/(norm(st)*norm(sr)), 1e-4, ...
                'arms are not orthogonal after the solve');
            % and each is CIRCULAR: |b/a| = 1 with a +-90 degree phase
            for e = {st, sr}
                r = e{1}(2)/e{1}(1);
                testCase.verifyEqual(abs(r), 1, 'AbsTol', 1e-4);
                testCase.verifyEqual(abs(rad2deg(angle(r))), 90, 'AbsTol', 0.05);
            end
            % opposite handedness -- an orthogonal pair of circular states
            testCase.verifyLessThan(imag(st(2)/st(1))*imag(sr(2)/sr(1)), 0);
        end

        % -----------------------------------------------------------------
        function test_analyzer_basis_spans_the_sweep(testCase)
            % Three traces per arm reproduce the detector field at an angle
            % that is NOT one of the three.
            S = testCase.basis(testCase.AT, []);
            for th = [23.7 137.2]
                Ed = testCase.armField(testCase.AT, th, []);
                rel = max(abs(reshape(tTgPol.synth(S,th)-Ed,[],1)))/max(abs(Ed(:)));
                testCase.verifyLessThan(rel, 1e-8, ...
                    sprintf('analyzer basis fails at theta = %g', th));
            end
        end

        % -----------------------------------------------------------------
        function test_analyzer_basis_error_follows_the_beta_squared_law(testCase)
            % NON-VACUITY for the test above.  Give the test optic real power
            % so the analyzer leg is no longer collimated; the residual must
            % then appear and grow as beta^2, beta = the largest ray angle off
            % the analyzer normal.  Without this, "exact" is indistinguishable
            % from "the term was never exercised".
            b = zeros(1,2);  rel = zeros(1,2);
            for k = 1:2
                Kr = [4000 2000];
                Gk = macos.design.twyman_green('polarizing', true, ...
                    'ngridpts', testCase.NGRID, 'to_Kr', Kr(k), 'qwp_ret', testCase.QWP);
                Gk.bt.emit('nv.in');
                Ak = testCase.setAlign(testCase.armDesc('nv.in', Gk.bt, Gk.T, 0), ...
                                       testCase.alphaT, testCase.oq);
                Ed = testCase.armField(Ak, 23.7, []);
                rel(k) = max(abs(reshape(tTgPol.synth(testCase.basis(Ak,[]),23.7)-Ed,[],1))) ...
                         / max(abs(Ed(:)));
                b(k) = testCase.analyzerBeta(Ak);
            end
            testCase.verifyGreaterThan(rel(2), 1e-6, ...
                'the powered rig shows no synthesis error to detect');
            q = (rel(1)/rel(2)) / (b(1)/b(2))^2;
            testCase.verifyEqual(q, 1, 'AbsTol', 0.05, ...
                'synthesis error does not follow the beta^2 law');
        end

        % -----------------------------------------------------------------
        function test_sweep_is_a_pure_two_theta_fringe(testCase)
            % I(theta) is quartic in (cos t, sin t), so DC + 2t + 4t and
            % NOTHING above -- the analyzer has to behave as an ideal rank-1
            % projector for that to hold.
            %
            % The frames must come from DIRECT TRACES.  A frame synthesized
            % from the quadratic basis is a degree-2 trig polynomial in 2t BY
            % CONSTRUCTION, so its 6t bin is zero by algebra and would pass
            % this test against any engine at all -- the exact shape of
            % vacuous gate the "meaningful tests" rule is about.  The
            % synthesized stack is computed too, and the two must agree on
            % the 4t term, which is what ties the cheap path to the honest
            % one.
            St = testCase.basis(testCase.AT, []);
            Sr = testCase.basis(testCase.AR, []);
            n = 12;  th = (0:n-1)/n*180;
            Id = zeros(size(St.A,1), size(St.A,2), n);
            Is = Id;
            for k = 1:n
                Ed = testCase.armField(testCase.AT, th(k), []) ...
                   + testCase.armField(testCase.AR, th(k), []);
                Id(:,:,k) = sum(abs(Ed).^2, 3);
                Is(:,:,k) = tTgPol.frame(St, Sr, th(k));
            end
            m = mean(Id,3) > 0.1*max(reshape(mean(Id,3),[],1));
            [vd, h4d, h6d] = tTgPol.harmonics(Id, m);
            [~,  h4s, h6s] = tTgPol.harmonics(Is, m);
            testCase.verifyGreaterThan(vd, 0.99, 'fringe visibility');
            testCase.verifyLessThan(h6d, 1e-8, ...
                sprintf('traced sweep has content above 4 theta (%.2e)', h6d));
            testCase.verifyEqual(h4d, h4s, 'RelTol', 0.05, ...
                'traced and synthesized 4-theta content disagree');
            % and the synthesized stack really is structurally clean, which is
            % why it cannot serve as the gate
            testCase.verifyLessThan(h6s, 1e-12);
        end

        % -----------------------------------------------------------------
        function test_grid_value_is_a_surface_height(testCase)
            % A uniform grid piston dz must read 4*pi*dz/lambda of fringe
            % phase, and must agree with translating the whole optic by dz.
            dz = 20e-6;  expect = 4*pi*dz/testCase.LAM;
            macos.write_grid_file('pist.txt', dz*ones(testCase.N_G));
            Gp = testCase.mkrig('pist.txt');  Gp.bt.emit('p.in');
            Ap = testCase.setAlign(testCase.armDesc('p.in', Gp.bt, Gp.T, 0), ...
                                   testCase.alphaT, testCase.oq);
            Sr = testCase.basis(testCase.AR, []);
            Sb = testCase.basis(testCase.AT, []);
            [d_grid, m] = testCase.psiDiff(testCase.basis(Ap,[]), Sb, Sr);
            At = testCase.AT;  At.shift = struct('elt', testCase.G.T.iTO, 'dz', dz);
            d_shift = testCase.psiDiff(testCase.basis(At,[]), Sb, Sr);
            testCase.verifyEqual(abs(median(d_grid(m)))/expect, 1, 'RelTol', 2e-3, ...
                'recovered piston has the wrong scale');
            testCase.verifyEqual(median(d_grid(m)), median(d_shift(m)), ...
                'AbsTol', 1e-3*expect, ...
                'a uniform grid piston and a rigid optic shift disagree');
        end

        % -----------------------------------------------------------------
        function test_the_alignment_step_is_load_bearing(testCase)
            % The counterfactual for the alignment: the SAME calibration on
            % the rig at its design azimuths must be materially wrong, or the
            % alignment could be deleted and every other gate would still pass.
            % Also the reason visibility is not the alignment metric: it barely
            % moves while the scale moves by ~12%.
            dz = 20e-6;  expect = 4*pi*dz/testCase.LAM;
            macos.write_grid_file('pist2.txt', dz*ones(testCase.N_G));
            Gp = testCase.mkrig('pist2.txt');  Gp.bt.emit('p2.in');
            Ap0 = testCase.armDesc('p2.in', Gp.bt, Gp.T, 0);   % design azimuths
            [d0, m0, v0] = testCase.psiDiff(testCase.basis(Ap0,[]), ...
                                            testCase.basis(testCase.AT0,[]), ...
                                            testCase.basis(testCase.AR0,[]));
            gain0 = abs(median(d0(m0)))/expect;
            testCase.verifyGreaterThan(abs(gain0-1), 0.05, ...
                'the unaligned rig is not measurably wrong -- alignment gate is vacuous');
            % contrast barely notices
            testCase.verifyGreaterThan(median(v0(m0)), 0.99, ...
                'visibility should stay high even misaligned -- that is the point');
        end

        % -----------------------------------------------------------------
        function test_two_actuators_calibrate_then_verify(testCase)
            % Shape, orientation, position and amplitude in one, WITHOUT
            % assuming an array convention the engine does not guarantee.
            %
            % doc/opd_conventions.md section 2 is explicit: the diffraction /
            % complex-field arrays "carry their own parity" and it is
            % DECK-DEPENDENT (the image flips through each intermediate
            % focus), so the row/column-to-(x,y) mapping must be PROBED, not
            % asserted.  A test that hard-coded one of the eight candidates
            % would be gating a fixture, not the engine.
            %
            % So do what a bench does: CALIBRATE on one known poke and VERIFY
            % on a second, different one.  Resolving the eightfold ambiguity
            % on actuator A costs the test nothing it should be checking;
            % actuator B then has to land in the right place under that same
            % choice, which still catches a wrong magnification, a shear, a
            % broken pupil map or a sign error in the recovery.
            na = testCase.NACT;
            [~, info] = dm_influence_map(testCase.N_G, testCase.DX_G, ...
                'nact',na, 'pitch',testCase.PITCH, 'pattern','zero');
            Sr = testCase.basis(testCase.AR, []);
            Sb = testCase.basis(testCase.AT, []);
            [L, t0, dxp] = testCase.affinePupilMap(testCase.AT, testCase.G.T);

            actA = [2 3];  actB = [6 7];      % distinct, and not related by
                                              % any of the eight symmetries
            [pkA, pixA] = testCase.pokePeak(actA, 150e-6, Sb, Sr);
            [pkB, pixB] = testCase.pokePeak(actB, 150e-6, Sb, Sr);
            xyA = [info.xact(actA(1)); info.xact(actA(2))];
            xyB = [info.xact(actB(1)); info.xact(actB(2))];

            % amplitude: the pupil imaging smooths a single actuator, but may
            % not lose or gain more than 15% of it
            testCase.verifyEqual(1e6*pkA, 150, 'RelTol', 0.15, 'actuator A amplitude');
            testCase.verifyEqual(1e6*pkB, 150, 'RelTol', 0.15, 'actuator B amplitude');

            % CALIBRATE the array convention on A
            cands = {[1 1 0], [1 -1 0], [-1 1 0], [-1 -1 0], ...
                     [1 1 1], [1 -1 1], [-1 1 1], [-1 -1 1]};
            bestd = inf;  bestc = 1;
            for k = 1:numel(cands)
                e = norm(tTgPol.mapPix(pixA, cands{k}, dxp, L, t0) - xyA);
                if e < bestd, bestd = e;  bestc = k; end
            end
            testCase.verifyLessThan(bestd, 0.5*testCase.PITCH, ...
                'no array convention puts actuator A at its own position');

            % VERIFY B under that SAME convention
            xyBm = tTgPol.mapPix(pixB, cands{bestc}, dxp, L, t0);
            testCase.verifyLessThan(norm(xyBm - xyB), 0.5*testCase.PITCH, ...
                sprintf(['actuator B at [%.2f %.2f] mm under the convention ' ...
                         'calibrated on A; truth [%.2f %.2f] mm'], ...
                        xyBm(1), xyBm(2), xyB(1), xyB(2)));

            % and the SEPARATION is convention-free -- a pure scale check
            testCase.verifyEqual(norm(xyBm - tTgPol.mapPix(pixA,cands{bestc},dxp,L,t0)), ...
                norm(xyB - xyA), 'RelTol', 0.15, ...
                'actuator separation is wrong -- the pupil magnification is off');
        end
    end

    % =====================================================================
    methods (Access = private)
        function G = mkrig(testCase, gf)
            G = macos.design.twyman_green('polarizing', true, ...
                'ngridpts', testCase.NGRID, 'qwp_ret', testCase.QWP, ...
                'pol_in_deg', 45, 'qwp_test_deg', 0, 'qwp_ref_deg', 45, ...
                'out_qwp_deg', 0, 'analyzer_deg', 0, ...
                'to_grid_file', gf, 'to_grid_n', testCase.N_G, ...
                'to_grid_dx', testCase.DX_G);
        end

        function A = armDesc(~, rx, b, ix, base_deg)
            nm = {b.E.name};
            A = struct('rx', rx, 'b', b, 'iPol', find(strcmp(nm,'PolIn'),1), ...
                'iQ', find(contains(nm,'QWP') & ~strcmp(nm,'OutQWP')), ...
                'base', base_deg, 'qwp_deg', base_deg, 'oq_deg', 0, ...
                'iTO', [], 'iRC', ix.iRC, 'iOQ', ix.iOutQWP, ...
                'iAn', ix.iAnalyzer, 'iDET', ix.iDET, 'shift', []);
            if isfield(ix,'iTO'), A.iTO = ix.iTO; end
        end

        function A = setAlign(~, A, qwp_deg, oq_deg)
            A.qwp_deg = qwp_deg;  A.oq_deg = oq_deg;
        end

        function [alpha, oq, rot] = solveAlign(testCase)
            az_r  = testCase.armAzimuth(testCase.AR0, 45);
            az_t0 = testCase.armAzimuth(testCase.AT0, 0);
            want  = az_r - 90;
            rot   = az_t0 - want;
            al = [0, -0.5*rot];  az = [az_t0, testCase.armAzimuth(testCase.AT0, al(2))];
            for it = 1:3
                if abs(az(end)-want) < 1e-7, break; end
                sl = (az(end)-az(end-1))/(al(end)-al(end-1));
                al(end+1) = al(end) + (want-az(end))/sl; %#ok<AGROW>
                az(end+1) = testCase.armAzimuth(testCase.AT0, al(end)); %#ok<AGROW>
            end
            alpha = al(end);  oq = az_r + 45;
        end

        function loadArm(testCase, A, an_deg, grid)
            macos.load_rx(A.rx);  b = A.b;
            if ~isempty(A.shift)
                p = macos.get_elt_psi(A.shift.elt);  v = macos.get_elt_vpt(A.shift.elt);
                macos.set_elt_vpt(A.shift.elt, v + A.shift.dz*p);
            end
            if nargin >= 4 && ~isempty(grid)
                macos.set_elt_grid(A.iTO, macos.get_elt_grid_spacing(A.iTO), grid);
            end
            macos.polarizer(A.iPol, 'axis', tTgPol.lax(b.E(A.iPol).psi, 45));
            qa = tTgPol.lax(b.E(A.iQ(1)).psi, A.qwp_deg);
            for j = 1:2
                macos.waveplate(A.iQ(j), 'axis', qa, 'retardance', testCase.QWP);
            end
            macos.waveplate(A.iOQ, 'axis', tTgPol.lax(b.E(A.iOQ).psi, A.oq_deg), ...
                            'retardance', testCase.QWP);
            macos.polarizer(A.iAn, 'axis', tTgPol.lax(b.E(A.iAn).psi, an_deg));
            macos.polarization('on', 'Ex',[1/sqrt(2) 0], 'Ey',[1/sqrt(2) 0]);
            macos.vector_diffraction(true);
        end

        function E = armField(testCase, A, an_deg, grid)
            testCase.loadArm(A, an_deg, grid);
            E = cat(3, macos.complex_field(A.iDET,'plane',1), ...
                       macos.complex_field(A.iDET,'plane',2), ...
                       macos.complex_field(A.iDET,'plane',3));
        end

        function I = gridIntensity(testCase, A, an_deg)
            testCase.loadArm(A, an_deg, []);
            I = macos.intensity(A.iDET);
        end

        function S = basis(testCase, A, grid)
            E0  = testCase.armField(A,  0, grid);
            E45 = testCase.armField(A, 45, grid);
            E90 = testCase.armField(A, 90, grid);
            S = struct('A', E0, 'C', E90, 'B', 2*E45 - E0 - E90);
        end

        function e = armState(testCase, A, iElt)
            testCase.loadArm(A, 0, []);
            macos.trace(iElt);  f = macos.ray_field(iElt);
            ok = f.status == 0;
            psi = A.b.E(iElt).psi(:);
            u1 = macos.design.Bench.perp(psi);  u2 = cross(psi, u1);
            e1 = f.Ex*u1(1) + f.Ey*u1(2) + f.Ez*u1(3);
            e2 = f.Ex*u2(1) + f.Ey*u2(2) + f.Ez*u2(3);
            r = e2(ok)./e1(ok);  a = median(abs(e1(ok)));
            e = [a; a*(median(real(r)) + 1i*median(imag(r)))];
        end

        function az = armAzimuth(testCase, A, qwp_deg)
            A.qwp_deg = qwp_deg;
            e = testCase.armState(A, A.iRC);
            az = 0.5*atan2d(2*real(conj(e(1))*e(2)), abs(e(1))^2 - abs(e(2))^2);
        end

        function b = analyzerBeta(testCase, A)
            macos.load_rx(A.rx);
            st = macos.trace(A.iAn);  ri = macos.get_ray_info(st.nRays);
            ok = ri.ok_trace(:) & ri.ok_pass(:);
            psiA = A.b.E(A.iAn).psi(:);
            b = max(acos(min(abs(psiA.' * ri.dir(:,ok)), 1)));
        end

        function [dphi, m, vis] = psiDiff(testCase, Sx, Sb, Sr)
            [px, I, vis] = tTgPol.psiFrames(Sx, Sr, testCase.THETAS);
            pb = tTgPol.psiFrames(Sb, Sr, testCase.THETAS);
            dphi = angle(exp(1i*(px - pb)));
            m = I{1} > 0.1*max(I{1}(:));
        end

        function [dxp, cen, rad] = detGeom(testCase, A)
            macos.load_rx(A.rx);
            I = macos.intensity(A.iDET);
            dxp = macos.dx_at(A.iDET, 'mm');
            N = size(I,1);  [cg, rg] = meshgrid(1:N, 1:N);
            w = I/sum(I(:));
            cen = [sum(cg(:).*w(:)); sum(rg(:).*w(:))];
            rad = sqrt(sum(((cg(:)-cen(1)).^2 + (rg(:)-cen(2)).^2).*w(:)));
        end

        function [pkval, pix] = pokePeak(testCase, act, amp, Sb, Sr)
            % Drive ONE actuator, recover differentially, and return the peak
            % height and its (col-cx, row-cy) pixel offset from the beam
            % centroid.
            C = zeros(testCase.NACT);  C(act(1), act(2)) = amp;
            M = dm_influence_map(testCase.N_G, testCase.DX_G, 'nact',testCase.NACT, ...
                                 'pitch',testCase.PITCH, 'act',C);
            [d, m] = testCase.psiDiff(testCase.basis(testCase.AT, M), Sb, Sr);
            h = d * testCase.LAM/(4*pi);
            [pkval, ipk] = max(abs(h(m)));
            idx = find(m);  N = size(h,1);
            [cg, rg] = meshgrid(1:N, 1:N);
            pix = [cg(idx(ipk)) - mean(cg(m)); rg(idx(ipk)) - mean(rg(m))];
        end

        function [L, t0, dxp] = affinePupilMap(~, A, ix)
            % detector mm (about the chief) -> DM mm (about the DM vertex),
            % measured from the trace: one pair per surviving ray.
            macos.load_rx(A.rx);
            s1 = macos.trace(ix.iTO);   ito  = macos.get_ray_info(s1.nRays);
            s2 = macos.trace(ix.iDET);  idet = macos.get_ray_info(s2.nRays);
            ok = ito.ok_trace(:) & ito.ok_pass(:) & idet.ok_trace(:) & idet.ok_pass(:);
            p1 = macos.get_elt_psi(ix.iTO);  v1 = macos.get_elt_vpt(ix.iTO);
            u1 = macos.design.Bench.perp(p1);  w1 = cross(p1, u1);
            xy_to = [u1.'; w1.'] * (ito.pos - v1);
            p2 = macos.get_elt_psi(ix.iDET);
            u2 = macos.design.Bench.perp(p2);  w2 = cross(p2, u2);
            xy_d = [u2.'; w2.'] * (idet.pos - idet.pos(:,1));
            Aaf = [xy_d(:,ok).' ones(nnz(ok),1)] \ xy_to(:,ok).';
            L = Aaf(1:2,:).';  t0 = Aaf(3,:).';
            dxp = macos.dx_at(ix.iDET, 'mm');
        end
    end

    % =====================================================================
    methods (Static, Access = private)
        function a = lax(psi, deg)
            u1 = macos.design.Bench.perp(psi(:));  u2 = cross(psi(:), u1);
            a = cosd(deg)*u1 + sind(deg)*u2;  a = a(:).';
        end

        function E = synth(S, th)
            c = cosd(th);  s = sind(th);
            E = c^2*S.A + c*s*S.B + s^2*S.C;
        end

        function I = frame(Sx, Sr, th)
            I = sum(abs(tTgPol.synth(Sx,th) + tTgPol.synth(Sr,th)).^2, 3);
        end

        function [psi, I, vis] = psiFrames(Sx, Sr, th)
            I = cell(1,4);
            for q = 1:4, I{q} = tTgPol.frame(Sx, Sr, th(q)); end
            psi = atan2(I{2}-I{4}, I{1}-I{3});
            % visibility from the FITTED harmonic; a min/max over four samples
            % of a cos(2t) fringe touches its extremes only by luck and reads
            % low, which looks like a contrast loss that is not there.
            t2 = 2*deg2rad(th(:));
            M = [ones(4,1) cos(t2) sin(t2)];
            Sm = zeros(4, numel(I{1}));
            for q = 1:4, Sm(q,:) = I{q}(:).'; end
            c = M \ Sm;
            vis = reshape(sqrt(c(2,:).^2 + c(3,:).^2)./max(abs(c(1,:)),eps), size(I{1}));
        end

        function xy = mapPix(pix, c, dxp, L, t0)
            % One of the eight (row,col)->(x,y) array conventions applied to a
            % pixel offset, then the trace-measured affine pupil map.
            % C = [sign1 sign2 swap].
            p = [c(1)*pix(1); c(2)*pix(2)];
            if c(3), p = flipud(p); end
            xy = L*(p*dxp) + t0;
        end

        function [vis, h4, h6] = harmonics(I, m)
            % Visibility and the 4t / 6t content of a stack of analyzer frames
            % spanning [0,180) uniformly; FFT bin k holds the 2*(k-1)-theta
            % harmonic.
            n = size(I,3);
            F = fft(I, [], 3)/n;
            h0 = abs(F(:,:,1));  h1 = 2*abs(F(:,:,2));
            h2 = 2*abs(F(:,:,3));  h3 = 2*abs(F(:,:,4));
            vis = median(h1(m)./h0(m));
            h4  = mean(h2(m))/mean(h1(m));
            h6  = mean(h3(m))/mean(h1(m));
        end

        function D = readDeck(fn)
            L = regexp(fileread(fn), '\n', 'split');
            D = struct('element',{{}}, 'surface',{{}}, 'proptype',{{}});
            for k = 1:numel(L)
                tok = regexp(L{k}, '^\s*(\w+)=\s*(\S+)', 'tokens', 'once');
                if isempty(tok), continue; end
                switch tok{1}
                case 'Element',  D.element{end+1}  = tok{2};
                case 'Surface',  D.surface{end+1}  = tok{2};
                case 'PropType', D.proptype{end+1} = tok{2};
                end
            end
        end
    end
end
