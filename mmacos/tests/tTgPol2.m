classdef tTgPol2 < matlab.unittest.TestCase
%TTGPOL2  Polarization PSI Twyman-Green with a REAL polarizing beamsplitter.
%   Gates templates/90_polarization/tg_psi_dm_v2 -- the v1 rig
%   (templates/90_polarization/tg_psi_dm, gated by tTgPol) with its
%   perfect-conductor plate + compensator + ideal per-arm TrPolarizer
%   replaced by a CEMENTED MacNeille CUBE: one coated interface at 45 deg
%   between two prisms of the same glass, so the split is real coating
%   physics rather than bookkeeping.
%
%   v1 IS NOT SUPERSEDED.  tTgPol still runs, unchanged, and the plate rig is
%   built here as the counterexample several gates below are scored against.
%
%   WHAT IS ALREADY GATED ELSEWHERE and deliberately not repeated: the
%   polarizing elements (tPolElement), vector propagation (tVecChain), the
%   coated-branch conventions and the radiometric factor (tJonesPupil,
%   tPolRadiometric), the published-Mueller anchor (tPolExternal), and the
%   PSI machinery itself (tTgPol).
%
%   WHAT THIS CLASS PINS -- five things the cube adds:
%
%   (1) THE STACK REALLY REACHES THE ENGINE.  The deck writes "Coating=" in
%       OPTICAL thickness (waves at the deck Wavelen); the parser scales by
%       Wavelen/IndRef; coat_get returns PHYSICAL thickness.  The round trip
%       is the one place those two units meet, so it is a real check.
%
%   (2) THE COATED 45-DEG DIAGONAL IS THE TEXTBOOK STACK.  Engine R and T,
%       measured per polarization from the two arms' own decks, against
%       macos.design.thinfilm_rt -- Macleod's characteristic matrix, written
%       from the textbook and NEVER transcribed from elemsub.F (an
%       "analytic" copied out of the engine is circular in exactly the
%       coefficient it should check: REVIEW_POL_SP_SIGN_2026-07-27).  R + T
%       = 1 is a free cross-deck check: the two arms measure R and T of the
%       SAME coating.
%
%   (3) THE MacNEILLE p-NULL NEEDS MORE THAN BREWSTER.  Brewster at the H/L
%       interfaces equalizes the tilted p admittances, so for p the stack is
%       ONE HOMOGENEOUS SLAB -- and its boundaries with the PRISM are not
%       Brewster.  An even number of quarter waves makes that slab a
%       half-wave absentee (R_p = 0); an odd number makes it a quarter-wave
%       layer (R_p = 2.1%).  Both satisfy the textbook condition, so the
%       odd-QW stack is the non-vacuity counterexample for every p-null
%       assertion here.
%
%   (4) THE CUBE NEEDS NO ALIGNMENT AND NO COMPENSATOR.  Each arm's state
%       sits ON a coating eigenaxis, where a diattenuator cannot rotate it,
%       so the 7.479-degree arm rotation v1 has to null is structurally
%       absent; and every cube traversal is a/2 -> diagonal -> a/2 from
%       whichever port, so the two arms' glass paths are identical by
%       construction.  Gated BOTH ways -- v1 must still be misaligned here,
%       or the comparison is vacuous.
%
%   (5) AND IT INVERTS THE v1 ERROR BUDGET.  On the REFLECTING return the
%       coating re-projects the arm onto its own eigenaxis, so a waveplate
%       azimuth error is cleaned to r_p (zero); on the TRANSMITTING return
%       it is cleaned only to the extinction ratio t_s/t_p.  Either way the
%       error costs CONTRAST, not SCALE -- the opposite of v1, where it cost
%       12% of scale and 0.2% of contrast.
%
%   The full DM-map closure lives in the example (run by hand).

    properties (Constant)
        ModelSize = 128
        LAM   = 6.328e-4     % mm
        NGRID = 31
        N_G   = 128
        DX_G  = 0.7          % 128 nodes at 0.7 mm = 88.9 mm span
        QWP   = 0.25
        THETAS = [0 45 90 135]
    end

    properties
        PBS          % macos.design.pbs_macneille -- the installed design
        PBSq         % the odd-QW counterexample stack
        G            % the v2 cube rig (flat test optic)
        AT           % test-arm descriptor  (transmits out, reflects back)
        AR           % reference-arm descriptor (reflects out, transmits back)
        V1           % the v1 plate rig, for the non-vacuity comparisons
        B1T
        B1R
        wdir
        oldcd
    end

    methods (TestClassSetup)
        function setupClass(testCase)
            macos.init(testCase.ModelSize);
            testCase.wdir = tempname;  mkdir(testCase.wdir);
            testCase.oldcd = pwd;  cd(testCase.wdir);
            testCase.addTeardown(@() cd(testCase.oldcd));
            testCase.addTeardown(@() rmdir(testCase.wdir, 's'));

            testCase.PBS  = macos.design.pbs_macneille();
            testCase.PBSq = macos.design.pbs_macneille('design','qw');

            macos.write_grid_file('flat.txt', zeros(testCase.N_G));
            testCase.G = testCase.mkcube('flat.txt');
            testCase.G.bt.emit('t2.in');  testCase.G.br.emit('r2.in');
            testCase.AT = testCase.setAlign( ...
                testCase.armDesc('t2.in', testCase.G.bt, testCase.G.T, ...
                                 testCase.G.P.qwp_test_deg), ...
                testCase.G.P.qwp_test_deg, testCase.G.P.out_qwp_deg);
            testCase.AR = testCase.setAlign( ...
                testCase.armDesc('r2.in', testCase.G.br, testCase.G.R, ...
                                 testCase.G.P.qwp_ref_deg), ...
                testCase.G.P.qwp_ref_deg, testCase.G.P.out_qwp_deg);

            testCase.V1 = testCase.mkplate('flat.txt');
            testCase.V1.bt.emit('t1.in');  testCase.V1.br.emit('r1.in');
            testCase.B1T = testCase.armDesc('t1.in', testCase.V1.bt, testCase.V1.T, ...
                                            testCase.V1.P.qwp_test_deg);
            testCase.B1R = testCase.armDesc('r1.in', testCase.V1.br, testCase.V1.R, ...
                                            testCase.V1.P.qwp_ref_deg);
        end
    end

    methods (Test)

        % -----------------------------------------------------------------
        function test_the_emitted_stack_round_trips_through_the_engine(testCase)
            % Coating= is written in waves at the deck Wavelen and the parser
            % scales it by Wavelen/IndRef; coat_get hands back PHYSICAL
            % thickness.  A round trip therefore exercises that conversion --
            % it is not a tautology.  Also pin the design itself: the
            % MacNeille condition and the quarter-wave-AT-ANGLE thickness.
            S = testCase.PBS;
            testCase.verifyEqual(S.theta_H + S.theta_L, 90, 'AbsTol', 1e-9, ...
                'the H/L interfaces are not at Brewster');
            testCase.verifyEqual(S.n_glass, S.n_glass_mn, 'RelTol', 1e-12);
            testCase.verifyEqual(mod(S.qw_total, 2), 0, 'AbsTol', 1e-12, ...
                'the symmetric design must be an EVEN number of quarter waves');

            macos.load_rx('t2.in');
            iD = testCase.G.T.iPBSf(2);   iF = testCase.G.T.iPBSf(1);
            c = macos.coating(iD);
            testCase.verifyEqual(c.n_layer, size(S.layers,1));
            testCase.verifyEqual(c.thickness(:), S.thk(:), 'RelTol', 1e-12, ...
                'Coating= thickness does not round-trip through the parser');
            testCase.verifyEqual(c.index(:), S.layers(:,1), 'AbsTol', 1e-12);
            testCase.verifyEqual(c.extinc(:), zeros(size(S.thk(:))), 'AbsTol', 0);
            % and the faces carry their single-layer MgF2 quarter wave
            cf = macos.coating(iF);
            testCase.verifyEqual(cf.n_layer, 1);
            testCase.verifyEqual(cf.thickness(1), testCase.LAM/(4*1.38), 'RelTol', 1e-12);
            % nothing exceeds the engine's mCoat = 10 (the Rx parser does NOT
            % bound-check this -- an over-long stack loads silently and
            % corrupts memory, which is why pbs_macneille asserts)
            testCase.verifyLessThanOrEqual(c.n_layer, 10);
        end

        % -----------------------------------------------------------------
        function test_engine_coated_diagonal_matches_the_macleod_analytic(testCase)
            % The whole v2 claim.  Probe the diagonal with a PURE s and a PURE
            % p input and take the amplitude ratio ACROSS it, from the two
            % arms' own decks: the test arm's diagonal transmits, the
            % reference arm's reflects, so together they give R and T of the
            % same coating and R + T = 1 becomes a free cross-deck check.
            %
            % Every normalization subtlety collapses here because the cube is
            % CEMENTED (n_inc == n_sub): the engine's radiometric factor
            % sqrt(n_sub*cos_sub/(n_inc*cos_inc)) and Macleod's
            % tangential-vs-Fresnel factor cos_sub/cos_inc are both
            % identically 1.  This is NOT a degenerate check of the coating
            % itself -- the AOI is 45 deg, where s and p differ by three
            % orders of magnitude.
            E = testCase.probeRT('t2.in', testCase.G.T.iPBSf, ...
                                 'r2.in', testCase.G.R.iPBSf);
            A = testCase.PBS.rt;
            testCase.verifyEqual(E.Rs, A.Rs, 'RelTol', 1e-7, 'R_s');
            testCase.verifyEqual(E.Tp, A.Tp, 'RelTol', 1e-7, 'T_p');
            % T_s is ~4e-4 and R at a 45-deg interface varies quadratically
            % with AOI, so the beam's residual divergence shows up here first
            testCase.verifyEqual(E.Ts, A.Ts, 'RelTol', 1e-4, 'T_s');
            testCase.verifyEqual(E.Rs + E.Ts, 1, 'AbsTol', 1e-9, ...
                'R_s + T_s /= 1 -- the two decks disagree about one coating');
            testCase.verifyEqual(E.Rp + E.Tp, 1, 'AbsTol', 1e-9, 'R_p + T_p /= 1');
            % the MacNeille p-null itself
            testCase.verifyLessThan(E.Rp, 1e-9, ...
                'the p-null is not there -- r_p should vanish at Brewster');
            % a real polarizer: three orders of extinction in transmission
            testCase.verifyGreaterThan(E.Tp/E.Ts, 1e3);
        end

        % -----------------------------------------------------------------
        function test_the_p_null_needs_an_even_quarter_wave_stack(testCase)
            % NON-VACUITY for every p-null assertion above.  H(LH)^4
            % satisfies the SAME Brewster condition at every H/L interface --
            % and is a 2.1%-R_p polarizer, because its p slab is an ODD
            % number of quarter waves and so a quarter-wave layer rather than
            % a half-wave absentee.  If this stack passed the p-null gate,
            % that gate would be measuring nothing.
            Gq = testCase.mkcube('flat.txt', 'pbs_coat', testCase.PBSq.layers);
            Gq.bt.emit('tq.in');  Gq.br.emit('rq.in');
            E = testCase.probeRT('tq.in', Gq.T.iPBSf, 'rq.in', Gq.R.iPBSf);
            testCase.verifyEqual(E.Rp, testCase.PBSq.rt.Rp, 'RelTol', 1e-3, ...
                'the engine and the analytic disagree on the odd-QW stack');
            testCase.verifyGreaterThan(E.Rp, 1e-2, ...
                'the odd-QW stack must FAIL the p-null gate, or that gate is vacuous');
            testCase.verifyEqual(E.Rp + E.Tp, 1, 'AbsTol', 1e-9);
        end

        % -----------------------------------------------------------------
        function test_a_bare_cemented_interface_carries_no_light(testCase)
            % The structural check that the COATING is what makes the cube a
            % beamsplitter, not some geometric artefact of the prism: strip
            % the stack and the cemented interface is glass against the SAME
            % glass, i.e. optically nothing.  R = 0 and T = 1 exactly, for
            % both polarizations -- and the consequence is stronger than
            % "the arms come out parallel", which is what a first reading
            % suggests: each arm REFLECTS off the diagonal once (the test arm
            % on its return, the reference arm on the way out), so with R = 0
            % BOTH arms are extinguished and the rig delivers nothing at all.
            % Measure that, and do not report an azimuth for a zero field --
            % arm_state divides by it.
            Gb = testCase.mkcube('flat.txt', 'pbs_coat', zeros(0,3), 'ar_faces', false);
            Gb.bt.emit('tb.in');  Gb.br.emit('rb.in');
            E = testCase.probeRT('tb.in', Gb.T.iPBSf, 'rb.in', Gb.R.iPBSf);
            testCase.verifyEqual(E.Ts, 1, 'AbsTol', 1e-12, 'bare interface: T_s');
            testCase.verifyEqual(E.Tp, 1, 'AbsTol', 1e-12, 'bare interface: T_p');
            testCase.verifyLessThan(E.Rs, 1e-12, 'bare interface: R_s');
            testCase.verifyLessThan(E.Rp, 1e-12, 'bare interface: R_p');
            % ... so both arms go dark.  Score it against the coated rig.
            Ab = testCase.setAlign(testCase.armDesc('tb.in', Gb.bt, Gb.T, ...
                     Gb.P.qwp_test_deg), Gb.P.qwp_test_deg, Gb.P.out_qwp_deg);
            Rb = testCase.setAlign(testCase.armDesc('rb.in', Gb.br, Gb.R, ...
                     Gb.P.qwp_ref_deg),  Gb.P.qwp_ref_deg,  Gb.P.out_qwp_deg);
            Pbare = sum(sum(sum(abs(testCase.armField(Ab,0)).^2))) + ...
                    sum(sum(sum(abs(testCase.armField(Rb,0)).^2)));
            Pcoat = sum(sum(sum(abs(testCase.armField(testCase.AT,0)).^2))) + ...
                    sum(sum(sum(abs(testCase.armField(testCase.AR,0)).^2)));
            testCase.verifyLessThan(Pbare/Pcoat, 1e-9, ...
                sprintf(['an uncoated cemented interface must deliver no light ' ...
                         '(bare/coated = %.3e)'], Pbare/Pcoat));
        end

        % -----------------------------------------------------------------
        function test_the_arms_leave_orthogonal_with_no_alignment_step(testCase)
            % v1 has to solve a +3.768-degree waveplate clock here.  The cube
            % does not, because each arm's state sits ON a coating eigenaxis
            % (the test arm on p, the reference arm on s), where a
            % diattenuator cannot rotate anything.  Gated both ways: v1 must
            % still be several degrees off, or this is not a comparison.
            sep2 = abs(tTgPol2.wrap180(testCase.armAzimuth(testCase.AT, testCase.AT.base) - ...
                                       testCase.armAzimuth(testCase.AR, testCase.AR.base))) - 90;
            testCase.verifyLessThan(abs(sep2), 1e-3, ...
                sprintf('v2 arms %.3e deg from orthogonal', sep2));
            sep1 = abs(tTgPol2.wrap180(testCase.armAzimuth(testCase.B1T, testCase.B1T.base) - ...
                                       testCase.armAzimuth(testCase.B1R, testCase.B1R.base))) - 90;
            testCase.verifyGreaterThan(abs(sep1), 1, ...
                'v1 was supposed to be misaligned -- the comparison is vacuous');
            % and the pair the analyzer sees is orthogonal CIRCULAR, with the
            % output QWP at a DESIGN azimuth rather than a solved one
            st = testCase.armState(testCase.AT, testCase.AT.iOQ);
            sr = testCase.armState(testCase.AR, testCase.AR.iOQ);
            testCase.verifyLessThan(abs(st'*sr)/(norm(st)*norm(sr)), 1e-4);
            for e = {st, sr}
                r = e{1}(2)/e{1}(1);
                testCase.verifyEqual(abs(r), 1, 'AbsTol', 1e-4);
                testCase.verifyEqual(abs(rad2deg(angle(r))), 90, 'AbsTol', 0.05);
            end
            testCase.verifyLessThan(imag(st(2)/st(1))*imag(sr(2)/sr(1)), 0, ...
                'the two circular states have the same handedness');
        end

        % -----------------------------------------------------------------
        function test_the_cube_needs_no_compensator(testCase)
            % Every traversal is a/2 -> diagonal -> a/2 whichever port you
            % enter by, so the arms balance BY CONSTRUCTION.  v1 needs a
            % compensator plate and an internal-V return to do the same job.
            d = abs(testCase.G.bt.path_len - testCase.G.br.path_len);
            testCase.verifyLessThan(d, 1e-9, ...
                sprintf('arm paths differ by %.3e mm', d));
            testCase.verifyEmpty(find(contains({testCase.G.bt.E.name}, 'Comp'), 1), ...
                'the cube rig should carry no compensator');
            % and the two beams land on the same detector pixels
            [dxT, cT, rT] = testCase.detGeom(testCase.AT);
            [dxR, cR, rR] = testCase.detGeom(testCase.AR);
            testCase.verifyEqual(dxT, dxR, 'RelTol', 1e-9);
            testCase.verifyLessThan(norm(cT-cR), 0.05);
            testCase.verifyLessThan(abs(rT-rR), 0.05);
        end

        % -----------------------------------------------------------------
        function test_psi_closure_on_a_known_piston(testCase)
            % Same chain gate as v1: a uniform grid piston dz must recover as
            % 4*pi*dz/lambda, and must agree with translating the whole optic
            % by dz.  On the cube rig this runs at the DESIGN azimuths -- no
            % alignment solve in front of it.
            dz = 20e-6;  expect = 4*pi*dz/testCase.LAM;
            macos.write_grid_file('pist.txt', dz*ones(testCase.N_G));
            Gp = testCase.mkcube('pist.txt');  Gp.bt.emit('p2.in');
            Ap = testCase.setAlign(testCase.armDesc('p2.in', Gp.bt, Gp.T, ...
                     Gp.P.qwp_test_deg), Gp.P.qwp_test_deg, Gp.P.out_qwp_deg);
            Sr = testCase.basis(testCase.AR);
            Sb = testCase.basis(testCase.AT);
            [d_grid, m, v] = testCase.psiDiff(testCase.basis(Ap), Sb, Sr);
            At = testCase.AT;
            At.shift = struct('elt', testCase.G.T.iTO, 'dz', dz);
            d_shift = testCase.psiDiff(testCase.basis(At), Sb, Sr);
            testCase.verifyEqual(abs(median(d_grid(m)))/expect, 1, 'RelTol', 1e-3, ...
                'recovered piston has the wrong scale');
            testCase.verifyEqual(median(d_grid(m)), median(d_shift(m)), ...
                'AbsTol', 1e-3*expect, ...
                'a uniform grid piston and a rigid optic shift disagree');
            testCase.verifyGreaterThan(median(v(m)), 0.999, 'fringe visibility');
        end

        % -----------------------------------------------------------------
        function test_the_reflecting_arm_is_cleaned_and_the_error_costs_contrast(testCase)
            % The v2 error budget, and the inversion of the v1 finding.
            % Turn each arm's waveplate 10 degrees off design:
            %   * the arm that REFLECTS on its return is re-projected onto the
            %     coating's own eigenaxis, so it is cleaned to r_p = 0;
            %   * the arm that TRANSMITS is cleaned only to t_s/t_p;
            %   * v1, with no cleanup at all, moves by ~2x the error.
            % And in v2 the scale does not move at all -- the error shows up
            % as CONTRAST, i.e. on the fringe monitor, which is the opposite
            % of the v1 case where it hid there.
            e = 10;
            az_r = testCase.armAzimuth(testCase.AR, testCase.AR.base);
            az_t = testCase.armAzimuth(testCase.AT, testCase.AT.base);
            sep_refl = abs(tTgPol2.wrap180( ...
                testCase.armAzimuth(testCase.AT, testCase.AT.base + e) - az_r)) - 90;
            sep_tran = abs(tTgPol2.wrap180( ...
                testCase.armAzimuth(testCase.AR, testCase.AR.base + e) - az_t)) - 90;
            az_r1 = testCase.armAzimuth(testCase.B1R, testCase.B1R.base);
            sep_v1 = abs(tTgPol2.wrap180( ...
                testCase.armAzimuth(testCase.B1T, testCase.B1T.base + e) - az_r1)) - 90;

            testCase.verifyLessThan(abs(sep_refl), 1e-3, ...
                'the REFLECTING arm should be cleaned to r_p (= 0)');
            testCase.verifyGreaterThan(abs(sep_tran), 10*abs(sep_refl), ...
                'the transmitting arm cannot be cleaned better than the reflecting one');
            testCase.verifyLessThan(abs(sep_tran), 1, ...
                'the transmitting arm should still be cleaned to ~t_s/t_p');
            testCase.verifyGreaterThan(abs(sep_v1), 5*abs(sep_tran), ...
                'v1 must be materially more sensitive than either v2 arm');

            % scale vs contrast, on the same 20 nm piston
            dz = 20e-6;  expect = 4*pi*dz/testCase.LAM;
            macos.write_grid_file('pist3.txt', dz*ones(testCase.N_G));
            Gp = testCase.mkcube('pist3.txt');  Gp.bt.emit('p3.in');
            Sr = testCase.basis(testCase.AR);
            g = zeros(1,2);  vv = zeros(1,2);
            for q = 1:2
                de = (q-1)*e;
                Ab = testCase.setAlign(testCase.AT, testCase.AT.base + de, testCase.AT.oq_deg);
                Ap = testCase.setAlign(testCase.armDesc('p3.in', Gp.bt, Gp.T, 0), ...
                         Gp.P.qwp_test_deg + de, Gp.P.out_qwp_deg);
                [d, m, v] = testCase.psiDiff(testCase.basis(Ap), testCase.basis(Ab), Sr);
                g(q) = abs(median(d(m)))/expect;   vv(q) = median(v(m));
            end
            testCase.verifyEqual(g(2), 1, 'RelTol', 1e-3, ...
                'a 10-degree waveplate error must NOT move the v2 scale');
            testCase.verifyLessThan(vv(2), vv(1), ...
                'the error has to show up somewhere -- expected as contrast');
        end

        % -----------------------------------------------------------------
        function test_output_port_efficiency_against_the_declared_stack(testCase)
            % A PBS earns its place by putting BOTH returns in the output
            % port.  Score the delivered power against the plate rig, and
            % against the budget the DECLARED stack predicts: per arm,
            % T_p * R_s through the diagonal and four AR'd faces.
            Sb = testCase.basis(testCase.AT);
            Sr = testCase.basis(testCase.AR);
            P2 = sum(sum(sum(abs(Sb.A + Sr.A).^2)));
            S1b = testCase.basis(testCase.setAlign(testCase.B1T, testCase.B1T.base, 0));
            S1r = testCase.basis(testCase.setAlign(testCase.B1R, testCase.B1R.base, 0));
            P1 = sum(sum(sum(abs(S1b.A + S1r.A).^2)));
            testCase.verifyGreaterThan(P2/P1, 2, ...
                sprintf('cube/plate delivered power %.3f -- expected > 2', P2/P1));
            % the declared-stack budget: T of one MgF2 quarter wave on the
            % prism glass, from the same textbook analytic
            ar = macos.design.thinfilm_rt([1.38, testCase.LAM/(4*1.38)], 1.0, ...
                                          testCase.PBS.n_glass, 0, testCase.LAM);
            testCase.verifyGreaterThan(ar.Ts, 0.99, 'the AR face should transmit > 99%');
            testCase.verifyEqual(ar.Ts, ar.Tp, 'RelTol', 1e-12, ...
                'at normal incidence s and p must be identical');
            E = testCase.probeRT('t2.in', testCase.G.T.iPBSf, ...
                                 'r2.in', testCase.G.R.iPBSf);
            testCase.verifyGreaterThan(E.Tp*E.Rs*ar.Ts^4, 0.97, ...
                'per-arm throughput budget');
        end
    end

    % =====================================================================
    methods (Access = private)
        function G = mkcube(testCase, gf, varargin)
            G = macos.design.twyman_green('pbs','cube', 'polarizing',true, ...
                'ngridpts', testCase.NGRID, 'qwp_ret', testCase.QWP, ...
                'to_grid_file', gf, 'to_grid_n', testCase.N_G, ...
                'to_grid_dx', testCase.DX_G, varargin{:});
        end

        function G = mkplate(testCase, gf)
            G = macos.design.twyman_green('polarizing', true, ...
                'ngridpts', testCase.NGRID, 'qwp_ret', testCase.QWP, ...
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

        function loadArm(testCase, A, an_deg)
            macos.load_rx(A.rx);  b = A.b;
            if ~isempty(A.shift)
                p = macos.get_elt_psi(A.shift.elt);  v = macos.get_elt_vpt(A.shift.elt);
                macos.set_elt_vpt(A.shift.elt, v + A.shift.dz*p);
            end
            macos.polarizer(A.iPol, 'axis', tTgPol2.lax(b.E(A.iPol).psi, 45));
            qa = tTgPol2.lax(b.E(A.iQ(1)).psi, A.qwp_deg);
            for j = 1:2
                macos.waveplate(A.iQ(j), 'axis', qa, 'retardance', testCase.QWP);
            end
            macos.waveplate(A.iOQ, 'axis', tTgPol2.lax(b.E(A.iOQ).psi, A.oq_deg), ...
                            'retardance', testCase.QWP);
            macos.polarizer(A.iAn, 'axis', tTgPol2.lax(b.E(A.iAn).psi, an_deg));
            macos.polarization('on', 'Ex',[1/sqrt(2) 0], 'Ey',[1/sqrt(2) 0]);
            macos.vector_diffraction(true);
        end

        function E = armField(testCase, A, an_deg)
            testCase.loadArm(A, an_deg);
            E = cat(3, macos.complex_field(A.iDET,'plane',1), ...
                       macos.complex_field(A.iDET,'plane',2), ...
                       macos.complex_field(A.iDET,'plane',3));
        end

        function e = armState(testCase, A, iElt)
            testCase.loadArm(A, 0);
            macos.trace(iElt);  f = macos.ray_field(iElt);
            ok = f.status == 0;
            psi = A.b.E(iElt).psi(:);
            u1 = macos.design.Bench.perp(psi);  u2 = cross(psi, u1);
            e1 = f.Ex*u1(1) + f.Ey*u1(2) + f.Ez*u1(3);
            e2 = f.Ex*u2(1) + f.Ey*u2(2) + f.Ez*u2(3);
            r  = e2(ok)./e1(ok);
            a  = median(abs(e1(ok)));
            e  = [a; a*(median(real(r)) + 1i*median(imag(r)))];
        end

        function az = armAzimuth(testCase, A, qwp_deg)
            A.qwp_deg = qwp_deg;
            e = testCase.armState(A, A.iRC);
            az = 0.5*atan2d(2*real(conj(e(1))*e(2)), abs(e(1))^2 - abs(e(2))^2);
        end

        function S = basis(testCase, A)
            E0  = testCase.armField(A, 0);
            E45 = testCase.armField(A, 45);
            E90 = testCase.armField(A, 90);
            S = struct('A', E0, 'C', E90, 'B', 2*E45 - E0 - E90);
        end

        function [d, m, v] = psiDiff(testCase, Sx, Sb, Sr)
            [px, v, m] = testCase.psiFrames(Sx, Sr);
            pb = testCase.psiFrames(Sb, Sr);
            d = angle(exp(1i*(px - pb)));  d(~m) = 0;
        end

        function [psi, vis, msk] = psiFrames(testCase, Sx, Sr)
            th = testCase.THETAS;  nt = numel(th);
            F = cell(1,nt);
            for q = 1:nt
                F{q} = sum(abs(tTgPol2.synth(Sx,th(q)) + tTgPol2.synth(Sr,th(q))).^2, 3);
            end
            psi = atan2(F{2}-F{4}, F{1}-F{3});
            t2 = 2*deg2rad(th(:));  M = [ones(nt,1) cos(t2) sin(t2)];
            Sm = zeros(nt, numel(F{1}));
            for q = 1:nt, Sm(q,:) = F{q}(:).'; end
            c = M \ Sm;
            vis = reshape(sqrt(c(2,:).^2 + c(3,:).^2)./max(abs(c(1,:)),eps), size(F{1}));
            Ibar = reshape(c(1,:), size(F{1}));
            msk = Ibar > 0.1*max(Ibar(:));
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

        function R = probeRT(~, rx_t, iT, rx_r, iR)
            % Engine R and T of the cemented diagonal, per polarization.  The
            % source frame for the +x chief is xGrid = yhat (the diagonal's p
            % axis) and yGrid = zhat (its s axis); PolIn is set to the probe
            % axis so it passes the state instead of projecting it.  The
            % amplitude ratio is taken across the diagonal only -- the glass
            % between the two reads is lossless, so the propagation phase has
            % unit modulus and drops out.
            P = {rx_t, iT, 'T'; rx_r, iR, 'R'};
            Q = struct();
            for k = 1:2
                for pol = {'s','p'}
                    macos.load_rx(P{k,1});
                    if strcmp(pol{1},'s'), ax = [0 0 1];  Ex = [0 0];  Ey = [1 0];
                    else,                  ax = [0 1 0];  Ex = [1 0];  Ey = [0 0]; end
                    ip = P{k,2};
                    macos.polarizer(ip(1)-1, 'axis', ax);
                    macos.polarization('on','Ex',Ex,'Ey',Ey);
                    macos.vector_diffraction(true);
                    macos.trace(ip(1));  f1 = macos.ray_field(ip(1));
                    macos.trace(ip(2));  f2 = macos.ray_field(ip(2));
                    ok = (f1.status == 0) & (f2.status == 0);
                    a1 = sqrt(abs(f1.Ex).^2 + abs(f1.Ey).^2 + abs(f1.Ez).^2);
                    a2 = sqrt(abs(f2.Ex).^2 + abs(f2.Ey).^2 + abs(f2.Ez).^2);
                    Q.([P{k,3} pol{1}]) = median(a2(ok)./a1(ok))^2;
                end
            end
            R = struct('Ts',Q.Ts, 'Tp',Q.Tp, 'Rs',Q.Rs, 'Rp',Q.Rp);
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
        function x = wrap180(x)
            x = mod(x + 90, 180) - 90;
        end
    end
end
