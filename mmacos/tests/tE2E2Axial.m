classdef tE2E2Axial < matlab.unittest.TestCase
%TE2E2AXIAL  Gates for e2e2 stage 1 -- the Korsch axial starting point.
%
%   The stage driver (design/examples/e2e2/s1_axial.m) runs three gates
%   before it believes any wavefront number, and this class runs the two
%   cheap ones plus a set of checks on the COMMITTED artifact.  It does not
%   re-solve: the CALIB solve takes minutes and its result is already
%   pinned by the deck the driver saved.
%
%   WHAT IS ACTUALLY BEING GATED, and why each is here:
%
%   (1) The CONIC SOLVER, against the shared TMA fixture.  Engine-free
%       arithmetic.  If seidel_seed has moved, every conic in every design
%       downstream is wrong in a way no field map would reveal.
%
%   (2) The FIRST-ORDER LAYOUT.  e2e2's design point is a validated CODE V
%       design scaled by 3/5, so tma_layout's R3 -- which it derives from
%       the f/# constraint alone -- must land on the radius CODE V's own
%       `CUY UMY -0.025` marginal-angle solve produced.  This is a check on
%       the BUILDER that a de-novo layout cannot make, and it is the reason
%       the design point was chosen this way.
%
%   (3) The SAVED DECK.  It must load, trace, pass the pupil gate (greatest
%       chord vs the declared Aperture, never a span -- macos PR #70), and
%       carry an exit-pupil pair whose FP_return still sits at the detector
%       station.  That last one is not pedantry: an earlier version of the
%       driver fitted the detector on the unsolved K = 0 design, which
%       walked it 1.8 m, and add_pupil then derived FP_return and the
%       ExitPupil sphere from the abandoned station.  The deck loaded, the
%       solve converged, and the declared exit pupil belonged to a design
%       that no longer existed.  Note the gate is not exact equality: the
%       joint solve MOVES the detector after add_pupil, by design -- that
%       is the FPA piston doing its job.  The bar separates a 20 um
%       refinement from a 1.8 m excursion.
%
%   (4) The ANCHOR.  Stage 1 hands stage 2 the residual the field bias will
%       then spoil, and stage 2's collapse measurement is only attributable
%       if this number is known and stable.  At the 0.6 deg box it is 16.5
%       nm -- 46% of the diffraction bar, against 1.8% at the original 0.2
%       deg box.  Widening the field spends most of the budget here; what
%       is left for stages 2-4, added in quadrature, is 31.7 nm.

    properties (Constant)
        MODEL = 256
        ANCHOR_MAX_NM = 10.0  % stage-1 residual bar over the used box,
                              % rung 4.  Tracks the field: 0.638 nm at the
                              % original 0.2 deg box, 16.5 at 0.6, and
                              % 4.1 at the adopted 0.4 -- the field is the
                              % dominant term in this number, so the bar
                              % moves when P.fov_half_deg does.  Set where
                              % a regression shows rather than where the
                              % current value sits.
    end

    properties
        P
        exdir
        deck
    end

    methods (TestClassSetup)
        function setup(tc)
            here = fileparts(mfilename('fullpath'));
            root = fileparts(here);
            run(fullfile(root,'mmacos_setup.m'));
            tc.exdir = fullfile(root,'design','examples','e2e2');
            addpath(tc.exdir);
            tc.P    = e2e2_params();
            tc.deck = fullfile(tc.exdir,'s1_axial.in');
            macos.init(tc.MODEL);
        end
    end

    methods (Test)   % ---- engine-free gates ----------------------------

        function test_conic_solver_reproduces_the_tma_fixture(tc)
            root = fileparts(fileparts(mfilename('fullpath')));
            fxdir = fullfile(fileparts(root), 'optical_design', 'fixtures');
            if ~isfolder(fxdir)
                fxdir = fullfile(getenv('HOME'),'dev','MACOS_resources', ...
                                 'optical_design','fixtures');
            end
            f = fullfile(fxdir,'tma_fixture.json');
            tc.assumeTrue(exist(f,'file') == 2, 'tma_fixture.json not reachable');
            fx = jsondecode(fileread(f));
            K  = macos.design.seidel_seed( ...
                    [fx.layout_m.R1, fx.layout_m.R2, fx.layout_m.R3], ...
                    [fx.layout_m.t_M1_M2, fx.layout_m.t_M2_M3], fx.layout_m.D);
            tc.verifyLessThanOrEqual( ...
                max(abs(K - [fx.conics.K1, fx.conics.K2, fx.conics.K3])), ...
                tc.P.fixture_tol, ...
                ['the conic solver no longer reproduces the fixture.  ' ...
                 'STOP AND FIX -- do not widen this bar and do not edit ' ...
                 'the fixture; regenerate it with make_tma_fixture.py.']);
        end

        function test_layout_derives_the_reference_radii(tc)
            P = tc.P;
            [R, tsp, lay] = macos.design.tma_layout(P.D_m, P.primary_fnum, ...
                    P.system_fnum, 'secondary_mag', P.secondary_mag, ...
                    'int_focus_m', P.int_focus_m, 'm3_behind_m', P.m3_behind_m);
            tc.verifyLessThanOrEqual(abs(R(3)-P.R_ref_m(3))/P.R_ref_m(3), ...
                P.R3_tol_rel, ...
                ['the builder no longer derives the reference R3 from the ' ...
                 'f/# constraint -- the layout or the design point moved']);
            tc.verifyEqual(R(1:2), P.R_ref_m(1:2), 'RelTol', 1e-5);
            tc.verifyEqual(tsp,    P.t_ref_m,      'RelTol', 1e-5);
            tc.verifyEqual(lay.EFL, P.system_fnum*P.D_m, 'RelTol', 1e-9, ...
                'paraxial EFL must be f/# x D by construction');
        end

        function test_params_are_self_consistent(tc)
        %  The design point claims to be a 3/5 scaling of a f/20 system off
        %  an f/1.2358 primary.  R1 = 2*primary_fnum*D must therefore BE the
        %  reference R1, with no free parameter in between.
            P = tc.P;
            tc.verifyEqual(2*P.primary_fnum*P.D_m, P.R_ref_m(1), 'RelTol', 1e-6);
            tc.verifyEqual(P.dl_rms_m, P.dl_waves*P.lambda_m, 'RelTol', 1e-12, ...
                'the RMS bar in metres and in waves disagree');
            tc.verifyEqual(P.fov_arcmin, P.fov_half_deg*60, 'RelTol', 1e-12, ...
                'the half-field is stated twice and the two disagree');
        end
    end

    methods (Test)   % ---- the committed artifact ------------------------

        function test_saved_deck_loads_traces_and_holds_its_pupil(tc)
            tc.assumeTrue(exist(tc.deck,'file') == 2, ...
                's1_axial.in not built yet -- run s1_axial.m');
            n = macos.load_rx(tc.deck);
            tc.verifyGreaterThanOrEqual(n, 6, 'expected at least 6 elements');
            tr = macos.trace(n);
            ri = macos.get_ray_info(tr.nRays);
            npass = nnz(logical(ri.ok_pass) & logical(ri.ok_trace));
            tc.verifyGreaterThan(npass, 0.85*tr.nRays, ...
                'most rays should survive an on-axis anastigmat with a hole');
            g = pupil_gate('elt', 1, 'rtol', tc.P.pupil_tol_rel, 'quiet', true);
            tc.verifyTrue(g.ok, g.msg);
        end

        function test_exit_pupil_pair_tracks_the_detector(tc)
        %  FP_return is frozen at whatever station add_pupil saw.  The joint
        %  solve then refines the detector with its FPA piston, so a SMALL
        %  residual is expected and is exactly the refinement -- 0.48 mm at
        %  the 0.6 deg box, 2e-4 of the exit-pupil radius.  What must not happen is the
        %  failure in the class header: the detector fitted on the unsolved
        %  K = 0 design walked 1.8 m, add_pupil derived FP_return and the
        %  ExitPupil sphere from the abandoned station, and the saved deck
        %  declared an exit pupil belonging to a design that no longer
        %  existed.  The bar separates the two by five orders of magnitude.
            tc.assumeTrue(exist(tc.deck,'file') == 2, 's1_axial.in not built');
            n = macos.load_rx(tc.deck);
            V = zeros(3,n);
            for k = 1:n, V(:,k) = macos.get_elt_vpt(k); end
            iret = n - 2;   % M1 M2 M3 | FP_return ExitPupil FP
            d = norm(V(:,iret) - V(:,n));
            % 10 mm: still 180x below the 1.8 m failure, and 20x above the
            % refinement the joint solve legitimately applies.  The two are
            % separated by orders of magnitude, so the bar does not need to
            % be tight -- and a tight one would just track the field size.
            tc.verifyLessThan(d, 1e-2, sprintf( ...
                ['FP_return sits %.4g m from the detector.  A refinement is ' ...
                 'sub-millimetre; this is the stale-station failure.'], d));
            tc.verifyGreaterThan(d, 0, ...
                ['FP_return exactly coincides with the detector -- the joint ' ...
                 'solve''s FPA piston did nothing, so the detector was not ' ...
                 'in the DOF set']);
        end

        function test_pupil_gate_is_bias_invariant(tc)
        %  A COLLIMATED source lays its lattice down perpendicular to its
        %  own chief direction, so the traced pupil cannot depend on where
        %  the telescope is pointed.  Any bias dependence is the
        %  MEASUREMENT leaking, not the engine.
        %
        %  This caught a real one.  pupil_gate projected ray offsets along
        %  get_ray_info's .dir at the measured element -- the OUTGOING
        %  direction, which is neither the travel direction nor common to
        %  all rays.  On axis that is accidentally exact (outgoing chief
        %  antiparallel to the axis, sag along the axis), so it read
        %  perfectly until stage 2 biased the field: at 1.5 deg it reported
        %  1.0025 x the declared semi-diameter and failed its own gate.
        %  0.1517 m of M1 rim sag times sin(1.5 deg) is 0.0040 m against the
        %  0.0037 m excess -- the whole discrepancy, and the engine's pupil
        %  was right.  Projecting along the INCOMING chief removes it.
            P = tc.P;   S1 = load(fullfile(tc.exdir,'s1_axial.mat'));
            bias = [0 30 90 150];
            r = nan(size(bias));   nout = nan(size(bias));
            for i = 1:numel(bias)
                t = macos.design.Telescope('family','TMA', ...
                        'aperture_diameter_m', P.D_m, 'wavelength_m', P.lambda_m, ...
                        'model_size', tc.MODEL, 'grid_npts', P.grid_npts);
                t.add_mirror('M1','radius_m',S1.R(1),'conic',S1.K(1), ...
                             'spacing_after_m',S1.tsp(1));
                t.add_mirror('M2','radius_m',S1.R(2),'conic',S1.K(2), ...
                             'spacing_after_m',S1.tsp(2),'convex',true);
                t.add_mirror('M3','radius_m',S1.R(3),'conic',S1.K(3), ...
                             'spacing_after','derive');
                t.add_focal_plane('FP','ap_r',P.fp_body_r);
                % the hole is MEASURED by the drivers now (floored at the
                % secondary's shadow), so the test uses what stage 1
                % actually declared rather than a retired constant
                t.set_hole('M1', S1.r_hole);
                if bias(i) > 0, t.set_field_bias(bias(i)); end
                t.build();
                g = pupil_gate('elt', 1, 'rtol', P.pupil_tol_rel, 'quiet', true);
                r(i) = g.r_ratio;   nout(i) = g.n_outside;
            end
            tc.verifyEqual(r, repmat(r(1), size(r)), 'AbsTol', 1e-12, sprintf( ...
                ['the traced pupil moved with the field bias (%s) -- a ' ...
                 'collimated lattice cannot do that, so the measurement is ' ...
                 'projecting along the wrong axis'], mat2str(r,8)));
            tc.verifyEqual(nout, zeros(size(nout)), ...
                'rays outside the declared pupil at some bias');
            tc.verifyLessThanOrEqual(r(1), 1 + P.pupil_tol_rel, ...
                'the traced pupil exceeds the declared aperture');
        end

        function test_stage1_is_a_negligible_anchor(tc)
        %  (4) stage 2 measures a COLLAPSE against this; it has to be small
        %  enough that the collapse is attributable to the field bias.
            tc.assumeTrue(exist(tc.deck,'file') == 2, 's1_axial.in not built');
            P = tc.P;
            F = macos.design.field_grid(P.fov_arcmin, 5, 'units','arcmin');
            [L, info] = strict_ladder_deck(tc.deck, F, 'lambda', P.lambda_m);
            ok = all(isfinite(L),2);
            tc.verifyGreaterThan(nnz(ok), 0, 'no field scored');
            tc.verifyLessThan(max(L(ok,4))*1e9, tc.ANCHOR_MAX_NM, ...
                'the axial anchor has degraded');
            tc.verifyGreaterThan(min(info.strehl(ok,4)), 0.90, ...
                'the axial anchor should still be well corrected');
            tc.verifyLessThan(max(L(ok,4))*1e9, P.dl_rms_m*1e9, ...
                'the anchor must be far inside the diffraction-limit bar');
        end
    end
end
