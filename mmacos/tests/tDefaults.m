classdef tDefaults < matlab.unittest.TestCase
%TDEFAULTS  One default per option (Dave 2026-10-10; BRIEF_to_bindings A.6) -- twin
%   of pymacos tests/test_defaults.py.  Every session/user option the api exposes
%   has ONE default, held by the engine; a veneer inherits it (reads the api getter)
%   instead of restating it.  Each test reads the same deck as test_defaults.py and
%   asserts the SAME numbers (measured 2026-10-16, mmacos and pymacos alike) for
%   the default and for its override.  The option table is cmdref 25_defaults.md.
%
%   The mmacos-only leg: the dw_* harvests re-applied opd_ref = 'chief' after their
%   reload, overriding a deck's UseChfRay4OPD= N (measured on tst_opdref_mean.in:
%   w_nom == opd() under 'chief' to 0, off the deck's 'mean' by 2.518e-12, and the
%   session left on 'chief' afterwards).  Fixed: the default '' keeps the deck's.
    properties (Constant)
        Model = 128
        % measured 2026-10-16 (e5hex1, model 128, a FRESH load + trace)
        SpotShiftElt   = [0.01003577993452497, -5877.054058107832]
        SpotShiftChief = [0.01013346493612515, -5938.921990992478]
        E5ValidPx      = 10245
        MeanDeckChiefMinusMean = -2.534833896820362e-12   % opd() units, constant
        MeanDeckValidPx = 1756
    end
    methods (Static, Access = private)
        function p = mean_deck()
            home = getenv('MACOS_HOME');
            if isempty(home), home = fullfile(getenv('HOME'), 'dev', 'macos', 'macos_f90'); end
            p = fullfile(fileparts(home), 'ZGD_test_files', 'tst_opdref_mean.in');
            assert(isfile(p), 'tst_opdref_mean.in not found at %s', p);
        end
        function ne = e5()
            macos.init(tDefaults.Model);
            macos.load_rx(rx_fixture_path('e5hex1.in'));
            ne = macos.num_elt();  macos.trace(ne);
        end
    end
    methods (Test)
        function test_session_modes_default(tc)
            tDefaults.e5();
            md = macos.modes();
            tc.verifyEqual([md.pol md.vecdif md.centroid md.lneg md.cir], ...
                logical([0 0 0 1 0]));
            tc.verifyEmpty(md.in_force);
        end
        function test_opd_reference_default_and_deck_override(tc)
            tDefaults.e5();
            tc.verifyEqual(macos.opd_ref(), 'chief');
            macos.load_rx(tDefaults.mean_deck());          % UseChfRay4OPD= N
            tc.verifyEqual(macos.opd_ref(), 'mean');
            ne = macos.num_elt();  macos.trace(ne);
            Wm = macos.opd();  K = macos.opd_mask();
            macos.opd_ref('chief');
            tc.verifyEqual(macos.opd_ref(), 'chief');
            macos.trace(ne);  d = macos.opd() - Wm;  d = d(K);
            tc.verifyEqual(nnz(K), tc.MeanDeckValidPx);
            tc.verifyEqual(mean(d), tc.MeanDeckChiefMinusMean, 'RelTol', 1e-6);
            tc.verifyLessThan(std(d), 1e-20, 'the two references differ by a constant');
        end
        function test_harvest_keeps_the_decks_reference(tc)
            rx = tDefaults.mean_deck();
            m = macos.Session(tc.Model);
            o = macos.dw_dx(m, rx, 'elts', 2, 'dofs', 0);
            tc.verifyEqual(m.opd_ref(), 'mean', 'the harvest must not switch the deck''s reference');
            m.load_rx(rx);  m.trace(o.wf_elt);
            K = macos.opd_mask();  W = macos.opd();
            tc.verifyEqual(o.w_nom_2d(K), W(K), 'AbsTol', 0, 'default = the deck''s mean');
            oc = macos.dw_dx(m, rx, 'elts', 2, 'dofs', 0, 'opd_ref', 'chief');
            m.load_rx(rx);  m.opd_ref('chief');  m.trace(o.wf_elt);  Wc = macos.opd();
            tc.verifyEqual(oc.w_nom_2d(K), Wc(K), 'AbsTol', 0, 'override = the chief');
            tc.verifyGreaterThan(max(abs(Wc(K) - W(K))), 1e-12, 'the two must differ (else vacuous)');
        end
        function test_fex_axis_default_is_the_session_axis(tc)
            tDefaults.e5();
            f = macos.fex();  f1 = macos.fex(1);  f0 = macos.fex(0);
            tc.verifyEqual(f.xp, f1.xp, 'AbsTol', 0, 'default = chief (session)');
            tc.verifyGreaterThan(max(abs(f.xp - f0.xp)), 1e-8, 'centroid override moves it');
            tc.verifyEqual(macos.sxp(), macos.sxp(1), 'AbsTol', 0);
            tc.verifyNotEqual(macos.sxp(0), macos.sxp(1), 'sxp accepts the centroid axis');
        end
        function test_spot_centre_default_is_the_element(tc)
            ne = tDefaults.e5();
            s = macos.spot(ne);  se = macos.spot(ne, 'at', 'elt');  sc = macos.spot(ne, 'at', 'chief');
            % each spot re-traces; a re-trace is not bit-idempotent (a few ulp)
            tc.verifyEqual(s.pts, se.pts, 'AbsTol', 1e-9, 'default = the element vertex');
            tc.verifyGreaterThan(max(abs(s.centroid - sc.centroid)), 1, 'the chief override moves the centre');
            tc.verifyEqual(mmacos('spc_option_get'), 1, ...
                'a per-call chief centre does not change the session option');
            s2 = macos.spot(ne);
            tc.verifyEqual(s2.pts, se.pts, 'AbsTol', 1e-9, 'so the next default call is the element again');
            % mmacos's .shift is the api's 4-vector [chief xy, vertex xy] (whatever 'at'
            % is); pymacos returns the half vpt_center selects -- same numbers
            tc.verifyEqual(s.shift(3:4), tc.SpotShiftElt, 'RelTol', 1e-9);
            tc.verifyEqual(s.shift(1:2), tc.SpotShiftChief, 'RelTol', 1e-9);
        end
        function test_obscuration_option_default_is_positive(tc)
            ne = tDefaults.e5();
            tc.verifyEqual(macos.obs(), 'positive');      % the query (obs_opt_get)
            tc.verifyEqual(nnz(macos.opd_mask()), tc.E5ValidPx);
            macos.obs('negative');
            c = onCleanup(@() macos.obs('positive')); %#ok<NASGU>
            tc.verifyEqual(macos.obs(), 'negative');
            macos.trace(ne);
            tc.verifyEqual(nnz(macos.opd_mask()), 0);
            macos.obs('positive');  macos.trace(ne);
            tc.verifyEqual(nnz(macos.opd_mask()), tc.E5ValidPx);
        end
        function test_calib_beam_rows_leave_the_spot_centre_alone(tc)
            % CALIB's SPOT rows (macos_ops.F) set the session spcOption to the
            % chief ray and never restored it (found 2026-10-16); twin of the
            % pymacos leg.
            m = macos.Session(tc.Model);
            m.load_rx(rx_fixture_path('Rx_AsphCalib.in'));     % a SPOT target
            tc.verifyEqual(mmacos('spc_option_get'), 1);
            m.calib_set_iter(2);  m.calib();
            tc.verifyEqual(mmacos('spc_option_get'), 1, 'CALIB must restore the session SPCENTER');
        end
        function test_ffcut_default_off(tc)
            tDefaults.e5();
            [on, npix] = macos.ffcut();
            tc.verifyEqual([double(on) npix], [0 0]);
        end
        function test_opd_orientation_default_raw(tc)
            tDefaults.e5();
            W = macos.opd();
            tc.verifyEqual(W, macos.opd('orient', 'raw'));
            tc.verifyEqual(W.', macos.opd('orient', 'xy'));
            tc.verifyEqual(macos.opd_mask(), macos.opd_mask('orient', 'raw'));
        end
    end
end
