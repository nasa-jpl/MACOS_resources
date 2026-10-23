classdef tAsphCalib < matlab.unittest.TestCase
%TASPHCALIB  CALIB on an aspheric coefficient: the differential step, and the
%   LM failure path through the mex.
%
%   Two engine defects found by the dyson5 native-optimize stage (TO, beat 4c
%   sections 3.5 and 3.6, 2026-10-01), fixed in design_optim.F the same day:
%     * the finite-difference step for an aspheric coefficient was a FIXED
%       1e-10 (x 1e-5 per order) in base units -- on a deck in metres that
%       moves an h^4 sag by ~1e-15 m, round-off, so the derivative column was
%       zero and the LM reported "gaussj: singular matrix" at its first step.
%       The step is now 1e-3 of the coefficient (a zero coefficient takes a
%       sag-based step at the element's circular aperture).
%     * that failure branch executed a bare STOP before its own rtn_flg=1:
%       the CLI died or survived by accident, the mex took MATLAB down.  It
%       now restores the last accepted optical state and returns the flag
%       (which the mex wrapper raises as a catchable MATLAB error).
%   Fixture Rx_AsphCalib.in: a collimated beam on an f = 1 m paraboloid in
%   METRES whose h^4 term is spoiled (5e-3 m^-3), SPOT target at the focus,
%   OptAsph= 1 1.  Non-vacuity: the pre-fix engine fails the first test with
%   the singular matrix (measured on TO's reproducer and on this fixture's
%   step arithmetic: 1e-10 x 0.1^4 = 1e-14 m of sag), and the second test
%   could not be RUN against it -- the STOP ends the MATLAB process.
    properties (Constant)
        ModelSize = 128
        RxName    = 'Rx_AsphCalib.in'
        SeedCoef  = 5e-3
    end
    methods (Access = private)
        function c = asph_coefs_from_save(~, m, wd, name)
            % The SAVEd AsphCoef line of element 1 (nAsphCoef= N, then N values).
            out = fullfile(wd, name);  m.save_rx(out);
            L = splitlines(string(fileread(out)));
            i1 = find(startsWith(strtrim(L), "iElt=") & endsWith(strtrim(L), "1"), 1);
            i2 = find(startsWith(strtrim(L), "iElt=") & endsWith(strtrim(L), "2"), 1);
            B = L(i1:i2-1);
            ia = find(startsWith(strtrim(B), "AsphCoef="), 1);
            tc_assert = ~isempty(ia);
            assert(tc_assert, 'no AsphCoef= line in the SAVEd element 1');
            c = sscanf(char(extractAfter(B(ia), "=")), '%f')';
        end
        function p = variant_no_aperture_zero_term(tc, wd)
            % Same deck, element 1 with NO aperture and the (zero) h^8 term
            % freed: the legacy absolute step applies, 1e-10 x 1e-5^2 = 1e-20,
            % and its sag contribution at the 100 mm edge (1e-28 m) is below
            % the double-precision resolution of the 5e-7 m asphere sag, so
            % the derivative column is EXACTLY zero and the LM must FAIL
            % (the h^6 term's 1e-15 step sits at ~10 ulp and gives a noise
            % column the LM wanders on -- not a deterministic failure).  The
            % failure must come back as a flag with the optics untouched.
            L = splitlines(string(fileread(rx_fixture_path(tc.RxName))));
            L(startsWith(strtrim(L), "OptAsph="))  = "          OptAsph=  1 3";
            L(startsWith(strtrim(L), "ApVec="))    = [];
            k = find(startsWith(strtrim(L), "ApType=  Circular"), 1);
            L(k) = "           ApType=  None";
            p = fullfile(wd, 'asph_noap.in');
            fid = fopen(p, 'w');  fprintf(fid, '%s\n', L);  fclose(fid);
        end
    end
    methods (Test)
        function test_asphere_step_scales_with_the_coefficient_and_calib_converges(tc)
            wd = tempname;  mkdir(wd);  c = onCleanup(@() rmdir(wd, 's')); %#ok<NASGU>
            m = macos.Session(tc.ModelSize);
            m.load_rx(rx_fixture_path(tc.RxName));
            c0 = tc.asph_coefs_from_save(m, wd, 'seed.in');
            tc.assertEqual(c0(1), tc.SeedCoef, 'RelTol', 1e-12, 'the seed must carry the spoiled term');
            r = m.calib();
            tc.verifyTrue(r.converged, sprintf('CALIB must converge on the asphere term (rtn_flag %d)', r.rtn_flag));
            c1 = tc.asph_coefs_from_save(m, wd, 'after.in');
            tc.verifyLessThan(abs(c1(1)), 0.1 * tc.SeedCoef, ...
                sprintf('the h^4 term must be driven toward zero: %.3e -> %.3e', c0(1), c1(1)));
        end
        function test_lm_failure_returns_a_flag_restores_the_optics_and_keeps_the_host(tc)
            wd = tempname;  mkdir(wd);  c = onCleanup(@() rmdir(wd, 's')); %#ok<NASGU>
            rx = tc.variant_no_aperture_zero_term(wd);
            m = macos.Session(tc.ModelSize);
            m.load_rx(rx);
            kr0 = m.get_elt_kr(1);
            c0 = tc.asph_coefs_from_save(m, wd, 'seed.in');
            % The engine returns rtn_flg=1 and calib_run answers OK=FAIL, which
            % the mex wrapper raises as an ordinary MATLAB error ("calib_run
            % failed") after the engine has printed the reason.  Pre-fix the
            % STOP on this path ended the MATLAB process instead.
            tc.verifyError(@() m.calib(), ?MException, ...
                'a zero derivative column must surface as a MATLAB error, not end the host');
            c1 = tc.asph_coefs_from_save(m, wd, 'after.in');
            tc.verifyEqual(c1, c0, 'AbsTol', 0, 'the optics must be back at the pre-optimization state');
            tc.verifyEqual(m.get_elt_kr(1), kr0, 'AbsTol', 0, 'KrElt must be untouched');
            % and the session is still usable
            m2 = macos.Session(tc.ModelSize);
            n = m2.load_rx(rx_fixture_path(tc.RxName));
            tc.verifyEqual(n, 2, 'the engine must load a deck after the failed optimization');
        end
    end
end
