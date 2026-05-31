classdef tSrsBugFlatZ < matlab.unittest.TestCase
%TSRSBUGFLATZ  Regression for the SRS clobbers-Flat-zElt bug.
%   On FFSegDemoAll.in:
%     Elt 10 is a FocalPlane / Flat with a big-magnitude sentinel
%     zElt (~1e19 from the file; canonically INF=1e22).
%     Elt 9 is a curved Reference (Conic, KrElt=-2.6).
%   Workflow: load → SXP (set exit pupil at Elt 9) → SRS 10 9.
%   Expected: SRS moves VptElt(:,10) to lie on Elt 9's reference
%   sphere along the chief ray, but PRESERVES zElt(10) at its big
%   sentinel value.  A Flat encodes "next defined plane" in zElt,
%   not a physical distance, per Dave's invariant
%   ("Flat should always have big Kr or f / z", 2026-05-07).
%
%   Bug (2026-05-31): the symmetric-Returns / Flat-slave / curved-
%   slave-to branch in macos_cmd_loop.inc:2257 overwrites zp with
%   `zElt(iSlv2) + path-difference`, clobbering the Flat's sentinel.
%   The asymmetric-Returns branch already does the right thing
%   (`zp = zElt(iSlv1)`); the symmetric branch was inconsistent.

    properties (Constant)
        ModelSize = 128
        RxName    = 'FFSegDemoAll.in'
        FlatElt   = 10        % FocalPlane Flat slave
        SphereElt = 9         % Reference Conic slave-to
        BigZThreshold = 1e18  % BIG; pre-fix zElt drops well below this
    end

    properties
        rx_path
    end

    methods (TestClassSetup)
        function setupClass(testCase)
            testCase.rx_path = rx_fixture_path(testCase.RxName);
            macos.init(testCase.ModelSize);
        end
    end

    methods (TestMethodSetup)
        function setupMethod(testCase)
            macos.load_rx(testCase.rx_path);
        end
    end

    methods (Test)
        function test_loaded_zelt_is_sentinel(testCase)
            % Sanity: the as-loaded zElt(10) is the file's big value
            % (1e19) — well above BIG, our pre/post-condition anchor.
            z0 = macos.get_elt_z(testCase.FlatElt);
            testCase.verifyGreaterThan(z0, testCase.BigZThreshold, ...
                sprintf('zElt(Flat) loaded as %.3e — expected ≥ BIG=%.0e', ...
                    z0, testCase.BigZThreshold));
        end

        function test_srs_preserves_flat_zelt(testCase)
            % The bug.  load → STOP obj 0,0,0 → SXP → SRS 10 9
            % must not change zElt(10).
            z_before = macos.get_elt_z(testCase.FlatElt);
            % STOP at object (0,0,0).
            mmacos('stop_obj_set', 0.0, 0.0, 0.0);
            % SXP (mode=1 = standard exit pupil; ignores XP arg in get mode).
            mmacos('sxp_fnd', 1);  % SXP standard mode
            % SRS slave=Flat(10), slave-to=Sphere(9), no link.
            mmacos('srs_run', testCase.FlatElt, testCase.SphereElt, 0);
            z_after = macos.get_elt_z(testCase.FlatElt);

            testCase.verifyGreaterThan(z_after, testCase.BigZThreshold, ...
                sprintf(['SRS bug: zElt(Flat) was %.3e (>= BIG), is now %.3e (< BIG). ' ...
                         'A Flat''s zElt is a "propagate to next plane" sentinel; ' ...
                         'SRS must not overwrite it.'], z_before, z_after));
        end
    end
end
