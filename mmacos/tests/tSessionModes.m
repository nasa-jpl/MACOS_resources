classdef tSessionModes < matlab.unittest.TestCase
%TSESSIONMODES  Session modes survive load_rx, and load_rx says so (Dave 2026-10-11,
%   PLAN_CONSOLIDATION 4 item 2; CC's extension to the bindings on the one-semantics rule).
%   POLARIZATION / VECTOR / CENTROID / LNEG / CIR are SESSION modes set at the engine's
%   allocation, no longer reset by a load (before 2026-10-11 every load reset them --
%   the CLI's CENTROID before an `old` was lost silently).  The CLI's LOAD prints the
%   non-default ones; the bindings' load_rx raises warning 'mmacos:modesInForce' with
%   the same list (macos.modes, from the engine's ModesGet -- one source).
%   Pre-change mex: no modes_get, and a load reset polarization -- red.
    properties (Constant)
        Rx = 'Rx_AsphCalib.in'
    end
    methods (Test)
        function test_a_default_load_is_silent(tc)
            macos.init(128);
            tc.verifyWarningFree(@() macos.load_rx(rx_fixture_path(tc.Rx)));
            md = macos.modes();
            tc.verifyEmpty(md.in_force);
            tc.verifyFalse(md.pol);  tc.verifyTrue(md.lneg);
        end
        function test_polarization_survives_load_rx_and_is_named(tc)
            macos.init(128);  macos.load_rx(rx_fixture_path(tc.Rx));
            macos.polarization('on', 'Ex', [1 0], 'Ey', [0 0]);
            c = onCleanup(@() macos.polarization('off')); %#ok<NASGU>
            tc.verifyWarning(@() macos.load_rx(rx_fixture_path(tc.Rx)), 'mmacos:modesInForce');
            md = macos.modes();
            tc.verifyTrue(md.pol, 'polarization survives the load');
            tc.verifyEqual(md.in_force{1}, 'POLARIZATION');
            macos.polarization('off');
            tc.verifyWarningFree(@() macos.load_rx(rx_fixture_path(tc.Rx)), ...
                'back at the defaults, the load is silent again');
        end

        % --- modes_set (Dave 2026-10-28): the two modes that had no setter.
        % Pre-change mex: mmacos('modes_set') is an unknown command -- red.
        function test_centroid_set_before_load_survives_and_is_named(tc)
            % A session mode is set BEFORE the load (init only, no Rx) and
            % stays through it: the allocation rule, and the CLI's own order
            % (CENTROID, then LOAD).
            macos.init(128);
            macos.messages();                      % empty the engine's buffer
            md = macos.modes('centroid', true);
            c = onCleanup(@() macos.modes('centroid', false)); %#ok<NASGU>
            tc.verifyTrue(md.centroid, 'the setter returns the new modes');
            % The command-stack loader matched CENTROID against CENTER on three
            % letters and pushed CENTER's arguments, which the loop then read as
            % commands (smacosutil.F LoadStack; fixed 2026-10-28).  Red before.
            msgs = macos.messages();
            tc.verifyFalse(any(contains(msgs, 'Unknown command')), ...
                sprintf('stray commands after CENTROID: %s', strjoin(cellstr(msgs), ' | ')));
            tc.verifyEqual(md.in_force, {'CENTROID'});
            tc.verifyWarning(@() macos.load_rx(rx_fixture_path(tc.Rx)), 'mmacos:modesInForce');
            md = macos.modes();
            tc.verifyTrue(md.centroid, 'CENTROID survives the load');
            tc.verifyTrue(md.lneg, 'a name left out is left alone');
            md = macos.modes('centroid', false);
            tc.verifyFalse(md.centroid);
            tc.verifyEmpty(md.in_force);
            tc.verifyWarningFree(@() macos.load_rx(rx_fixture_path(tc.Rx)));
        end
        function test_lneg_set_and_read_back(tc)
            macos.init(128);  macos.load_rx(rx_fixture_path(tc.Rx));
            md = macos.modes('lneg', false);
            c = onCleanup(@() macos.modes('lneg', true)); %#ok<NASGU>
            tc.verifyFalse(md.lneg);
            tc.verifyEqual(md.in_force, {'NOLNEG'});
            tc.verifyFalse(md.centroid, 'a name left out is left alone');
            % both at once, and a 0/1 numeric is accepted as a logical
            md = macos.modes('centroid', 1, 'lneg', 1);
            tc.verifyTrue(md.centroid);  tc.verifyTrue(md.lneg);
            tc.verifyEqual(md.in_force, {'CENTROID'});
            md = macos.modes('centroid', 0);
            tc.verifyEmpty(md.in_force);
        end
        function test_fex_follows_the_session_axis(tc)
            % macos.fex() with no mode reads the session (A.6): under
            % CENTROID it is macos.fex(0), under CHIEFRAY macos.fex(1).
            % e5hex1 (a STOP, more than 3 elements; tDefaults' fex fixture).
            % Its two axes may coincide, so the test compares the omitted-mode
            % result with BOTH explicit ones and asks only that it equals the
            % session's.
            macos.init(128);  macos.load_rx(rx_fixture_path('e5hex1.in'));
            macos.modes('centroid', true);
            c = onCleanup(@() macos.modes('centroid', false)); %#ok<NASGU>
            s0 = macos.fex(0);  s = macos.fex();
            tc.verifyEqual(s.xp, s0.xp, 'AbsTol', 0, 'CENTROID session: fex() == fex(0)');
            tc.verifyGreaterThan(max(abs(s.xp - macos.fex(1).xp)), 1e-8, ...
                'the two axes must differ on e5hex1 (else vacuous)');
            macos.modes('centroid', false);
            s1 = macos.fex(1);  s = macos.fex();
            tc.verifyEqual(s.xp, s1.xp, 'AbsTol', 0, 'CHIEFRAY session: fex() == fex(1)');
        end
        function test_obs_is_reported_and_reset_by_load(tc)
            % OBS is NOT a session mode (Dave 2026-10-28, PLAN 1.2 (cccc)(2)):
            % modes() reports it in .obs and never in .in_force, and a load_rx
            % resets it to POSITIVE with the engine saying so (the message
            % buffer carries the line).  Red on the 7e9ed0a4 engine (OBS kept
            % across the load and named in in_force) and before it (silent).
            macos.init(128);  macos.load_rx(rx_fixture_path(tc.Rx));
            macos.obs('all');
            c = onCleanup(@() macos.obs('positive')); %#ok<NASGU>
            md = macos.modes();
            tc.verifyEqual(md.obs, 'all');
            tc.verifyEmpty(md.in_force, 'OBS is not a session mode');
            macos.messages();
            tc.verifyWarningFree(@() macos.load_rx(rx_fixture_path(tc.Rx)));
            tc.verifyEqual(macos.obs(), 'positive', 'a load resets OBS to POSITIVE');
            msgs = macos.messages();
            tc.verifyTrue(any(contains(msgs, 'OBS ALL reset to POSITIVE')), ...
                'the engine says it reset OBS');
            macos.obs('negative');
            tc.verifyEqual(macos.modes().obs, 'negative');
            macos.messages();
            macos.load_rx(rx_fixture_path(tc.Rx));
            tc.verifyEqual(macos.obs(), 'positive');
            tc.verifyTrue(any(contains(macos.messages(), 'OBS NEGATIVE reset to POSITIVE')));
            macos.messages();
            macos.load_rx(rx_fixture_path(tc.Rx));
            tc.verifyFalse(any(contains(macos.messages(), 'reset to POSITIVE')), ...
                'a load with OBS POSITIVE says nothing');
        end
    end
end
