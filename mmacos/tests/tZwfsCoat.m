classdef tZwfsCoat < matlab.unittest.TestCase
    % The ZWFS / PDI benches' OAP coating knob (templates/40_benches/
    % zwfs_dm96, PLAN_CONSOLIDATION 1.2 (nnn), Dave 2026-10-26): the default
    % depends on the rig -- protected aluminum on the OAP rig (the
    % 2026-09-17 decision, made for that rig), none on the lens rig -- and a
    % coating asked of the lens rig is a WARNING that names the ignored
    % setting, not the refusal a bare zwfs_run / pdi_run used to stop at
    % ('zwfs_run: bench.coat_oap needs bench.optics ''oap''').
    methods (TestClassSetup)
        function paths(tc)
            here = fileparts(mfilename('fullpath'));
            d = fullfile(fileparts(here), 'templates', '40_benches', 'zwfs_dm96');
            addpath(d);
            tc.addTeardown(@() rmpath(d));
        end
    end

    methods (Test)
        function test_sheet_default_is_the_rig_default(tc)
            P = zwfs_params();
            tc.verifyEqual(P.bench.coat_oap, 'auto');
            P = tc.verifyWarningFree(@() zwfs_coat_rig(P));
            tc.verifyEqual(P.bench.coat_oap, 'none', ...
                'the lens rig (the sheet''s rig) is uncoated by default');
        end

        function test_oap_rig_defaults_to_protected_aluminum(tc)
            P = zwfs_params();  P.bench.optics = 'oap';
            P = tc.verifyWarningFree(@() zwfs_coat_rig(P));
            tc.verifyEqual(P.bench.coat_oap, 'protectedAl');
        end

        function test_explicit_coat_on_the_lens_rig_warns_and_is_ignored(tc)
            P = zwfs_params();  P.bench.coat_oap = 'protectedAl';
            P = tc.verifyWarning(@() zwfs_coat_rig(P), 'zwfs_run:coat_oap_ignored');
            tc.verifyEqual(P.bench.coat_oap, 'none');
            % the warning names the ignored setting
            lastwarn('');
            ws = warning('off', 'zwfs_run:coat_oap_ignored');
            c = onCleanup(@() warning(ws));
            zwfs_coat_rig(setfield(zwfs_params(), 'bench', 'coat_oap', 'bareAl')); %#ok<SFLD>
            msg = lastwarn();
            tc.verifySubstring(msg, 'bench.coat_oap ''bareAl''');
            tc.verifySubstring(msg, 'lens');
        end

        function test_explicit_coat_on_the_oap_rig_is_kept(tc)
            P = zwfs_params();  P.bench.optics = 'oap';  P.bench.coat_oap = 'bareAl';
            P = tc.verifyWarningFree(@() zwfs_coat_rig(P));
            tc.verifyEqual(P.bench.coat_oap, 'bareAl');
            P.bench.coat_oap = 'none';
            P = tc.verifyWarningFree(@() zwfs_coat_rig(P));
            tc.verifyEqual(P.bench.coat_oap, 'none');
        end
    end
end
