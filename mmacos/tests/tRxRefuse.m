classdef tRxRefuse < matlab.unittest.TestCase
%TRXREFUSE  A value with no natural zero that cannot be read REFUSES the load; the host lives.
%   PLAN_CONSOLIDATION item 1b (2026-10-09; Dave's ruling, PLAN_CONSOLIDATION 4a).  The parser read
%   geometry vectors, frames, aperture/obscuration groups, integer lists and scalars with bare
%   list-directed internal READs: a short line or a token that is not a number was a Fortran runtime
%   abort -- it kills MATLAB when the engine is the mex.  Unlike the coefficient blocks (tRxShortCoef:
%   zero is natural there, so they pad), these have no natural default, so every such READ now carries
%   IOSTAT and RxBad (elt_mod): one message naming the key and the element, the load-failure exit
%   (nElt = 0, LOAD_SUCCESS false), MATLAB alive.  macos.load_rx raises an ordinary, catchable error.
%
%   THE MUST-FAIL LEG runs in a matlab -batch SUBPROCESS: three refusals -- Rx_ShortVec.in (psiElt= with
%   2 of 3 values), Rx_BadScalar.in (KrElt= minus200), a header ChfRayDir= with 2 of 3 values -- each
%   caught, each naming its key, then a good deck loads and traces in the SAME session.  The pre-fix mex
%   aborts at the first load: the subprocess exits nonzero.  These decks stay refusals forever (CC keeps
%   the short-psiElt deck in macos cli_tests/must_fail/ as the permanent negative control).
%   Item 5c (2026-10-09): Rx_RxStop.in (EdgeSensors= 0) -- the parser printed its error and then executed
%   STOP, which ended MATLAB; three such STOPs (EdgeSensors < 1, nPetals before nRings, tMetElt before
%   nMetPos) now refuse the load through the same exit.
%   The engine's lines are read from its message buffer, echoed by the child (MSG_ECHO_ / MSG_TEXT_;
%   PLAN_CONSOLIDATION 1.2 (ee)): the Fortran unit-6 copy never reached a nested MATLAB's output on
%   macOS, so this test was filtered on the Mac until 2026-10-22.
    properties (Constant)
        ModelSize = 128
    end
    methods (Test)
        function test_refusals_name_the_key_and_the_host_lives(tc)
            wd = tempname;  mkdir(wd);  c = onCleanup(@() rmdir(wd, 's'));
            mm = fileparts(fileparts(mfilename('fullpath')));
            good = rx_fixture_path('Rx_ShortCoef.in');
            s = string(fileread(good));  old = "        ChfRayDir=  0.0D+00  0.0D+00  1.0D+00";
            tc.assertEqual(count(s, old), 1, 'header anchor');
            hdr = fullfile(wd, 'short_chfraydir.in');  fid = fopen(hdr, 'w');  fprintf(fid, '%s', replace(s, old, "        ChfRayDir=  0.0D+00  0.0D+00"));  fclose(fid);
            bad = {rx_fixture_path('Rx_ShortVec.in'), rx_fixture_path('Rx_BadScalar.in'), hdr, rx_fixture_path('Rx_RxStop.in')};
            scr = fullfile(wd, 'leg.m');  fid = fopen(scr, 'w');
            fprintf(fid, 'run(''%s'');\nmacos.init(%d);\n', fullfile(mm, 'mmacos_setup.m'), tc.ModelSize);
            for k = 1:numel(bad)
                fprintf(fid, 'try, macos.load_rx(''%s''); fprintf(''LOADED %d\\n''); catch, fprintf(''CAUGHT %d\\n''); end\n', bad{k}, k, k);
                fprintf(fid, '%s\n', msg_echo_());
            end
            fprintf(fid, 'macos.load_rx(''%s''); t = macos.trace(macos.num_elt()); fprintf(''GOOD %%.10e\\n'', t.rmsWFE);\n', good);
            fprintf(fid, '%s\n', msg_echo_());
            fclose(fid);
            [st, out] = system(sprintf('cd %s && MACOS_HOME="%s" "%s" -batch "run(''%s'')" 2>&1', wd, macos_home_(), fullfile(matlabroot, 'bin', 'matlab'), scr));   % MACOS_HOME: a system() from MATLAB on macOS drops the parent's environment (CCMac 2026-10-19)
            tc.verifyEqual(st, 0, sprintf('a refused load must not kill MATLAB (pre-fix: the engine aborts)\n%s', out));
            for k = 1:numel(bad), tc.verifyTrue(contains(out, sprintf('CAUGHT %d', k)), sprintf('bad deck %d is refused with a catchable error', k)); end
            m = msg_text_(out);
            tc.verifyEqual(count(m, "Rx load refused: psiElt (elt   1)"), 1, 'the short psiElt= names its key and element');
            tc.verifyEqual(count(m, "Rx load refused: KrElt (elt   1)"), 1, 'the bad KrElt= names its key and element');
            tc.verifyEqual(count(m, "Rx load refused: ChfRayDir (header)"), 1, 'the short header ChfRayDir= names its key');
            tc.verifyTrue(contains(m, 'EdgeSensor must be positive integer'), 'item 5c: the parser''s own Rx error message, then a refusal (was STOP)');
            g = regexp(out, 'GOOD (\S+)', 'tokens', 'once');
            tc.assertNotEmpty(g, 'a good deck loads and traces after the refusals, in the same session');
            tc.verifyGreaterThan(str2double(g{1}), 0, 'and traces to a real OPD');
        end
    end
end
