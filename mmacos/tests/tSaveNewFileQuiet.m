classdef tSaveNewFileQuiet < matlab.unittest.TestCase
%TSAVENEWFILEQUIET  api save_rx on a NEW file says nothing stray (PLAN_CONSOLIDATION 1.2 (v), 2026-10-22).
%   Twin of pymacos test_save_new_file_quiet.  The wrapper passed CARG(2) = 'YES' as the standing
%   answer to CSAVE's 'Replace?' prompt; for a file that did not exist CSAVE never asked, the token
%   fell through to the SMACOS loop as a command, and the loop printed '** Unknown command = YES'.
%   Now the wrapper answers only when the file exists, and a blank second token is not pushed.
%   The engine's stdout is Fortran unit 6, which evalc does not see, so the saves run in a
%   matlab -batch SUBPROCESS and its output is read: a save to a new file, then a second save over
%   it (the Replace? path); both write the deck, neither prints 'Unknown command'.  The pre-fix
%   mex prints the line on the first save -> red.
%   The SMACOS loop's line is read from the engine's message buffer, echoed by the child (MSG_ECHO_ /
%   MSG_TEXT_; PLAN_CONSOLIDATION 1.2 (ee)) -- the unit-6 copy never reached a nested MATLAB's output on
%   macOS, so this test was filtered there.  Must-pass leg: the load's own 'being loaded' line is echoed,
%   so the channel works and the absence of 'Unknown command' means something.
    properties (Constant)
        ModelSize = 128
        Base      = 'Rx_SaveKeys.in'
    end
    methods (Test)
        function test_save_new_then_existing_prints_no_stray_command(tc)
            wd = tempname;  mkdir(wd);  c = onCleanup(@() rmdir(wd, 's'));
            mm = fileparts(fileparts(mfilename('fullpath')));
            out = fullfile(wd, 'saved.in');
            scr = fullfile(wd, 'leg.m');  fid = fopen(scr, 'w');
            fprintf(fid, 'run(''%s'');\nmacos.init(%d);\nmacos.load_rx(''%s'');\n', ...
                fullfile(mm, 'mmacos_setup.m'), tc.ModelSize, rx_fixture_path(tc.Base));
            fprintf(fid, '%s\n', msg_echo_());
            fprintf(fid, 'fprintf(''MARK1\\n'');\nmacos.save_rx(''%s'');\nfprintf(''MARK2\\n'');\n', out);
            fprintf(fid, '%s\n', msg_echo_());
            fprintf(fid, 'f = fopen(''%s'', ''w''); fprintf(f, ''%%%% overwritten\\n''); fclose(f);\n', out);
            fprintf(fid, 'macos.save_rx(''%s'');\nfprintf(''MARK3\\n'');\n', out);
            fprintf(fid, '%s\n', msg_echo_());
            fclose(fid);
            [st, txt] = system(sprintf('cd %s && MACOS_HOME="%s" "%s" -batch "run(''%s'')" 2>&1', ...
                wd, macos_home_(), fullfile(matlabroot, 'bin', 'matlab'), scr));
            tc.assertEqual(st, 0, txt);
            tc.assertTrue(contains(txt, 'MARK3'), ['the subprocess ran both saves' newline txt]);
            tc.verifyTrue(contains(fileread(out), 'iElt='), 'the second save replaced the overwritten file');
            m = msg_text_(txt);
            tc.assertTrue(contains(m, 'Rx_SaveKeys.in') && contains(m, 'being loaded'), ['the child echoed the load''s lines' newline txt]);
            tc.verifyFalse(contains(txt, 'Unknown command') || contains(m, 'Unknown command'), ['a stray answer reached the SMACOS loop' newline txt]);
        end
    end
end
