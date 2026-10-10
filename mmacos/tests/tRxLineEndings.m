classdef tRxLineEndings < matlab.unittest.TestCase
%TRXLINEENDINGS  LF, CRLF and CR-only decks load to one answer through one mechanism.
%   PLAN_CONSOLIDATION item 5 (2026-10-09; Dave's ruling 4a, CC's conditions).  MBFile6 loads a deck
%   that holds any CR from an LF copy written to the system temp dir (RxNormalizeEOL,
%   validate_prescription_mod) and deletes the copy on every exit of the load.  Why: ifx reads a CR-only
%   (classic Mac) deck as ONE record -- 13 corpus decks failed or crashed on it -- while gfortran splits
%   records on CR; the copy makes both compilers, the CLI and both bindings take the same path.
%
%   In a matlab -batch SUBPROCESS with TMPDIR pointed at an empty directory: the LF / CRLF / CR twins of
%   one deck (Rx_Eol_*.in = macos ZGD_test_files/tst_eol_*.in) trace to the identical rms; the CRLF and
%   CR loads each print ONE note naming the ORIGINAL deck, the LF load none; the temp name never appears;
%   the temp directory is empty afterwards.  The red leg on the pre-fix mex (gfortran, which already read
%   the CR twin natively) is the normalization itself: no note, the deck not routed through the copy.
%   pymacos's twin (test_rx_line_endings, ifx) is the one where the CR twin did not load at all.
    properties (Constant)
        ModelSize = 128
    end
    methods (Test)
        function test_three_line_endings_one_answer(tc)
            wd = tempname;  mkdir(wd);  c = onCleanup(@() rmdir(wd, 's'));
            td = fullfile(wd, 'tmpdir');  mkdir(td);
            mm = fileparts(fileparts(mfilename('fullpath')));
            t = {'lf', 'crlf', 'cr'};  d = cellfun(@(x) rx_fixture_path(['Rx_Eol_' x '.in']), t, 'uni', 0);
            scr = fullfile(wd, 'leg.m');  fid = fopen(scr, 'w');
            fprintf(fid, 'setenv(''TMPDIR'', ''%s'');\nrun(''%s'');\nmacos.init(%d);\n', td, fullfile(mm, 'mmacos_setup.m'), tc.ModelSize);
            for k = 1:3
                fprintf(fid, 'macos.load_rx(''%s''); r = macos.trace(macos.num_elt()); fprintf(''RMS %d %%.12e\\n'', r.rmsWFE);\n', d{k}, k);
            end
            fclose(fid);
            [st, out] = system(sprintf('cd %s && "%s" -batch "run(''%s'')" 2>&1', wd, fullfile(matlabroot, 'bin', 'matlab'), scr));
            tc.assertEqual(st, 0, out);
            v = zeros(1, 3);
            for k = 1:3
                g = regexp(out, sprintf('RMS %d (\\S+)', k), 'tokens', 'once');
                tc.assertNotEmpty(g, sprintf('twin %s loads and traces', t{k}));  v(k) = str2double(g{1});
            end
            tc.verifyGreaterThan(v(1), 0);
            tc.verifyEqual(v(2:3), [v(1) v(1)], 'the three twins trace to the identical rms');
            tc.verifyEqual(count(string(out), "CR line endings normalized for this load"), 2, 'one note per CR deck, none for LF');
            for k = 2:3, tc.verifyTrue(contains(out, [d{k} ': CR line endings normalized']), 'the note names the ORIGINAL deck'); end
            tc.verifyFalse(contains(out, 'macos_rx_'), 'the temp name never appears');
            tc.verifyEmpty(dir(fullfile(td, 'macos_rx_*')), 'the LF copy is deleted');
        end
    end
end
