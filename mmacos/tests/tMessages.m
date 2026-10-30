classdef tMessages < matlab.unittest.TestCase
%TMESSAGES  macos.messages -- the engine's message buffer (PLAN_CONSOLIDATION 1.2 (ee), 2026-10-22);
%   twin of pymacos test_messages.py.  The load path's notes, warnings and refusals, CALIB's cause
%   lines and the SMACOS loop's 'Unknown command' print to Fortran unit 6, which evalc never sees and
%   which reaches no MATLAB-visible stream on macOS; the engine (emsg_mod) also keeps each line and
%   macos.messages reads it.  In-process (a refused load is a catchable error since PLAN item 1b).
%   Legs: a refused load is named in the buffer AND in load_rx's error; messages(false) keeps, the
%   default empties; past 500 kept lines the first entry counts the dropped ones.  The pre-(ee) mex has
%   no messages_count command: every leg is red.
    properties (Constant)
        Model = 128
    end
    methods (Static, Access = private)
        function p = short_psi_deck(wd)
            s = splitlines(string(fileread(rx_fixture_path('Rx_SaveKeys.in'))));
            k = find(startsWith(strtrim(s), "psiElt="), 1);
            s(k) = "           psiElt=  0.0E+00  0.0E+00";      % two of three values
            p = fullfile(wd, 'short_psi.in');  fid = fopen(p, 'w');  fprintf(fid, '%s\n', s);  fclose(fid);
        end
    end
    methods (Test)
        function test_a_refused_load_is_named_in_the_buffer_and_the_error(tc)
            wd = tempname;  mkdir(wd);  c = onCleanup(@() rmdir(wd, 's')); %#ok<NASGU>
            macos.init(tc.Model);  macos.messages();
            p = tMessages.short_psi_deck(wd);
            try
                macos.load_rx(p);  tc.verifyFail('the short psiElt= deck must be refused');
            catch e
                tc.verifyTrue(contains(e.message, 'Rx load refused: psiElt (elt'), e.message);
            end
            m = macos.messages();
            tc.verifyTrue(any(contains(m, 'Rx load refused: psiElt (elt')), strjoin(m, newline));
            tc.verifyTrue(any(contains(m, 'short_psi.in') & contains(m, 'being loaded')), strjoin(m, newline));
        end
        function test_keep_and_empty(tc)
            macos.init(tc.Model);  macos.messages();
            macos.load_rx(rx_fixture_path('Rx_HdrUnknown.in'));
            a = macos.messages(false);
            tc.verifyTrue(any(contains(a, 'Unidentified header string = OptFooKey')), strjoin(a, newline));
            tc.verifyEqual(macos.messages(), a, 'messages(false) kept the lines');
            tc.verifyEmpty(macos.messages(), 'the default read emptied the buffer');
        end
        function test_the_buffer_counts_what_it_dropped(tc)
            macos.init(tc.Model);  macos.messages();
            nd = 0;
            for k = 1:400                                  % each load keeps several lines; 500 are kept
                macos.load_rx(rx_fixture_path('Rx_HdrUnknown.in'));
                [nl, ~, nd] = mmacos('messages_count');
                if nd > 0, break; end
            end
            tc.assertGreaterThan(nd, 0);  tc.verifyEqual(nl, 500);
            m = macos.messages();
            tc.verifyEqual(m(1), string(sprintf('(%d earlier messages dropped)', nd)));
            tc.verifyNumElements(m, 501);
            tc.verifyEmpty(macos.messages());
        end
    end
end
