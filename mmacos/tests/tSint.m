classdef tSint < matlab.unittest.TestCase
%TSINT  macos.sint -- the interpolated-surface setup (the CLI's SINT through the api's sint_cmd;
%   BRIEF_to_bindings B.2, 2026-10-23); twin of pymacos test_sint.py.
%   Deck: Rx_TraceRewrites.in (= macos ZGD tst_trace_rewrites.in), the f = 1 m paraboloid whose
%   element 1 is Surface= Interpolated; with no data it traces as its base conic.  The data file is
%   the deck's name + '.srf1.txt', with or without its '.in' (the CLI's spelling keeps it, the
%   bindings' drops it; each tries its own first), on the 15 x 15 grid of cli_tests/cmd/SINT_short_file.jou
%   (a 1e-6 piston + 1e-4 r^2 deviation, 225 points).  CliRms is what that journal's `opd 2` printed
%   (both compilers, 2026-10-23).  Legs: no file -> not set, both names in macos.messages; the file
%   under either name -> set, the CLI's RMS; a header of 226 for the 225 rows -> the same RMS and the warning (the pre-fix
%   engine traced an unread row: RMS 6.17e-7).  The pre-B.2 mex has no sint_cmd: every leg is red.
    properties (Constant)
        Model = 256
        CliRms = 5.630563199e-07
    end
    properties
        Deck
    end
    methods (TestMethodSetup)
        function load_deck(tc)
            wd = tempname;  mkdir(wd);  tc.addTeardown(@() rmdir(wd, 's'));
            tc.Deck = fullfile(wd, 'Rx_TraceRewrites.in');
            copyfile(rx_fixture_path('Rx_TraceRewrites.in'), tc.Deck);
            macos.init(tc.Model);  macos.load_rx(tc.Deck);  macos.messages();
        end
    end
    methods (Static, Access = private)
        function write_grid(p, nHeader)
            fid = fopen(p, 'w');  fprintf(fid, '%d 1\n', nHeader);
            for i = -7:7
                for j = -7:7
                    x = i*0.0175;  y = j*0.0175;
                    fprintf(fid, '%g %g %.12g\n', x, y, 1e-6 + 1e-4*(x*x + y*y));
                end
            end
            fclose(fid);
        end
    end
    properties (TestParameter)
        Name = {'Rx_TraceRewrites.in.srf1.txt', 'Rx_TraceRewrites.srf1.txt'}
    end
    methods (Test)
        function test_no_data_file_is_not_set_and_says_why(tc)
            r = macos.sint();
            tc.verifyEqual(r.n_set, 0);
            tc.verifyEqual(r.set, [-1, zeros(1, macos.num_elt() - 1)]);
            m = macos.messages();
            for name = ["Rx_TraceRewrites.srf1.txt", "Rx_TraceRewrites.in.srf1.txt"]
                tc.verifyTrue(any(endsWith(strtrim(m), filesep + name + " does not exist.")), name + newline + strjoin(m, newline));
            end
            tc.verifyTrue(any(contains(m, 'not correctly set for element   1')), strjoin(m, newline));
        end
        function test_the_data_file_sets_the_surface_and_traces_as_the_cli(tc, Name)
            t0 = macos.trace(2);
            tSint.write_grid(fullfile(fileparts(tc.Deck), Name), 225);
            r = macos.sint();
            tc.verifyEqual(r.n_set, 1);  tc.verifyEqual(r.set(1), 1);  tc.verifyFalse(any(r.set(2:end)));
            t = macos.trace(2);
            tc.verifyEqual(t.rmsWFE, tc.CliRms, 'RelTol', 1e-5);
            tc.verifyGreaterThan(t.rmsWFE, 1e3 * t0.rmsWFE);
            tc.verifyFalse(any(contains(macos.messages(), 'declares')));
        end
        function test_a_short_file_uses_the_points_read_and_warns(tc)
            tSint.write_grid([tc.Deck '.srf1.txt'], 226);
            r = macos.sint();
            tc.verifyEqual(r.set(1), 1);
            t = macos.trace(2);
            tc.verifyEqual(t.rmsWFE, tc.CliRms, 'RelTol', 1e-5);
            m = macos.messages();
            tc.verifyTrue(any(contains(m, 'element   1') & contains(m, 'declares 226 points, 225 read')), strjoin(m, newline));
        end
    end
end
