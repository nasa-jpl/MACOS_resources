classdef tTraceKeepsRx < matlab.unittest.TestCase
%TTRACEKEEPSRX  Tracing a deck must not change the prescription SAVE writes.
%   PLAN_CONSOLIDATION item 3 (2026-10-09).  CTRACE's Return branch wrote the RUNNING medium index into
%   the Return element's IndRef (IndRef(iElt) = CurIndRef) after every ray.  Nothing in a trace reads a
%   Return's IndRef as a medium (ReturnSrf uses the incoming index; CPROPAGATE resets its medium only
%   after a Refractor or a LensArray), so the stamp's one visible effect was on the prescription: after a
%   LensArray in glass, load -> trace -> SAVE wrote the lenslet index (1.51242597) onto the next Return
%   where load -> SAVE wrote the deck's 1.0 (Rx_SaveKeys.in = macos ZGD_test_files/tst_save_keys.in,
%   elt 9 ExitPupil).  PLAN recorded it as a lensarr trace-time OVERRUN; it is not -- no array bound is
%   crossed.  The gate: SAVE after load -> trace(nElt) == SAVE after load, byte for byte.  Red pre-fix:
%   the IndRef line of the Return.
    properties (Constant)
        ModelSize = 256
    end
    methods (Test)
        function test_save_after_a_trace_equals_save_after_load(tc)
            wd = tempname;  mkdir(wd);  c = onCleanup(@() rmdir(wd, 's'));
            copyfile(rx_fixture_path('Rx_SaveKeys.in'), fullfile(wd, 'Rx_SaveKeys.in'));
            copyfile(rx_fixture_path('tst_save_ampl.dat'), fullfile(wd, 'tst_save_ampl.dat'));
            old = cd(wd);  back = onCleanup(@() cd(old));
            m = macos.Session(tc.ModelSize);
            m.load_rx('Rx_SaveKeys.in');  m.save_rx(fullfile(wd, 'a.in'));
            m.load_rx('Rx_SaveKeys.in');  m.trace(m.num_elt());  m.save_rx(fullfile(wd, 'b.in'));
            A = splitlines(string(fileread(fullfile(wd, 'a.in'))));  B = splitlines(string(fileread(fullfile(wd, 'b.in'))));
            tc.assertEqual(numel(B), numel(A), 'the two SAVEs have the same number of lines');
            d = find(A ~= B);
            tc.verifyEmpty(d, sprintf('SAVE after a trace differs from SAVE after load at line(s) %s: %s', mat2str(d(:)'), strjoin(B(d), ' | ')));
        end
    end
end
