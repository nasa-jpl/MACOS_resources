classdef tRxShortAsph < matlab.unittest.TestCase
%TRXSHORTASPH  A short AsphCoef= line pads with zero instead of killing the host.
%   A line with fewer values than nAsphCoef (default 4) was an uncaught
%   end-of-file in the parser's list-directed internal READ -- a Fortran
%   runtime abort that took the MATLAB process with it (TO, dyson5 beat 3,
%   2026-10-01).  Now ReadRealsPad (elt_mod) reads what the line carries,
%   pads the rest with 0 and warns once.  There is no coefficient getter in
%   the bindings, so the gate is a load -> SAVE round trip.  SAVE writes
%   `nAsphCoef= N` (the last NONZERO index) and exactly N values, so a short
%   line of two values SAVEs as N = 2 with those two values, and a line of
%   four nonzero values SAVEs as N = 4 (the control, distinct by construction).
%   The pre-fix engine cannot run the short leg at all (the process dies),
%   which is this gate's non-vacuity.
    properties (Constant)
        ModelSize = 128
        Base      = 'Rx_AsphShort_base.in'   % TO's dyson5 R2 deck (two Aspheric faces)
        Elt       = 2
    end
    methods (Access = private)
        function [p, vals] = variant(tc, wd, mode)
            L = splitlines(string(fileread(rx_fixture_path(tc.Base))));
            iel = find(strtrim(L) == sprintf("iElt=  %d", tc.Elt), 1);
            tc.assertNotEmpty(iel, 'element header');
            ia = iel - 1 + find(startsWith(strtrim(L(iel:end)), "AsphCoef="), 1);
            base = sscanf(char(extractAfter(L(ia), "=")), '%f')';
            tc.assertGreaterThanOrEqual(numel(base), 2, 'the base deck carries coefficients');
            switch mode
                case 'short', vals = base(1:2);                         % two values on the line
                case 'full4', vals = [base(1:2) 1.5e-3 -2.5e-4];         % four NONZERO values
            end
            L(ia) = "         AsphCoef=  " + join(string(arrayfun(@(x) sprintf('%.15E', x), vals, 'uni', 0)), "  ");
            p = fullfile(wd, [mode '.in']);  fid = fopen(p, 'w');  fprintf(fid, '%s\n', L);  fclose(fid);
        end
        function [v, n] = saved_coefs(tc, rx, wd, tag)
            m = macos.Session(tc.ModelSize);
            m.load_rx(rx);
            out = fullfile(wd, [tag '_saved.in']);  m.save_rx(out);
            L = splitlines(string(fileread(out)));
            i2 = find(strtrim(L) == sprintf("iElt=  %d", tc.Elt), 1);
            i3 = find(strtrim(L) == sprintf("iElt=  %d", tc.Elt + 1), 1);
            B = L(i2:i3-1);
            n = sscanf(char(extractAfter(B(startsWith(strtrim(B), "nAsphCoef=")), "=")), '%d');
            ia = find(startsWith(strtrim(B), "AsphCoef="), 1);
            v = sscanf(char(extractAfter(B(ia), "=")), '%f')';
            k = ia + 1;      % SAVE wraps long lists over value-only continuation lines
            while numel(v) < n && k <= numel(B) && ~contains(B(k), "=")
                v = [v, sscanf(char(B(k)), '%f')']; %#ok<AGROW>
                k = k + 1;
            end
        end
    end
    methods (Test)
        function test_short_line_loads_and_saves_its_two_values(tc)
            wd = tempname;  mkdir(wd);  c = onCleanup(@() rmdir(wd, 's'));
            macos.init(tc.ModelSize);
            [rx, vals] = tc.variant(wd, 'short');
            [v, n] = tc.saved_coefs(rx, wd, 'short');
            tc.verifyEqual(n, 2, 'SAVE reports the last nonzero index: the padded tail is zero');
            tc.verifyEqual(v, vals, 'RelTol', 1e-12, 'the two given values survive the pad');
        end
        function test_four_nonzero_values_round_trip(tc)
            wd = tempname;  mkdir(wd);  c = onCleanup(@() rmdir(wd, 's'));
            macos.init(tc.ModelSize);
            [rx, vals] = tc.variant(wd, 'full4');
            [v, n] = tc.saved_coefs(rx, wd, 'full4');
            tc.verifyEqual(n, 4, 'four nonzero coefficients SAVE as four');
            tc.verifyEqual(v, vals, 'RelTol', 1e-12, 'the control: a full line round-trips unchanged');
        end
    end
end
