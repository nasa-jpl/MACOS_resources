classdef tRxShortCoef < matlab.unittest.TestCase
%TRXSHORTCOEF  Coefficient blocks with short or re-wrapped lines load instead of killing the host.
%   PLAN_CONSOLIDATION item 1 (2026-10-09).  The parser read ZernCoef=, MonCoef= and the
%   FF/MonZern coefficient blocks with bare list-directed internal READs: a line with fewer
%   values than the block expects was an uncaught end-of-file -- a Fortran runtime abort
%   that kills MATLAB when the engine is the mex -- and a block written on ONE line when the
%   parser expected groups of six read the NEXT keyword as data and died the same way.  Now
%   every block goes through ReadCoefBlock (elt_mod): one line OR wrapped six per line, a
%   short line pads with zero and prints one line naming the key, the element and the count,
%   a continuation line with no number on it (the next keyword) is put back.  Sibling of
%   tRxShortAsph (the AsphCoef= / AnaCoef= first cut).
%
%   Legs, on Rx_ShortCoef.in (= macos ZGD_test_files/tst_short_coef.in: element 1 a
%   Surface= Zernike mirror, nZernCoef= 4, a ZernCoef= line of 3 values) and variants of it:
%     1 THE MUST-FAIL LEG, in a matlab -batch SUBPROCESS: the short ZernCoef= deck loads,
%       its 4th coefficient is 0, exactly one note is printed.  The pre-fix mex aborts at the
%       load, so the subprocess exits nonzero -- the engine abort itself, as the red leg.
%     2 ZernCoef= with all 8 of nZernCoef= 8 values on one line (pre-fix: read 6, then the
%       next keyword's line as data -> abort).
%     3 MonZernCoef= wrapped 6 + 1 when 8 are declared (pre-fix: load refused).
%     4 MonCoef= cut to 3 lines of 6 before the next keyword (pre-fix: abort): loads, the 18
%       values survive, and the keyword after the block (lMon=) is still read.
%     5 the control: a full ZernCoef= line round-trips with no pad.
    properties (Constant)
        ModelSize = 128
        Base      = 'Rx_ShortCoef.in'
        Zern3     = "         ZernCoef=  2.0D-04  1.0D-04  5.0D-05"
    end
    methods (Access = private)
        function p = variant(tc, wd, tag, from, to)
            s = string(fileread(rx_fixture_path(tc.Base)));
            for k = 1:numel(from)
                tc.assertEqual(count(s, from(k)), 1, sprintf('variant %s: anchor %d', tag, k));
                s = replace(s, from(k), to(k));
            end
            p = fullfile(wd, [tag '.in']);  fid = fopen(p, 'w');  fprintf(fid, '%s', s);  fclose(fid);
        end
    end
    methods (Test)
        function test_short_zerncoef_loads_in_a_subprocess(tc)
            wd = tempname;  mkdir(wd);  c = onCleanup(@() rmdir(wd, 's'));
            mm = fileparts(fileparts(mfilename('fullpath')));
            scr = fullfile(wd, 'leg1.m');  fid = fopen(scr, 'w');
            fprintf(fid, 'run(''%s'');\nmacos.init(%d);\nmacos.load_rx(''%s'');\n', fullfile(mm, 'mmacos_setup.m'), tc.ModelSize, rx_fixture_path(tc.Base));
            fprintf(fid, 'c = macos.get_elt_zrn_coef(1, (1:4).'');\nfprintf(''COEF %%.10e %%.10e %%.10e %%.10e\\n'', c);\n');
            fclose(fid);
            [st, out] = system(sprintf('cd %s && "%s" -batch "run(''%s'')" 2>&1', wd, fullfile(matlabroot, 'bin', 'matlab'), scr));
            tc.verifyEqual(st, 0, sprintf('the short ZernCoef= deck must not kill MATLAB (pre-fix: the engine aborts at load)\n%s', out));
            t = regexp(out, 'COEF (\S+) (\S+) (\S+) (\S+)', 'tokens', 'once');
            tc.assertNotEmpty(t, 'the subprocess read the coefficients back');
            tc.verifyEqual(str2double(t), [2e-4 1e-4 5e-5 0], 'RelTol', 1e-12, 'three given, the fourth padded with 0');
            tc.verifyEqual(count(string(out), "ZernCoef (elt"), 1, 'exactly ONE note line');
        end
        function test_zerncoef_on_one_line_beyond_a_group(tc)
            wd = tempname;  mkdir(wd);  c = onCleanup(@() rmdir(wd, 's'));
            v = [2e-4 1e-4 5e-5 3e-5 -2e-5 1.5e-5 -1e-5 7e-6];
            p = tc.variant(wd, 'zern8', ["        nZernCoef=  4", tc.Zern3], ...
                ["        nZernCoef=  8", "         ZernCoef=  " + join(compose("%.6E", v), "  ")]);
            macos.init(tc.ModelSize);  macos.load_rx(p);
            tc.verifyEqual(macos.get_elt_zrn_coef(1, (1:8).').', v, 'RelTol', 1e-12, 'all 8 read from one line');
        end
        function test_monzerncoef_wrapped_short_loads(tc)
            wd = tempname;  mkdir(wd);  c = onCleanup(@() rmdir(wd, 's'));
            p = tc.variant(wd, 'monzern', ["          Surface=  Zernike", "         ZernType=  ANSI" + newline + "        nZernCoef=  4" + newline + tc.Zern3], ...
                ["          Surface=  FreeForm", "      MonZernType=  ANSI" + newline + "     nMonZernCoef=   8" + newline + ...
                 "     MonZernModes=  4 5 6 7 8 9 10 11" + newline + ...
                 "      MonZernCoef=  1.0D-04  2.0D-04  3.0D-04  4.0D-04  5.0D-04  6.0D-04" + newline + ...
                 "                    7.0D-04"]);
            macos.init(tc.ModelSize);  macos.load_rx(p);
            tc.verifyEqual(macos.get_elt_mon_zrn_coef(1, (4:11).').', [1:7 0]*1e-4, 'RelTol', 1e-12, ...
                'six + one read, the eighth padded with 0 (pre-fix: the load was refused)');
        end
        function test_moncoef_cut_short_keeps_the_next_keyword(tc)
            wd = tempname;  mkdir(wd);  c = onCleanup(@() rmdir(wd, 's'));
            v = (1:18)*1e-9;  r = @(k) join(compose("%.6E", v(k)), "  ");
            p = tc.variant(wd, 'moncoef', ["          Surface=  Zernike", "         ZernType=  ANSI" + newline + "        nZernCoef=  4" + newline + tc.Zern3], ...
                ["          Surface=  Monomial", "          MonCoef=  " + r(1:6) + newline + "                    " + r(7:12) + newline + "                    " + r(13:18)]);
            macos.init(tc.ModelSize);  macos.load_rx(p);
            out = fullfile(wd, 'saved.in');  macos.save_rx(out);
            L = splitlines(string(fileread(out)));  i1 = find(strtrim(L) == "iElt=  1", 1);  i2 = find(strtrim(L) == "iElt=  2", 1);  B = L(i1:i2-1);
            im = find(startsWith(strtrim(B), "MonCoef="), 1);
            tc.assertNotEmpty(im, 'SAVE writes the MonCoef block');
            w = sscanf(char(extractAfter(B(im), "=")), '%f')';  k = im + 1;
            while k <= numel(B) && ~contains(B(k), "="), w = [w, sscanf(char(B(k)), '%f')']; k = k + 1; end %#ok<AGROW>
            tc.verifyEqual(w(1:18), v, 'RelTol', 1e-12, 'the 18 given values survive');
            tc.verifyEqual(w(19:end), zeros(1, numel(w) - 18), 'the rest padded with 0');
            il = find(startsWith(strtrim(B), "lMon="), 1);
            tc.assertNotEmpty(il, 'the keyword after the block was not eaten');
            tc.verifyEqual(sscanf(char(extractAfter(B(il), "=")), '%f'), 10, 'RelTol', 1e-12, 'lMon= still read (10)');
        end
        function test_full_zerncoef_control(tc)
            wd = tempname;  mkdir(wd);  c = onCleanup(@() rmdir(wd, 's'));
            p = tc.variant(wd, 'full', tc.Zern3, tc.Zern3 + "  3.0D-05");
            macos.init(tc.ModelSize);  macos.load_rx(p);
            tc.verifyEqual(macos.get_elt_zrn_coef(1, (1:4).').', [2e-4 1e-4 5e-5 3e-5], 'RelTol', 1e-12, 'a full line is read unchanged');
        end
    end
end
