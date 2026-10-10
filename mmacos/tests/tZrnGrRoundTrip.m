classdef tZrnGrRoundTrip < matlab.unittest.TestCase
%TZRNGRROUNDTRIP  The IRIS save_rx -> reload SIGSEGV, closed (macos db9236b, CCMac 2026-10-10).
%   Two engine defects that chained on IRIS (REPORT_iris_save_crash.md, CLOSURE):
%     A  SAVE dropped NSCount= (PrtSingleEltInfo never wrote it), so a reloaded
%        non-sequential group lost its hit budget (0 = unlimited), a ray
%        over-searched, the grid surface solve diverged and the grid index
%        xi = (xData.rhom)/dAct grew without bound;
%     B  past 2^31 IDFLOOR(xi) saturated, i0+1 WRAPPED to INT_MIN, the integer
%        bounds test passed and GridMat was read ~2e9 out of bounds -- SIGSEGV,
%        the host killed (the mex takes MATLAB down).
%   Each defect has its own public fixture (= macos ZGD_test_files/tst_zrngr_*.in,
%   copied to the shared Rx corpus): a Cassegrain whose primary is a single-member
%   NSReflector group with a flat 64x64 ZrnGrData grid and NSCount= 1.
%     1 A: load -> trace -> save_rx: the saved deck carries NSCount= 1, and the
%       reload traces to the same ray set and the same OPD.
%     2 B, in a matlab -batch SUBPROCESS (the pre-fix engine can kill the host):
%       the overflow twin (GridSrfdx= 1e-10, xi ~ 1e10) traces to completion with
%       the rejected samples COUNTED (macos.grid_idx_ovf > 0) and the grid inert
%       (the same pass set as the finite deck).
%     3 the control: the finite deck rejects nothing (grid_idx_ovf == 0).
%   Pre-fix (mex at d69b408 = engine 47cd323), MEASURED 2026-10-10:
%     leg 1 FAILS twice over -- the save has no NSCount= line, and the reload
%       is REFUSED with no message (the old writer's blank ZernType= block; the
%       parser read ZernType_MaxMode(0), out of the array).  The red.
%     leg 2 cannot fail on x86_64 Linux, by HARDWARE: an out-of-range double ->
%       INT conversion gives INT_MIN there (cvttsd2si "integer indefinite";
%       floor(1e10) = -2147483648 with gfortran AND ifx, measured), so i0+1
%       does not wrap and the OLD guard (i0 < 1) already catches it -- the
%       overflow deck traces bit-identically to the finite one (max diff 0).
%       ARM saturates to INT_MAX, i0+1 wraps to INT_MIN, the guard passes:
%       B is an Apple-Silicon defect, red only on CCMac's Mac (crash_opd).  A
%       bounds-checked debug build cannot show it either (gfortran's aborts at
%       start-up on a legacy A(1) dummy in math_mod LZERO; ifx's runs clean).
%       Here leg 2 is the no-host-kill + counted regression gate.
    properties (Constant)
        Model = 256           % the decks' nGridpts= 256
        RT    = 'Rx_ZrnGrRoundTrip.in'
        OV    = 'Rx_ZrnGrOverflow.in'
        Grid  = 'tst_zrngr_grid.txt'
        NPass = 32168         % CCMac's CLI count (opd nElt), both decks
    end
    properties
        wd
        cwd0
    end
    methods (TestClassSetup)
        function stage(tc)
            % GridFile= is a bare name resolved from the cwd: run from a scratch
            % copy, which also takes the save_rx output.
            tc.wd = tempname;  mkdir(tc.wd);
            for f = {tc.RT, tc.OV, tc.Grid}
                copyfile(rx_fixture_path(f{1}), fullfile(tc.wd, f{1}));
            end
            tc.cwd0 = cd(tc.wd);
            tc.addTeardown(@() cd(tc.cwd0));
            tc.addTeardown(@() rmdir(tc.wd, 's'));
        end
    end
    methods (Access = private)
        function [np, W] = trace_(tc)
            n  = macos.num_elt();
            tr = macos.trace(n);
            ri = macos.get_ray_info(tr.nRays);
            np = nnz(ri.ok_pass(2:end));   % as the CLI's nPassRays (the chief is ray 1)
            W  = macos.opd();
            tc.assertGreaterThan(np, 0, 'the deck must trace');
        end
    end
    methods (Test)
        function test_save_keeps_nscount_and_the_reload_traces_the_same(tc)
            macos.init(tc.Model);  macos.load_rx(tc.RT);
            [np0, W0] = tc.trace_();
            tc.verifyEqual(np0, tc.NPass, 'nominal pass count (the flat grid is inert)');
            macos.save_rx('saved.in');
            txt = fileread('saved.in');
            % asserted BEFORE any newer api call, so a pre-fix mex fails HERE
            tc.verifyEqual(numel(regexp(txt, '^\s*NSCount=\s*1\s*$', 'lineanchors')), 1, ...
                'the saved deck must carry NSCount= 1 (pre-fix SAVE dropped it)');
            blk = extractBetween(string(txt), "EltName=  Primary", "EltName=  Secondary");
            tc.verifyFalse(contains(blk, "ZernType"), ['no Zernike block for an element that ' ...
                'declares none (the old blank block made the reload refuse SILENTLY)']);
            M0 = macos.opd_mask();
            macos.init(tc.Model);  macos.load_rx('saved.in');
            [np1, W1] = tc.trace_();  M1 = macos.opd_mask();
            tc.verifyEqual(np1, np0, 'the reload passes the same rays');
            tc.verifyEqual(M1, M0, 'the same pupil');
            tc.verifyEqual(W1(M1), W0(M0), 'AbsTol', 1e-15, 'the same OPD');
        end
        function test_overflow_deck_completes_and_counts_in_a_subprocess(tc)
            mm  = fileparts(fileparts(mfilename('fullpath')));
            scr = fullfile(tc.wd, 'ovf_leg.m');  fid = fopen(scr, 'w');
            fprintf(fid, 'run(''%s'');\ncd(''%s'');\nmacos.init(%d);\n', ...
                fullfile(mm, 'mmacos_setup.m'), tc.wd, tc.Model);
            fprintf(fid, ['macos.load_rx(''%s'');\ntr = macos.trace(macos.num_elt());\n' ...
                'ri = macos.get_ray_info(tr.nRays);\nfprintf(''NPASS %%d\\n'', nnz(ri.ok_pass(2:end)));\n' ...
                'fprintf(''NOVF %%d\\n'', macos.grid_idx_ovf());\n'], tc.OV);
            fclose(fid);
            [st, out] = system(sprintf('cd %s && "%s" -batch "run(''%s'')" 2>&1', ...
                tc.wd, fullfile(matlabroot, 'bin', 'matlab'), scr));
            tc.verifyEqual(st, 0, sprintf(['the overflow deck must not kill MATLAB ' ...
                '(pre-fix: GridMat read ~2e9 out of bounds)\n%s'], out));
            np = str2double(regexp(out, 'NPASS (\d+)', 'tokens', 'once'));
            nv = str2double(regexp(out, 'NOVF (\d+)', 'tokens', 'once'));
            tc.verifyEqual(np, tc.NPass, 'the trace completes with the grid inert');
            tc.verifyGreaterThan(nv, 0, 'the off-grid samples are counted, not silent');
        end
        function test_finite_deck_rejects_nothing(tc)
            macos.init(tc.Model);  macos.load_rx(tc.RT);
            tc.trace_();
            tc.verifyEqual(macos.grid_idx_ovf(), 0, ...
                'a finite in-range grid index is the old path, bit for bit');
        end
    end
end
