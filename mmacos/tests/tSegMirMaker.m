classdef tSegMirMaker < matlab.unittest.TestCase
    % SegMirMaker batch-mode regression (design-layer Sprint 2D, S0).
    %
    % The macos.design.segmirmaker_run driver scripts the SegMirMaker
    % interactive dialog over stdin.  These tests pin (a) the answer
    % ordering the driver assembles and (b) the generator's output
    % against the committed references in segmirmaker/test_in:
    % a batch run must reproduce e5pie / e5hex2 BYTE-IDENTICALLY
    % (the binary is deterministic and built -fp-model strict).
    %
    % Runs no engine session (the binary is standalone); skips cleanly
    % if the SegMirMaker binary has not been built on this machine.

    properties
        tin     % segmirmaker/test_in
        bin     % SegMirMaker executable
    end

    methods (TestClassSetup)
        function locate(tc)
            here = fileparts(mfilename('fullpath'));            % mmacos/tests
            res_root = fileparts(fileparts(here));              % MACOS_resources
            tc.tin = fullfile(res_root, 'segmirmaker', 'test_in');
            % Locate whichever build tree exists (ifx on Linux, gfortran on
            % macOS / gfortran builds); ifx preferred when both are present.
            smmdir = fullfile(res_root, 'segmirmaker');
            tc.bin = '';
            for tag = ["build_release_ifx", "build_release_gfortran", ...
                       "build_debug_ifx", "build_debug_gfortran"]
                cand = fullfile(smmdir, tag, 'SegMirMaker');
                if isfile(cand), tc.bin = cand; break; end
            end
            tc.assumeTrue(isfolder(tc.tin), ...
                'segmirmaker/test_in not found');
            tc.assumeTrue(~isempty(tc.bin) && isfile(tc.bin), ...
                'SegMirMaker binary not built (source ./makesegmirmaker.sh)');
        end
    end

    methods (Test)
        function test_pie_reproduces_committed(tc)
            out = macos.design.segmirmaker_run( ...
                fullfile(tc.tin, 'e5mono.in'), ...
                'name', 'e5pie', 'grid', 'Pie', 'rings', 1, ...
                'gap', 50, 'dofs', 6, 'meas_config', 1);
            % byte-identical modulo the sign of printed zeros (-0.0E+00 vs
            % 0.0E+00 flips with FP round-off noise -- inconsequential)
            tc.verifyEqual(normalize_zero_signs(fileread(out.presc)), ...
                normalize_zero_signs(fileread(fullfile(tc.tin, 'e5pie.presc'))), ...
                'e5pie.presc not byte-identical to the committed reference');
            tc.verifyEqual(normalize_zero_signs(fileread(out.hx)), ...
                normalize_zero_signs(fileread(fullfile(tc.tin, 'e5pieHx.m'))), ...
                'e5pieHx.m not byte-identical to the committed reference');
        end

        function test_hex2_reproduces_committed(tc)
            % 2 rings, all-adjacency edge sensors (config 2).  Note the
            % grid-type string is echoed verbatim into the .presc
            % header, so the committed lowercase 'hex' must be matched.
            out = macos.design.segmirmaker_run( ...
                fullfile(tc.tin, 'e5mono.in'), ...
                'name', 'e5hex2', 'grid', 'hex', 'rings', 2, ...
                'gap', 50, 'dofs', 6, 'meas_config', 2);
            tc.verifyEqual(normalize_zero_signs(fileread(out.presc)), ...
                normalize_zero_signs(fileread(fullfile(tc.tin, 'e5hex2.presc'))), ...
                'e5hex2.presc not byte-identical to the committed reference');
            tc.verifyEqual(normalize_zero_signs(fileread(out.hx)), ...
                normalize_zero_signs(fileread(fullfile(tc.tin, 'e5hex2Hx.m'))), ...
                'e5hex2Hx.m not byte-identical to the committed reference');
            % Hx dimension sanity from the committed run: 19 segments,
            % 42 shared edges x 2 locations x 3 axes (2026-07-19 model,
            % no anchor row) -> nMeas=252, nState=114.
            hxtxt = fileread(out.hx);
            tc.verifySubstring(hxtxt, 'nMeas=   252;');
            tc.verifySubstring(hxtxt, 'nState=   114;');
        end

        function test_size_args_mutually_exclusive(tc)
            tc.verifyError(@() macos.design.segmirmaker_run( ...
                fullfile(tc.tin, 'e5mono.in'), ...
                'seg_size', 100, 'ap_diam', 8000), ...
                'macos:design:segmirmaker_run:sizeargs');
        end
    end
end
