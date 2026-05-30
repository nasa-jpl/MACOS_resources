classdef tCodeVObsMasksCirc < matlab.unittest.TestCase
%TCODEVOBSMASKSCIRC  Port of pymacos TestObsMasksCirc (test_masks.py).
%   Installs a circular OBSCURATION (rays land outside it).  Disables
%   the original aperture mask first.

    properties (Constant)
        ModelSize = 128
    end
    properties
        tmp_rx
    end

    methods (TestClassSetup)
        function setupClass(testCase), macos.init(testCase.ModelSize); end
    end
    methods (TestMethodSetup)
        function setupMethod(testCase)
            testCase.tmp_rx = [tempname(), '.in'];
        end
    end
    methods (TestMethodTeardown)
        function teardownMethod(testCase)
            if exist(testCase.tmp_rx, 'file'), delete(testCase.tmp_rx); end
        end
    end

    methods (Static, Access = private)
        function cases = mask_sets()
            cases = [4.75, 0, 0];
            for dx_ = [-1, 0, 1]
                for dy_ = [-1, 0, 1]
                    for r = [4.1, 8.1, 10.1]
                        cases(end+1, :) = [r, dx_, dy_]; %#ok<AGROW>
                    end
                end
            end
            for w = linspace(0, 2*pi, 13)
                if w == 2*pi, continue, end
                cases(end+1, :) = [4.90, cos(w)*5.0, sin(w)*5.0]; %#ok<AGROW>
            end
        end
    end

    methods (Access = private)
        function run_one_combo(testCase, srf, dx_fact, line_id, line_id_obs, lines_src)
            cases = tCodeVObsMasksCirc.mask_sets();
            for i = 1:size(cases, 1)
                rad = cases(i, 1); dx = cases(i, 2); dy = cases(i, 3);

                obs_block = [
                    "   nObs=  1"
                    "ObsType=  Circular"
                    sprintf(" ObsVec=  %23.16e %23.16e %23.16e", ...
                            rad, dx*dx_fact, dy)
                ];
                lines = lines_src;
                lines(line_id - 1) = "ApType=  None";
                lines(line_id)     = "";
                lines = [lines(1:line_id_obs - 1); obs_block; lines(line_id_obs + 1:end)];

                pts = ray_pos_at_srf_in_tangent_plane( ...
                    testCase.tmp_rx, lines, srf);

                r2 = (pts(:,1) - dx).^2 + (pts(:,2) - dy).^2;
                testCase.verifyTrue(all(r2 >= rad^2), sprintf( ...
                    'srf=%d rad=%g dx=%g dy=%g: %d / %d rays inside obs', ...
                    srf, rad, dx, dy, sum(r2 < rad^2), numel(r2)));
            end
        end
    end

    methods (Test)
        function test_parabola_srf2(testCase)
            p = rx_mask_params('parabola');
            lines = readlines(p.rx_path, 'EmptyLineRule', 'read');
            testCase.run_one_combo(2, p.srfs.S2.dx_fact, p.srfs.S2.line_id, p.srfs.S2.line_id_obs, lines);
        end
        function test_parabola_srf4(testCase)
            p = rx_mask_params('parabola');
            lines = readlines(p.rx_path, 'EmptyLineRule', 'read');
            testCase.run_one_combo(4, p.srfs.S4.dx_fact, p.srfs.S4.line_id, p.srfs.S4.line_id_obs, lines);
        end
        function test_parabola_glb_srf3(testCase)
            p = rx_mask_params('parabola_glb');
            lines = readlines(p.rx_path, 'EmptyLineRule', 'read');
            testCase.run_one_combo(3, p.srfs.S3.dx_fact, p.srfs.S3.line_id, p.srfs.S3.line_id_obs, lines);
        end
        function test_parabola_glb_srf5(testCase)
            p = rx_mask_params('parabola_glb');
            lines = readlines(p.rx_path, 'EmptyLineRule', 'read');
            testCase.run_one_combo(5, p.srfs.S5.dx_fact, p.srfs.S5.line_id, p.srfs.S5.line_id_obs, lines);
        end
    end
end
