classdef tCodeVApeMasksEllipse < matlab.unittest.TestCase
%TCODEVAPEMASKSELLIPSE  Port of pymacos TestApeMasksEllipse (test_masks.py).
%   For each (rx, srf), installs an elliptical aperture and asserts
%   every passed ray sits inside the analytic ellipse.

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
            cases = [2, 4, 0, 0];
            for dx_ = [-1, 0, 1]
                for dy_ = [-1, 0, 1]
                    for side = [2.1, 4.1, 8.1, 10.1]
                        cases(end+1, :) = [1.5*side, side, dx_, dy_]; %#ok<AGROW>
                    end
                end
            end
            for w = linspace(0, 2*pi, 13)
                if w == 2*pi, continue, end
                cases(end+1, :) = [3, 5, cos(w)*4.75, sin(w)*4.75]; %#ok<AGROW>
            end
        end
    end

    methods (Access = private)
        function run_one_combo(testCase, srf, dx_fact, line_id, lines_src)
            cases = tCodeVApeMasksEllipse.mask_sets();
            for i = 1:size(cases, 1)
                a = cases(i, 1); b = cases(i, 2);
                dx = cases(i, 3); dy = cases(i, 4);

                lines = lines_src;
                lines(line_id - 1) = "ApType=  Elliptical";
                lines(line_id)     = sprintf( ...
                    " ApVec= %23.16e %23.16e %23.16e %23.16e", ...
                    a, b, dx * dx_fact, dy);

                pts = ray_pos_at_srf_in_tangent_plane( ...
                    testCase.tmp_rx, lines, srf);

                % Inside the ellipse: (b*(x-dx))^2 + (a*(y-dy))^2 <= (a*b)^2
                lhs = (b*(pts(:,1)-dx)).^2 + (a*(pts(:,2)-dy)).^2;
                testCase.verifyTrue(all(lhs <= (a*b)^2), sprintf( ...
                    'srf=%d a=%g b=%g dx=%g dy=%g: %d / %d rays outside', ...
                    srf, a, b, dx, dy, sum(lhs > (a*b)^2), numel(lhs)));
            end
        end
    end

    methods (Test)
        function test_parabola_srf2(testCase)
            p = rx_mask_params('parabola');
            lines = readlines(p.rx_path, 'EmptyLineRule', 'read');
            testCase.run_one_combo(2, p.srfs.S2.dx_fact, p.srfs.S2.line_id, lines);
        end
        function test_parabola_srf4(testCase)
            p = rx_mask_params('parabola');
            lines = readlines(p.rx_path, 'EmptyLineRule', 'read');
            testCase.run_one_combo(4, p.srfs.S4.dx_fact, p.srfs.S4.line_id, lines);
        end
        function test_parabola_glb_srf3(testCase)
            p = rx_mask_params('parabola_glb');
            lines = readlines(p.rx_path, 'EmptyLineRule', 'read');
            testCase.run_one_combo(3, p.srfs.S3.dx_fact, p.srfs.S3.line_id, lines);
        end
        function test_parabola_glb_srf5(testCase)
            p = rx_mask_params('parabola_glb');
            lines = readlines(p.rx_path, 'EmptyLineRule', 'read');
            testCase.run_one_combo(5, p.srfs.S5.dx_fact, p.srfs.S5.line_id, lines);
        end
    end
end
