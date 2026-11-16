classdef tCodeVApeMasksRect < matlab.unittest.TestCase
%TCODEVAPEMASKSRECT  Port of pymacos TestApeMasksRect (test_masks.py).

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
            cases = [7, 7, 0, 0];
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
            cases = tCodeVApeMasksRect.mask_sets();
            for i = 1:size(cases, 1)
                wx = cases(i, 1); wy = cases(i, 2);
                dx = cases(i, 3); dy = cases(i, 4);

                % ApVec = xmin, xmax, ymin, ymax — coords reflect dx_fact.
                xv = [dx*dx_fact - wx/2, dx*dx_fact + wx/2];
                yv = [dy - wy/2, dy + wy/2];
                xmin_w = min(xv); xmax_w = max(xv);
                ymin_w = min(yv); ymax_w = max(yv);

                lines = lines_src;
                lines(line_id - 1) = "ApType=  Rectangular";
                lines(line_id)     = sprintf( ...
                    " ApVec=  %23.16e %23.16e %23.16e %23.16e", ...
                    xmin_w, xmax_w, ymin_w, ymax_w);

                pts = ray_pos_at_srf_in_tangent_plane( ...
                    testCase.tmp_rx, lines, srf);

                % Check rays in the un-flipped bounds (local frame removes
                % the dx_fact convention).
                xv0 = [dx - wx/2, dx + wx/2];
                yv0 = [dy - wy/2, dy + wy/2];
                xmin = min(xv0); xmax = max(xv0);
                ymin = min(yv0); ymax = max(yv0);

                in_x = all(pts(:,1) >= xmin & pts(:,1) <= xmax);
                in_y = all(pts(:,2) >= ymin & pts(:,2) <= ymax);
                testCase.verifyTrue(in_x && in_y, sprintf( ...
                    'srf=%d wx=%g wy=%g dx=%g dy=%g: rays out of bounds', ...
                    srf, wx, wy, dx, dy));
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
