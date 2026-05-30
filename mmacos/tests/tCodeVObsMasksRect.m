classdef tCodeVObsMasksRect < matlab.unittest.TestCase
%TCODEVOBSMASKSRECT  Port of pymacos TestObsMasksRect.
%   Two sub-tests: axis-aligned and rotated rectangle obscurations.

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

        function cases = mask_sets_rot()
            cases = [5, 9, 0, 0, 0];
            phi_vals = linspace(-6, 354, 13); phi_vals(end) = [];
            for dx_ = [-1, 0, 1]
                for dy_ = [-1, 0, 1]
                    for side = [4, 8, 10.1]
                        for phi = phi_vals
                            cases(end+1, :) = [1.5*side, side, dx_, dy_, phi]; %#ok<AGROW>
                        end
                    end
                end
            end
            for w = linspace(0, 2*pi, 13)
                if w == 2*pi, continue, end
                for phi = phi_vals
                    cases(end+1, :) = [3, 5, cos(w)*4.75, sin(w)*4.75, phi]; %#ok<AGROW>
                end
            end
        end
    end

    methods (Access = private)
        function run_rect(testCase, srf, dx_fact, line_id, line_id_obs, lines_src)
            cases = tCodeVObsMasksRect.mask_sets();
            for i = 1:size(cases, 1)
                wx = cases(i, 1); wy = cases(i, 2);
                dx = cases(i, 3); dy = cases(i, 4);

                xv = [dx*dx_fact - wx/2, dx*dx_fact + wx/2];
                yv = [dy - wy/2, dy + wy/2];
                xmin_w = min(xv); xmax_w = max(xv);
                ymin_w = min(yv); ymax_w = max(yv);

                obs_block = [
                    "   nObs=  1"
                    "ObsType=  Rectangular"
                    sprintf(" ObsVec=  %23.16e %23.16e %23.16e %23.16e", ...
                            xmin_w, xmax_w, ymin_w, ymax_w)
                ];
                lines = lines_src;
                lines(line_id - 1) = "ApType=  None";
                lines(line_id)     = "";
                lines = [lines(1:line_id_obs - 1); obs_block; lines(line_id_obs + 1:end)];

                pts = ray_pos_at_srf_in_tangent_plane( ...
                    testCase.tmp_rx, lines, srf);

                xv0 = [dx - wx/2, dx + wx/2];
                yv0 = [dy - wy/2, dy + wy/2];
                xmin = min(xv0); xmax = max(xv0);
                ymin = min(yv0); ymax = max(yv0);

                xp = pts(:,1); yp = pts(:,2);
                outside_box = (xp <= xmin) | (xp >= xmax) | (yp <= ymin) | (yp >= ymax);
                testCase.verifyTrue(all(outside_box), sprintf( ...
                    'srf=%d wx=%g wy=%g dx=%g dy=%g: %d rays inside box', ...
                    srf, wx, wy, dx, dy, sum(~outside_box)));
            end
        end

        function run_rect_rot(testCase, srf, dx_fact, line_id, line_id_obs, lines_src)
            cases = tCodeVObsMasksRect.mask_sets_rot();
            for i = 1:size(cases, 1)
                wx = cases(i, 1); wy = cases(i, 2);
                dx = cases(i, 3); dy = cases(i, 4);
                phi_deg = cases(i, 5);
                phi_ = phi_deg * pi / 180;

                obs_block = [
                    "   nObs=  1"
                    "ObsType=  RotRectangular"
                    sprintf(" ObsVec=  %23.16e %23.16e %23.16e %23.16e %23.16e", ...
                            wx, wy, dx*dx_fact, dy, phi_)
                ];
                lines = lines_src;
                lines(line_id - 1) = "ApType=  None";
                lines(line_id)     = "";
                lines = [lines(1:line_id_obs - 1); obs_block; lines(line_id_obs + 1:end)];

                pts = ray_pos_at_srf_in_tangent_plane( ...
                    testCase.tmp_rx, lines, srf);

                c = cos(phi_ * dx_fact);
                s = sin(phi_ * dx_fact);
                pp = ([c, s; -s, c] * (pts - [dx, dy]).').';
                xp = pp(:,1); yp = pp(:,2);
                outside_box = (xp <= -wx/2) | (xp >= wx/2) | (yp <= -wy/2) | (yp >= wy/2);
                testCase.verifyTrue(all(outside_box), sprintf( ...
                    'srf=%d wx=%g wy=%g dx=%g dy=%g phi=%g: %d inside', ...
                    srf, wx, wy, dx, dy, phi_deg, sum(~outside_box)));
            end
        end
    end

    methods (Test)
        function test_parabola_srf2(testCase)
            p = rx_mask_params('parabola');
            lines = readlines(p.rx_path, 'EmptyLineRule', 'read');
            testCase.run_rect(2, p.srfs.S2.dx_fact, p.srfs.S2.line_id, p.srfs.S2.line_id_obs, lines);
        end
        function test_parabola_srf4(testCase)
            p = rx_mask_params('parabola');
            lines = readlines(p.rx_path, 'EmptyLineRule', 'read');
            testCase.run_rect(4, p.srfs.S4.dx_fact, p.srfs.S4.line_id, p.srfs.S4.line_id_obs, lines);
        end
        function test_parabola_glb_srf3(testCase)
            p = rx_mask_params('parabola_glb');
            lines = readlines(p.rx_path, 'EmptyLineRule', 'read');
            testCase.run_rect(3, p.srfs.S3.dx_fact, p.srfs.S3.line_id, p.srfs.S3.line_id_obs, lines);
        end
        function test_parabola_glb_srf5(testCase)
            p = rx_mask_params('parabola_glb');
            lines = readlines(p.rx_path, 'EmptyLineRule', 'read');
            testCase.run_rect(5, p.srfs.S5.dx_fact, p.srfs.S5.line_id, p.srfs.S5.line_id_obs, lines);
        end

        function test_rot_parabola_srf2(testCase)
            p = rx_mask_params('parabola');
            lines = readlines(p.rx_path, 'EmptyLineRule', 'read');
            testCase.run_rect_rot(2, p.srfs.S2.dx_fact, p.srfs.S2.line_id, p.srfs.S2.line_id_obs, lines);
        end
        function test_rot_parabola_srf4(testCase)
            p = rx_mask_params('parabola');
            lines = readlines(p.rx_path, 'EmptyLineRule', 'read');
            testCase.run_rect_rot(4, p.srfs.S4.dx_fact, p.srfs.S4.line_id, p.srfs.S4.line_id_obs, lines);
        end
        function test_rot_parabola_glb_srf3(testCase)
            p = rx_mask_params('parabola_glb');
            lines = readlines(p.rx_path, 'EmptyLineRule', 'read');
            testCase.run_rect_rot(3, p.srfs.S3.dx_fact, p.srfs.S3.line_id, p.srfs.S3.line_id_obs, lines);
        end
        function test_rot_parabola_glb_srf5(testCase)
            p = rx_mask_params('parabola_glb');
            lines = readlines(p.rx_path, 'EmptyLineRule', 'read');
            testCase.run_rect_rot(5, p.srfs.S5.dx_fact, p.srfs.S5.line_id, p.srfs.S5.line_id_obs, lines);
        end
    end
end
