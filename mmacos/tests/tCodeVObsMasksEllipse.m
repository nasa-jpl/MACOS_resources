classdef tCodeVObsMasksEllipse < matlab.unittest.TestCase
%TCODEVOBSMASKSELLIPSE  Port of pymacos TestObsMasksEllipse.
%   Two sub-tests: axis-aligned ellipse and rotated ellipse.

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
                    for side = [1, 2, 4, 6, 8, 10.1]
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
            cases = [2, 4, 0, 0, 30];
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
        function run_ellipse(testCase, srf, dx_fact, line_id, line_id_obs, lines_src)
            cases = tCodeVObsMasksEllipse.mask_sets();
            for i = 1:size(cases, 1)
                a = cases(i, 1); b = cases(i, 2);
                dx = cases(i, 3); dy = cases(i, 4);

                obs_block = [
                    "   nObs=  1"
                    "ObsType=  Elliptical"
                    sprintf(" ObsVec=  %23.16e %23.16e %23.16e %23.16e", ...
                            a, b, dx*dx_fact, dy)
                ];
                lines = lines_src;
                lines(line_id - 1) = "ApType=  None";
                lines(line_id)     = "";
                lines = [lines(1:line_id_obs - 1); obs_block; lines(line_id_obs + 1:end)];

                pts = ray_pos_at_srf_in_tangent_plane( ...
                    testCase.tmp_rx, lines, srf);

                lhs = (b*(pts(:,1)-dx)).^2 + (a*(pts(:,2)-dy)).^2;
                testCase.verifyTrue(all(lhs >= (a*b)^2), sprintf( ...
                    'srf=%d a=%g b=%g dx=%g dy=%g: %d / %d rays inside', ...
                    srf, a, b, dx, dy, sum(lhs < (a*b)^2), numel(lhs)));
            end
        end

        function run_ellipse_rot(testCase, srf, dx_fact, line_id, line_id_obs, lines_src)
            cases = tCodeVObsMasksEllipse.mask_sets_rot();
            for i = 1:size(cases, 1)
                a = cases(i, 1); b = cases(i, 2);
                dx = cases(i, 3); dy = cases(i, 4);
                phi_deg = cases(i, 5);
                phi_ = phi_deg * pi / 180;

                obs_block = [
                    "   nObs=  1"
                    "ObsType=  RotElliptical"
                    sprintf(" ObsVec=  %23.16e %23.16e %23.16e %23.16e %23.16e", ...
                            a, b, dx*dx_fact, dy, phi_)
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

                lhs = (b*pp(:,1)).^2 + (a*pp(:,2)).^2;
                testCase.verifyTrue(all(lhs >= (a*b)^2), sprintf( ...
                    'srf=%d a=%g b=%g dx=%g dy=%g phi=%g: %d / %d inside', ...
                    srf, a, b, dx, dy, phi_deg, sum(lhs < (a*b)^2), numel(lhs)));
            end
        end
    end

    methods (Test)
        function test_parabola_srf2(testCase)
            p = rx_mask_params('parabola');
            lines = readlines(p.rx_path, 'EmptyLineRule', 'read');
            testCase.run_ellipse(2, p.srfs.S2.dx_fact, p.srfs.S2.line_id, p.srfs.S2.line_id_obs, lines);
        end
        function test_parabola_srf4(testCase)
            p = rx_mask_params('parabola');
            lines = readlines(p.rx_path, 'EmptyLineRule', 'read');
            testCase.run_ellipse(4, p.srfs.S4.dx_fact, p.srfs.S4.line_id, p.srfs.S4.line_id_obs, lines);
        end
        function test_parabola_glb_srf3(testCase)
            p = rx_mask_params('parabola_glb');
            lines = readlines(p.rx_path, 'EmptyLineRule', 'read');
            testCase.run_ellipse(3, p.srfs.S3.dx_fact, p.srfs.S3.line_id, p.srfs.S3.line_id_obs, lines);
        end
        function test_parabola_glb_srf5(testCase)
            p = rx_mask_params('parabola_glb');
            lines = readlines(p.rx_path, 'EmptyLineRule', 'read');
            testCase.run_ellipse(5, p.srfs.S5.dx_fact, p.srfs.S5.line_id, p.srfs.S5.line_id_obs, lines);
        end

        function test_rot_parabola_srf2(testCase)
            p = rx_mask_params('parabola');
            lines = readlines(p.rx_path, 'EmptyLineRule', 'read');
            testCase.run_ellipse_rot(2, p.srfs.S2.dx_fact, p.srfs.S2.line_id, p.srfs.S2.line_id_obs, lines);
        end
        function test_rot_parabola_srf4(testCase)
            p = rx_mask_params('parabola');
            lines = readlines(p.rx_path, 'EmptyLineRule', 'read');
            testCase.run_ellipse_rot(4, p.srfs.S4.dx_fact, p.srfs.S4.line_id, p.srfs.S4.line_id_obs, lines);
        end
        function test_rot_parabola_glb_srf3(testCase)
            p = rx_mask_params('parabola_glb');
            lines = readlines(p.rx_path, 'EmptyLineRule', 'read');
            testCase.run_ellipse_rot(3, p.srfs.S3.dx_fact, p.srfs.S3.line_id, p.srfs.S3.line_id_obs, lines);
        end
        function test_rot_parabola_glb_srf5(testCase)
            p = rx_mask_params('parabola_glb');
            lines = readlines(p.rx_path, 'EmptyLineRule', 'read');
            testCase.run_ellipse_rot(5, p.srfs.S5.dx_fact, p.srfs.S5.line_id, p.srfs.S5.line_id_obs, lines);
        end
    end
end
