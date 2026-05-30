classdef tCodeVObsMasksPolygon < matlab.unittest.TestCase
%TCODEVOBSMASKSPOLYGON  Port of pymacos TestObsMasksPolygon.
%   Three sub-tests:
%     - hexagon via ObsVec  (local-frame vertices)
%     - rectangle via ObsVec (local-frame vertices)
%     - hexagon via PolyObsVec (global-frame vertices)
%   Pymacos's polygon-obscuration set is the heavyweight slice of
%   test_masks.py (PolyObsVec by itself produces hundreds of cases
%   per Rx/srf combo).

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
        function cases = hex_mask_sets()
            cases = [5, 4, 0];
            for dx_ = [-1, 0, 1]
                for dy_ = [-1, 0, 1]
                    for side = [2, 4, 6, 8]
                        cases(end+1, :) = [side, dx_, dy_]; %#ok<AGROW>
                    end
                end
            end
            for w = linspace(0, 2*pi, 13)
                if w == 2*pi, continue, end
                cases(end+1, :) = [5, cos(w)*4.75, sin(w)*4.75]; %#ok<AGROW>
            end
        end

        function cases = rect_mask_sets()
            cases = zeros(0, 4);
            for dx_ = [-1, 0, 1]
                for dy_ = [-1, 0, 1]
                    for wx = [2, 4, 6]
                        for wy = [3, 5, 7]
                            cases(end+1, :) = [wx, wy, dx_, dy_]; %#ok<AGROW>
                        end
                    end
                end
            end
            for w = linspace(0, 2*pi, 13)
                if w == 2*pi, continue, end
                cases(end+1, :) = [3, 5, cos(w)*4.75, sin(w)*4.75]; %#ok<AGROW>
            end
        end

        function cases = hex_glb_mask_sets()
            cases = [7, 0, 0];
            for dx_ = [-1, 0, 1]
                for dy_ = [-1, 0, 1]
                    for side = [2, 4, 6, 8]
                        cases(end+1, :) = [side, dx_, dy_]; %#ok<AGROW>
                    end
                end
            end
            for w = linspace(0, 2*pi, 13)
                if w == 2*pi, continue, end
                cases(end+1, :) = [5, cos(w)*4.75, sin(w)*4.75]; %#ok<AGROW>
            end
        end
    end

    methods (Access = private)
        function run_hex(testCase, srf, dx_fact, line_id, line_id_obs, lines_src)
            cases = tCodeVObsMasksPolygon.hex_mask_sets();
            for i = 1:size(cases, 1)
                side = cases(i, 1); dx = cases(i, 2); dy = cases(i, 3);
                v = hexagon(side, 0, 0);
                bounds = poly_lines(v);

                obs_block = [
                    "   nObs=  1"
                    "ObsType=  Polygonal"
                    sprintf(" ObsVec=  %23.16e %23.16e 6", dx*dx_fact, dy)
                    sprintf("          %23.16e %23.16e", v(1,1), v(1,2))
                    sprintf("          %23.16e %23.16e", v(2,1), v(2,2))
                    sprintf("          %23.16e %23.16e", v(3,1), v(3,2))
                    sprintf("          %23.16e %23.16e", v(4,1), v(4,2))
                    sprintf("          %23.16e %23.16e", v(5,1), v(5,2))
                    sprintf("          %23.16e %23.16e", v(6,1), v(6,2))
                ];
                lines = lines_src;
                lines(line_id - 1) = "ApType=  None";
                lines(line_id)     = "";
                lines = [lines(1:line_id_obs - 1); obs_block; lines(line_id_obs + 1:end)];

                pts = ray_pos_at_srf_in_tangent_plane( ...
                    testCase.tmp_rx, lines, srf);

                [inside, outside] = chk_polygon_pts(pts - [dx, dy], bounds);
                testCase.verifyTrue(outside && ~inside, sprintf( ...
                    'srf=%d side=%g dx=%g dy=%g: inside=%d outside=%d', ...
                    srf, side, dx, dy, inside, outside));
            end
        end

        function run_rect_poly(testCase, srf, dx_fact, line_id, line_id_obs, lines_src)
            cases = tCodeVObsMasksPolygon.rect_mask_sets();
            for i = 1:size(cases, 1)
                wx = cases(i, 1); wy = cases(i, 2);
                dx = cases(i, 3); dy = cases(i, 4);
                v = rectangular_polygon(wx, wy, 0, 0);
                bounds = poly_lines(v);

                obs_block = [
                    "   nObs=  1"
                    "ObsType=  Polygonal"
                    sprintf(" ObsVec=  %23.16e %23.16e 4", dx*dx_fact, dy)
                    sprintf("          %23.16e %23.16e", v(1,1), v(1,2))
                    sprintf("          %23.16e %23.16e", v(2,1), v(2,2))
                    sprintf("          %23.16e %23.16e", v(3,1), v(3,2))
                    sprintf("          %23.16e %23.16e", v(4,1), v(4,2))
                ];
                lines = lines_src;
                lines(line_id - 1) = "ApType=  None";
                lines(line_id)     = "";
                lines = [lines(1:line_id_obs - 1); obs_block; lines(line_id_obs + 1:end)];

                pts = ray_pos_at_srf_in_tangent_plane( ...
                    testCase.tmp_rx, lines, srf);

                [inside, outside] = chk_polygon_pts(pts - [dx, dy], bounds);
                testCase.verifyTrue(outside && ~inside, sprintf( ...
                    'srf=%d wx=%g wy=%g dx=%g dy=%g: inside=%d outside=%d', ...
                    srf, wx, wy, dx, dy, inside, outside));
            end
        end

        function run_hex_glb(testCase, srf, dx_fact, line_id, line_id_obs, rx_path, dz)
            % Global PolyObsVec form.
            macos.load_rx(rx_path);
            vpt  = macos.get_elt_vpt(srf);
            c    = macos.get_elt_csys(srf);
            csys = c.csys(1:3, 1:3, 1);

            cases = tCodeVObsMasksPolygon.hex_glb_mask_sets();
            lines_src = readlines(rx_path, 'EmptyLineRule', 'read');
            for i = 1:size(cases, 1)
                side = cases(i, 1); dx = cases(i, 2); dy = cases(i, 3);
                v = hexagon(side, 0, 0);
                bounds = poly_lines(v);

                shift = [dx * dx_fact; dy; 0];
                v3 = [v, repmat(dz, 6, 1)].';
                vs = (csys * (shift + v3)).' + vpt.';

                obs_block = [
                    "        nObs=  1"
                    "     ObsType=  Polygonal"
                    "  PolyObsVec=  6"
                    sprintf("              %23.16e %23.16e %23.16e", vs(1,1), vs(1,2), vs(1,3))
                    sprintf("              %23.16e %23.16e %23.16e", vs(2,1), vs(2,2), vs(2,3))
                    sprintf("              %23.16e %23.16e %23.16e", vs(3,1), vs(3,2), vs(3,3))
                    sprintf("              %23.16e %23.16e %23.16e", vs(4,1), vs(4,2), vs(4,3))
                    sprintf("              %23.16e %23.16e %23.16e", vs(5,1), vs(5,2), vs(5,3))
                    sprintf("              %23.16e %23.16e %23.16e", vs(6,1), vs(6,2), vs(6,3))
                ];
                lines = lines_src;
                lines(line_id - 1) = "ApType=  None";
                lines(line_id)     = "";
                lines = [lines(1:line_id_obs - 1); obs_block; lines(line_id_obs + 1:end)];

                pts = ray_pos_at_srf_in_tangent_plane( ...
                    testCase.tmp_rx, lines, srf);

                [inside, outside] = chk_polygon_pts( ...
                    pts - [dx*dx_fact, dy], bounds);
                testCase.verifyTrue(outside && ~inside, sprintf( ...
                    'srf=%d side=%g dx=%g dy=%g dz=%g: inside=%d outside=%d', ...
                    srf, side, dx, dy, dz, inside, outside));
            end
        end
    end

    methods (Test)
        % --- Hexagon via ObsVec ---
        function test_hex_parabola_srf2(testCase)
            p = rx_mask_params('parabola');
            lines = readlines(p.rx_path, 'EmptyLineRule', 'read');
            testCase.run_hex(2, p.srfs.S2.dx_fact, p.srfs.S2.line_id, p.srfs.S2.line_id_obs, lines);
        end
        function test_hex_parabola_srf4(testCase)
            p = rx_mask_params('parabola');
            lines = readlines(p.rx_path, 'EmptyLineRule', 'read');
            testCase.run_hex(4, p.srfs.S4.dx_fact, p.srfs.S4.line_id, p.srfs.S4.line_id_obs, lines);
        end
        function test_hex_parabola_glb_srf3(testCase)
            p = rx_mask_params('parabola_glb');
            lines = readlines(p.rx_path, 'EmptyLineRule', 'read');
            testCase.run_hex(3, p.srfs.S3.dx_fact, p.srfs.S3.line_id, p.srfs.S3.line_id_obs, lines);
        end
        function test_hex_parabola_glb_srf5(testCase)
            p = rx_mask_params('parabola_glb');
            lines = readlines(p.rx_path, 'EmptyLineRule', 'read');
            testCase.run_hex(5, p.srfs.S5.dx_fact, p.srfs.S5.line_id, p.srfs.S5.line_id_obs, lines);
        end

        % --- Rectangle via ObsVec ---
        function test_rect_parabola_srf2(testCase)
            p = rx_mask_params('parabola');
            lines = readlines(p.rx_path, 'EmptyLineRule', 'read');
            testCase.run_rect_poly(2, p.srfs.S2.dx_fact, p.srfs.S2.line_id, p.srfs.S2.line_id_obs, lines);
        end
        function test_rect_parabola_srf4(testCase)
            p = rx_mask_params('parabola');
            lines = readlines(p.rx_path, 'EmptyLineRule', 'read');
            testCase.run_rect_poly(4, p.srfs.S4.dx_fact, p.srfs.S4.line_id, p.srfs.S4.line_id_obs, lines);
        end
        function test_rect_parabola_glb_srf3(testCase)
            p = rx_mask_params('parabola_glb');
            lines = readlines(p.rx_path, 'EmptyLineRule', 'read');
            testCase.run_rect_poly(3, p.srfs.S3.dx_fact, p.srfs.S3.line_id, p.srfs.S3.line_id_obs, lines);
        end
        function test_rect_parabola_glb_srf5(testCase)
            p = rx_mask_params('parabola_glb');
            lines = readlines(p.rx_path, 'EmptyLineRule', 'read');
            testCase.run_rect_poly(5, p.srfs.S5.dx_fact, p.srfs.S5.line_id, p.srfs.S5.line_id_obs, lines);
        end

        % --- Hexagon via PolyObsVec (global) --- 4 srf x dz=(0,75) ---
        function test_hex_glb_parabola_srf2_dz0(testCase)
            p = rx_mask_params('parabola');
            testCase.run_hex_glb(2, p.srfs.S2.dx_fact, p.srfs.S2.line_id, p.srfs.S2.line_id_obs, p.rx_path, 0);
        end
        function test_hex_glb_parabola_srf2_dzp75(testCase)
            p = rx_mask_params('parabola');
            testCase.run_hex_glb(2, p.srfs.S2.dx_fact, p.srfs.S2.line_id, p.srfs.S2.line_id_obs, p.rx_path, 75);
        end
        function test_hex_glb_parabola_srf4_dz0(testCase)
            p = rx_mask_params('parabola');
            testCase.run_hex_glb(4, p.srfs.S4.dx_fact, p.srfs.S4.line_id, p.srfs.S4.line_id_obs, p.rx_path, 0);
        end
        function test_hex_glb_parabola_srf4_dzp75(testCase)
            p = rx_mask_params('parabola');
            testCase.run_hex_glb(4, p.srfs.S4.dx_fact, p.srfs.S4.line_id, p.srfs.S4.line_id_obs, p.rx_path, 75);
        end
        function test_hex_glb_parabola_glb_srf3_dz0(testCase)
            p = rx_mask_params('parabola_glb');
            testCase.run_hex_glb(3, p.srfs.S3.dx_fact, p.srfs.S3.line_id, p.srfs.S3.line_id_obs, p.rx_path, 0);
        end
        function test_hex_glb_parabola_glb_srf3_dzp75(testCase)
            p = rx_mask_params('parabola_glb');
            testCase.run_hex_glb(3, p.srfs.S3.dx_fact, p.srfs.S3.line_id, p.srfs.S3.line_id_obs, p.rx_path, 75);
        end
        function test_hex_glb_parabola_glb_srf5_dz0(testCase)
            p = rx_mask_params('parabola_glb');
            testCase.run_hex_glb(5, p.srfs.S5.dx_fact, p.srfs.S5.line_id, p.srfs.S5.line_id_obs, p.rx_path, 0);
        end
        function test_hex_glb_parabola_glb_srf5_dzp75(testCase)
            p = rx_mask_params('parabola_glb');
            testCase.run_hex_glb(5, p.srfs.S5.dx_fact, p.srfs.S5.line_id, p.srfs.S5.line_id_obs, p.rx_path, 75);
        end
    end
end
