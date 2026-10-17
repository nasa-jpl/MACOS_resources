classdef tSrt < matlab.unittest.TestCase
%TSRT  macos.srt -- one ray from a normalized source-aperture point (the CLI's SRT through
%   the api's srt_trace; BRIEF_to_bindings B.2, 2026-10-17); twin of pymacos test_srt.py.
%   Deck: the manual's Cassegrain (primary R = 10.8, K = -1 at the origin; detector at z = 1.5).
%   The CLI numbers are what `srt 5` printed (both compilers, 2026-10-17).  The analytic legs
%   come from the geometry: a ray at radius r meets the paraboloid at z = -r^2/(2R), and an
%   on-axis Cassegrain focuses every ray on the detector axis.
    properties (Constant)
        Model = 128
        R = 10.8
        CliPos2 = [-6.0e-01; -4.0e-01; -2.40740740740737e-02]
        CliPos3 = [-1.48921405137844e-01; -9.92809367585626e-02; -4.06568259532358e+00]
        CliPos5 = [-3.20217463656292e-11; -2.13479789401561e-11; 1.5]
        CliDir2 = [1.10617958347559e-01; 7.37453055650393e-02; -9.91123250256060e-01]
        CliDir3 = [2.67432537941554e-02; 1.78288358627703e-02; 9.99483332021240e-01]
        % CLI `ray 40` at MODEL 128 (both compilers): crossings at 2, 3, 5; direction leaving 2
        Ray40Pos2 = [6.12244897959183e-01; 1.75510204081633e+00; -1.59964212442347e-01]
        Ray40Pos3 = [1.52887879747765e-01; 4.38278588610260e-01; -4.09147466310857e+00]
        Ray40Pos5 = [3.28707894237112e-11; 9.42290689920355e-11; 1.5]
        Ray40Dir2 = [-1.10116697619937e-01; -3.15667866510485e-01; -9.42458546015684e-01]
    end
    methods (Static, Access = private)
        function ne = cass()
            home = getenv('MACOS_HOME');
            if isempty(home), home = fullfile(getenv('HOME'), 'dev', 'macos', 'macos_f90'); end
            macos.init(tSrt.Model);
            macos.load_rx(fullfile(fileparts(home), 'docs', 'macos-manual', 'examples', 'Cassegrain.in'));
            ne = macos.num_elt();
        end
    end
    methods (Test)
        function test_srt_reproduces_the_cli(tc)
            ne = tSrt.cass();
            r = macos.srt(0.3, 0.2);
            tc.verifyTrue(r.ok);  tc.verifyEqual([r.status r.fail_elt], [0 0]);
            tc.verifyEqual(r.pos(:,2), tc.CliPos2, 'RelTol', 1e-12, 'AbsTol', 1e-15);
            tc.verifyEqual(r.pos(:,3), tc.CliPos3, 'RelTol', 1e-12, 'AbsTol', 1e-15);
            tc.verifyEqual(r.pos(:,5), tc.CliPos5, 'RelTol', 1e-12, 'AbsTol', 1e-15);
            tc.verifyEqual(r.dir(:,2), tc.CliDir2, 'RelTol', 1e-12, 'AbsTol', 1e-15);
            tc.verifyEqual(r.dir(:,3), tc.CliDir3, 'RelTol', 1e-12, 'AbsTol', 1e-15);
            tc.verifyEqual([r.n_prev r.n_cur], ones(1, 2*ne));
        end
        function test_srt_geometry_paraboloid_sag_and_axial_focus(tc)
            ne = tSrt.cass();
            for xy = [0.3 0.2; 0.5 0; -0.1 0.45].'
                r = macos.srt(xy(1), xy(2));
                p = r.pos(:,2);
                tc.verifyEqual(p(3), -(p(1)^2 + p(2)^2) / (2*tc.R), 'RelTol', 1e-12, 'AbsTol', 1e-15);
                tc.verifyLessThan(hypot(r.pos(1,ne), r.pos(2,ne)), 1e-9, 'focus on axis');
                tc.verifyEqual(r.pos(3,ne), 1.5, 'AbsTol', 1e-12);
                tc.verifyEqual(vecnorm(r.dir), ones(1, ne), 'AbsTol', 1e-12);
            end
        end
        function test_srt_obscured_ray_is_traced_and_flagged(tc)
            ne = tSrt.cass();
            for xy = [0.5 0; 0 0].'
                r = macos.srt(xy(1), xy(2));
                tc.verifyTrue(r.ok);  tc.verifyEqual([r.status r.fail_elt], [1 1]);
                tc.verifyEqual(r.pos(3,ne), 1.5, 'AbsTol', 1e-12);
            end
        end
        function test_srt_end_element_and_refusal(tc)
            tSrt.cass();
            r = macos.srt(0.3, 0.2, 3);
            tc.verifyTrue(all(any(r.pos(:,1:3), 1)));  tc.verifyFalse(any(r.pos(:,4:end), 'all'));
            tc.verifyError(@() macos.srt(0.6, 0), ?MException);
        end
        function test_srt_leaves_the_full_trace_unchanged(tc)
            ne = tSrt.cass();
            macos.trace(ne);  W0 = macos.opd();  K0 = macos.opd_mask();
            macos.srt(0.3, 0.2);
            macos.trace(ne);
            tc.verifyEqual(macos.opd_mask(), K0);
            tc.verifyEqual(macos.opd() .* K0, W0 .* K0, 'AbsTol', 0);
        end
        function test_ray_reproduces_the_cli(tc)
            tSrt.cass();
            r = macos.ray(40);
            tc.verifyTrue(r.ok);  tc.verifyEqual([r.status r.fail_elt], [0 0]);
            tc.verifyEqual(r.pos(:,2), tc.Ray40Pos2, 'RelTol', 1e-12, 'AbsTol', 1e-15);
            tc.verifyEqual(r.pos(:,3), tc.Ray40Pos3, 'RelTol', 1e-12, 'AbsTol', 1e-15);
            tc.verifyEqual(r.pos(:,5), tc.Ray40Pos5, 'RelTol', 1e-12, 'AbsTol', 1e-15);
            tc.verifyEqual(r.dir(:,2), tc.Ray40Dir2, 'RelTol', 1e-12, 'AbsTol', 1e-15);
        end
        function test_ray_geometry_and_chief(tc)
            ne = tSrt.cass();
            for i = [40 77]
                r = macos.ray(i);  p = r.pos(:,2);
                tc.verifyEqual(p(3), -(p(1)^2 + p(2)^2) / (2*tc.R), 'RelTol', 1e-12, 'AbsTol', 1e-15);
                tc.verifyLessThan(hypot(r.pos(1,ne), r.pos(2,ne)), 1e-9);
            end
            tc.verifyEqual(macos.ray(1).pos, macos.srt(0, 0).pos, 'AbsTol', 0);
            t = macos.trace(ne);
            tc.verifyError(@() macos.ray(t.nRays + 1), ?MException);
        end
        function test_ray_leaves_the_full_trace_unchanged(tc)
            ne = tSrt.cass();
            macos.trace(ne);  W0 = macos.opd();  K0 = macos.opd_mask();
            macos.ray(40);  macos.trace(ne);
            tc.verifyEqual(macos.opd_mask(), K0);
            tc.verifyEqual(macos.opd() .* K0, W0 .* K0, 'AbsTol', 0);
        end
    end
end
