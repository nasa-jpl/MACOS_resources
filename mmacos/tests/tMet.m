classdef tMet < matlab.unittest.TestCase
    % Sprint 2D S3: engine metrology wrappers (met_calc/met_get ->
    % macos.met) + the Q8-style closed-form checks for SrfMetCalc:
    %   (1) baseline beam lengths == hand-computed point distances,
    %   (2) gauge change under a global translation == the LOS
    %       projection -u.d (analytic),
    %   (3) motion orthogonal to the LOS -> ~zero reading (second
    %       order d^2/2L only).
    % Fixture: e5mono + hand-inserted met blocks — m2 (elt 2) carries
    % two launcher points beamed 1-1/2-2 onto two points on fpa
    % (elt 5).  (Reciprocity swap deferred.)

    properties
        wd
        fixture
        a; b; c; d   % met points (mm, global): a,b on m2; c,d on fpa
    end

    methods (TestClassSetup)
        function make(tc)
            here = fileparts(mfilename('fullpath'));
            res_root = fileparts(fileparts(here));
            tin = fullfile(res_root, 'segmirmaker', 'test_in');
            tc.wd = tempname; mkdir(tc.wd);
            copyfile(fullfile(tin, 'flat.txt'), fullfile(tc.wd, 'flat.txt'));
            % met points: element Vpts (+ offsets), global frame, mm
            tc.a = [0; -5471.177517626807; -21308.82954482988];   % m2 Vpt
            tc.b = tc.a + [100; 0; 0];
            tc.c = [0; -6571.126153057798; 3678.032705099662];    % fpa Vpt
            tc.d = tc.c + [0; 100; 0];
            lines = readlines(fullfile(tin, 'e5mono.in'));
            v = @(p) sprintf('  %.15E  %.15E  %.15E', p);
            im2 = find(strtrim(lines) == "EltName=  m2", 1);
            metm2 = [ ...
                "          nMetPos=  2"; string(v(tc.a)); string(v(tc.b)); ...
                "          tMetElt=  5  2"; ...
                "  1  0"; ...
                "  0  1"];
            lines = [lines(1:im2); metm2; lines(im2+1:end)];
            ifpa = find(strtrim(lines) == "EltName=  fpa", 1);
            metfpa = ["          nMetPos=  2"; string(v(tc.c)); string(v(tc.d))];
            lines = [lines(1:ifpa); metfpa; lines(ifpa+1:end)];
            tc.fixture = fullfile(tc.wd, 'e5mono_met.in');
            writelines(lines, tc.fixture);

            old = cd(tc.wd);
            macos.init(512);
            macos.load_rx(tc.fixture);
            cd(old);
        end
    end

    methods (Test)
        function test_baseline_lengths(tc)
            m = macos.met('native');
            tc.verifyEqual(m.n, 2);
            expect = [norm(tc.a - tc.c); norm(tc.b - tc.d)];
            tc.verifyEqual(m.l, expect, 'RelTol', 1e-12);
            % SI conversion: BaseUnits mm -> metres.
            msi = macos.met();
            tc.verifyEqual(msi.l, expect*1e-3, 'RelTol', 1e-12);
        end

        function test_los_projection_and_null(tc)
            l0 = macos.met('native').l;
            u1 = (tc.a - tc.c)/norm(tc.a - tc.c);   % tgt->src LOS, beam 1
            % (2) translate fpa (the TARGET, elt 5) along a global step
            % with a component on the LOS: dl = -u.d exactly (the model
            % is |s - t|; translation moves t rigidly).
            dmm = [0.05; -0.02; 0.04];
            macos.perturb(5, 'translation', dmm*1e-3, 'frame', 'global');
            l1 = macos.met('native').l;
            tc.verifyEqual(l1(1) - l0(1), -dot(u1, dmm), 'RelTol', 5e-6);
            macos.perturb(5, 'translation', -dmm*1e-3, 'frame', 'global');
            % (3) null test: step orthogonal to beam 1's LOS reads ~0
            % (second order |d|^2/2L ~ 4e-10 mm).
            dperp = cross(u1, [1;0;0]); dperp = 0.01*dperp/norm(dperp);
            macos.perturb(5, 'translation', dperp*1e-3, 'frame', 'global');
            l2 = macos.met('native').l;
            tc.verifyLessThan(abs(l2(1) - l0(1)), 1e-7);
            macos.perturb(5, 'translation', -dperp*1e-3, 'frame', 'global');
        end

        function test_no_met_rx_returns_empty(tc)
            here = fileparts(mfilename('fullpath'));
            res_root = fileparts(fileparts(here));
            tin = fullfile(res_root, 'segmirmaker', 'test_in');
            old = cd(tin); restore = onCleanup(@() cd(old));
            macos.load_rx(fullfile(tin, 'e5mono.in'));
            m = macos.met();
            tc.verifyEqual(m.n, 0);
            tc.verifyEmpty(m.l);
            % restore the met fixture for any later test ordering
            cd(tc.wd);
            macos.load_rx(tc.fixture);
        end

        function test_add_met_stewart_truss(tc)
            % S3 emitter: 6 launchers/segment + 6 around an aft element
            % -> 3 hub fiducials; engine lengths must equal the emitted
            % point geometry exactly (buffer order = source elements in
            % read order, launcher-major).
            here = fileparts(mfilename('fullpath'));
            res_root = fileparts(fileparts(here));
            tin = fullfile(res_root, 'segmirmaker', 'test_in');
            bin = fullfile(res_root, 'segmirmaker', ...
                           'build_release_ifx', 'SegMirMaker');
            tc.assumeTrue(isfile(bin), 'SegMirMaker not built');
            seg = macos.design.segment_rx(fullfile(tin, 'e5mono.in'), ...
                'elt', 1, 'rings', 1, 'grid', 'Pie', 'gap', 50, ...
                'dofs', 6, 'meas_config', 1);
            % after the 7-seg splice: m2 = elt 8 (hub), fpa = elt 11
            am = macos.design.add_met(seg.in, seg, 'hub', 8, ...
                'r_fid', 300, 'nf', 3, 'extra_sources', 11);
            tc.verifyEqual(am.n_beams, 7*6 + 6);
            old = cd(seg.run.workdir); restore = onCleanup(@() cd(old));
            macos.load_rx(am.in);
            m = macos.met('native');
            tc.verifyEqual(m.n, am.n_beams);
            expect = vecnorm(am.src_pts - am.tgt_pts)';
            tc.verifyEqual(m.l, expect, 'RelTol', 1e-12);
            % dldx FD channel vs analytic: straight-line model =>
            % dl/d(local trans) = u.(triad axis), u = (s-t)/|s-t|
            % (source points move with the segment).  Seg1 owns beams
            % 1:6 / columns 1:6; other segments' columns read zero.
            dm = macos.design.dmet_dx([seg.seg_elts 11], ...
                'dstep_trans', 1e-6, 'dstep_rot', 1e-6);
            tc.verifySize(dm.dldx, [48, 48]);
            u = am.src_pts - am.tgt_pts; u = u ./ vecnorm(u);
            f1 = seg.frames(1); T1 = [f1.xhat f1.yhat f1.zhat];
            tc.verifyEqual(dm.dldx(1:6, 4:6), u(:,1:6)'*T1, 'AbsTol', 1e-7);
            tc.verifyEqual(max(abs(dm.dldx(1:6, 7:12)), [], 'all'), 0);
            % rotations couple through the moment arms: nonzero
            tc.verifyGreaterThan(max(abs(dm.dldx(1:6, 1:3)), [], 'all'), 0);
        end
    end
end
