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
    end
end
