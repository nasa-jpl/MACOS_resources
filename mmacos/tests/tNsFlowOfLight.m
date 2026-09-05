classdef tNsFlowOfLight < matlab.unittest.TestCase
%TNSFLOWOFLIGHT  Non-sequential candidate selection obeys flow of light.
%   Gates the surfsub LNSFlowL fix (2026-09-05, BRIEF_luis_round3): the
%   NS candidate probe used ConSrf's |L^2-mpr| proximity metric, which
%   has no flow-of-light sense.  Two measured failure flavors on the
%   PRE-fix engine, both asserted here as physical laws (self-contained
%   -- no reference files):
%     * reflective (Rx_Cass_NS.in, full-disk launch): every ray inside
%       launch r ~ 1.005 took a wrong turn (the metric flipped between
%       the secondary hyperboloid's two sheets) and died at the flat
%       Return -- a razor-sharp unphysical boundary far outside the
%       physical secondary shadow (r 0.52).
%     * refractive (Luneberg.in, the manual's own NS example): 20 of
%       100 fan rays SKIPPED a shell-exit refraction (a sphere entered
%       but never exited -- impossible through nested concentric
%       shells).
%   The corner-cube example (flat NS faces, single-root) is bit-stable
%   across the fix and is smoke-checked for full ray survival.
%   Rx_Luneberg.in / Rx_CornerCube.in are verbatim copies of the manual
%   examples (docs/macos-manual/examples/) so the gate is hermetic.

    properties (Constant)
        ModelSize = 256
    end

    methods (TestClassSetup)
        function setupClass(testCase) %#ok<MANU>
            macos.init(256);
        end
    end

    methods (Test)
        function test_ns_wrong_turns_confined_to_physical_shadow(testCase)
            % Full-disk launch of the NS Cass twin: geometric losses must
            % be confined to the secondary's physical shadow (r <= 0.55).
            % Pre-fix engines lose every ray inside r ~ 1.005.
            src = fileread(rx_fixture_path('Rx_Cass_NS.in'));
            src = strrep(src, 'Obscratn=2.4E+00', 'Obscratn=0.0E+00');
            tmp = [tempname '.in'];
            fd = fopen(tmp, 'w');  fwrite(fd, src);  fclose(fd);
            cln = onCleanup(@() delete(tmp)); %#ok<NASGU>
            m = macos.Session(testCase.ModelSize);
            m.load_rx(tmp);
            s1 = m.trace(1);
            r1 = m.get_ray_info(s1.nRays);
            rr = hypot(r1.pos(1,:), r1.pos(2,:));
            m.trace(m.num_elt() - 1);
            st = m.get_ray_status(s1.nRays);
            miss = st.status(:).' == 2;
            testCase.verifyGreaterThan(nnz(st.status == 0), 0.9 * s1.nRays * (1 - 0.55^2/4), ...
                'annulus survival collapsed');
            testCase.verifyEqual(nnz(miss & (rr > 0.55)), 0, ...
                'NS wrong turns outside the physical secondary shadow');
        end

        function test_luneberg_shell_crossings_pair(testCase)
            % Nested concentric shells: every sphere a ray enters must be
            % exited -- each shell element crossed an EVEN number of times,
            % radii unimodal (in, then out).  Pre-fix: 20 of 100 fan rays
            % skip an exit refraction.
            m = macos.Session(testCase.ModelSize); %#ok<NASGU>
            macos.load_rx(rx_fixture_path('Rx_Luneberg.in'));
            b = macos.draw_rays3d('YZ');
            n_pair = 0;  n_uni = 0;  nr = 0;
            for r = 1:b.nray
                np = b.nper(r);
                el = b.elt(1:np, r);
                sh = el >= 1 & el <= 14;          % Shell1..Shell14
                if nnz(sh) < 2, continue; end
                nr = nr + 1;
                c = histcounts(el(sh), 0.5:1:14.5);
                if any(mod(c, 2) ~= 0), n_pair = n_pair + 1; end
                rr = vecnorm(b.P(:, sh, r));
                d = diff(rr);
                turn = find(d > 1e-12, 1);
                if ~isempty(turn) && any(d(turn:end) < -1e-12)
                    n_uni = n_uni + 1;
                end
            end
            testCase.verifyGreaterThan(nr, 50, 'fan did not trace');
            testCase.verifyEqual(n_pair, 0, ...
                sprintf('%d rays enter a shell and never exit it', n_pair));
            testCase.verifyEqual(n_uni, 0, ...
                sprintf('%d rays have non-unimodal crossing radii', n_uni));
        end

        function test_corner_cube_survives(testCase)
            % Flat NS faces (single-root class): the fix is a no-op there;
            % every ray retroreflects.
            m = macos.Session(testCase.ModelSize);
            m.load_rx(rx_fixture_path('Rx_CornerCube.in'));
            s = m.trace(m.num_elt());
            st = m.get_ray_status(s.nRays);
            testCase.verifyEqual(nnz(st.status ~= 0), 0, ...
                'corner cube lost rays');
        end
    end
end
