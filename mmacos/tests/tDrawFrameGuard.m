classdef tDrawFrameGuard < matlab.unittest.TestCase
    % macos.design.assert_draw_frame_global -- the consumption-side guard on
    % reading macos.draw_rays' U/V as global coordinates.
    %
    % draw_rays projects onto the SOURCE / RAY-GRID basis (DRAW sets
    % xDraw/yDraw from xGrid/yGrid/zGrid), NOT the global frame.  The design
    % layer pins its own emission to xGrid=(+1,0,0) so its consumers may read
    % U/V as global; heritage / SegMirMaker decks carry xGrid=(-1,0,0) and
    % break that silently.  Three Telescope sites (aperture_full_field,
    % draw_plane_, resolve_section_poles_) depend on the pin, so they now
    % assert it.
    %
    % These tests use REAL decks from the segmentation corpus -- one of each
    % handedness -- and cross-check against draw_rays3d, which returns true
    % global (x,y,z) (engine Draw3DVec = RayPos verbatim).  That is arbiter B
    % of the round-2 routing audit; see segmirmaker/SEG_AUDIT_STATUS.md.

    properties
        heritage   % a deck with xGrid = (-1,0,0)  (the e5/e2e corpus)
        globalfr   % a deck with xGrid = (+1,0,0)
    end

    methods (TestClassSetup)
        function locate(tc)
            here = fileparts(mfilename('fullpath'));         % mmacos/tests
            res_root = fileparts(fileparts(here));
            tc.heritage = fullfile(res_root, 'segmirmaker', 'test_in', 'e5pie.in');
            tc.globalfr = fullfile(getenv('HOME'), 'dev', 'macos', ...
                                   'ZGD_test_files', 'SegDemo3.in');
            tc.assumeTrue(isfile(tc.heritage), 'heritage fixture missing');
            tc.assumeTrue(isfile(tc.globalfr), 'global-frame fixture missing');
            macos.init(512);
        end
    end

    methods (Test)
        function test_fixture_handedness_is_engine_truth(tc)
            % The premise of every other test here: confirm from the ENGINE
            % (src_mod xGrid via get_src_csys), not from the file text.
            load_from(tc.heritage);
            s = macos.get_src_csys();
            tc.verifyLessThan(s.xDir(1), -0.9, 'heritage fixture is not xGrid=-x');
            load_from(tc.globalfr);
            s = macos.get_src_csys();
            tc.verifyGreaterThan(s.xDir(1), 0.9, 'global fixture is not xGrid=+x');
        end

        function test_guard_fires_on_heritage_deck(tc)
            load_from(tc.heritage);
            tc.verifyError(@() macos.design.assert_draw_frame_global('XY', 'tDrawFrameGuard'), ...
                'macos:design:drawFrame:notGlobal');
            % 'XZ' also reads xGrid (as its V axis) -> must fire too
            tc.verifyError(@() macos.design.assert_draw_frame_global('XZ', 'tDrawFrameGuard'), ...
                'macos:design:drawFrame:notGlobal');
        end

        function test_guard_passes_on_global_frame_deck(tc)
            load_from(tc.globalfr);
            macos.trace(macos.num_elt());   % establish the ray grid (zGrid)
            s = macos.get_src_csys();
            tc.assumeGreaterThan(norm(s.zDir), 0.9, 'zGrid not established');
            macos.design.assert_draw_frame_global('XY', 'tDrawFrameGuard');
            macos.design.assert_draw_frame_global('XZ', 'tDrawFrameGuard');
            macos.design.assert_draw_frame_global('YZ', 'tDrawFrameGuard');
        end

        function test_guard_rejects_bad_plane(tc)
            load_from(tc.globalfr);
            tc.verifyError(@() macos.design.assert_draw_frame_global('QQ', 'x'), ...
                'macos:design:drawFrame:plane');
        end

        function test_what_the_guard_prevents(tc)
        % The substance: on the heritage deck b.U really IS minus global X,
        % so an unguarded read mirrors the pupil.  Compare draw_rays' U/V
        % against draw_rays3d's TRUE GLOBAL x/y for the very same crossings
        % (both getters enumerate crossings identically -- draw_rays3d docs).
            load_from(tc.heritage);
            macos.trace(macos.num_elt());
            b2 = macos.draw_rays('XY',   1, 7);
            b3 = macos.draw_rays3d('XY', 1, 7);
            sel = b2.elt > 0;
            tc.assumeGreaterThan(nnz(sel), 50, 'not enough crossings to compare');
            gx = squeeze(b3.P(1,:,:));  gy = squeeze(b3.P(2,:,:));
            U  = b2.U(sel);  V = b2.V(sel);  GX = gx(sel);  GY = gy(sel);
            sc = max(abs(GX));
            % U is MINUS global X on this deck ...
            tc.verifyLessThan(max(abs(U + GX))/sc, 1e-6, ...
                'b.U is not -X here; the fixture''s handedness assumption changed');
            % ... i.e. reading U as +X is wrong by the full pupil diameter
            tc.verifyGreaterThan(max(abs(U - GX))/sc, 1.0, ...
                'reading b.U as global X should be grossly wrong on a -xGrid deck');
            % ... and V is NOT global Y either, though it looks close: this
            % deck's yGrid carries the source tilt (0, ~1, 3.49e-4), so
            % V = Y*yGrid(2) + Z*yGrid(3).  Verify V is exactly that
            % projection of the true global position -- which is the second
            % reason there is no clean inverse back to global x,y, and why the
            % guard errors instead of transforming.
            sg = macos.get_src_csys();
            gz = squeeze(b3.P(3,:,:));  GZ = gz(sel);
            Vproj = GX*sg.yDir(1) + GY*sg.yDir(2) + GZ*sg.yDir(3);
            tc.verifyLessThan(max(abs(V - Vproj))/sc, 1e-9, ...
                'b.V is not RayPos.yGrid -- draw_rays/draw_rays3d disagree');
            tc.verifyGreaterThan(max(abs(V - GY))/max(abs(GY)), 1e-3, ...
                'b.V should differ from global Y here (yGrid carries the source tilt)');
        end

        function test_design_layer_path_still_runs(tc)
        % The guard must not fire on the design layer's own decks -- it pins
        % xGrid=(+1,0,0) on purpose.  Exercise a real Telescope through the
        % guarded call site rather than calling the predicate directly.
            t = macos.design.Telescope('family', 'TMA', ...
                    'aperture_diameter_m', 1.0, 'model_size', 512, 'grid_npts', 33);
            t.add_mirror('M1','radius_m',8.0,'spacing_after_m',3.0);
            t.add_mirror('M2','radius_m',2.0,'spacing_after_m',4.5);
            t.add_mirror('M3','radius_m',4.0,'spacing_after','derive');
            t.build();
            macos.design.assert_draw_frame_global('XY', 'tDrawFrameGuard');
            s = macos.get_src_csys();
            tc.verifyGreaterThan(s.xDir(1), 0.9, ...
                'design layer no longer emits xGrid=(+1,0,0) -- the three guarded sites assume it');
        end
    end
end

function load_from(rx)
% GridFile= paths in these decks are relative to the deck's own directory.
here = pwd;  c = onCleanup(@() cd(here)); %#ok<NASGU>
cd(fileparts(rx));
macos.load_rx(rx);
end
