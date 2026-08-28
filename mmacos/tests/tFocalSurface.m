classdef tFocalSurface < matlab.unittest.TestCase
%TFOCALSURFACE  design/src/focal_surface -- the best-focus-surface finder.
%
%   Companion to the 2026-08-28 engine fix that makes FEX/SXP reference a
%   CURVED iElt+1's actual surface.  That fix makes the DECLARED focal
%   surface load-bearing, and this is the tool that measures the real one.
%
%   Fixture: the jwst OTE zoom deck (templates/50_sensitivities/zoom_5x5),
%   the only corpus deck family whose iElt+1 is curved.  Model 256, ng 63,
%   stop 25, 3x3 field grid at the deck's own +-1 arcmin.
%
%   The whole class runs ONE measurement in TestClassSetup -- the scan is
%   ~9 FEX runs plus 5 traces each, too slow to repeat per method.

    properties (Constant)
        FOV   = 2.90888e-4      % half-field (rad) = 1 arcmin
        STOPE = 25              % the FSM is the pupil; deck carries no ApStop
        NG    = 63
        MODEL = 256
    end

    properties
        rx, fs, wd
    end

    methods (TestClassSetup)
        function setup(tc)
            import matlab.unittest.fixtures.PathFixture
            here = fileparts(mfilename('fullpath'));
            root = fileparts(here);
            tc.applyFixture(PathFixture({fullfile(root,'src'), ...
                                         fullfile(root,'design','src')}));
            tc.wd = tempname();  mkdir(tc.wd);
            src = fullfile(root, 'templates', '50_sensitivities', 'zoom_5x5');
            tc.rx = fullfile(tc.wd, 'jwst_ote_designc.in');
            copyfile(fullfile(src, 'jwst_ote_designc.in'), tc.rx);
            copyfile(fullfile(src, 'flat64.txt'), fullfile(tc.wd, 'flat64.txt'));
            old = cd(tc.wd);  c = onCleanup(@() cd(old)); %#ok<NASGU>

            macos.init(tc.MODEL);
            tc.fs = focal_surface(tc.rx, 'fov_rad', tc.FOV, 'grid', '3x3', ...
                'fit', 'sphere', 'stop_elt', tc.STOPE, 'ngridpts', tc.NG, ...
                'model_size', tc.MODEL, 'verbose', false, ...
                'write', fullfile(tc.wd, 'emitted.in'));
        end
    end

    methods (Test)

        function test_cloud_is_complete_and_verified(tc)
            % Every (config,field) point solved, and the a4 the solve
            % claims to null really is nulled: the verify residual must be
            % orders below the a4 the FEX radius started with.
            tc.verifyEqual(size(tc.fs.pts, 1), 9);
            tc.verifyTrue(all([tc.fs.pt.ok]));
            a4_start = max(abs([tc.fs.pt.a4]));
            a4_resid = max(abs([tc.fs.pt.a4_resid]));
            tc.verifyLessThan(a4_resid, 0.05 * a4_start, ...
                'the Z4-null solve must actually null Z4');
        end

        function test_null_radii_are_pinned(tc)
            % TIGHT regression pin, measured in this configuration
            % (STOP-ENFORCED-CHIEF order -- the CLI convention, adopted by
            % ruling 2026-08-28; model 256, ng 63, 3x3 grid).  Grid order
            % is meshgrid over (dthx,dthy) ascending, so points 1,3,7,9
            % are the corners (LL, UL, LR, UR) in the A/B report's naming.
            % (The pre-ruling runner-order values differed by up to
            % 0.72 mm at the -y corners -- the re-aim moves the FEX
            % vertex; the sphere CENTRE moved only 8e-5 mm.)
            want = [-3017.5824141975, -3017.5470569491, -3017.5166544465, ...
                    -3017.5790358204, -3017.5444265155, -3017.5149790258, ...
                    -3017.5721202838, -3017.5392847693, -3017.5118052792];
            tc.verifyEqual([tc.fs.pt.R_null], want, 'RelTol', 1e-7);
        end

        function test_null_radii_match_the_ab_report_stop_orders_now_agree(tc)
            % Cross-check against the A/B report's V4 column
            % (macos/REPORT_wnom_cli_ab.md §5), which measured the
            % focus-nulling radius independently, in the CLI stop order.
            %
            % HISTORY OF THIS TOLERANCE.  focal_surface originally used
            % the runner stop order (stop set once, before the field),
            % which offset these radii by up to 0.76 mm with a systematic
            % sign pattern (-y corners long, +y short) -- this test used
            % to assert that offset was present.  The stop-enforced-chief
            % ruling (Dave 2026-08-28) put BOTH flows in the CLI order,
            % and the offset closed: measured residual 5.5-6.2e-4 mm,
            % near-uniform across corners -- the model-256-vs-128 +
            % report-rounding class, no sign pattern.  The bound sits
            % ~8x above that measurement and ~150x below the old offset,
            % so it fails against either the pre-ruling flow or a
            % returning stop-order asymmetry.
            R = abs([tc.fs.pt.R_null]);
            got  = R([1 3 7 9]);                       % LL UL LR UR
            v4   = [3017.58303, 3017.51721, 3017.57273, 3017.51236];
            tc.verifyEqual(got, v4, 'AbsTol', 5e-3);
        end

        function test_cloud_is_independent_of_the_engine_fix(tc)
            % The cloud point is where a4 = 0, wherever FEX started, so it
            % must not depend on the FEX radius convention.  Measured
            % pre/post the 2026-08-28 fix (fs_pre.mat / fs_post.mat in
            % ~/dev/MACOS_sandbox/xp_tst/fs_fix): the FEX start radius
            % moves by up to 0.487 mm while R_null moves by <= 1.1e-4 mm.
            % Here the same invariant is checked WITHIN one engine: the
            % solved radius must be far from the radius FEX handed it, yet
            % the residual a4 tiny -- i.e. the solve, not FEX, set it.
            dfex = abs([tc.fs.pt.rad] - [tc.fs.pt.R_null]);
            tc.verifyGreaterThan(max(dfex), 1e-2, ...
                'the solve must actually move off the FEX radius');
            tc.verifyLessThan(max(abs([tc.fs.pt.a4_resid])), 1e-6);
        end

        function test_sphere_beats_plane_on_this_deck(tc)
            % Reported, never auto-selected: the chosen model is 'sphere'
            % because the caller said so.  On a deck with a genuinely
            % curved focal surface the plane residual must be far worse --
            % if it were not, the 'sphere' choice would deserve a second
            % look.
            tc.verifyEqual(tc.fs.fit.kind, 'sphere');
            tc.verifyEqual(tc.fs.other.kind, 'plane');
            tc.verifyGreaterThan(tc.fs.other.rms, 20 * tc.fs.fit.rms);
        end

        function test_fitted_radius_agrees_with_the_deck_within_its_sigma(tc)
            % The deliverable number: how good was the deck author?  The
            % fit must land within its OWN reported uncertainty of the
            % deck's declared -3017.5606.  This is deliberately a test of
            % the uncertainty too -- a sigma so tight the deck falls
            % outside it would mean the sigma is wrong, and a sigma so
            % loose it admits anything would mean the fit says nothing.
            R = tc.fs.fit.radius;
            sig = tc.fs.fit.sigma(4);
            tc.verifyEqual(R, abs(tc.fs.deck.kr), 'AbsTol', 2*sig);
            tc.verifyLessThan(sig, 0.02 * R, ...
                'a sigma above 2% of R would make the fit uninformative');
            tc.verifyGreaterThan(sig, 0, 'sigma must be finite and positive');
        end

        function test_it_finds_both_focal_surface_elements(tc)
            % The jwst deck carries the SAME sphere at elt 26 (the focal
            % Return) and elt 28 (the detector Reference).  Auto-detection
            % must find BOTH -- moving one without the other would leave
            % the deck inconsistent.  Element 27, the exit pupil, has a
            % different radius and must NOT be swept in.
            tc.verifyEqual(sort(tc.fs.deck.elts), [26 28]);
            tc.verifyFalse(ismember(27, tc.fs.deck.elts));
        end

        function test_emitted_deck_keeps_the_psi_hemisphere(tc)
            % The pupil_find psi-hemisphere defect is the cautionary tale:
            % the emitted normal is COPIED into the existing element's
            % hemisphere, never derived from a fresh sign rule.
            e = tc.fs.deck.emit;
            tc.verifyGreaterThan(dot(e.psi, tc.fs.deck.psi), 0.99);
            tc.verifyEqual(norm(e.psi), 1, 'AbsTol', 1e-12);
            % Kr sign convention copied too (this deck is Kr < 0).
            tc.verifyEqual(sign(e.kr), sign(tc.fs.deck.kr));
            tc.verifyEqual(e.kc, 0);
        end

        function test_emitted_deck_loads_and_its_fex_radius_follows_the_fit(tc)
            % End of the chain: the emitted deck must load, and with the
            % FIXED engine its FEX radius must be the chief-ray distance
            % to the FITTED surface, not to the old one.
            out = fullfile(tc.wd, 'emitted.in');
            tc.verifyTrue(isfile(out));
            old = cd(tc.wd);  c = onCleanup(@() cd(old)); %#ok<NASGU>
            m = macos.Session(tc.MODEL);
            m.load_rx(out);
            nE = m.num_elt();
            tc.verifyEqual(macos.get_elt_kr(nE), tc.fs.deck.emit.kr, ...
                'RelTol', 1e-9);
            tc.verifyEqual(macos.get_elt_kr(26), tc.fs.deck.emit.kr, ...
                'RelTol', 1e-9);
            m.set_src_sampling(tc.NG);  m.stop(int32(tc.STOPE));
            macos.fex(nE-1);
            xp = macos.get_xp();
            % on axis the fitted-surface leg and the old one differ by the
            % vertex shift only -- a few microns -- so pin loosely but
            % non-vacuously against the ORIGINAL deck's on-axis radius.
            tc.verifyEqual(abs(xp.rad), 3017.5444, 'AbsTol', 1e-2);
        end

        function test_emitted_deck_nulls_the_corner_focus(tc)
            % The point of the whole exercise.  On the emitted deck the
            % engine's own FEX radius lands on the focus-nulling radius,
            % so the corner nominals must collapse to the A/B report's V4
            % column -- the values that report had to construct by hand.
            out = fullfile(tc.wd, 'emitted.in');
            old = cd(tc.wd);  c = onCleanup(@() cd(old)); %#ok<NASGU>
            m = macos.Session(tc.MODEL);
            m.load_rx(out);  m.set_src_sampling(tc.NG);  m.stop(int32(tc.STOPE));
            nom = m.get_src_fov();  wfe = m.num_elt() - 1;
            f = tc.FOV;
            dxy  = [0 0; -f +f; +f +f; -f -f; +f -f];   % C UL UR LL LR
            % V3 = post-fix FEX on the ORIGINAL deck; the emitted deck
            % must be at or below it at every corner.
            v3 = [6.8464e-06, 7.3329e-06, 7.9558e-06, 1.1470e-05, 9.9154e-06];
            got = zeros(1,5);
            for k = 1:5
                v = nom.src_dir(:) + [dxy(k,1); dxy(k,2); 0];
                m.load_rx(out);  m.set_src_sampling(tc.NG);
                m.set_src_fov('src_pos', nom.src_pos, 'src_dir', v/norm(v), ...
                              'zSrc', nom.zSrc);
                m.modify();  m.stop(int32(tc.STOPE));
                macos.fex(wfe);
                m.trace(wfe);
                W = m.opd();  msk = W ~= 0;
                got(k) = sqrt(mean(W(msk).^2));
            end
            tc.verifyLessThanOrEqual(got(2:5), v3(2:5) * 1.02, ...
                'the measured focal surface must not be worse than the declared one');
            tc.verifyLessThan(max(got(2:5) ./ v3(2:5)), 0.9, ...
                'and at the corners it must be materially better');
            % centre field is untouched by a focal-surface change on axis
            tc.verifyEqual(got(1), v3(1), 'RelTol', 5e-3);
        end

        function test_write_refuses_to_overwrite_the_input(tc)
            tc.verifyError(@() focal_surface(tc.rx, 'fov_rad', tc.FOV, ...
                'grid', '2x2', 'stop_elt', tc.STOPE, 'ngridpts', tc.NG, ...
                'model_size', tc.MODEL, 'verbose', false, 'write', tc.rx), ...
                'macos:focal_surface:write');
        end

    end
end
