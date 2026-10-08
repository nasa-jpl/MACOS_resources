classdef tTmaLongslit < matlab.unittest.TestCase
%TTMALONGSLIT  The tma_longslit template (templates/10_telescopes/tma_longslit).
%
%   The long-slit TMA front end: the SBG VSWIR zig-zag (Bradley et al. 2024,
%   Fig. 4b) at the dyson5 3k spec, stop at M2, telecentric at the slit.
%   What is asserted, cheapest first:
%     1  first order: the stop-at-M2 solve is telecentric, EFL and back focus
%        exact, NO intermediate focus, and every mirror needs R_t/R_s =
%        1/cos^2(AOI) -- the reason a tilted sphere cannot do it;
%     2  the emitted section meets first order IN THE ENGINE: chiefs within
%        the telecentric tolerance, plate scale, every ray admitted, the chief
%        through the M2 pole (the deck's element stop);
%     3  the construction gate: h^4/h^6 terms on the off-axis sections do not
%        bend the chief off the layout (it stays on every pole and leaves
%        along the slit normal) -- conic + asphere solved together;
%     3b the pole-frame freeform (Surface= Monomial, degree 3..6) changes the
%        figure and leaves the chief path exactly where it was;
%     4  the cone rows BITE (the must-fail leg): a beam 30 % too wide puts
%        rays below F/1.7 and the cone rows charge exactly the hinge of the
%        F/# the engine measures; the nominal seed has no ray below F/1.7;
%     5  the record re-scores: the emitted rung decks in the template
%        directory reproduce their recorded tables (skipped when no record).
%
%   Model 128, 21-point grid (SUITE_FAST); the record itself is model 256.
    properties
        tdir
        P
        FO
        out
    end

    methods (TestClassSetup)
        function setup(tc)
            h = fileparts(mfilename('fullpath'));
            run(fullfile(fileparts(h), 'mmacos_setup.m'));
            tc.tdir = fullfile(fileparts(h), 'templates', '10_telescopes', 'tma_longslit');
            addpath(tc.tdir);
            tc.out = tempname;  mkdir(tc.out);
            tc.P = tma_longslit_params(struct('model', 128, 'ngridpts', 21, 'nfield', 3, 'outdir', tc.out, 'tag', 'ttls'));
            tc.FO = tls_first_order(tc.P);
            macos.init(128);
        end
    end
    methods (TestClassTeardown)
        function teardown(tc)
            if exist(tc.out, 'dir'), rmdir(tc.out, 's'); end
        end
    end

    methods (Test)
        function test_first_order_is_telecentric_with_the_stop_at_M2(tc)
            FO = tc.FO;  P = tc.P;
            tc.verifyLessThan(abs(FO.exit_slope), 1e-12);
            tc.verifyEqual(FO.efl, P.f_m, 'RelTol', 1e-12);
            tc.verifyEqual(FO.bfd, P.legs_m(3), 'RelTol', 1e-12);
            tc.verifyEqual(FO.f_k(3), P.legs_m(2), 'RelTol', 1e-12, 'stop at M2: M2 sits at M3''s front focus');
            tc.verifyTrue(all(isnan(FO.int_focus)), 'this family has no real intermediate image');
            tc.verifyEqual(sign(FO.phi), [1 -1 1], 'concave / convex / concave');
            tc.verifyEqual(FO.Rt./FO.Rs, 1./cosd(P.aoi_deg).^2, 'RelTol', 1e-12, ...
                'the first order needs R_t/R_s = 1/cos^2(AOI): a tilted sphere cannot do it');
        end

        function test_the_section_meets_first_order_in_the_engine(tc)
            P = tc.P;  deck = fullfile(tc.out, 'ttls_section.in');
            G = tls_section(P, tc.FO, deck);
            tc.verifyEqual([G.m.K], -ones(1, 3), 'AbsTol', 1e-12, 'theta = AOI seeds three paraboloids');
            M = tls_measure(P, G, deck);
            tc.verifyLessThan(max(M.chief_deg), P.telecentric_deg);
            tc.verifyEqual(M.plate_local, P.f_m, 'RelTol', 1e-3);
            tc.verifyEqual(M.pass, ones(1, numel(M.pass)));
            tc.verifyLessThan(max(M.stop_miss_m), 1e-9, 'the deck''s element stop puts the chief on the M2 pole');
            tc.verifyTrue(M.clear.pass, 'the seed section clears');
        end

        function test_asphere_keeps_the_chief_on_the_poles(tc)
            P = tc.P;  X = tls_design(P, tc.FO);  X.asph = [200 -150; 100 80; -300 120];
            deck = fullfile(tc.out, 'ttls_asph.in');  G = tls_section(P, X, deck);
            macos.load_rx(deck);  nE = macos.num_elt();
            for k = 1:nE
                s = macos.trace(k);  ri = macos.get_ray_info(s.nRays);
                if k <= 3, ref = G.m(k).pole; else, ref = G.slit.point; end
                tc.verifyLessThan(norm(ri.pos(:, 1) - ref), 1e-12, sprintf('chief on station %d', k));
            end
            tc.verifyLessThan(acos(min(1, abs(ri.dir(:, 1).'*G.slit.normal))), 1e-6, 'chief leaves along the slit normal');
            tc.verifyNotEqual([G.m.K], -ones(1, 3), 'the conics absorbed the aspheres');
        end

        function test_the_pole_frame_freeform_is_first_order_neutral(tc)
            % degree >= 3 in the section frame about the pole: the figure changes, the chief path does not
            P = tc.P;  X = tls_design(P, tc.FO);
            G0 = tls_section(P, X, fullfile(tc.out, 'ttls_mon0.in'));  M0 = tls_measure(P, G0, fullfile(tc.out, 'ttls_mon0.in'), 'clearance', false);
            rng(7);  X.mon = 50*randn(3, size(X.mon_terms, 1));  X.declare_mon = true;
            deck = fullfile(tc.out, 'ttls_mon.in');  G = tls_section(P, X, deck);
            tc.verifyNotEmpty(regexp(fileread(deck), 'Surface=  Monomial', 'once'));
            macos.load_rx(deck);  nE = macos.num_elt();
            for k = 1:nE
                s = macos.trace(k);  ri = macos.get_ray_info(s.nRays);
                if k <= 3, ref = G.m(k).pole; else, ref = G.slit.point; end
                tc.verifyLessThan(norm(ri.pos(:, 1) - ref), 1e-12, sprintf('chief on station %d', k));
            end
            tc.verifyLessThan(acos(min(1, abs(ri.dir(:, 1).'*G.slit.normal))), 1e-9);
            M = tls_measure(P, G, deck, 'clearance', false);
            tc.verifyGreaterThan(max(abs(M.rms_um - M0.rms_um)), 10, 'the freeform reaches the figure (non-vacuous)');
        end

        function test_the_cone_rows_bite(tc)
            % the must-fail leg: a beam 30 % too wide runs below F/1.7, and the cone rows charge EXACTLY the hinge of
            % the F/# the engine measures there; the nominal seed has no ray below F/1.7
            P = tc.P;  X = tls_design(P, tc.FO);  fd = [0 P.strip_half_deg];
            G = tls_section(P, X, fullfile(tc.out, 'ttls_cone0.in'));
            M0 = tls_measure(P, G, fullfile(tc.out, 'ttls_cone0.in'), 'clearance', false, 'fields_deg', fd);
            tc.verifyGreaterThanOrEqual(min([M0.fno_x M0.fno_y]), P.cone_fnum(1), 'the nominal seed: no ray below F/1.7');
            Pw = P;  Pw.D_m = 1.30*P.D_m;  Pw.solve_fields_deg = fd;
            Gw = tls_section(Pw, X, fullfile(tc.out, 'ttls_conew.in'));
            Mw = tls_measure(Pw, Gw, fullfile(tc.out, 'ttls_conew.in'), 'clearance', false, 'fields_deg', fd);
            tc.verifyLessThan(max([Mw.fno_x Mw.fno_y]), Pw.cone_fnum(1) - Pw.cone_tol, 'a 30 % wide beam runs below F/1.7 in both sections');
            [~, Rw] = tls_figure(Pw, X, {'slit_dz'}, 'maxfev', 0, 'quiet', true);
            hinge = @(F) max(0, F - P.cone_fnum(2) - P.cone_tol) + max(0, P.cone_fnum(1) - P.cone_tol - F);
            want = sum((P.w_cone*hinge([Mw.fno_x Mw.fno_y])).^2);
            tc.verifyGreaterThan(want, 0);
            tc.verifyEqual(Rw.rows.cone, want, 'RelTol', 1e-9, 'the cone rows are the hinge of the measured F/#');
        end

        function test_the_record_rescores(tc)
            % PINS THE DECK OF RECORD (R9 ffo, 2026-10-07; was R7 ffw): that rung of tls_figure.mat re-traced in the
            % engine must reproduce its recorded per-field rms / chief / FWHM to 1e-9.  R9 = R7 (the reweighted merit:
            % along-slit spot rows x3, outer fields x2/x3, from R4 -- the pixel floor) + the OFF rows x1000 (chief -
            % centroid across the slit: the point-source shift t5f's chief launch reads as smile, 0.140 -> 0.016 px)
            % with the floor held.  A re-solve that changes the record must re-pin it with the mechanism, never a
            % tolerance bump.
            f = fullfile(tc.tdir, 'tls_figure.mat');
            tc.assumeTrue(exist(f, 'file') == 2, 'no recorded ladder in the template directory');
            Z = load(f);  S = Z.S;  j = find(strcmp({S.rungs.name}, 'ffo'), 1);   % the deck of record BY NAME
            tc.assertNotEmpty(j, 'the record holds the deck of record, R9 ffo');  r = S.rungs(j);
            tc.verifyLessThan(max([r.M.fwhm_x_px r.M.fwhm_y_px]), 1.05, 'R9: every field at the pixel floor');
            tc.verifyGreaterThanOrEqual(min([r.M.fno_x r.M.fno_y]), S.P.cone_fnum(1), 'R9: no ray below F/1.7');
            off = (r.M.y_m - r.M.cy_m)*1e6;  off = off - off(r.M.fields_deg == 0);
            tc.verifyLessThan(max(abs(off)), 0.1, 'R9: the chief - centroid offset across the slit < 0.1 um (R7: 2.43)');
            % re-scored at the class's model (128): a geometric trace does not depend on the model size, and a
            % 128 -> 256 transition inside the fast batch would risk the macos_init_all heap bug for later classes
            deck = fullfile(tc.tdir, sprintf('%s_R%d_%s.in', S.P.tag, j - 1, r.name));
            M = tls_measure(S.P, r.G, deck, 'clearance', false);
            tc.verifyEqual(M.rms_um, r.M.rms_um, 'RelTol', 1e-9);
            tc.verifyEqual(M.chief_deg, r.M.chief_deg, 'AbsTol', 1e-9);
            tc.verifyEqual(M.fwhm_x_px, r.M.fwhm_x_px, 'AbsTol', 1e-9);
        end
    end
end
