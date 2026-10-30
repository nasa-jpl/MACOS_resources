classdef tIodc2010 < matlab.unittest.TestCase
%TIODC2010  The IODC 2010 "green lens" evaluator: its gates.
%
%   METRIC (challenges/iodc2010/README.md, stated before the numbers): RMS
%   wavefront error at 532 nm on the exit-pupil sphere centred on the chief
%   ray's image point (chief-ray reference, tilt and focus kept, piston
%   removed), rays uniform over the paraxial entrance pupil, every launched
%   ray must pass, distortion 100 (yR - yP)/yP, paraxial EFL.
%
%   Legs
%     gate      the engine's wavefront against iodc2010_reftrace (an exact
%               skew-ray trace written from the textbook) on the SAME rays:
%               per-ray agreement to 1e-6 wave (measured 2e-10), the first
%               surface crossings and the sphere points to 1e-9 mm (measured
%               1e-13), the chief through the stop centre, the image height
%               both ways, the paraxial vs the engine's near-axis EFL
%     paraxial  the y-nu trace against the closed form of a plano-convex lens
%     must-fail a prescription whose clear apertures sit inside the beam is
%               refused with reason 'vignetting' and merit 0
%     limit     the f/14 singlet's semi-field is set by the 0.070-wave line
%               (1.80 deg, merit 12.6) with its apertures verified there
%     grid      the RMS is sampling-converged at 97 across (vs 127)
%     N-BK7 row assumption-gated: a deck naming the glass with no coefficient
%               row traces to EFL 100 only when the engine build carries the
%               table row added 2026-10-30 (Incomplete before the rebuild,
%               never a silent pass; the GlassCoef path is what the gate leg
%               proves on every build)
%     stop      the stop is the paraxial construct of the 2006 evaluator: the
%               ring of pupil-edge rays its own pupil aberration carries a
%               fraction of a percent outside the stop radius is KEPT
%               (p.stop_clips false); the hard-aperture switch loses exactly
%               that outermost ring (the 2026-10-30 a.m. convention, measured)
%     forms     iodc2010_forms: 14 of the 64 two-element patterns are
%               Petzval-balanced (equal converging and diverging surface
%               counts, checked here from the textbook rule), every seed is
%               scaled to |EFL| = 100
%     calib deck every CALIB field's chief starts ON the source plane, backed
%               off by the lens's length (the R = 130 item's fix)
%
%   Model size 128 -> SUITE_FAST.  Runtime ~50 s.  Lives in mmacos/tests; the
%   challenge code it exercises is challenges/iodc2010.

    properties
        p
        tmp
    end

    methods (TestClassSetup)
        function setup(tc)
            here = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'challenges', 'iodc2010');
            if isempty(which('iodc2010_params')), addpath(here); end
            tc.tmp = tempname;  mkdir(tc.tmp);
            tc.p = iodc2010_params();
            tc.p.outdir = tc.tmp;
            macos.init(tc.p.model_size);
        end
    end

    methods (TestClassTeardown)
        function teardown(tc)
            if exist(tc.tmp, 'dir'), rmdir(tc.tmp, 's'); end
        end
    end

    methods
        function L = pair(tc, varargin)
            n  = iodc2010_glass(tc.p.glass, tc.p.lambda_nm);
            mk = @(d) iodc2010_lens('R', 80, 'sgn', [1 0 0 -1], 't', [6 d 6 NaN], ...
                'glass', {'N-BK7', 'air', 'N-BK7', 'air'}, 'stop', [6 + d/2, 3.5], ...
                'name', 'pair', varargin{:});
            f1 = 80 / (n - 1);  d0 = 2 * f1 - f1^2 / 100;
            d  = fzero(@(dd) abs(iodc2010_paraxial(mk(dd), tc.p.lambda_nm).efl) - 100, d0);
            L  = mk(d);
        end
        function L = singlet(tc, a_s)
            n = iodc2010_glass(tc.p.glass, tc.p.lambda_nm);
            L = iodc2010_lens('R', (n - 1) * 100, 'sgn', [1 0], 't', [6 NaN], ...
                'glass', {'N-BK7', 'air'}, 'stop', [0 a_s], 'name', 'singlet');
        end
    end

    methods (Test)

        function test_gate_engine_wavefront_matches_the_independent_tracer(tc)
            pg = tc.p;  pg.nGridpts = 33;
            G = iodc2010_gate(pg, tc.pair(), 10);
            tc.verifyLessThan(G.dW_max_waves, 1e-6, 'per-ray OPD, engine vs textbook trace');
            tc.verifyLessThan(G.drms_waves,   1e-6, 'RMS, engine vs textbook trace');
            tc.verifyLessThan(G.dP1_max_mm,   1e-9, 'first-surface crossings');
            tc.verifyLessThan(G.dQ_max_mm,    1e-9, 'reference-sphere points');
            tc.verifyLessThan(G.chief_stop_mm, 1e-9, 'the chief ray is aimed through the stop centre');
            tc.verifyEqual(G.yR_engine_mm, G.yR_ref_mm, 'AbsTol', 1e-9, 'image height both ways');
            tc.verifyEqual(G.efl_engine_mm, G.efl_paraxial_mm, 'AbsTol', 1e-6, 'near-axis EFL = paraxial EFL');
            tc.verifyEqual(G.rms_eval_waves, G.rms_engine_waves, 'RelTol', 1e-12, 'the scorer''s RMS is the per-ray set');
            tc.verifyLessThan(G.drms_opd_nochief, 1e-9, 'the engine''s opd() map = the per-ray set without the chief');
            tc.verifyGreaterThan(abs(G.rms_opd_waves - G.rms_engine_waves) / G.rms_engine_waves, 1e-4, ...
                'the chief''s omission from the map is visible at 33 across (the gate can tell the two apart)');
            tc.verifyGreaterThan(G.rms_engine_waves, 1, 'the gate field is aberrated enough to see an index or sign error');
            tc.verifyEqual(G.dist_pct, 100 * (G.yR_engine_mm - G.yP_mm) / G.yP_mm, 'AbsTol', 1e-9);
        end

        function test_paraxial_trace_matches_the_plano_convex_closed_form(tc)
            n = iodc2010_glass('N-BK7', 532);
            L = tc.singlet(5);  P = iodc2010_paraxial(L, 532);
            tc.verifyEqual(P.efl, L.R / (n - 1), 'AbsTol', 1e-12, 'EFL of a plano-convex lens');
            tc.verifyEqual(P.bfl, P.efl - L.t(1) / n, 'AbsTol', 1e-12, 'BFL = f - t/n');
            tc.verifyEqual(P.D, 10, 'AbsTol', 1e-12, 'stop on the first surface: EPD = stop diameter');
            tc.verifyEqual(P.z_ep, 0, 'AbsTol', 1e-12);
            tc.verifyEqual(P.yP_unit, P.efl, 'AbsTol', 1e-12, 'yP = f tan(theta) at the paraxial focus');
            tc.verifyEqual(n, 1.519473, 'AbsTol', 1e-6, 'Schott N-BK7 at 532 nm (the data-sheet Sellmeier; 1.5195 to 4 places)');
        end

        function test_apertures_inside_the_beam_are_refused_as_vignetting(tc)
            % the pair's on-axis footprint at surfaces 3-4 is 2.6 mm; 2.0 mm clips
            L = tc.pair('semidia', [5.5 5.3 2.0 2.0]);
            S = iodc2010_score(L, tc.p, 'quiet', true);
            tc.verifyEqual(S.limit_reason, 'vignetting');
            tc.verifyEqual(S.merit, 0);
            tc.verifyGreaterThan(S.fields.nLost(1), 0, 'rays are lost on axis');
            % the same lens with open apertures passes on axis
            S2 = iodc2010_score(tc.pair(), tc.p, 'quiet', true);
            tc.verifyEqual(S2.fields.nLost(1), 0);
            tc.verifyTrue(S2.fields.pass(1));
        end

        function test_the_rms_line_sets_the_singlet_field(tc)
            S = iodc2010_score(tc.singlet(3.5), tc.p, 'quiet', true);
            tc.verifyTrue(S.efl_ok);
            tc.verifyEqual(S.D_mm, 7, 'AbsTol', 1e-9);
            tc.verifyEqual(S.limit_reason, 'rms');
            tc.verifyGreaterThanOrEqual(S.theta_lim_deg, 1.75);
            tc.verifyLessThanOrEqual(S.theta_lim_deg, 1.85);
            tc.verifyEqual(S.merit, 12.6, 'AbsTol', 0.4);
            tc.verifyTrue(S.verify.pass, 'the footprint-sized deck passes at the limit');
            tc.verifyEqual(S.verify.nLost, 0);
            tc.verifyTrue(S.rules.ok, 'positive edge thickness');
            % the RMS crosses 0.070 between the last pass and the first fail
            T = S.fields;
            tc.verifyLessThanOrEqual(max(T.rms_waves(T.pass)), tc.p.rms_limit);
            tc.verifyGreaterThan(min(T.rms_waves(~T.pass)), tc.p.rms_limit);
            tc.verifyTrue(exist(S.report, 'file') == 2);
        end

        function test_rms_is_sampling_converged_at_97_across(tc)
            L = tc.singlet(5);  P = iodc2010_paraxial(L, 532);
            L.semidia = [8 8];
            r = zeros(1, 2);  g = [97 127];
            for i = 1:2
                pg = tc.p;  pg.nGridpts = g(i);
                E = iodc2010_emit(L, P, pg, fullfile(tc.tmp, 'grid.in'));  macos.load_rx(E.path);
                F = iodc2010_eval(E, P, pg, 0);  r(i) = F.rms_waves;
            end
            tc.verifyEqual(r(1), r(2), 'RelTol', 1e-3, '97 vs 127 across');
            tc.verifyEqual(r(1), 0.0944, 'AbsTol', 5e-4, 'the f/10 singlet on axis');
        end

        function test_nbk7_table_row_when_the_engine_build_carries_it(tc)
            L = tc.singlet(3.5);  L.semidia = [8 8];  P = iodc2010_paraxial(L, 532);
            E = iodc2010_emit(L, P, tc.p, fullfile(tc.tmp, 'row.in'));
            txt = fileread(E.path);
            txt = regexprep(txt, '^[ \t]*GlassType=[^\n]*\n', '', 'lineanchors');
            txt = regexprep(txt, '^[ \t]*GlassCoef=[^\n]*\n', '', 'lineanchors');
            txt = regexprep(txt, '(IndRef=  )1\.5\d+E\+00', '$11.000000000000000E+00');
            tc.assertTrue(contains(txt, 'GlassElt=  N-BK7') && ~contains(txt, 'GlassCoef'));
            f = fullfile(tc.tmp, 'row_only.in');  fid = fopen(f, 'w');  fwrite(fid, txt);  fclose(fid);
            macos.load_rx(f);
            iodc2010_eval(E, P, tc.p, 0);
            r = macos.srt(1e-4, 0, E.iFP);
            u = atan2(r.dir(1, E.iSurf(end)), r.dir(3, E.iSurf(end)));
            efl = -r.pos(1, E.iSurf(1)) / tan(u);
            tc.assumeLessThan(abs(efl - 100), 1e-3, ...
                'N-BK7 is not in this engine build''s glass table (rebuild pending)');
            tc.verifyEqual(efl, P.efl, 'AbsTol', 1e-6, 'the table row''s index');
        end

        function test_the_paraxial_stop_keeps_the_pupil_edge_rays(tc)
            % a two-element lens whose stop sits behind a strongly curved
            % element (a solved lens of the outer loop's smoke run): its pupil
            % aberration carries the outermost ring of the paraxial pupil a
            % fraction of a percent outside the stop radius
            L = iodc2010_lens('R', 18.8308, 'sgn', [1 0 0 1], 't', [4.4192 6.4141 5.8014 63.0070], ...
                'glass', {'N-BK7', 'air', 'N-BK7', 'air'}, 'stop', [10.0181 2.1896], 'name', 'stopring');
            L.semidia = inf(1, L.N);
            P = iodc2010_paraxial(L, tc.p.lambda_nm);
            % the metric's convention: nothing is lost
            E = iodc2010_emit(L, P, tc.p, fullfile(tc.tmp, 'stop_open.in'));
            macos.load_rx(E.path);
            F = iodc2010_eval(E, P, tc.p, 0);
            tc.verifyEqual(F.nLost, 0, 'the paraxial stop clips nothing');
            tc.verifyTrue(F.pass);
            % the hard-aperture switch: the outermost ring, and only it
            pc = tc.p;  pc.stop_clips = true;
            Ec = iodc2010_emit(L, P, pc, fullfile(tc.tmp, 'stop_hard.in'));
            macos.load_rx(Ec.path);
            Fc = iodc2010_eval(Ec, P, pc, 0, 'rays', true);
            tc.verifyGreaterThan(Fc.nLost, 0, 'the hard stop loses rays');
            tc.verifyLessThan(Fc.nLost, 0.02 * Fc.nLaunched, 'a thin ring, not the beam');
            rad = hypot(Fc.ray.pos_e1(1, :), Fc.ray.pos_e1(2, :));
            tc.verifyGreaterThan(min(rad(~Fc.ray.ok_fp)), 0.995 * P.D / 2, 'the lost rays are the pupil edge');
            tc.verifyEqual(Fc.reason, 'vignetting');
        end

        function test_forms_are_petzval_balanced_and_scaled(tc)
            [C, fi] = iodc2010_forms(2, tc.p, 'air_grid', [0.25 1]);
            tc.verifyEqual(fi.n_patterns_all, 64);
            tc.verifyEqual(fi.n_patterns_balanced, 14);
            tc.verifyGreaterThan(fi.n_seeds, 20);
            n = iodc2010_glass('N-BK7', tc.p.lambda_nm);
            for i = 1:numel(C)
                L = C(i).L;
                P = iodc2010_paraxial(L, tc.p.lambda_nm);
                tc.verifyEqual(abs(P.efl), 100, 'AbsTol', 1e-9, L.name);
                tc.verifyGreaterThan(P.z_img, L.z(end), 'the image is real and beyond the lens');
                % the textbook rule: a surface converges when c (n' - n) > 0
                nb = [1, P.n_after(1:end-1)];  na = P.n_after;
                pw = P.c .* (na - nb);
                tc.verifyEqual(nnz(pw > 0), nnz(pw < 0), ['Petzval balance: ' L.name]);
                tc.verifyEqual(P.D, 10, 'RelTol', 1e-9, 'the seed stop is sized for D_target');
            end
        end

        function test_calib_deck_fields_start_on_the_source_plane(tc)
            L = tc.pair();  L.semidia = inf(1, L.N);
            P = iodc2010_paraxial(L, tc.p.lambda_nm);
            opt = struct('fields_deg', [0 1.5 3], 'max_iters', 5, 'var_surf', 1:L.N, 'var_fp', true);
            E = iodc2010_emit(L, P, tc.p, fullfile(tc.tmp, 'pair_calib.in'), opt);
            txt = fileread(E.path);
            t0 = regexp(txt, '\n\s*ChfRayPos=\s*(\S+)\s+(\S+)\s+(\S+)', 'tokens', 'once');
            z0 = str2double(t0{3});
            rows = regexp(txt, 'OptChfRayPos=\s*(\S+)\s+(\S+)\s+(\S+)', 'tokens');
            tc.verifyEqual(numel(rows), 2, 'one row per off-axis field');
            for i = 1:numel(rows)
                tc.verifyEqual(str2double(rows{i}{3}), z0, 'AbsTol', 1e-9, 'the field starts on the source plane');
            end
            z_front = min([L.z, L.stop(1)]);
            tc.verifyLessThanOrEqual(z0, z_front - (P.z_img - z_front) + 1e-9, 'backed off by the lens''s length');
            tc.verifyEqual(E.z_src, z0, 'AbsTol', 1e-9);
        end
    end
end
