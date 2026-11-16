classdef tLens < matlab.unittest.TestCase
%TLENS  macos.design.Lens, macos.design.glass and Bench.add_optic: the lens primitive.
%
%   Legs
%     zwfs bytes   the ZWFS DM-gauge test arm (zwfs_dm96's bench, twyman_green
%                  'fieldlens') re-built with its field lens placed by
%                  add_optic(Lens.plano(...)) instead of add_lens emits the SAME
%                  deck, byte for byte; a minimal bench does the same for the
%                  'collimate' singlet (bench_layout's L1 form)
%     engine EFL   a convex-plano singlet and a cemented N-BK7 / N-SF6 doublet
%                  on a collimated bench: the engine's near-axis EFL (two srt
%                  heights, the h^2 aberration term extrapolated out) equals the
%                  paraxial y-nu solve to 1e-9 relative at 587.56 nm.  The leg
%                  can tell: the same doublet deck with its GlassElt/GlassType/
%                  GlassCoef lines stripped (the engine then traces the 7-digit
%                  IndRef) misses by more than 1e-8
%     overlap      a must-fail: a biconvex lens whose faces cross inside its
%                  clear aperture is refused (macos:design:Bench:overlap); the
%                  same lens thick enough is placed
%     paraxial     the closed forms of a thick convex-plano lens (EFL R/(n-1),
%                  BFL EFL - t/n, H at the front vertex); flip() swaps BFL and
%                  FFL and keeps the EFL
%     iodc         the IODC 2010 green-lens spec is a special case: Lens.from_iodc
%                  of the challenge's gate pair has iodc2010_paraxial's EFL and BFL,
%                  and macos.design.glass reads iodc2010_glass's indices, both to
%                  round-off
%
%   Model size 128 -> SUITE_FAST.

    properties (Constant)
        ModelSize = 128
        LamNm = 587.5618
    end
    properties
        tmp
    end

    methods (TestClassSetup)
        function setup(tc)
            macos.init(tc.ModelSize);
            tc.tmp = tempname;  mkdir(tc.tmp);
        end
    end
    methods (TestClassTeardown)
        function teardown(tc)
            if exist(tc.tmp, 'dir'), rmdir(tc.tmp, 's'); end
        end
    end

    methods
        function efl = engine_efl(~, O)
            % near-axis EFL of the loaded deck: two srt heights, Richardson on h^2
            hs = [1e-4 2e-4];  E = zeros(1, 2);
            for i = 1:2
                r  = macos.srt(hs(i), 0, O.i_last);
                ax = [0; 0; 1];
                h  = r.pos(:, O.i_first);  hv = h - ax * (ax.' * h);  e1 = hv / norm(hv);
                d  = r.dir(:, O.i_last);
                E(i) = -norm(hv) / ((e1.' * d) / (ax.' * d));
            end
            efl = (4 * E(1) - E(2)) / 3;
        end
        function [O, P, f] = collimated_bench(tc, L, tag)
            lam_mm = tc.LamNm * 1e-6;
            P = L.paraxial(tc.LamNm);
            b = macos.design.Bench(tag, 'dir', [0; 0; 1], 'zsource', 1e22, ...
                'aperture', 0.8 * L.D, 'wavelen', lam_mm, 'ngridpts', 21);
            O = b.add_optic(L, 50);
            b.add_detector(P.bfl, 'FP');
            f = fullfile(tc.tmp, [tag '.in']);  b.emit(f);
        end
    end

    methods (Test)

        function test_zwfs_field_lens_deck_is_byte_identical(tc)
            here = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'templates', '40_benches', 'zwfs_dm96');
            addpath(here);  c = onCleanup(@() rmpath(here));
            Pz = zwfs_params();
            Pz.bench.MASK_TRIM = -5.582;                 % the record's number; 'scan' traces, the deck's FL does not care
            bf = fieldnames(Pz.bench);  bf = bf(~strncmp(bf, 'coat_', 5));
            ba = cell(1, 2 * numel(bf));
            for i = 1:numel(bf), ba{2*i-1} = bf{i};  ba{2*i} = Pz.bench.(bf{i}); end
            G = macos.design.twyman_green(ba{:}, 'ngridpts', Pz.NGRID, ...
                'to_grid_file', Pz.grid.flat_file, 'to_grid_n', Pz.grid.N_G, 'to_grid_dx', Pz.grid.DX_G);
            bt = G.bt;  bt.wavelen = Pz.LAM;
            f1 = fullfile(tc.tmp, 'zwfs_add_lens.in');  bt.emit(f1);

            names = {bt.E.name};
            ip = find(strcmp(names, 'FLpow'));  iq = find(strcmp(names, 'FLflat'));
            tc.assertEqual([numel(ip) numel(iq)], [1 1], 'the ZWFS arm has one field lens FL');
            tc.assertEqual(iq, ip + 1);
            % the same bench up to the element before FL, then FL by add_optic, then the tail
            b2 = macos.design.Bench(bt.name, 'pos', bt.src_pos, 'dir', bt.src_dir, 'wavelen', bt.wavelen, ...
                'aperture', bt.aperture, 'ngridpts', bt.ngridpts, 'zsource', bt.zsource, 'baseunits', bt.baseunits);
            b2.E = bt.E(1:ip-1);
            b2.pos = bt.E(ip-1).vpt;  b2.dir = bt.E(ip).psi;  b2.path_len = bt.E(ip-1).s;
            GP = G.P;
            if strcmp(bt.E(ip-1).name, 'MaskSphereOut')
                d_out = 0.6 * GP.D_MASK_FL;  d_fl = GP.D_MASK_FL - d_out;    % twyman_green's own arithmetic
            else
                d_fl = GP.D_MASK_FL;
            end
            Lfl = macos.design.Lens.plano(GP.FL_F, GP.FL_D, 'mode', 'focus', 'n', GP.N_GLASS, ...
                'name', 'FL', 'edge_margin', GP.EDGE_MARGIN, 'Kc', GP.FL_Kc);
            O = b2.add_optic(Lfl, d_fl);
            tc.verifyEqual(O.idx, [ip iq]);
            b2.E = [b2.E, bt.E(iq+1:end)];
            f2 = fullfile(tc.tmp, 'zwfs_add_optic.in');  b2.emit(f2);
            t1 = fileread(f1);  t2 = fileread(f2);
            tc.verifyTrue(strcmp(t1, t2), 'add_optic(Lens.plano) re-emits the ZWFS test arm byte for byte');
            tc.verifyGreaterThan(numel(t1), 10000, 'a whole bench deck was compared');
        end

        function test_collimate_singlet_is_byte_identical(tc)
            mk = @() macos.design.Bench('c', 'aperture', 0.1, 'ngridpts', 21);
            b1 = mk();  b1.add_baffle(40, 10);
            b1.add_lens(200, 250, 30, 'mode', 'collimate', 'n', 1.5168, 'name', 'L1');
            b1.add_detector(100);
            b2 = mk();  b2.add_baffle(40, 10);
            O = b2.add_optic(macos.design.Lens.plano(250, 30, 'mode', 'collimate', 'n', 1.5168, 'name', 'L1'), 200);
            b2.add_detector(100);
            tc.verifyEqual(O.at, 3 - 1, 'DIST located the powered (back) face');
            f1 = fullfile(tc.tmp, 'c1.in');  f2 = fullfile(tc.tmp, 'c2.in');
            b1.emit(f1);  b2.emit(f2);
            tc.verifyTrue(strcmp(fileread(f1), fileread(f2)));
        end

        function test_engine_efl_matches_paraxial(tc)
            S = macos.design.Lens('R', [51.68 Inf], 't', 5, 'glass', {'N-BK7'}, 'D', 25, 'name', 'S');
            Dbl = macos.design.Lens('R', [61.47 -44.64 -129.94], 't', [6 2.5], ...
                'glass', {'N-BK7', 'N-SF6'}, 'D', 25, 'name', 'AC');
            for L = {S, Dbl}
                [O, P, f] = tc.collimated_bench(L{1}, ['efl_' L{1}.name]);
                macos.load_rx(f);
                efl = tc.engine_efl(O);
                tc.verifyEqual(efl, P.efl, 'RelTol', 1e-9, sprintf('%s: engine EFL vs y-nu', L{1}.name));
            end
            % the leg can tell: strip the glass rows -> the engine traces the 7-digit IndRef
            txt = fileread(f);
            txt = regexprep(txt, '[ ]*(GlassElt|GlassType|GlassCoef)=[^\n]*\n', '');
            tc.assertEmpty(strfind(txt, 'GlassCoef'));
            f0 = fullfile(tc.tmp, 'efl_AC_noglass.in');  fid = fopen(f0, 'w');  fwrite(fid, txt);  fclose(fid);
            macos.load_rx(f0);
            efl0 = tc.engine_efl(O);
            tc.verifyGreaterThan(abs(efl0 / P.efl - 1), 1e-8, 'the 1e-9 gate sees a 7-digit index');
        end

        function test_overlapping_faces_are_refused(tc)
            L = macos.design.Lens('R', [20 -20], 't', 1, 'glass', {'N-BK7'}, 'D', 20, 'name', 'X');
            tc.verifyLessThan(L.edge_thickness(), 0);
            b = macos.design.Bench('x', 'dir', [0; 0; 1]);
            tc.verifyError(@() b.add_optic(L, 10), 'macos:design:Bench:overlap');
            tc.verifyEmpty(b.E, 'nothing was placed');
            L.t = 7;
            O = b.add_optic(L, 10);
            tc.verifyEqual(numel(O.idx), 2);
            tc.verifyGreaterThan(L.edge_thickness(), 0);
        end

        function test_paraxial_closed_forms_and_flip(tc)
            n = macos.design.glass('N-BK7', tc.LamNm);
            R = 51.68;  t = 5;
            L = macos.design.Lens('R', [R Inf], 't', t, 'glass', {'N-BK7'}, 'D', 25);
            P = L.paraxial(tc.LamNm);
            tc.verifyEqual(P.efl, R / (n - 1), 'RelTol', 1e-14);
            tc.verifyEqual(P.bfl, P.efl - t / n, 'RelTol', 1e-14);
            tc.verifyEqual(P.h1, 0, 'AbsTol', 1e-12, 'front principal plane at the curved vertex');
            tc.verifyEqual(P.h2, -t / n, 'RelTol', 1e-12);
            Pf = L.flip().paraxial(tc.LamNm);
            tc.verifyEqual(Pf.efl, P.efl, 'RelTol', 1e-14);
            tc.verifyEqual(L.flip().R, [Inf -R], 'a plano face flips to +Inf');
            tc.verifyEqual(Pf.bfl, P.ffl, 'RelTol', 1e-14);
            tc.verifyEqual(Pf.ffl, P.bfl, 'RelTol', 1e-14);
            tc.verifyEqual(macos.design.glass('n-bk7', 587.5618), 1.5168, 'AbsTol', 5e-6, 'Schott n_d');
            tc.verifyError(@() macos.design.glass('BK', 500), 'macos:design:glass:unknown');
        end

        function test_iodc_spec_is_a_special_case(tc)
            here = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'challenges', 'iodc2010');
            addpath(here);  c = onCleanup(@() rmpath(here));
            mk = @(d) iodc2010_lens('R', 80, 'sgn', [1 0 0 -1], 't', [6 d 6 NaN], ...
                'glass', {'N-BK7', 'air', 'N-BK7', 'air'}, 'stop', [6 + d/2, 7], 'name', 'pair');
            Li = mk(40);
            Pi = iodc2010_paraxial(Li, 532);
            P  = macos.design.Lens.from_iodc(Li).paraxial(532);
            tc.verifyEqual(P.efl, Pi.efl, 'RelTol', 1e-14);
            tc.verifyEqual(P.bfl, Pi.bfl, 'RelTol', 1e-14);
            for g = {'N-BK7', 'N-SF6'}
                for lam = [486.13 532 656.27]
                    tc.verifyEqual(macos.design.glass(g{1}, lam), iodc2010_glass(g{1}, lam), 'AbsTol', 1e-15);
                end
            end
        end
    end
end
