classdef tTelescopeRx < matlab.unittest.TestCase
%TTELESCOPERX  The emitted telescope and end-to-end decks trace as the chain predicts.
%   telescope_geom builds the three-mirror fore-optics (with a fold flat
%   and an aspheric mirror in this fixture) placed on the Dyson's slit;
%   e2e_geom prepends it to the spectrometer chain; spectrometer_rx writes
%   both as MACOS decks with a COLLIMATED source.  This gate loads each
%   deck, declares the stop (M2's vertex for the telescope alone, the
%   grating for the instrument), writes the chain's own aimed launch per
%   field, and checks that the ENGINE's chief and EVERY ray land where the
%   SAME chain (chain_trace, the Dyson's tracer) says, traced from the
%   engine's own launch points along its own launch directions (ray_hist
%   slot 1 -> 2): 1e-9 m at the slit and at the FPA.  On the line: the
%   collimated-source header (zSource 1e22, Aperture = the beam, ChfRayPos
%   = the launch point), KrElt = -|R| with psi toward the centre of
%   curvature on a convex mirror, Surface= Aspheric on a REFLECTOR (the
%   AsphCoef sag convention, pinned on the Dyson's refracting face until
%   now), the flat fold's psi, a mid-chain pass-through Reference (the
%   slit) and the stop aim through it, and the Dyson's own conventions
%   after it.  Teeth: a wrong fold normal, a flipped asphere, or a chief
%   not re-aimed through the grating all fail at the millimetre.
%   Size 128 -> SUITE_FAST.

    properties (Constant)
        Model = 128
        PD = struct('Fno',1.8,'pixel_m',18e-6,'npix',[3000 500],'band_m',[380e-9 2500e-9], ...
                    'lambda_ref_m',1e-6,'order',1,'y_slit',6e-3,'block_r',0.22,'glass','Silica', ...
                    'face_offset',0.5e-3,'Rg_factor',1,'offner_R',0.5, ...
                    'men_z',0.24,'men_t',0.004,'men_ca',0.5,'men_cb',0.5, 'slit_px', 2)   % R4's form
        Fields = [-0.2 0 0.2]                 % rad along the slit (the 24.6 deg field is +-0.214)
        TolRay = 1e-9                         % m
    end

    properties
        tmpdir
    end

    methods (TestClassSetup)
        function setup(tc)
            run(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'mmacos_setup.m'));
            tc.tmpdir = tempname;  mkdir(tc.tmpdir);
            macos.init(tc.Model);
        end
    end

    methods (TestClassTeardown)
        function teardown(tc)
            if exist(tc.tmpdir, 'dir'), rmdir(tc.tmpdir, 's'); end
        end
    end

    methods
        function [GT, GD] = build(tc)
            GD = spectrometer_geom('dyson', tc.PD);
            f = 0.126;  D = 0.07;  fov = 3000*18e-6/f;
            G0 = telescope_geom(struct('f', f, 'D', D, 'fov', fov, 'R', [0.3 0.1 0.3], 't', [0.1 0.1 0.1]), GD);
            Sd = telescope_seed(f, D, G0.pupil.L_app, 0.11, 0.55);
            tc.assertTrue(Sd.ok, 'the first-order seed closes');
            % a fixture with teeth: conics on all three, an h^4 + h^6 asphere on
            % M1 and M3 (a REFLECTOR's Aspheric sag), a fold, a bias
            Pt = struct('f', f, 'D', D, 'fov', fov, 'bias', -8*pi/180, 'R', Sd.R, 't', Sd.t, ...
                        'Kc', [-0.8 -1.5 0.3], 'A', [2.0 -300; 0 0; -1.5 200], 'fold_d', Sd.t3 - 0.025, ...
                        'fold_dir', [0;1;0], 'lambda_c', GD.src.lambda_c);
            GT = telescope_geom(Pt, GD);
            tc.assertEqual(GT.surf(1).kind, 'asph');  tc.assertEqual(GT.surf(3).kind, 'asph');
        end
        function M = emit(tc, G, iStop, file)
            d0 = G.field_dir(0);
            if isfield(G, 'launch_field'), [p0, ~, ok] = G.launch_field(0, G.src.lambda_c); else, [p0, ok] = G.aim_pt(d0, G.src.lambda_c); end
            tc.assertTrue(ok, 'the centre-field chief aims through the stop');
            F = G.footprints('nx', 3, 'nlam', 1, 'nring', 2);
            M = spectrometer_rx(G, file, 'ngridpts', 21, 'apertures', true, 'margin', 5e-3, 'footprints', F, ...
                                'source', struct('dir', d0, 'pos', p0, 'aperture', G.src.D_src), 'wavelen', G.src.lambda_c);
            M.iStop = iStop;
            macos.load_rx(file);
            tc.assertEqual(macos.num_elt(), M.nElt, 'deck loads with every element');
        end
        function check_rays(tc, G, M, label)
            lam = G.src.lambda_c;
            for th = tc.Fields
                d = G.field_dir(th);
                if isfield(G, 'launch_field'), [p0, ~, ok] = G.launch_field(th, lam); else, [p0, ok] = G.aim_pt(d, lam); end
                tc.assertTrue(ok, 'chain aim');
                macos.stop(M.iStop);                       % stop FIRST, then the chain's exact chief
                macos.set_src_fov('src_pos', p0, 'src_dir', d, 'zSrc', 1e22);
                macos.set_src_wvl(lam);  macos.modify();
                s = macos.trace(M.iStop);  ri = macos.get_ray_info(s.nRays);
                tc.assertTrue(ri.ok_trace(1), 'chief traces to the stop');
                tc.verifyLessThan(norm(ri.pos(:,1) - G.surf(M.iStop).vpt), 1e-9, sprintf('%s: stop aim through the vertex (field %.2f)', label, th));
                macos.modify();
                s = macos.trace(M.nElt);  ri = macos.get_ray_info(s.nRays);
                macos.ray_hist('on');  macos.modify();
                s1 = macos.trace(1);  h = macos.ray_hist(s1.nRays);  macos.ray_hist('off');
                P0 = squeeze(h.P(:,:,1));  P1 = squeeze(h.P(:,:,2));
                okl = logical(h.ok(:,2));  Dd = P1 - P0;  Dd = Dd ./ vecnorm(Dd);
                tc.assertEqual(numel(okl), numel(ri.ok_trace), 'same ray count');
                okr = ri.ok_trace(:) & ri.ok_pass(:) & okl(:);
                tc.assertGreaterThan(nnz(okr), 0.9*s.nRays, sprintf('%s: most rays reach the end (field %.2f)', label, th));
                tc.verifyEqual(nnz(ri.ok_trace & ~ri.ok_pass), 0, sprintf('%s: declared apertures (footprint + 5 mm) vignette nothing', label));
                % the chief
                [pk, ~, okk] = G.trace(P0(:,1), Dd(:,1), lam);
                tc.assertTrue(okk, 'chain traces the engine chief');
                tc.verifyLessThan(norm(ri.pos(:,1) - pk(:,end)), tc.TolRay, sprintf('%s: chief at the end (field %.2f)', label, th));
                % every ray
                idx = find(okr);  dmax = 0;
                for k = idx(:)'
                    [pk, ~, okk] = G.trace(P0(:,k), Dd(:,k), lam);
                    tc.assertTrue(okk, 'chain traces every engine ray');
                    dmax = max(dmax, norm(ri.pos(:,k) - pk(:,end)));
                end
                tc.verifyLessThan(dmax, tc.TolRay, sprintf('%s: every ray at the end, max |engine - chain| (field %.2f)', label, th));
            end
        end
    end

    methods (Test)
        function test_telescope_deck_traces_as_the_chain_says(tc)
            [GT, ~] = tc.build();
            M = tc.emit(GT, GT.iStop, fullfile(tc.tmpdir, 'tel.in'));
            tc.verifyEqual(M.nElt, numel(GT.surf) + 1, 'M1 M2 M3 fold + PreSlit + Slit');
            tc.check_rays(GT, M, 'telescope');
        end

        function test_end_to_end_deck_traces_as_the_chain_says(tc)
            [GT, GD] = tc.build();
            GE = e2e_geom(GT, GD);
            M = tc.emit(GE, GE.iG, fullfile(tc.tmpdir, 'e2e.in'));
            tc.verifyEqual(M.iG, GD.iG + numel(GT.surf), 'the grating index follows the telescope');
            tc.verifyEqual(M.iSlit, numel(GT.surf), 'the slit is a mid-chain Reference');
            tc.check_rays(GE, M, 'end-to-end');
            % the centre field lands on the FPA where the Dyson's own chief lands
            [pd, ~, okd] = GD.trace(GD.slit, GD.src.chief_dir, GD.src.lambda_c);
            [p0, d0, ok] = GE.launch_field(0, GD.src.lambda_c);  tc.assertTrue(okd && ok);
            [pe, ~, oke] = GE.trace(p0, d0, GD.src.lambda_c);  tc.assertTrue(oke);
            tc.verifyLessThan(norm(pe(:, end) - pd(:, end)), 1e-9, 'the placed telescope feeds the Dyson''s own chief');
        end
    end
end
