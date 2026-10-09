classdef tSpectrometerRx < matlab.unittest.TestCase
%TSPECTROMETERRX  The emitted spectrometer deck traces as the chain predicts.
%   spectrometer_geom builds a concentric Dyson / Offner chain and solves
%   the chief aim, the groove period and the focus by exact ray trace;
%   spectrometer_rx writes it as a MACOS deck.  This gate loads the deck,
%   declares the grating the stop (macos.stop), and for 3 wavelengths x 3
%   slit positions checks that the ENGINE's chief ray lands on the FPA
%   where the SAME chain traced from the ENGINE's own source point and
%   chief direction says it must (1e-9 m, i.e. 1e-4 pixel) -- the two are
%   exact traces of one geometry, so every emission convention is on the
%   line: KrElt = -R with psi toward the CoC on concave AND convex spheres,
%   IndRef = the medium after a Refractor with GlassElt carrying the
%   Sellmeier, the Grating's h1HOE/OrderHOE/RuleWidth signs, the point
%   source's Aperture = full cone angle, the ChfRayPos-before-first-surface
%   rule, the Reference-not-Return tail, and the stop aim through the
%   grating vertex.  Teeth: the engine's band must span the FPA's spectral
%   height (9 mm) to 1 % -- an inert grating, a wrong order sign or a
%   wrong RuleWidth unit all fail it -- and EVERY engine ray, re-traced by
%   the chain along the engine's own launch direction, must land within
%   1e-9 m of the engine's ray (a per-ray, every-surface check).
%   Size 128 -> SUITE_FAST.

    properties (Constant)
        Model = 128
        P = struct('Fno',1.8,'pixel_m',18e-6,'npix',[3000 500],'band_m',[380e-9 2500e-9], ...
                   'lambda_ref_m',1e-6,'order',1,'y_slit',6e-3,'block_r',0.22,'glass','Silica', ...
                   'face_offset',0.5e-3,'Rg_factor',1,'offner_R',0.5)
        Xs = [-0.027 0 0.027]                 % slit positions along x (the 54 mm slit)
        TolChief = 1e-9                       % m
    end

    properties (TestParameter)
        form = {'offner', 'dyson', 'dyson_asph', 'dyson_apertures', 'dyson_fold'}
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
        function [G, M] = build(tc, form)
            P = tc.P;
            if strcmp(form, 'offner'), P.Fno = 2.8; end      % the Offner's own speed
            ap = false;
            if strcmp(form, 'dyson_asph')
                % the block's convex face as conic + h^4 + h^6 (engine AsphCoef
                % convention: coef(i) on h^(2i+2) of the sag along +psi) -- pins
                % the sag sign; a sphere-only chain misses by 0.26 mm here
                form = 'dyson';  P.block_Kc = -0.3;  P.block_asph = [2.0 -40];
            elseif strcmp(form, 'dyson_apertures')
                % DECLARED apertures (addendum 6): ApType Circular, ApVec =
                % (footprint radius + margin, xc, yc) in the aperture frame
                % xObs = global x, yObs = psi x xObs (tracesub.F) -- the test
                % below also asserts that NOT ONE ray is vignetted, which pins
                % the frame's sign (a flipped yObs decentres every aperture)
                form = 'dyson';  ap = true;
            elseif strcmp(form, 'dyson_fold')
                % R5: entrance plate + mirror-coated fold prism (a Reflector
                % INSIDE glass, GlassElt carried; two glass-to-glass planes;
                % the FPA folded to normal +y) -- the engine must land every
                % ray where the chain says, in the folded frame
                form = 'dyson';  ap = true;  P.fold_h = 8e-3;  P.face_offset = 17e-3;
            end
            G = spectrometer_geom(form, P);
            file = fullfile(tc.tmpdir, ['spec_' form '.in']);
            M = spectrometer_rx(G, file, 'ngridpts', 21, 'apertures', ap, 'margin', 5e-3);
            macos.load_rx(file);
            tc.assertEqual(macos.num_elt(), M.nElt, 'deck loads with every element');
        end
    end

    methods (Test)

        function test_engine_chief_lands_where_the_chain_says(tc, form)
            [G, M] = tc.build(form);
            lams = [tc.P.band_m(1), G.src.lambda_c, tc.P.band_m(2)];
            yb = nan(1, 2);                 % engine band edges at the slit centre
            for xs = tc.Xs
                for j = 1:3
                    lam = lams(j);
                    slit = G.slit + [xs; 0; 0];
                    % the chain's own aim for THIS slit point, so the engine's
                    % stop re-aim is a no-op (the engine's source point is
                    % ChfRayPos + zSource*ChfRayDir and would otherwise move
                    % with the re-aimed direction)
                    da = G.aim(slit, lam);
                    % after LOAD the engine holds ChfRayPos = the physical source
                    % (it folds the deck's ChfRayPos + zSource*ChfRayDir in once);
                    % set_src_fov writes ChfRayPos raw, so hand it the SLIT POINT
                    % ORDER MATTERS: macos.stop aims IMMEDIATELY from the source
                    % state it finds, and its first pass on this deck is 3.6 mrad
                    % short of converged (a second call finishes it -- measured
                    % 2026-09-30).  Declare the stop FIRST, then write the chain's
                    % exact chief; the engine keeps it.
                    macos.stop(M.iG);
                    macos.set_src_fov('src_pos', slit, 'src_dir', da, 'zSrc', -G.src.zsrc_gap);
                    macos.set_src_wvl(lam);  macos.modify();
                    % the engine's aimed chief hits the grating vertex
                    s = macos.trace(M.iG);  ri = macos.get_ray_info(s.nRays);
                    tc.assertTrue(ri.ok_trace(1), 'chief traces to the grating');
                    tc.verifyLessThan(norm(ri.pos(:,1) - G.surf(G.iG).vpt), 1e-9, ...
                        sprintf('%s: stop aim through the grating vertex (x=%.3f, %.2f um)', form, xs, lam*1e6));
                    % the engine's chief at the FPA vs the chain traced from the
                    % engine's own source point and direction
                    macos.modify();
                    s = macos.trace(M.nElt);  ri = macos.get_ray_info(s.nRays);
                    % the engine's PHYSICAL source point = the common point of
                    % its launched rays (ray_hist slot 1 -> slot 2).  After the
                    % stop aim the engine has moved ChfRayPos onto that point,
                    % so ChfRayPos + zSource*ChfRayDir is NOT the source any
                    % more -- measured 2026-09-30, see the reference doc sec. 2.
                    macos.ray_hist('on');  macos.modify();
                    s1 = macos.trace(1);  h = macos.ray_hist(s1.nRays);  macos.ray_hist('off');
                    P0 = squeeze(h.P(:,:,1));  P1 = squeeze(h.P(:,:,2));
                    okl = logical(h.ok(:,2));  D = P1 - P0;  D = D ./ vecnorm(D);   % column, like ri.ok_*
                    A = zeros(3);  bb = zeros(3,1);
                    for k = find(okl)'
                        Mk = eye(3) - D(:,k)*D(:,k)';  A = A + Mk;  bb = bb + Mk*P0(:,k);
                    end
                    p0 = A\bb;
                    tc.verifyLessThan(norm(p0 - slit), 1e-9, 'engine source point = the slit point');
                    f = macos.get_src_fov();
                    [pts, ~, ok] = G.trace(p0, f.src_dir, lam);
                    tc.assertTrue(ok, 'chain traces the engine chief');
                    tc.verifyLessThan(norm(ri.pos(:,1) - pts(:,end)), tc.TolChief, ...
                        sprintf('%s: chief at the FPA (x=%.3f, %.2f um)', form, xs, lam*1e6));
                    % EVERY ray: the chain traced along the engine's own launch
                    % directions (ray_hist slot 1 -> slot 2) from the engine's
                    % own source point must land where the engine's rays land --
                    % a per-ray, every-surface check, not a centroid of two
                    % differently sampled cones (meaningless on an aberrated
                    % field)
                    tc.assertEqual(numel(okl), numel(ri.ok_trace), 'same ray count at elt 1 and the FPA');
                    okr = ri.ok_trace(:) & ri.ok_pass(:) & okl(:);
                    tc.assertGreaterThan(nnz(okr), 0.9*s.nRays, 'most rays reach the FPA');
                    if M.apertures
                        tc.verifyEqual(nnz(ri.ok_trace & ~ri.ok_pass), 0, ...
                            'declared apertures (footprint + 5 mm) must not vignette the beam they were cut from');
                    end
                    idx = find(okr);  dmax = 0;
                    for k = idx(:)'
                        [pk, ~, okk] = G.trace(p0, D(:,k), lam);
                        tc.assertTrue(okk, 'chain traces every engine ray');
                        dmax = max(dmax, norm(ri.pos(:,k) - pk(:,end)));
                    end
                    tc.verifyLessThan(dmax, tc.TolChief, ...
                        sprintf('%s: every ray at the FPA, max |engine - chain| (x=%.3f, %.2f um)', form, xs, lam*1e6));
                    if xs == 0 && j ~= 2, yb(1 + (j == 3)) = (ri.pos(:, 1) - G.fpa.center(:))'*G.fpa.yhat(:); end   % along the FPA's dispersion axis
                end
            end
            % teeth: the engine's band spans the FPA spectral height
            H = tc.P.npix(2)*tc.P.pixel_m;
            tc.verifyEqual(abs(yb(2) - yb(1)), H, 'RelTol', 1e-2, ...
                sprintf('%s: band edges %.3f / %.3f mm span the 9 mm FPA', form, yb*1e3));
            % and the band sits on the solved FPA centre
            tc.verifyEqual(mean(yb), 0, 'AbsTol', 0.5e-3, ...
                sprintf('%s: band centre on the FPA centre', form));
        end

        function test_links_make_the_return_pass_follow_the_first_pass(tc)
            % 'links' writes Link= on the return-pass copy of every surface
            % the beam crosses twice (and on the pre-FPA Reference, which
            % follows the FPA): the engine then applies a PERTURB (and CALIB's
            % ROC / CONIC / ASPH perturbs, the same LnkElt loop in macos_ops.F)
            % to both passes -- one physical surface.  The block's flat face
            % is written with opposite normals on its two passes and must NOT
            % be linked (a PIST would move the passes apart).  Must-PASS leg:
            % the same deck without links leaves the copy where it was.
            P = tc.P;  P.men_z = 0.24;  P.men_t = 0.004;  P.men_ca = 0.5;  P.men_cb = 0.5;   % R4's meniscus
            G = spectrometer_geom('dyson', P);
            dz = [0; 0; 1e-4];
            for links = [true false]
                file = fullfile(tc.tmpdir, sprintf('spec_links_%d.in', links));
                M = spectrometer_rx(G, file, 'ngridpts', 21, 'apertures', true, 'margin', 5e-3, 'links', links);
                macos.load_rx(file);
                ix = @(nm) find(strcmp(M.names, nm), 1);
                if links
                    tc.verifyEqual(M.link(ix('MenA_in')), ix('MenA_out'), 'MenA_in follows MenA_out');
                    tc.verifyEqual(M.link(ix('MenB_in')), ix('MenB_out'), 'MenB_in follows MenB_out');
                    tc.verifyEqual(M.link(ix('BlockSphereIn')), ix('BlockSphereOut'), 'BlockSphereIn follows BlockSphereOut');
                    tc.verifyEqual(M.link(ix('PreFPA')), ix('FPA'), 'the pre-FPA Reference follows the FPA');
                    tc.verifyEqual(M.link(ix('BlockFaceIn')), 0, 'the flat face (opposite normals per pass) is not linked');
                    tc.verifyEqual(M.link(ix('BlockFaceOut')), 0, 'the flat face (opposite normals per pass) is not linked');
                else
                    tc.verifyTrue(all(M.link == 0), 'no links without the option');
                end
                a = ix('MenA_out');  b = ix('MenA_in');
                va0 = macos.get_elt_vpt(a);  vb0 = macos.get_elt_vpt(b);
                macos.perturb(a, 'translation', dz);
                da = macos.get_elt_vpt(a) - va0;  db = macos.get_elt_vpt(b) - vb0;
                tc.verifyEqual(norm(da), 1e-4, 'AbsTol', 1e-12, 'the first pass moved by the piston');
                if links
                    tc.verifyEqual(db, da, 'AbsTol', 1e-12, 'the return-pass copy moved WITH its first pass');
                else
                    tc.verifyEqual(norm(db), 0, 'AbsTol', 1e-15, 'without the link the copy stays (the leg that proves the gate bites)');
                end
            end
        end

        function test_opt_block_configures_calib_fields_and_wavelengths(tc)
            % the 'opt' block: field 1 is the header's ChfRayDir/Pos (the
            % engine's parse counts it), the others OptChfRayDir/Pos pairs,
            % Wavelen + ArrWaveLen the lambda list -- CALIB reports what it
            % was given.  The ENGINE leg runs at ONE field x ONE wavelength:
            % CALIB's derivative loop steps the SPOT objective at the
            % wavefront-map stride (design_optim.F ~:792), a heap stomp on
            % the second (field, wavelength) that kills the host process
            % (pinned in the bounds-checked CLI, BRIEF_dyson5_beat4c.md 3.4)
            % -- the multi-field leg runs again since macos 0d257ff.
            G = spectrometer_geom('dyson', tc.P);
            O1 = struct('fovs', struct('slit', G.slit, 'dir', G.aim(G.slit, G.src.lambda_c)), 'wavelens', G.src.lambda_c, ...
                        'weights', 1, 'target', 'SPOT', 'wf_elt', [], 'max_iters', 1, ...
                        'var', struct('name', 'FPA', 'mask', [0 0 0 0 0 1 0 0], 'asph', []));
            file1 = fullfile(tc.tmpdir, 'spec_opt1.in');
            M1 = spectrometer_rx(G, file1, 'ngridpts', 21, 'apertures', true, 'margin', 5e-3, 'links', true, 'opt', O1);
            macos.load_rx(file1);
            tc.assertEqual(macos.num_elt(), M1.nElt, 'the 1x1 opt deck loads with every element');
            macos.calib_set_iter(1);
            r1 = macos.calib();
            tc.verifyEqual(r1.n_fov, 1, 'CALIB sees the header field');
            tc.verifyEqual(r1.n_wavelength, 1, 'CALIB sees the header wavelength');
            W = tc.P.npix(1)*tc.P.pixel_m;  xs = [-W/2 0 W/2];
            fovs = struct('slit', {}, 'dir', {});
            for i = 1:3
                sl = G.slit + [xs(i); 0; 0];  fovs(end+1) = struct('slit', sl, 'dir', G.aim(sl, G.src.lambda_c));  %#ok<AGROW>
            end
            O = struct('fovs', fovs, 'wavelens', [tc.P.band_m(1) tc.P.band_m(2)], 'weights', [1 1 1], 'target', 'SPOT', ...
                       'wf_elt', [], 'max_iters', 1, 'var', struct('name', 'FPA', 'mask', [0 0 0 0 0 1 0 0], 'asph', []));
            file = fullfile(tc.tmpdir, 'spec_opt.in');
            M = spectrometer_rx(G, file, 'ngridpts', 21, 'apertures', true, 'margin', 5e-3, 'links', true, 'opt', O);
            macos.load_rx(file);
            tc.assertEqual(macos.num_elt(), M.nElt, 'the 3x2 opt deck loads with every element');
            % what the emitter wrote (its own output, not an engine fact)
            txt = fileread(file);
            tc.verifyEqual(numel(regexp(txt, 'OptChfRayDir=', 'match')), 2, 'two off-centre fields written (the header is field 1)');
            tc.verifyEqual(numel(regexp(txt, 'OptChfRayPos=', 'match')), 2, 'two off-centre field positions written');
            tc.verifyEqual(numel(regexp(txt, 'ArrWaveLen=', 'match')), 1, 'the second wavelength written as ArrWaveLen');
            tc.verifyEqual(numel(regexp(txt, 'OptRayGrid=', 'match')), 0, 'OptRayGrid is not written (it corrupts the heap, beat 4c 3.3)');
            % the multi-field CALIB leg: a heap stomp until macos 0d257ff (the
            % SPOT derivative columns advanced by opd_size); runs since
            macos.calib_set_iter(1);
            r = macos.calib();
            tc.verifyEqual(r.n_fov, 3, 'CALIB sees the 3 fields (header + 2 OptChfRay pairs)');
            tc.verifyEqual(r.n_wavelength, 2, 'CALIB sees the 2 wavelengths (Wavelen + ArrWaveLen)');
        end

        function test_clearance_sees_a_beam_through_a_body(tc)
            % (addendum 46, 2026-10-06) the gate's distance to SAMPLED body
            % points cannot go negative, so a leg passing straight through a
            % mirror or grating disc read +0..1 mm (the sample spacing); the
            % crossing test (segment x surface inside aperture + mount) makes
            % it a penetration.  Must-FAIL: the F/1.8 Offner at the 0.22 R
            % ring, both beams through the grating (-33 mm when found; the
            % pre-fix gate read +0.2).  Must-PASS: the same Offner at 0.30 R
            % and the F/2.8 sibling at 0.22 R (no body crossed, unchanged).
            % Also pins the no-box-body path (the Offner): an empty 0x3 body
            % table, not a cell2table error.  Chain-only, no engine.
            P = tc.P;  P.Fno = 1.8;  P.offner_R = 0.5;  P.grating_model = 'planes';
            Pc = struct('mount_margin_m', 5e-3);
            P.y_slit = 0.22*0.5;  C = spectrometer_clearance(spectrometer_geom('offner', P), Pc, 'quiet', true);
            tc.verifyLessThan(C.min_mm, -10, 'F/1.8 Offner at 0.22 R: the beams cross the grating body');
            tc.verifySubstring(C.table.body{1}, 'Grating', 'the crossed body is the grating');
            tc.verifyEqual(height(C.body_table), 0, 'no box bodies: an empty body table');
            P.y_slit = 0.30*0.5;  C = spectrometer_clearance(spectrometer_geom('offner', P), Pc, 'quiet', true);
            tc.verifyGreaterThan(C.min_mm, 5, 'F/1.8 Offner at 0.30 R clears');
            P.Fno = 2.8;  P.y_slit = 0.22*0.5;  C = spectrometer_clearance(spectrometer_geom('offner', P), Pc, 'quiet', true);
            tc.verifyGreaterThan(C.min_mm, 5, 'F/2.8 Offner at 0.22 R clears (the record''s sibling)');
        end

    end
end
