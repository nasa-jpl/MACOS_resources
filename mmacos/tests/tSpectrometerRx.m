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
        form = {'offner', 'dyson', 'dyson_asph'}
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
            if strcmp(form, 'dyson_asph')
                % the block's convex face as conic + h^4 + h^6 (engine AsphCoef
                % convention: coef(i) on h^(2i+2) of the sag along +psi) -- pins
                % the sag sign; a sphere-only chain misses by 0.26 mm here
                form = 'dyson';  P.block_Kc = -0.3;  P.block_asph = [2.0 -40];
            end
            G = spectrometer_geom(form, P);
            file = fullfile(tc.tmpdir, ['spec_' form '.in']);
            M = spectrometer_rx(G, file, 'ngridpts', 21);
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
                    idx = find(okr);  dmax = 0;
                    for k = idx(:)'
                        [pk, ~, okk] = G.trace(p0, D(:,k), lam);
                        tc.assertTrue(okk, 'chain traces every engine ray');
                        dmax = max(dmax, norm(ri.pos(:,k) - pk(:,end)));
                    end
                    tc.verifyLessThan(dmax, tc.TolChief, ...
                        sprintf('%s: every ray at the FPA, max |engine - chain| (x=%.3f, %.2f um)', form, xs, lam*1e6));
                    if xs == 0 && j ~= 2, yb(1 + (j == 3)) = ri.pos(2, 1); end
                end
            end
            % teeth: the engine's band spans the FPA spectral height
            H = tc.P.npix(2)*tc.P.pixel_m;
            tc.verifyEqual(abs(yb(2) - yb(1)), H, 'RelTol', 1e-2, ...
                sprintf('%s: band edges %.3f / %.3f mm span the 9 mm FPA', form, yb*1e3));
            % and the band sits on the solved FPA centre
            tc.verifyEqual(mean(yb), G.fpa.center(2), 'AbsTol', 0.5e-3, ...
                sprintf('%s: band centre on the FPA centre', form));
        end
    end
end
