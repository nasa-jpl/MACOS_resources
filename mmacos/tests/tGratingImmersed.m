classdef tGratingImmersed < matlab.unittest.TestCase
%TGRATINGIMMERSED  A reflection grating IMMERSED in glass (dyson5 gate 1a).
%   Fixture Rx_GratingImmersed.in: a collimated beam enters a fused-silica
%   block (Refractor, GlassElt= Silica) and meets a concave 100 l/mm grating
%   that is the block's back face, so the grating sees GLASS on its incident
%   side.  elemsub.F's Snells_Law_Grating is written in the immersed form,
%       u = (na*i + m*lambda0/d * s) / nb,   lambda0 = VACUUM wavelength,
%   and tracesub.F hands it na = CurIndRef (the medium the ray is in) and
%   nb = IndRef(iElt) AS WRITTEN on the grating element -- and, unlike the
%   Reflector branch, never overwrites IndRef(iElt) with CurIndRef.  So the
%   glass has to be declared ON the grating element (GlassElt= Silica in the
%   fixture); the mirror habit IndRef= 1 gives nb = 1 and a wrong direction.
%
%   Closed form checked per ray, written from the grating equation (NOT
%   transcribed from the engine): with N the sphere normal at the hit point,
%   s = unit(RuleDir projected into the tangent plane), u = N x s,
%       nb (r.s) = na (i.s) + m lambda0/d        (dispersion direction)
%       nb (r.u) = na (i.u)                      (along the grooves)
%       |r| = 1,  sign(r.N) = -sign(i.N)         (reflection)
%   at three wavelengths with na = nb = n_Silica(lambda) from Malitson's
%   Sellmeier (the engine's own table row, evaluated here independently).
%
%   Must-PASS legs (gates fail closed): the IMMERSED law holds to 1e-10 AND
%   the un-immersed law (n = 1 in the grating term) misses by > 1e-3; the
%   all-air control (every GlassElt removed) obeys the air form to 1e-10;
%   the IndRef= 1 trap fails the immersed law by > 1e-3 and is pinned to
%   what the engine actually does with it (nb = 1), so a future engine
%   reconciliation of nb shows up here as a deliberate change.
%   Size 128 -> SUITE_FAST.

    properties (Constant)
        Model   = 128
        RxName  = 'Rx_GratingImmersed.in'
        Lambdas = [0.5e-6 1.0e-6 2.0e-6]     % m (fixture WaveUnits = m)
        Tol     = 1e-10
        MissMin = 1e-3                        % non-vacuity floor
        % grating + geometry exactly as written in the fixture
        Order   = 1
        RuleW   = 1.0e-5                      % m  (100 lines/mm)
        RuleDir = [1 0 0]
        Vpt     = [0 0 0.05]
        Psi     = [0 0 -1]                    % psi -> centre of curvature
        Rcurv   = 0.1
        % Malitson fused silica = macos_glass_list.txt row 'Silica' (C in um^2)
        SiB = [0.6961663 0.4079426 0.8974794]
        SiC = [0.004679148 0.01351206 97.934]
    end

    properties
        rx_path
        tmpdir
    end

    methods (TestClassSetup)
        function setup(tc)
            tc.rx_path = rx_fixture_path(tc.RxName);
            tc.tmpdir  = tempname;  mkdir(tc.tmpdir);
            macos.init(tc.Model);
        end
    end

    methods (TestClassTeardown)
        function teardown(tc)
            if exist(tc.tmpdir, 'dir'), rmdir(tc.tmpdir, 's'); end
        end
    end

    methods (Static)
        function k = nkey(txt)
            % number of GlassElt= KEYWORD lines (the header comments mention
            % the keyword too, and a count on the bare string caught them)
            k = numel(regexp(txt, '^[ \t]*GlassElt=', 'lineanchors', 'match'));
        end
    end

    methods
        function n = silica(tc, lam_m)
            L2 = (lam_m*1e6)^2;
            n  = sqrt(1 + sum(tc.SiB .* L2 ./ (L2 - tc.SiC)));
        end

        function p = variant(tc, kind)
            % Build a deck variant from the ONE fixture by text substitution,
            % asserting that exactly the intended lines changed.
            txt = fileread(tc.rx_path);
            switch kind
                case 'immersed'
                    p = tc.rx_path;  return
                case 'trap'   % grating element keeps IndRef= 1, loses its glass
                    i2 = strfind(txt, 'iElt=  2');  i3 = strfind(txt, 'iElt=  3');
                    tc.assertTrue(isscalar(i2) && isscalar(i3), 'fixture markers');
                    blk  = txt(i2:i3-1);
                    blk2 = regexprep(blk, '^[ \t]*GlassElt=[^\n]*\n', '', 'lineanchors');
                    tc.assertEqual(tc.nkey(blk) - tc.nkey(blk2), 1, ...
                                   'trap variant must remove exactly one GlassElt line');
                    txt = [txt(1:i2-1) blk2 txt(i3:end)];
                case 'air'    % every GlassElt line removed (IndRef= 1 is already written)
                    n0  = tc.nkey(txt);
                    txt = regexprep(txt, '^[ \t]*GlassElt=[^\n]*\n', '', 'lineanchors');
                    tc.assertEqual(n0, 4, 'fixture carries 4 GlassElt keyword lines');
                    tc.assertEqual(tc.nkey(txt), 0);
                otherwise
                    error('unknown variant %s', kind);
            end
            p = fullfile(tc.tmpdir, ['Rx_GratingImmersed_' kind '.in']);
            fid = fopen(p, 'w');  fwrite(fid, txt);  fclose(fid);
        end

        function [I, P, R] = trace_pair(tc, path, lam)
            % macos.get_ray_info after macos.trace(k) returns RayDir as CTRACE
            % leaves it: the OUTGOING direction from element k (the surface
            % routines overwrite RayDir in place; the api's "direction before
            % Srf" comment is stale).  So the grating's INCOMING direction is
            % the outgoing one from the flat face (trace(1)), and the
            % DIFFRACTED direction + hit point come from trace(2).  Neither
            % reading depends on what a Return element does to a direction.
            macos.load_rx(path);
            macos.set_src_wvl(lam);  macos.modify();
            s1 = macos.trace(1);  ri1 = macos.get_ray_info(s1.nRays);
            macos.modify();
            s2 = macos.trace(2);  ri2 = macos.get_ray_info(s2.nRays);
            tc.assertEqual(s1.nRays, s2.nRays);
            ok = ri1.ok_trace & ri2.ok_trace;
            tc.assertGreaterThan(nnz(ok), 50, 'too few rays survive the grating');
            I = ri1.dir(:, ok);  P = ri2.pos(:, ok);  R = ri2.dir(:, ok);
            % the flat face is at normal incidence: the incoming direction is
            % +z to round-off, whichever direction convention ray_info uses
            tc.assertLessThan(max(abs(I(3,:) - 1)), 1e-12, 'incoming +z at the grating');
        end

        function [res_s, res_u, res_norm, refl_ok] = residuals(tc, I, P, R, na, nb, lam)
            C = tc.Vpt(:) + tc.Rcurv*tc.Psi(:);      % centre of curvature
            G = tc.Order*lam/tc.RuleW;               % m*lambda0/d
            m = size(I, 2);
            res_s = zeros(1, m);  res_u = res_s;  res_norm = res_s;  refl_ok = false(1, m);
            for k = 1:m
                N = P(:,k) - C;  N = N/norm(N);
                s = tc.RuleDir(:) - (tc.RuleDir(:)'*N)*N;  s = s/norm(s);
                u = cross(N, s);
                res_s(k)    = nb*(R(:,k)'*s) - (na*(I(:,k)'*s) + G);
                res_u(k)    = nb*(R(:,k)'*u) -  na*(I(:,k)'*u);
                res_norm(k) = norm(R(:,k)) - 1;
                refl_ok(k)  = (R(:,k)'*N) * (I(:,k)'*N) < 0;
            end
        end
    end

    methods (Test)

        function test_immersed_grating_obeys_the_immersed_equation(tc)
            p = tc.variant('immersed');
            for lam = tc.Lambdas
                n = tc.silica(lam);
                [I, P, R] = tc.trace_pair(p, lam);
                [rs, ru, rn, refl] = tc.residuals(I, P, R, n, n, lam);
                tc.verifyLessThan(max(abs(rs)), tc.Tol, sprintf('dispersion law at %.1f um', lam*1e6));
                tc.verifyLessThan(max(abs(ru)), tc.Tol, sprintf('groove law at %.1f um', lam*1e6));
                tc.verifyLessThan(max(abs(rn)), tc.Tol, 'unit direction');
                tc.verifyTrue(all(refl), 'every ray reflected');
                % non-vacuity: the AIR form (no index in the grating term) must miss
                [rs_air] = tc.residuals(I, P, R, 1, 1, lam);
                tc.verifyGreaterThan(max(abs(rs_air)), tc.MissMin, ...
                    'the un-immersed law must fail on the immersed deck (else the gate is vacuous)');
            end
        end

        function test_air_control_obeys_the_air_form(tc)
            p = tc.variant('air');
            for lam = tc.Lambdas
                [I, P, R] = tc.trace_pair(p, lam);
                [rs, ru, rn, refl] = tc.residuals(I, P, R, 1, 1, lam);
                tc.verifyLessThan(max(abs(rs)), tc.Tol, sprintf('air dispersion law at %.1f um', lam*1e6));
                tc.verifyLessThan(max(abs(ru)), tc.Tol, 'air groove law');
                tc.verifyLessThan(max(abs(rn)), tc.Tol, 'unit direction');
                tc.verifyTrue(all(refl), 'every ray reflected');
            end
        end

        function test_indref_one_on_an_immersed_grating_is_a_trap(tc)
            % The mirror habit (IndRef= 1 on a reflecting element) inside glass:
            % the engine takes nb = 1 and the immersed law FAILS.  Pinned to the
            % engine's actual assignment (na = n, nb = 1) so that a future
            % reconciliation of nb with the incident medium changes this test
            % on purpose, not silently.
            p = tc.variant('trap');
            lam = 1.0e-6;  n = tc.silica(lam);
            [I, P, R] = tc.trace_pair(p, lam);
            rs_immersed = tc.residuals(I, P, R, n, n, lam);
            tc.verifyGreaterThan(max(abs(rs_immersed)), tc.MissMin, ...
                'IndRef= 1 on the grating must NOT satisfy the immersed law');
            [rs_engine, ru_engine] = tc.residuals(I, P, R, n, 1, lam);
            tc.verifyLessThan(max(abs(rs_engine)), tc.Tol, 'engine assigns nb = IndRef(iElt) = 1');
            tc.verifyLessThan(max(abs(ru_engine)), tc.Tol, 'engine assigns nb = IndRef(iElt) = 1');
        end

        function test_wavelength_reaches_the_grating_through_set_src_wvl(tc)
            % The dispersion term must move with the runtime wavelength: the
            % mean diffracted tangential component shifts by the closed-form
            % amount between 0.5 and 2.0 um (an engine that ignored
            % set_src_wvl would show zero shift and fail the per-lambda law
            % above, but this states the observable directly).
            p = tc.variant('immersed');
            ms = zeros(1, 2);  lams = [0.5e-6 2.0e-6];
            for j = 1:2
                [I, P, R] = tc.trace_pair(p, lams(j));
                C = tc.Vpt(:) + tc.Rcurv*tc.Psi(:);
                v = zeros(1, size(R,2));
                for k = 1:size(R,2)
                    N = P(:,k) - C;  N = N/norm(N);
                    s = tc.RuleDir(:) - (tc.RuleDir(:)'*N)*N;  s = s/norm(s);
                    v(k) = R(:,k)'*s - I(:,k)'*s;      % per-ray tangential kick
                end
                ms(j) = mean(v);
            end
            expect = tc.Order*lams/tc.RuleW ./ [tc.silica(lams(1)) tc.silica(lams(2))];
            tc.verifyEqual(ms, expect, 'AbsTol', 1e-9, 'kick = m*lambda0/(n d) per wavelength');
        end
    end
end
