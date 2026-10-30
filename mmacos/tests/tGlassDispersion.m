classdef tGlassDispersion < matlab.unittest.TestCase
%TGLASSDISPERSION  GlassElt dispersion is applied at TRACE time (dyson5 gate 1b).
%   Fixture Rx_GlassPlate.in: a collimated beam meets a fused-silica face
%   tilted 30 deg (Refractor, GlassElt= Silica) and is caught inside the
%   glass.  tracesub.F's CTRACE re-evaluates every GlassElt's Sellmeier
%   index at the CURRENT wavelength before tracing, so the refracted angle
%   must follow macos.set_src_wvl.  macos_api_mod has no index getter (the
%   IndRef_ line in its audit list is an unchecked TODO), so the index is
%   READ OFF THE RAY: n = sin(theta_i)/sin(theta_t) with theta_i = 30 deg
%   exactly, compared with Malitson's Sellmeier (the engine table's own row,
%   evaluated here independently) to 1e-12.
%
%   Must-PASS legs: Silica matches at three wavelengths AND the index MOVES
%   (n(0.5 um) - n(2.0 um) > 0.02); the fixed-index control (GlassElt line
%   removed, IndRef= 1.5) reads 1.5 at every wavelength and misses the
%   Sellmeier by > 1e-2 -- the null that proves the gate can see an engine
%   that ignores GlassElt.  CaF2 is assumption-gated: it reports Incomplete
%   (never a silent pass) until the engine build carries the table row added
%   on macos dev-candidate.  Size 128 -> SUITE_FAST.

    properties (Constant)
        Model   = 128
        RxName  = 'Rx_GlassPlate.in'
        Lambdas = [0.5e-6 1.0e-6 2.0e-6]     % m (fixture WaveUnits = m)
        Tol     = 1e-12
        Psi     = [0.5 0 -0.86602540378443860]   % 30 deg tilted face normal
        % macos_glass_list.txt rows, C in um^2
        SiB  = [0.6961663 0.4079426 0.8974794]
        SiC  = [0.004679148 0.01351206 97.934]
        CaB  = [0.5675888 0.4710914 3.8484723]           % Malitson 1963
        CaC  = [0.050263605^2 0.1003909^2 34.649040^2]
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
        function n = sellmeier(B, C, lam_m)
            L2 = (lam_m*1e6)^2;
            n  = sqrt(1 + sum(B .* L2 ./ (L2 - C)));
        end
    end

    methods
        function p = variant(tc, kind)
            txt = fileread(tc.rx_path);
            switch kind
                case 'silica'
                    p = tc.rx_path;  return
                case 'fixed'   % no glass: every element IndRef= 1.5 (header stays 1)
                    k = strfind(txt, 'nElt=');  tc.assertTrue(isscalar(k));
                    head = txt(1:k-1);  body = txt(k:end);
                    tc.assertEqual(numel(regexp(body, '^[ \t]*GlassElt=', 'lineanchors', 'match')), 2);
                    body = regexprep(body, '^[ \t]*GlassElt=[^\n]*\n', '', 'lineanchors');
                    body = strrep(body, 'IndRef=1.0E+00', 'IndRef=1.5E+00');
                    tc.assertEqual(numel(strfind(body, 'IndRef=1.5E+00')), 2);
                    txt  = [head body];
                case 'caf2'
                    tc.assertEqual(numel(strfind(txt, 'GlassElt=  Silica')), 2);
                    txt = strrep(txt, 'GlassElt=  Silica', 'GlassElt=  CaF2');
                otherwise
                    error('unknown variant %s', kind);
            end
            p = fullfile(tc.tmpdir, ['Rx_GlassPlate_' kind '.in']);
            fid = fopen(p, 'w');  fwrite(fid, txt);  fclose(fid);
        end

        function n = measure(tc, path, lam)
            macos.load_rx(path);
            macos.set_src_wvl(lam);  macos.modify();
            % ray_info after trace(k) = the OUTGOING direction from element k
            s  = macos.trace(1);  ri = macos.get_ray_info(s.nRays);
            ok = ri.ok_trace;
            tc.assertGreaterThan(nnz(ok), 50, 'rays lost at the plate');
            d  = ri.dir(:, ok);
            N  = tc.Psi(:)/norm(tc.Psi);  i0 = [0;0;1];
            sin_i = norm(cross(i0, N));
            sin_t = zeros(1, size(d,2));
            for k = 1:size(d,2), sin_t(k) = norm(cross(d(:,k), N)); end
            nk = sin_i ./ sin_t;
            tc.assertLessThan(max(nk) - min(nk), 1e-12, 'collimated rays refract identically');
            n = mean(nk);
        end
    end

    methods (Test)

        function test_silica_index_follows_malitson_at_three_wavelengths(tc)
            p = tc.variant('silica');
            n = zeros(size(tc.Lambdas));
            for j = 1:numel(tc.Lambdas)
                n(j) = tc.measure(p, tc.Lambdas(j));
                tc.verifyEqual(n(j), tc.sellmeier(tc.SiB, tc.SiC, tc.Lambdas(j)), ...
                    'AbsTol', tc.Tol, sprintf('Silica at %.1f um', tc.Lambdas(j)*1e6));
            end
            tc.verifyGreaterThan(n(1) - n(3), 0.02, 'the index must MOVE with wavelength');
        end

        function test_fixed_index_control_does_not_move(tc)
            % The null: an engine that ignored GlassElt would look like this
            % deck.  It must read 1.5 everywhere and miss the Sellmeier.
            p = tc.variant('fixed');
            for lam = tc.Lambdas
                n = tc.measure(p, lam);
                tc.verifyEqual(n, 1.5, 'AbsTol', tc.Tol, 'fixed IndRef reads back');
                tc.verifyGreaterThan(abs(n - tc.sellmeier(tc.SiB, tc.SiC, lam)), 1e-2, ...
                    'the gate distinguishes a fixed index from the Sellmeier');
            end
        end

        function test_caf2_when_the_engine_table_carries_it(tc)
            % Added to macos_f90/macos_glass_list.txt (dev-candidate) as the
            % Malitson 1963 Sellmeier; an engine built before that row leaves
            % the element at its written IndRef= 1 and this leg reports
            % Incomplete -- visible, never a silent pass.
            p = tc.variant('caf2');
            n1 = tc.measure(p, 1.0e-6);
            tc.assumeGreaterThan(n1, 1.3, ...
                'CaF2 is not in this engine build''s glass table (rebuild pending)');
            for lam = tc.Lambdas
                n = tc.measure(p, lam);
                tc.verifyEqual(n, tc.sellmeier(tc.CaB, tc.CaC, lam), 'AbsTol', tc.Tol, ...
                    sprintf('CaF2 at %.1f um', lam*1e6));
            end
        end
    end
end
