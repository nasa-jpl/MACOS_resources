classdef tLensTriplet < matlab.unittest.TestCase
%TLENSTRIPLET  The Cooke triplet template (templates/35_lenses/triplet).
%
%   Legs
%     seed     the thin-lens solve meets its three conditions from the textbook:
%              total power 1/f, zero axial color sum y^2 phi/V, Petzval sum
%              PETZVAL/f (each to 1e-10, recomputed here from the returned
%              powers and spacing, not from the template's own residual)
%     result   the defaults (SK16/F4/SK16, f 100, f/5, +/-20 deg): the
%              accepted lens is at EFL 100 (1e-9), every gap clears the 1 mm
%              floor, every edge thickness at the clear radius is positive, no
%              ray is lost at any scanned field; the merit falls from the
%              seed's 143.4 to 1.009 waves (pinned to 2 %); every field to 20 deg
%              is under 1.5 waves; distortion, reported not fitted, under 1 %
%              at the edge (measured -0.41 %); the solve reaches Taylor's
%              form (+ - + powers, biconcave flint)
%
%   Model 128 -> SUITE_FAST.  Runtime ~15 s.

    properties
        tmp
        out
    end
    methods (TestClassSetup)
        function setup(tc)
            h = fileparts(fileparts(mfilename('fullpath')));
            addpath(fullfile(h, 'templates', '35_lenses', 'triplet'));
            tc.tmp = tempname;  mkdir(tc.tmp);
            macos.init(128);
            tc.out = triplet('OUTDIR', tc.tmp);
        end
    end
    methods (TestClassTeardown)
        function teardown(tc)
            if exist(tc.tmp, 'dir'), rmdir(tc.tmp, 's'); end
        end
    end

    methods (Test)
        function test_thin_seed_meets_the_textbook_conditions(tc)
            x = tc.out.thin;  f = 100;  ld = 587.5618;
            nn = @(g, l) macos.design.glass(g, l);
            na = nn('SK16', ld);  nb = nn('F4', ld);
            Va = (na - 1) / (nn('SK16', 486.1327) - nn('SK16', 656.2725));
            Vb = (nb - 1) / (nn('F4', 486.1327) - nn('F4', 656.2725));
            p = [x(1) x(2) x(1)];  y = [1 0 0];  u = -p(1);
            y(2) = 1 + x(3) * u;  u = u - y(2) * p(2);  y(3) = y(2) + x(3) * u;  u = u - y(3) * p(3);
            tc.verifyEqual(-u, 1 / f, 'RelTol', 1e-10, 'total power');
            tc.verifyEqual(f * sum(y.^2 .* p ./ [Va Vb Va]), 0, 'AbsTol', 1e-10, 'axial color');
            tc.verifyEqual(f * sum(p ./ [na nb na]), 0.2, 'AbsTol', 1e-10, 'Petzval sum');
        end

        function test_result_pins(tc)
            o = tc.out;  L = o.lens;  S = o.scan;
            tc.verifyEqual(L.paraxial(587.5618).efl, 100, 'RelTol', 1e-9);
            tc.verifyGreaterThanOrEqual(min(L.t), 1.0 - 1e-12, 'gap floor');
            tc.verifyGreaterThan(min(L.edge_thickness(13)), 0, 'edge thickness at the clear radius');
            tc.verifyEqual(S.lost, zeros(size(S.lost)), 'no ray lost at any field');
            tc.verifyEqual(o.merit_seed, 143.4, 'RelTol', 0.02);
            tc.verifyEqual(o.merit, 1.009, 'RelTol', 0.02);
            tc.verifyLessThan(max(S.rms), 1.5);
            tc.verifyLessThan(abs(S.dist(end)), 1.0, 'distortion reported, not fitted');
            c = L.curv();
            tc.verifyEqual(sign([c(1) - c(2), c(3) - c(4), c(5) - c(6)]), [1 -1 1], '+ - + element powers');
            tc.verifyTrue(c(3) < 0 && c(4) > 0, 'biconcave flint');
        end
    end
end
