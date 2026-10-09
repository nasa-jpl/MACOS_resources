classdef tFFCut < matlab.unittest.TestCase
%TFFCUT  The far-field evanescent cut (engine 2026-10-03, dyson5 addendum 15).
%   A far-field leg maps spatial frequency f to x = lambda*dz*f, so |x| > dz
%   is |sin theta| > 1: frequencies above 1/lambda, no propagating energy.
%   A pupil sampled finer than lambda/2 makes the output window wider than
%   2 dz and those pixels EXIST; an energy-fraction metric over the window
%   counts them.  macos.ffcut(true) zeroes them (FFPROP and FFPropDFT).
%   Two legs, both load-bearing: Rx_FarFieldPinhole (20 um at 1 um, dx1
%   0.32 um): pixels zeroed, the inside bit-identical, the total lower;
%   Rx_Cass_FarField (dx1 >> lambda): zero pixels, bit-identical -- a cut
%   that zeroed everything would pass the pinhole leg alone.  Twin of
%   pymacos tests/test_ffcut.py.
    properties (Constant)
        ModelSize = 256
    end
    methods (Access = private)
        function [I, on, npix, dx] = run_(~, rx, on_)
            macos.load_rx(rx_fixture_path(rx));
            macos.ffcut(on_);
            nE = macos.num_elt();
            I = macos.intensity(nE);
            [on, npix] = macos.ffcut();
            dx = macos.dx_at(nE);
            macos.ffcut(false);
        end
    end
    methods (TestClassSetup)
        function setupClass(tc)
            macos.init(tc.ModelSize);
        end
    end
    methods (Test)
        function test_pinhole_cut_zeroes_only_the_evanescent_pixels(tc)
            [I0, on0, n0, dx] = tc.run_('Rx_FarFieldPinhole.in', false);
            [I1, on1, n1, ~]  = tc.run_('Rx_FarFieldPinhole.in', true);
            tc.verifyFalse(on0);  tc.verifyTrue(on1);
            tc.verifyEqual(n0, 0);  tc.verifyGreaterThan(n1, 0);
            n = size(I0, 1);  dz = 1e-3;
            tc.assertGreaterThan(n * dx / 2, dz, 'the fixture must have an output window wider than dz');
            i = (0:n-1) - floor(n/2);                % pixel n/2+1 is x = 0 (applyfac2 / FFEvanCut)
            [X, Y] = ndgrid(i * dx, i * dx);
            inside = (X.^2 + Y.^2) <= dz^2;
            tc.verifyTrue(isequal(I1(inside), I0(inside)), 'pixels with |x| <= dz must be bit-identical');
            tc.verifyTrue(all(I1(~inside) == 0), 'pixels with |x| > dz must be zero');
            tc.verifyEqual(n1, nnz(~inside));
            tc.verifyGreaterThan(sum(I0(~inside)), 0, 'the uncut deck must carry energy there, else the leg is vacuous');
            tc.verifyLessThan(sum(I1(:)), sum(I0(:)));
        end
        function test_cass_is_untouched(tc)
            [I0, ~, n0, ~] = tc.run_('Rx_Cass_FarField.in', false);
            [I1, ~, n1, ~] = tc.run_('Rx_Cass_FarField.in', true);
            tc.verifyEqual([n0 n1], [0 0]);
            tc.verifyTrue(isequal(I0, I1), 'a pupil sampled coarser than lambda/2 has no evanescent pixels: bit-identical');
        end
    end
end
