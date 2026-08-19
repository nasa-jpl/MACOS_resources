classdef tCoroContrast < matlab.unittest.TestCase
%TCOROCONTRAST  Unit tests for the ported contrast.py lambda/D machinery.
%   Pure math, no engine calls — pins the MATLAB port of
%   macos.radial_profile / macos.first_airy_null /
%   macos.lambda_over_D_pixels / macos.radial_contrast (+macos library,
%   hoisted out of templates/30_instruments/coro_experiments/ in the 2026-08 reorg)
%   against an analytic Airy pattern with a known first null.  Guards
%   the Sprint-1 E1 dark-zone merit from silent regressions in the port.
%
%   No PathFixture: the helpers are package functions, so src/ (already
%   on the path for every test run) is all they need.

    properties (Constant)
        LamD_true = 8.0           % lambda/D in pixels we build the Airy at
        N         = 192           % grid size
    end

    methods (Static)
        function I = airy_image(N, lamD_px)
            % Analytic Airy intensity centred on the (N-1)/2 array centre,
            % scaled so its first null sits at 1.22*lamD_px.  Airy:
            % I(x) = (2 J1(x)/x)^2, first zero of J1 at x=3.8317, which
            % corresponds to r = 1.22 lambda/D.
            c = (N - 1) / 2;
            [xx, yy] = meshgrid(0:N-1, 0:N-1);
            r = hypot(xx - c, yy - c);
            r_null = 1.22 * lamD_px;
            x = 3.8317 * r / r_null;       % x=3.8317 at the first null
            I = (2 * besselj(1, x) ./ x).^2;
            I(x == 0) = 1.0;               % limit (2 J1(x)/x)->1 as x->0
        end
    end

    methods (Test)
        function test_lambda_over_D_recovers_known(testCase)
            I = tCoroContrast.airy_image(testCase.N, testCase.LamD_true);
            lamD = macos.lambda_over_D_pixels(I);
            % 1-px radial binning -> first-null radius good to ~half a
            % bin -> lamD good to ~half a bin / 1.22.
            testCase.verifyEqual(lamD, testCase.LamD_true, 'AbsTol', 0.7);
        end

        function test_first_airy_null_near_122_lamD(testCase)
            I = tCoroContrast.airy_image(testCase.N, testCase.LamD_true);
            r_null = macos.first_airy_null(I);
            testCase.verifyEqual(r_null, 1.22 * testCase.LamD_true, ...
                'AbsTol', 1.0);
        end

        function test_radial_profile_peak_at_centre(testCase)
            I = tCoroContrast.airy_image(testCase.N, testCase.LamD_true);
            [r, m, ~, n] = macos.radial_profile(I);
            % First finite bin (near r=0) must hold the global peak.
            fin = find(isfinite(m));
            testCase.verifyEqual(m(fin(1)), max(m(isfinite(m))));
            testCase.verifyTrue(all(n(isfinite(m)) > 0));
            testCase.verifyTrue(r(1) < r(end));   % monotone bin centres
        end

        function test_radial_contrast_normalises_to_peak(testCase)
            I = tCoroContrast.airy_image(testCase.N, testCase.LamD_true);
            peak = max(I(:));
            [rl, c] = macos.radial_contrast(I, peak, testCase.LamD_true, 10.0);
            % On-axis contrast (first finite bin) is ~1 by construction.
            fin = find(isfinite(c));
            testCase.verifyEqual(c(fin(1)), 1.0, 'RelTol', 0.05);
            % Separation axis is expressed in lambda/D.
            testCase.verifyLessThanOrEqual(max(rl), 10.0 + 1);
        end

        function test_radial_contrast_flat_image(testCase)
            % A flat image -> contrast equals 1/peak everywhere it has
            % data (mean of a constant ring is the constant).
            I = 5.0 * ones(testCase.N);
            [~, c] = macos.radial_contrast(I, 5.0, testCase.LamD_true, 5.0);
            fin = isfinite(c);
            testCase.verifyEqual(c(fin), ones(1, nnz(fin)), 'AbsTol', 1e-12);
        end

        function test_dark_zone_metrics_flat_image(testCase)
            % Flat image -> every annulus pixel has contrast = const/peak;
            % mean=peak=floor=median equal it, energy = const*n_pix/peak.
            I = 5.0 * ones(testCase.N);
            m = macos.dark_zone_metrics(I, 5.0, testCase.LamD_true, 3, 7);
            testCase.verifyGreaterThan(m.n_pix, 0);
            testCase.verifyEqual(m.mean,   1.0, 'AbsTol', 1e-12);
            testCase.verifyEqual(m.peak,   1.0, 'AbsTol', 1e-12);
            testCase.verifyEqual(m.floor,  1.0, 'AbsTol', 1e-12);
            testCase.verifyEqual(m.median, 1.0, 'AbsTol', 1e-12);
            testCase.verifyEqual(m.energy, double(m.n_pix), 'AbsTol', 1e-9);
        end

        function test_dark_zone_metrics_one_sided(testCase)
            % One-sided (half-plane) region holds ~half the annulus
            % pixels; on a centro-symmetric Airy image the right-half
            % mean contrast equals the full-annulus mean.
            I = tCoroContrast.airy_image(testCase.N, testCase.LamD_true);
            pk = max(I(:));
            full  = macos.dark_zone_metrics(I, pk, testCase.LamD_true, 3, 7);
            right = macos.dark_zone_metrics(I, pk, testCase.LamD_true, 3, 7, ...
                                      'side', 'right');
            testCase.verifyLessThan(right.n_pix, full.n_pix);
            testCase.verifyGreaterThan(right.n_pix, 0.4 * full.n_pix);
            testCase.verifyLessThan(right.n_pix, 0.6 * full.n_pix);
            % centro-symmetric -> half-plane mean matches full mean
            testCase.verifyEqual(right.mean, full.mean, 'RelTol', 0.05);
        end

        function test_dark_zone_metrics_ordering(testCase)
            % On a real-ish (non-constant) image the metrics must order
            % floor <= median <= mean <= peak, and energy = mean*n_pix.
            I = tCoroContrast.airy_image(testCase.N, testCase.LamD_true);
            m = macos.dark_zone_metrics(I, max(I(:)), testCase.LamD_true, 3, 7);
            testCase.verifyLessThanOrEqual(m.floor,  m.median);
            testCase.verifyLessThanOrEqual(m.median, m.mean);
            testCase.verifyLessThanOrEqual(m.mean,   m.peak);
            testCase.verifyEqual(m.energy, m.mean * m.n_pix, 'RelTol', 1e-9);
        end
    end
end
