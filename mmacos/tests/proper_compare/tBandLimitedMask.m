classdef tBandLimitedMask < matlab.unittest.TestCase
%TBANDLIMITEDMASK  Port of pymacos test_band_limited_mask.py.
%   Phase 6b foundation: characterise band-limited Fourier mask
%   construction vs super-sampling.  Pure math, no macos/PROPER —
%   exercises the +apodizer build helpers across N in {128, 256, 512, 1024}.
%
%   3 assertion-bearing tests (the pymacos suite has 5 total: 2 are
%   ride-along smokes that print / write a 6-panel diagnostic plot
%   without assertions — skipped here for now).

    properties (Constant)
        R0_M   = 18e-3                   % aperture radius (m)
        Extent = 40e-3                   % physical grid extent (m)
        Nlist  = [128, 256, 512, 1024]
    end

    methods (Static, Access = private)
        function dx = dx_for(N, ext)
            dx = ext / N;
        end
    end

    methods (Test)
        function test_band_limited_circle_integral(testCase)
            % Sum of BL disc mask equals disc area / dx^2 to ~6+ digits.
            r0 = testCase.R0_M;
            for N = testCase.Nlist
                dx = tBandLimitedMask.dx_for(N, testCase.Extent);
                spec = apodizer.band_limited_circle(r0);
                mask = apodizer.build_band_limited_mask(N, dx, spec);
                expected = pi * r0^2 / dx^2;
                rel_err  = abs(sum(mask(:)) - expected) / expected;
                testCase.verifyLessThan(rel_err, 1e-6, sprintf( ...
                    'N=%d: BL sum %.4f vs expected %.4f -- rel err %.3e', ...
                    N, sum(mask(:)), expected, rel_err));
            end
        end

        function test_super_sampled_circle_integral(testCase)
            % Sum of K=16 super-sampled disc agrees with disc area to
            % the SS limit (~1e-4).  Baseline that BL beats.
            r0 = testCase.R0_M;
            for N = testCase.Nlist
                dx = tBandLimitedMask.dx_for(N, testCase.Extent);
                mask = apodizer.build_apodised_mask(N, dx, ...
                    apodizer.circle(r0), [], 16);
                expected = pi * r0^2 / dx^2;
                rel_err  = abs(sum(mask(:)) - expected) / expected;
                testCase.verifyLessThan(rel_err, 1e-3, sprintf( ...
                    'N=%d: SS sum %.4f vs expected %.4f -- rel err %.3e', ...
                    N, sum(mask(:)), expected, rel_err));
            end
        end

        function test_band_limited_invariant_across_N(testCase)
            % At fixed r0, the BL mask's central peak (Gibbs overshoot)
            % and its radial-mean at r=r0 should agree across N to
            % better than 1% / 2%.  The "low-N matches high-N" property
            % that makes BL the gold-standard apodiser builder.
            r0 = testCase.R0_M;
            peaks = zeros(numel(testCase.Nlist), 1);
            rings = zeros(numel(testCase.Nlist), 1);
            for k = 1:numel(testCase.Nlist)
                N = testCase.Nlist(k);
                dx = tBandLimitedMask.dx_for(N, testCase.Extent);
                mask = apodizer.build_band_limited_mask(N, dx, ...
                    apodizer.band_limited_circle(r0));

                peaks(k) = max(mask(:));

                [yy, xx] = ndgrid(0:N-1, 0:N-1);
                cy = (N - 1) / 2;
                cx = (N - 1) / 2;
                rr = hypot(yy - cy, xx - cx) * dx;
                ring = (rr > 0.95 * r0) & (rr < 1.05 * r0);
                rings(k) = mean(mask(ring));
            end
            peak_spread = (max(peaks) - min(peaks)) / mean(peaks);
            ring_spread = (max(rings) - min(rings)) / abs(mean(rings));
            testCase.verifyLessThan(peak_spread, 0.01, sprintf( ...
                'BL peak varies >1%% across N: peaks=%s spread=%.4f', ...
                mat2str(peaks', 4), peak_spread));
            testCase.verifyLessThan(ring_spread, 0.02, sprintf( ...
                'BL ring at r=r0 varies >2%% across N: rings=%s spread=%.4f', ...
                mat2str(rings', 4), ring_spread));
        end
    end
end
