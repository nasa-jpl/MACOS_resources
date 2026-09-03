classdef tNollMode < matlab.unittest.TestCase
%TNOLLMODE  The self-contained Noll Zernike evaluator (macos.noll_mode).
%   Replaces the JPL-internal ~/matlab/zernike_mode.m so mmacos distributes
%   self-contained (Luis, 2026-09-03).  At integration the new evaluator was
%   A/B'd against the original on circle/annulus/off-center supports, modes
%   1..15: worst relative difference 1.0e-10 (the original's 1e-9 azimuth
%   guard vs true atan2).  These gates pin the conventions analytically so
%   the equivalence outlives the external file.  SUITE_FAST.
    methods (Test)
        function piston_is_unity(tc)
            m = disk(128, 50);
            Z = macos.noll_mode(m, 1);
            tc.verifyEqual(max(abs(Z(m>0) - 1)), 0, 'AbsTol', 1e-12);
        end
        function defocus_shape_and_normalization(tc)
            m = disk(256, 100);
            Z = macos.noll_mode(m, 4);          % sqrt(3)(2r^2-1)
            tc.verifyEqual(min(Z(m>0)), -sqrt(3), 'AbsTol', 0.01);
            tc.verifyEqual(rms_support(Z, m), 1, 'AbsTol', 0.02);
        end
        function tilt_pair_axes_and_parity(tc)
            % Original convention: azimuth = -atan2(y,x), even index = cos.
            m = disk(128, 50);
            [x, y] = meshgrid(1:128, 1:128);
            Z2 = macos.noll_mode(m, 2);         % cos -> +x tilt
            Z3 = macos.noll_mode(m, 3);         % sin of NEGATED azimuth -> -y
            c2 = corrcoef(Z2(m>0), x(m>0));  tc.verifyGreaterThan(c2(1,2),  0.999);
            c3 = corrcoef(Z3(m>0), y(m>0));  tc.verifyLessThan(c3(1,2), -0.999);
        end
        function unit_rms_and_orthogonality_through_11(tc)
            m = disk(256, 100);
            B = zeros(nnz(m>0), 11);
            for j = 1:11, Z = macos.noll_mode(m, j); B(:,j) = Z(m>0); end
            B = B / sqrt(size(B,1));
            G = B.'*B;                          % ~identity on a filled disk
            tc.verifyEqual(max(abs(diag(G) - 1)), 0, 'AbsTol', 0.03);
            tc.verifyEqual(max(abs(G(~eye(11)))), 0, 'AbsTol', 0.03);
        end
        function noll_index_map_landmarks(tc)
            % (n,m) landmarks of the Noll ordering.
            m = disk(64, 25);
            [~, mm, nn] = macos.noll_mode(m, 4);   tc.verifyEqual([nn mm], [2 0]);
            [~, mm, nn] = macos.noll_mode(m, 11);  tc.verifyEqual([nn mm], [4 0]);
            [~, mm, nn] = macos.noll_mode(m, 7);   tc.verifyEqual([nn mm], [3 1]);
        end
        function offcenter_support_centers_itself(tc)
            % The support centroid, not the array center, is the origin.
            [x, y] = meshgrid(1:128, 1:128);
            m = double(hypot(x-80, y-50) <= 30);
            Z = macos.noll_mode(m, 4);
            [~, imn] = min(Z(:));  [r0, c0] = ind2sub([128 128], imn);
            tc.verifyLessThan(hypot(c0-80, r0-50), 2);
        end
    end
end
function m = disk(N, R)
    [x, y] = meshgrid(1:N, 1:N);
    m = double(hypot(x-(N+1)/2, y-(N+1)/2) <= R);
end
function v = rms_support(Z, m)
    v = sqrt(mean(Z(m>0).^2));
end
