classdef tZernikeGridBasis < matlab.unittest.TestCase
%TZERNIKEGRIDBASIS  WS3: engine-exact multi-convention Zernike grid basis.
%
%   macos.zernike_grid_basis realises a non-ANSI convention (Noll, Born&Wolf)
%   by REMAPPING the requested index to its ANSI index and evaluating the ANSI
%   polynomial -- exactly what the engine does (surfsub.F ZerntoMon6 / ZerntoMon2
%   are pure ordering permutations that relay to ZerntoMon1).  These tests pin
%   the remap against an INDEPENDENT (n,m) derivation of each convention's
%   ordering (the same ordering the engine tables encode), so a transcription
%   typo in the permutation table is caught without an engine round-trip.  They
%   also confirm the ANSI default is unchanged and that unsupported conventions
%   error rather than return a wrong basis.

    properties (Constant)
        N       = 64
        NModes  = 15    % orders 0..4 -- within ansi_zernike_eval's tabulated range
    end

    methods (Access = private)
        function a = ansi_index_(~, n, m)
            % 1-based ANSI/OSA single index of (n, signed m).
            a = (n*(n+2) + m)/2 + 1;
        end

        function [n, m] = noll_nm_(~, j)
            % Noll 1976 ordering -> (n, signed m); even j -> cos (m>=0),
            % odd j -> sin (m<0).  Mirrors macos.noll_mode's index math.
            n = 0;
            while (n+1)*(n+2)/2 < j, n = n + 1; end
            k = j - n*(n+1)/2 - 1;
            if mod(n,2) == 0, am = 2*floor((k+1)/2); else, am = 2*floor(k/2) + 1; end
            if am == 0
                m = 0;
            elseif mod(j,2) == 0
                m = am;      % cos
            else
                m = -am;     % sin
            end
        end

        function [n, m] = bornwolf_nm_(~, j)
            % Born & Wolf ordering -> (n, signed m): within radial order n the
            % index runs m = n, n-2, ..., -n (m>0 cos, m<0 sin), matching
            % surfsub.F ZerntoMon2's (n,m) column.
            n = 0;
            while 1 + (n+1)*(n+2)/2 <= j, n = n + 1; end
            start = 1 + n*(n+1)/2;   % 1-based index of this order's first mode
            p = j - start;           % 0..n
            m = n - 2*p;
        end
    end

    methods (Test)
        function test_ansi_default_unchanged(testCase)
            % The default convention is ANSI and byte-identical to the
            % explicit 'ansi' argument (no existing caller changes).
            modes = 1:testCase.NModes;
            d = macos.zernike_grid_basis(testCase.N, modes);
            a = macos.zernike_grid_basis(testCase.N, modes, 1.0, 'ansi');
            testCase.verifyTrue(isequal(d, a), ...
                'default convention must equal explicit ansi');
        end

        function test_noll_reindex_matches_nm_derivation(testCase)
            % Noll mode j must equal the ANSI mode at the index its (n,m)
            % maps to -- the engine's ZerntoMon6 reindex, cross-checked here
            % by an independent (n,m) derivation.
            for j = 1:testCase.NModes
                [n, m] = testCase.noll_nm_(j);
                a = testCase.ansi_index_(n, m);
                Bn = macos.zernike_grid_basis(testCase.N, j, 1.0, 'noll');
                Ba = macos.zernike_grid_basis(testCase.N, a, 1.0, 'ansi');
                testCase.verifyLessThan(max(abs(Bn(:) - Ba(:))), 1e-12, ...
                    sprintf('Noll mode %d must equal ANSI mode %d', j, a));
            end
        end

        function test_bornwolf_reindex_matches_nm_derivation(testCase)
            for j = 1:testCase.NModes
                [n, m] = testCase.bornwolf_nm_(j);
                a = testCase.ansi_index_(n, m);
                Bb = macos.zernike_grid_basis(testCase.N, j, 1.0, 'bornwolf');
                Ba = macos.zernike_grid_basis(testCase.N, a, 1.0, 'ansi');
                testCase.verifyLessThan(max(abs(Bb(:) - Ba(:))), 1e-12, ...
                    sprintf('Born&Wolf mode %d must equal ANSI mode %d', j, a));
            end
        end

        function test_conventions_differ_on_odd_modes(testCase)
            % Non-vacuity: the orderings genuinely differ where the (n,m)
            % assignment differs (e.g. index 7 -- coma vs trefoil), so a
            % no-op remap would be caught.
            j = 7;
            Bn = macos.zernike_grid_basis(testCase.N, j, 1.0, 'noll');
            Ba = macos.zernike_grid_basis(testCase.N, j, 1.0, 'ansi');
            s  = max(abs(Ba(:)));
            testCase.verifyGreaterThan(s, 0);
            testCase.verifyGreaterThan(max(abs(Bn(:) - Ba(:))) / s, 1e-2, ...
                'Noll and ANSI mode 7 must differ (ordering is a real remap)');
        end

        function test_unsupported_convention_errors(testCase)
            % Fringe / Hex / AnnularNoll are deferred -- requesting one must
            % error, never silently return a wrong basis.
            testCase.verifyError(@() macos.zernike_grid_basis( ...
                testCase.N, 4, 1.0, 'fringe'), ?MException);
            testCase.verifyError(@() macos.zernike_grid_basis( ...
                testCase.N, 4, 1.0, 'normhex'), ?MException);
        end
    end
end
