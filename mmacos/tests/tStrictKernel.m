classdef tStrictKernel < matlab.unittest.TestCase
%TSTRICTKERNEL  Gates for the shared strict-WFE metric kernel in design/src.
%
%   STRICT_REFS / STRICT_SPHERE_OPL / STRICT_WFE_DECK / STRICT_RUNGS /
%   STRICT_LADDER_DECK moved out of challenges/rodgers1 (where they were
%   example-local, with the 4-rung ladder duplicated verbatim in
%   pupil_audit.m and dense_field_check.m) into design/src, so the
%   e2e2 design flow and the Rodgers study score through ONE kernel.
%
%   The gates below are the invariants that make "one kernel" true:
%
%   (a) CROSS-KERNEL IDENTITY.  STRICT_LADDER_DECK's rung 1 and rung 2 are
%       the same numbers STRICT_WFE_DECK returns for 'strict-chief' and
%       'strict-centroid'.  Two independently-written drivers over the same
%       reference kernel must agree EXACTLY, not approximately -- they call
%       the same STRICT_REFS on the same rays.  This is what would break
%       first if the two paths drifted apart again.
%
%   (b) RUNG ORDERING.  Each rung grants strictly more reference freedom
%       than the one before, so its residual cannot be larger.  A violation
%       means a rung is not what its name says.  This is now EXACT for every
%       rung -- rung 3 evaluates its own starting point and keeps it when the
%       search cannot beat it -- so the gate is exact too, and any future
%       regression in the focus search trips immediately rather than hiding
%       inside a tolerance.  See STRICT_RUNGS.
%
%   (c) STREHL CONSISTENCY.  The exact aperture form
%       |mean(exp(i*2*pi*W/lambda))|^2 is bounded by 1 and, at small WFE,
%       tracks the Marechal approximation exp(-(2*pi*sigma/lambda)^2).
%
%   Fixture: the COMMITTED rodgers1 EPD-4060 stage-3 deck.  Nothing here
%   depends on its absolute WFE -- only on relations between numbers
%   computed from the same trace -- so it stays valid across engine
%   changes that move the deck's wavefront (macos PR #70 moved it ~12%).

    properties (Constant)
        NFIELD = 3          % fields per gate; each costs ~3 traces
        FOV_DEG = 0.1
    end

    properties
        deck
        lambda
    end

    methods (TestClassSetup)
        function setup(tc)
            here = fileparts(mfilename('fullpath'));
            root = fileparts(here);
            run(fullfile(root,'mmacos_setup.m'));
            tc.deck = fullfile(root,'challenges','rodgers1', ...
                               'rodgers1_epd4060_rodgersS3.in');
            tc.assumeTrue(exist(tc.deck,'file') == 2, ...
                'rodgers1 EPD-4060 stage-3 deck not present');
            tc.lambda = 1.0e-6;          % the Rodgers study wavelength
            macos.init(256);
        end
    end

    methods
        function F = fields_(tc)
            F = macos.design.field_grid(tc.FOV_DEG*60, tc.NFIELD, ...
                                        'units','arcmin');
        end
    end

    methods (Test)

        function test_ladder_rungs_1_2_equal_wfe_deck(tc)
        %  (a) the cross-kernel identity -- EXACT, not approximate.
            F = tc.fields_();
            L  = strict_ladder_deck(tc.deck, F);
            sc = strict_wfe_deck(tc.deck, F, 'reference','strict-chief');
            sn = strict_wfe_deck(tc.deck, F, 'reference','strict-centroid');
            ok = isfinite(L(:,1)) & isfinite(sc.wfe_m(:));
            tc.verifyGreaterThan(nnz(ok), 0, 'no field scored');
            tc.verifyEqual(L(ok,1), sc.wfe_m(ok), ...
                'rung 1 must BE strict_wfe_deck''s chief reference');
            tc.verifyEqual(L(ok,2), sn.wfe_m(ok), ...
                'rung 2 must BE strict_wfe_deck''s centroid reference');
        end

        function test_rungs_are_ordered_by_reference_freedom(tc)
        %  (b) each rung is more permissive than the last -- EXACTLY.
        %
        %  Rung 4's wavefront is rung 3's with a least-squares plane
        %  subtracted, so it cannot be larger.  Rung 3 slides the sphere
        %  along the chief by a bounded search whose domain CONTAINS u = 0,
        %  i.e. the rung-2 sphere; it now evaluates that point explicitly
        %  and keeps it when the search cannot beat it, so rung 3 <= rung 2
        %  holds by construction and not by the solver behaving well.
        %
        %  It did not always.  With FMINBND's default TolX (1e-4 BaseUnits
        %  = 0.1 mm of focus slide, ~31 nm of wavefront at f/20) the search
        %  could not resolve a minimum sharp at the nm level: on the e2e2
        %  axial TMA rung 3 read 2.7x WORSE than rung 2.  On this deck, 100x
        %  more aberrated, the same defect was only 1.7e-4 relative -- which
        %  is why an inexact gate would not have caught it.
            F = tc.fields_();
            L = strict_ladder_deck(tc.deck, F);
            ok = all(isfinite(L),2);
            tc.verifyGreaterThan(nnz(ok), 0, 'no field scored');
            tc.verifyLessThanOrEqual(L(ok,3), L(ok,2), ...
                ['best focus cannot be worse than the centroid sphere it ' ...
                 'starts from -- the focus search lost its ff(0) guard']);
            tc.verifyLessThanOrEqual(L(ok,4), L(ok,3), ...
                'removing LS tip/tilt cannot be worse than not removing it');
        end

        function test_strehl_is_bounded_and_marechal_consistent(tc)
        %  (c) the exact aperture form, not a psf-peak ratio.
            F = tc.fields_();
            [L, info] = strict_ladder_deck(tc.deck, F, 'lambda', tc.lambda);
            S = info.strehl;
            ok = all(isfinite(L),2) & all(isfinite(S),2);
            tc.verifyGreaterThan(nnz(ok), 0, 'no field scored');
            tc.verifyLessThanOrEqual(S(ok,:), 1 + 1e-12, ...
                'the exact Strehl form must never exceed 1');
            tc.verifyGreaterThan(min(S(ok,:),[],'all'), 0, ...
                'Strehl must be positive');
            % Marechal on the most-corrected rung, where sigma is smallest
            m = exp(-(2*pi*L(ok,4)/tc.lambda).^2);
            tc.verifyEqual(S(ok,4), m, 'RelTol', 0.05, ...
                'exact Strehl should track Marechal at small WFE');
        end

        function test_kernel_lives_in_design_src(tc)
        %  The hoist itself: one copy, in the shared library, reachable
        %  from mmacos_setup alone -- no example-local addpath needed.
            want = fullfile('design','src');
            for f = {'strict_refs','strict_sphere_opl','strict_wfe_deck', ...
                     'strict_rungs','strict_ladder_deck'}
                p = which(f{1});
                tc.verifyNotEmpty(p, sprintf('%s is not on the path', f{1}));
                tc.verifyTrue(contains(p, want), sprintf( ...
                    '%s resolves to %s, not the shared kernel in %s', ...
                    f{1}, p, want));
            end
        end
    end
end
