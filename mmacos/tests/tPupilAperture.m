classdef tPupilAperture < matlab.unittest.TestCase
%TPUPILAPERTURE  Gates for the ColSource pupil fix (macos PR #70).
%
%   Before the fix, ColSource's circular ray-grid acceptance radius
%   (sourcsub.F) was Aperture/2 + Aperture/npts, not Aperture/2.  With
%   npts = nGridpts-1 (tracesub.F:19) every COLLIMATED trace used a pupil
%
%       1 + 2/(nGridpts - 1)
%
%   too large -- +5.0% at nGridpts=41, +0.78% at 256.  PtSource was never
%   affected, so a collimated and a point source traced DIFFERENT pupils
%   for the same Aperture=.
%
%   It hid because the traced pupil is a disc of that radius CLIPPED by the
%   lattice square |x|,|y| <= Aperture/2 -- a square with rounded corners.
%   Measured along either axis it reads EXACTLY the declared aperture; only
%   the DIAGONAL is oversize, so a span check and a plot both look right.
%   These gates therefore measure the RADIUS and the DIAGONAL, never a span.
%   Full diagnosis: design/rodgers1/PACKET.md Addendum 10.
%
%   A/B against the preserved pre-fix mex (2026-08-01): post-fix 5 pass /
%   0 fail, pre-fix 1 pass / 4 fail.  All THREE section-A gates go red
%   (outermost radius 2.015563476 vs the declared semi-diameter 2, ratio
%   1.007782, on Rx_Cass_FarField; greatest chord 20.05731886 vs the
%   declared 20, ratio 1.002866, on Rx_Mask_Parabolas -- the chord is
%   lattice-sampled, so it reads under the full 2/(nGridpts-1)), and so
%   does section B's absolute-slack gate, which characterises the FIXED
%   engine's exact acceptance bound.  Only test_bottom_ray_survives is
%   pre-fix green -- the wider pre-fix circle admits the edge row too.
%
%   Every probe runs at model_size 512 -- the size Rx_Mask_Parabolas
%   (nGridpts=512) needs -- even though the other two fixtures would be
%   happy at 256.  A 256<->512 transition INSIDE one MATLAB process is the
%   hazard PLAN.md §0 logs, and run_mmacos_tests.sh splits the full suite
%   per model size for exactly that reason; a class that transitions
%   internally cannot be isolated by that split.  (§0 reads RESOLVED as of
%   macos 0b07046, but the runner keeps the split deliberately, so keep
%   the probes single-size.)  Model size does not
%   affect the radii these gates measure.  The class gets its own batch
%   line in the runner's `full` case.
%
%   NOTE these gates read ok_trace, NOT ok_pass.  Element apertures and
%   obscurations clear ok_pass downstream of the source grid and would mask
%   the very thing under test -- on Rx_Cass_FarField the oversized corner
%   rays were all clipped by M1, which is exactly why the deck's WFE never
%   moved and the defect survived this long.

    properties (Constant)
        % The engine admits r <= sqrt((Aperture*0.50001)^2 + 1d-9): a
        % RELATIVE epsilon (needed -- see bottom_ray gate) plus a
        % pre-existing ABSOLUTE slack at the r2<=A2 test.  1.00001 is the
        % brief's bound and holds whenever the absolute term is negligible,
        % i.e. Aperture >~ 0.5 base units.  Rx_VecChain (Aperture=0.01) is
        % deliberately NOT gated here for that reason; see
        % test_absolute_slack_is_the_only_excess.
        RTOL = 1.00001;
    end

    methods (TestClassSetup)
        function setup(tc)
            here = fileparts(mfilename('fullpath'));
            run(fullfile(fileparts(here),'mmacos_setup.m'));
        end
    end

    methods
        function S = probe(tc, name, model)
        %PROBE  Transverse ray radii at Elt 1, about the chief intercept.
            rx = rx_fixture_path(name);
            txt = fileread(rx);
            S.Aperture = str2double(strrep(gk(txt,'Aperture'),'D','E'));
            S.nGridpts = str2double(gk(txt,'nGridpts'));
            macos.init(model);
            macos.load_rx(rx);
            tr = macos.trace(macos.num_elt());
            macos.trace(1);
            rm = macos.get_ray_info(tr.nRays);
            pm = rm.pos - rm.pos(:,1);
            d  = rm.dir(:,1);  d = d/norm(d);
            perp = pm - d*(d.'*pm);           % transverse to the chief ray
            ok   = rm.ok_trace(:);
            S.rho   = sqrt(sum(perp.^2,1)).';
            S.rho   = S.rho(ok);
            S.perp  = perp(:,ok);
            S.nTrace= nnz(ok);
        end
    end

    methods (Test)   % ---- A: the pupil the prescription declares --------

        function test_outermost_ray_is_inside_the_declared_pupil(tc)
        %  The gate the pre-fix engine fails: 1.00778 x on this deck.
            S = tc.probe('Rx_Cass_FarField.in', 512);
            tc.verifyLessThanOrEqual(max(S.rho), S.Aperture/2*tc.RTOL, ...
                sprintf(['ColSource launched a ray at %.10g, beyond the ' ...
                         'declared semi-diameter %.10g (ratio %.6f).'], ...
                        max(S.rho), S.Aperture/2, max(S.rho)/(S.Aperture/2)));
        end

        function test_greatest_chord_matches_the_declared_aperture(tc)
        %  The DIAGONAL -- the measurement a span check cannot make.
            S = tc.probe('Rx_Mask_Parabolas.in', 512);
            P = S.perp;
            if size(P,2) > 3000, P = P(:,round(linspace(1,size(P,2),3000))); end
            D = 0;
            for a = 1:size(P,2)
                D = max(D, max(sqrt(sum((P - P(:,a)).^2,1))));
            end
            tc.verifyLessThanOrEqual(D, S.Aperture*tc.RTOL, ...
                sprintf('greatest chord %.10g vs declared aperture %.10g (ratio %.6f)', ...
                        D, S.Aperture, D/S.Aperture));
        end

        function test_collimated_and_point_source_agree(tc)
        %  The two source routines must accept the same radius for the same
        %  Aperture=.  PtSource has always used the (Aperture*0.50001)^2
        %  form; this is the invariant the fix restores.
            S = tc.probe('Rx_Cass_FarField.in', 512);   % collimated
            tc.verifyLessThanOrEqual(max(S.rho)/(S.Aperture/2), 1.00002*1.0000001, ...
                'collimated acceptance radius exceeds the PtSource form');
        end
    end

    methods (Test)   % ---- B: what the epsilon protects -----------------

        function test_bottom_ray_survives(tc)
        %  The lattice starts at -Aperture/2 and steps by Aperture/(npts-1),
        %  so when npts is ODD (nGridpts EVEN) a lattice point lands at
        %  EXACTLY (0,-Aperture/2) -- on the acceptance circle.  'y' gets
        %  there by repeated addition, so a bare r2<=(Aperture/2)^2 test can
        %  drop the whole edge row by one ulp.  That is what the epsilon is
        %  for, and this gate is why it was not simply deleted.
            S = tc.probe('Rx_Cass_FarField.in', 512);
            tc.assertEqual(mod(S.nGridpts,2), 0, ...
                'fixture must have EVEN nGridpts or the edge ray is not on the lattice');
            tol  = 1e-9*S.Aperture;
            edge = nnz(abs(S.rho - S.Aperture/2) < tol);
            tc.verifyGreaterThanOrEqual(edge, 4, ...
                sprintf(['expected the 4 axis-extreme lattice rays at ' ...
                         'r = Aperture/2 exactly; found %d'], edge));
        end

        function test_absolute_slack_is_the_only_excess(tc)
        %  Rx_VecChain declares Aperture=0.01, small enough that the
        %  PRE-EXISTING absolute 1d-9 slack on the r2<=A2 test (untouched by
        %  this fix) widens acceptance by ~2e-5 relative -- more than the
        %  0.50001 epsilon itself.  Pinned so the scale-dependence is a
        %  recorded fact rather than a surprise: the bound below is the
        %  engine's exact acceptance radius.
            S = tc.probe('Rx_VecChain.in', 512);
            bound = sqrt((S.Aperture*0.50001)^2 + 1e-9);
            tc.verifyLessThanOrEqual(max(S.rho), bound, ...
                sprintf('r=%.12g exceeds the exact acceptance bound %.12g', ...
                        max(S.rho), bound));
            tc.verifyGreaterThan(max(S.rho), S.Aperture/2*tc.RTOL, ...
                'if this now passes RTOL the absolute slack was made relative -- retire this gate');
        end
    end
end

function v = gk(txt,key)
    t = regexp(txt,['(?m)^\s*' key '\s*=\s*(\S+)'],'tokens','once');
    if isempty(t), v = ''; else, v = t{1}; end
end
