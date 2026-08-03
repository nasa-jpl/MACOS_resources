classdef tAfocalKernel < matlab.unittest.TestCase
%TAFOCALKERNEL  Gates for the AFOCAL (flat-reference) WFE metric kernel.
%
%   AFOCAL_PLANE_OPL / AFOCAL_REFS / AFOCAL_RUNGS / AFOCAL_WFE_DECK /
%   AFOCAL_LADDER_DECK in design/src are the R -> infinity limit of the
%   focal STRICT_* kernel: an afocal system delivers a collimated beam, so
%   the wavefront reference is a PLANE normal to the exit chief, not a
%   sphere anchored at an exit pupil.
%
%   The gates are the invariants that make "the plane limit of the same
%   algebra" true, and they are RELATIONS between numbers computed from the
%   same trace -- none depends on a deck's absolute WFE, so they stay valid
%   across engine changes that move it.
%
%   (a) THE R -> INFINITY LIMIT, measured.  On real rays, the difference
%       between AFOCAL_PLANE_OPL and STRICT_SPHERE_OPL with the sphere
%       centre pushed downstream at c = a + R*n falls as 1/R.  A decade
%       sweep must show a decade of improvement.  This is the "share the
%       algebra, do not fork it" claim, tested rather than asserted.
%
%   (b) RUNG ORDERING, EXACT.  Each rung is the previous one with a further
%       least-squares term removed, so rung k+1 <= rung k by construction.
%       Unlike STRICT_RUNGS' bounded focus search this needs no guard, and
%       the gate is correspondingly exact.
%
%   (c) CROSS-KERNEL IDENTITY.  AFOCAL_LADDER_DECK's three rungs ARE
%       AFOCAL_WFE_DECK's three 'reference' choices.  The two drivers carry
%       deliberately independent deck helpers (the STRICT_* precedent), so
%       this is what breaks first if they drift.
%
%   (d) ANCHOR INVARIANCE.  Sliding the reference anchor ALONG the chief is
%       pure piston on a parallel bundle, and O(s*theta^2) on one with a
%       residual divergence theta.  Both bounds are gated.  A metric that
%       moves more is measuring where its anchor sits, not the wavefront.
%
%   (e) BORESIGHT RECOVERY, and the 2/3 SPLIT.  The least-squares tilt
%       gradient of W is the transverse part of the exit ray direction, so
%       an injected boresight and an injected collimation error must both
%       come back exactly.  The two boresight ESTIMATORS (the LS gradient
%       and the mean ray direction) agree exactly on tilt-plus-defocus and
%       part company on odd terms by a computable factor -- 2/3 on a pure
%       coma monomial over a full disc -- which is what .bore_split_urad
%       measures.  Both are gated.
%
%   (f) THE POWER SIGN, pinned against a known bundle.  Read the focal
%       fixture at M3 and the beam is converging to the deck's own focal
%       plane; AFOCAL_REFS must report R_curv_m equal to that distance,
%       positive.  This is the gate that would catch a sign flip in the
%       collimation-error reporting, which is otherwise invisible.
%
%   (g) STREHL: the exact aperture form, bounded by 1 and Marechal-
%       consistent on the most-corrected rung.
%
%   FIXTURE.  The committed rodgers1 EPD-4060 stage-3 deck -- a FOCAL deck,
%   deliberately.  The plane metric is defined on any ray set (it reads huge
%   on a converging beam, which is the point of gate (f)), and using the
%   focal fixture keeps this class independent of the rodgers2 artifacts.
%   The afocal deck gate (h) runs additionally when they are present.

    properties (Constant)
        NFIELD  = 3
        FOV_DEG = 0.1
    end

    properties
        deck            % rodgers1 EPD-4060 stage-3 (focal)
        afocal_deck     % rodgers2 S3 (afocal), when committed
        lambda
        root
    end

    methods (TestClassSetup)
        function setup(tc)
            here = fileparts(mfilename('fullpath'));
            tc.root = fileparts(here);
            run(fullfile(tc.root,'mmacos_setup.m'));
            tc.deck = fullfile(tc.root,'design','rodgers1', ...
                               'rodgers1_epd4060_rodgersS3.in');
            tc.assumeTrue(exist(tc.deck,'file') == 2, ...
                'rodgers1 EPD-4060 stage-3 deck not present');
            tc.afocal_deck = fullfile(tc.root,'design','rodgers2', ...
                                      'rodgers2_S3_newconics.in');
            tc.lambda = 1.0e-6;
            macos.init(256);
        end
    end

    methods
        function F = fields_(tc)
            F = macos.design.field_grid(tc.FOV_DEG*60, tc.NFIELD, ...
                                        'units','arcmin');
        end

        function [P,D,L,p1,d1,a,truth] = synth_(~, tx, ty, c3, cm)
        %SYNTH_  A synthetic collimated bundle with a KNOWN wavefront.
        %
        %   The deck fixtures can only ever gate relations; this gates
        %   RECOVERY.  Build the wavefront on the z = 0 plane directly,
        %       Phi(u,v) = L0 + tx*u + ty*v + c3*(u^2+v^2) + cm*u*(u^2+v^2),
        %   put the rays where that plane is (P = (u,v,0), so their OPL to
        %   the reference plane IS Phi), and give each the direction its own
        %   wavefront normal, D = (dPhi/du, dPhi/dv, sqrt(1-...)).  Hand the
        %   kernel a CHIEF along +z: the bundle then carries a tilt, a power
        %   and (optionally) a coma relative to that reference, all known
        %   exactly, and AFOCAL_REFS must return them.
        %
        %   The sampling is a SQUARE lattice clipped to a disc, so it is
        %   symmetric under u -> -u and v -> -v.  That is load-bearing: the
        %   analytic 2/3 coma split assumes a symmetric pupil.
            if nargin < 5, cm = 0; end
            n = 24;  rho = 16.7e-3;               % a 33 mm exit beam
            g = linspace(-rho, rho, n);
            [U,V] = meshgrid(g,g);
            keep = hypot(U,V) <= rho;
            u = U(keep).';  v = V(keep).';
            r2 = u.^2 + v.^2;
            L0 = 1.0;
            L  = (L0 + tx*u + ty*v + c3*r2 + cm*u.*r2).';
            P  = [u; v; zeros(1,numel(u))];
            dx = tx + 2*c3*u + cm*(3*u.^2 + v.^2);
            dy = ty + 2*c3*v + cm*(2*u.*v);
            D  = [dx; dy; sqrt(max(0, 1 - dx.^2 - dy.^2))];
            p1 = [0;0;0];   d1 = [0;0;1];   a = [0;0;0];
            truth = struct('tilt',[tx ty], 'c3',c3, 'coma',cm, ...
                           'rho_max',max(hypot(u,v)), 'u',u, 'v',v);
        end

        function [P,D,L,p1,d1,a] = rays_(tc, ie)
        %RAYS_  Trace the fixture at its own nominal field and hand back the
        %   ray arrays at element IE, plus that element's vertex as anchor.
            txt = regexprep(fileread(tc.deck),'(ApType=\s*)\S+','$1None');
            tmp = [tempname '.in'];
            fid = fopen(tmp,'w'); fprintf(fid,'%s',txt); fclose(fid);
            c = onCleanup(@() delete(tmp)); %#ok<NASGU>
            macos.load_rx(tmp);
            tr = macos.trace(ie);
            ri = macos.get_ray_info(tr.nRays);
            ok = ri.ok_trace(:) & ri.ok_pass(:);  ok(1) = false;
            P = ri.pos(:,ok);  D = ri.dir(:,ok);  L = ri.opl(ok);
            p1 = ri.pos(:,1);  d1 = ri.dir(:,1)/norm(ri.dir(:,1));
            V = regexp(txt,'VptElt=\s*([^\n]*)','tokens');
            a = sscanf(strrep(V{ie}{1},'D','E'),'%f',3);
        end
    end

    methods (Test)

        function test_plane_is_the_R_to_infinity_sphere(tc)
        %  (a) the shared algebra, measured as a limit.
        %
        %  Read at M3, NOT at the focal plane.  At a focus the ray positions
        %  span microns, the sphere-vs-plane sag rho^2/(2R) drops below
        %  double-precision noise on a 20 m OPL, and the "limit" would be
        %  measuring round-off.  At M3 the bundle still fills the aperture,
        %  so the limit is optical, not numerical.
            [P,D,L,~,d1,a] = tc.rays_(3);
            Wp = afocal_plane_opl(P, D, L, a, d1);
            Rs = [1e4 1e5 1e6];
            e  = zeros(size(Rs));
            for i = 1:numel(Rs)
                R  = Rs(i);
                c  = a(:) + d1(:)*R;
                Ws = strict_sphere_opl(P, D, L, c, R);
                % piston-free comparison: the sphere and the plane differ by
                % a constant AND by the sag, and only the sag is the limit
                e(i) = std(Ws(:) - Wp(:));
            end
            tc.verifyGreaterThan(e(1), 0, 'no measurable sphere-plane gap');
            % THE LAW: the gap is the sphere's sag, rho^2/(2R), so e*R is a
            % constant of the bundle.  Two decades of R at 1% is a limit,
            % not a trend.
            eR = e .* Rs;
            tc.verifyEqual(eR(2), eR(1), 'RelTol', 0.01, ...
                'sphere->plane gap is not falling like 1/R (1e4 -> 1e5)');
            tc.verifyEqual(eR(3), eR(2), 'RelTol', 0.01, ...
                'sphere->plane gap is not falling like 1/R (1e5 -> 1e6)');
            % and the gap is bounded by the sag of the pupil it is measured
            % over -- i.e. it is geometry, not a coding error
            f = afocal_refs(P, D, L, P(:,1), d1, a);
            tc.verifyLessThan(e(3), f.rho_max^2/(2*Rs(3)), ...
                'the residual gap exceeds the sphere sag it should BE');
        end

        function test_rungs_are_ordered_by_reference_freedom(tc)
        %  (b) exactly ordered -- no tolerance.
            F = tc.fields_();
            L = afocal_ladder_deck(tc.deck, F);
            ok = all(isfinite(L),2);
            tc.verifyGreaterThan(nnz(ok), 0, 'no field scored');
            tc.verifyLessThanOrEqual(L(ok,2), L(ok,1), ...
                'removing LS tip/tilt cannot be worse than not removing it');
            tc.verifyLessThanOrEqual(L(ok,3), L(ok,2), ...
                'removing power as well cannot be worse than tip/tilt alone');
        end

        function test_ladder_rungs_equal_wfe_deck(tc)
        %  (c) the cross-kernel identity -- EXACT, not approximate.
            F  = tc.fields_();
            L  = afocal_ladder_deck(tc.deck, F);
            nm = {'chief','boresight','collimated'};
            for r = 1:3
                s = afocal_wfe_deck(tc.deck, F, 'reference', nm{r}, ...
                                    'init', false);
                ok = isfinite(L(:,r)) & isfinite(s.wfe_m(:));
                tc.verifyGreaterThan(nnz(ok), 0, 'no field scored');
                tc.verifyEqual(L(ok,r), s.wfe_m(ok), sprintf( ...
                    'rung %d must BE afocal_wfe_deck''s ''%s'' reference', ...
                    r, nm{r}));
            end
        end

        function test_anchor_slide_along_chief_is_pure_piston(tc)
        %  (d) for a COLLIMATED beam the anchor's position along the chief
        %  cannot change a statistic.  It is an afocal-beam property, not a
        %  general one: W shifts by s/(n.d), constant only when n.d is, so
        %  on a converging bundle sliding the anchor genuinely re-weights
        %  the metric (and moves the pupil coordinates the tilt/power fits
        %  live in).  Gated on the synthetic collimated bundle, where it
        %  must hold to round-off.
        %
        %  Two parts, because "pure piston" is exact only to second order in
        %  the beam's own angular spread.  W shifts by s/(n.d): with EVERY
        %  ray parallel (part 1) that is one number and the invariance is
        %  exact; with a residual divergence theta (part 2) it varies by
        %  s*theta^2/2 across the pupil, which is real physics -- a slid
        %  reference plane sees a slightly different wavefront -- and the
        %  gate holds it to that bound rather than pretending it is zero.
            [P,D,L,p1,d1,a] = tc.synth_(3e-6, -5e-6, 0);       % all parallel
            v0 = afocal_rungs(P, D, L, p1, d1, a);
            for s = [-0.37, +1.9]                 % metres along the chief
                v1 = afocal_rungs(P, D, L, p1, d1, a(:) + d1(:)*s);
                tc.verifyLessThan(max(abs(v1 - v0)), 1e-15, sprintf( ...
                    ['sliding the anchor %+g m along a PARALLEL bundle ' ...
                     'changed a rung -- the reference is not a plane ' ...
                     'normal to the chief'], s));
            end
            c3 = -1/(2*400);
            [P,D,L,p1,d1,a] = tc.synth_(3e-6, -5e-6, c3);
            v0 = afocal_rungs(P, D, L, p1, d1, a);
            th = max(vecnorm(D(1:2,:)));           % the beam's angular spread
            for s = [-0.37, +1.9]
                v1 = afocal_rungs(P, D, L, p1, d1, a(:) + d1(:)*s);
                tc.verifyLessThan(max(abs(v1 - v0)), abs(s)*th^2, sprintf( ...
                    ['sliding the anchor %+g m moved a rung by more than ' ...
                     's*theta^2 -- the residual is not the divergence ' ...
                     'term it should be'], s));
            end
        end

        function test_recovers_an_injected_tilt_and_power(tc)
        %  (e) RECOVERY, not merely a relation.  Inject a known boresight and
        %  a known collimation error into a synthetic bundle and require the
        %  kernel to hand both back, with the residual rung collapsing.
            tx = 4.0e-6;  ty = -7.0e-6;  c3 = -1/(2*250);   % converging, R=250 m
            [P,D,L,p1,d1,a,T] = tc.synth_(tx, ty, c3);
            [v, ~, f] = afocal_rungs(P, D, L, p1, d1, a);
            tc.verifyEqual(f.tilt(:).', [tx ty], 'AbsTol', 1e-14, ...
                'the LS tilt must return the injected boresight exactly');
            tc.verifyEqual(f.tilt_urad, hypot(tx,ty)*1e6, 'RelTol', 1e-9);
            tc.verifyEqual(f.power_coef, c3, 'RelTol', 1e-9, ...
                'the defocus coefficient must return the injected power');
            tc.verifyEqual(f.R_curv_m, 250, 'RelTol', 1e-6, ...
                'R_curv_m must be the injected wavefront radius, positive');
            tc.verifyEqual(f.rho_max, T.rho_max, 'RelTol', 1e-12);
            % rung 3 removes exactly what was injected -> round-off is left.
            % The bound is ABSOLUTE (a few eps on the 1 m OPL the bundle
            % carries), not relative: a relative bound would be asking the
            % residual to beat double precision.
            tc.verifyLessThan(v(3), 1e-15, ...
                ['rung 3 did not collapse on a wavefront that is PURE tilt ' ...
                 'plus power -- a fit term is wrong']);
            tc.verifyGreaterThan(v(1), 1e-9, 'the injected wavefront vanished');
            % and the two boresight estimators agree on a symmetric pupil
            % when the wavefront is tilt plus defocus only
            tc.verifyLessThan(f.bore_split_urad, 1e-6*f.tilt_urad, ...
                ['the LS wavefront tilt and the mean ray direction must be ' ...
                 'the same vector on a tilt-plus-defocus wavefront']);
        end

        function test_boresight_split_is_the_coma_indicator(tc)
        %  (e2) WHAT .bore_split_urad MEASURES.  On a pure coma monomial
        %  W = A*u*rho^2 over a symmetric disc the least-squares fit onto
        %  [1,u,v] returns 2*A*R^2/3 while the mean gradient returns A*R^2 --
        %  the two boresight estimators split by a fixed 2/3, independent of
        %  A and R.  Pinning that factor is what turns the split from an
        %  unexplained residual into a coma indicator.  (It is also why the
        %  identity is NOT gated tightly on real decks: the Rodgers2 exit
        %  wavefronts are coma-dominated and split by ~0.6 of the tilt --
        %  correctly.)
            A = 4.0e-3;                          % coma coefficient, 1/m^2
            [P,D,L,p1,d1,a,T] = tc.synth_(0, 0, 0, A);
            f = afocal_refs(P, D, L, p1, d1, a);
            R2 = mean(T.u.^2 + T.v.^2) * 2;      % <rho^2>*2 = R^2 on a disc
            ls = f.tilt(1);
            mn = mean(D(1,:));
            tc.verifyEqual(ls/mn, 2/3, 'RelTol', 0.02, ...
                ['the LS tilt and the mean ray direction must split by 2/3 ' ...
                 'on a pure coma monomial -- if they do not, one of the two ' ...
                 'estimators is not what its name says']);
            tc.verifyEqual(ls, 2*A*R2/3, 'RelTol', 0.02, ...
                'the LS coma-induced tilt is not 2*A*R^2/3');
            tc.verifyGreaterThan(f.bore_split_urad, 0.1*f.tilt_urad, ...
                'a coma-only wavefront must show a large boresight split');
        end

        %  THE BORESIGHT IDENTITY IS NOT GATED ON THE FOCAL FIXTURE, and the
        %  reason is worth recording.  On a strongly converging bundle the
        %  two estimators legitimately part company: the mean of the ray
        %  directions is tilt + 2*c3*(mean u, mean v), so with c3 = -1/(2f)
        %  = -0.098 /m a sub-millimetre asymmetry in the sampled pupil
        %  shifts it by hundreds of microradians, and the least-squares
        %  tilt picks up the rho^2 term through the same asymmetry.  On the
        %  rodgers1 f/20 deck read at M3 they differ by 41% -- correctly.
        %  The identity is an AFOCAL statement; it is gated exactly on the
        %  synthetic bundle above and on real rays in the afocal-deck gate
        %  below.

        function test_power_sign_and_magnitude(tc)
        %  (f) the collimation-error reporting, pinned against a real bundle.
        %  Read the FOCAL fixture at M3: the beam is converging to the deck's
        %  own focal plane, so the wavefront radius must come out POSITIVE
        %  and equal to the M3-to-FP distance.
            [P,D,L,p1,d1,a] = tc.rays_(3);
            f = afocal_refs(P, D, L, p1, d1, a);
            txt = fileread(tc.deck);
            V = regexp(txt,'VptElt=\s*([^\n]*)','tokens');
            v3 = sscanf(strrep(V{3}{1},'D','E'),'%f',3);
            v4 = sscanf(strrep(V{end}{1},'D','E'),'%f',3);
            fdist = norm(v4 - v3);
            tc.verifyGreaterThan(f.R_curv_m, 0, ...
                'a converging bundle must report a POSITIVE wavefront radius');
            tc.verifyEqual(f.R_curv_m, fdist, 'RelTol', 0.02, ...
                ['the fitted wavefront radius at M3 must be the M3-to-focal-' ...
                 'plane distance -- if it is not, the defocus coefficient is ' ...
                 'not a collimation error']);
            tc.verifyLessThan(f.divergence_urad, 0, ...
                'converging must report a negative divergence');
        end

        function test_strehl_is_bounded_and_marechal_consistent(tc)
        %  (g) the exact aperture form, not a psf-peak ratio.
            F = tc.fields_();
            [L, info] = afocal_ladder_deck(tc.deck, F, 'lambda', tc.lambda);
            S = info.strehl;
            ok = all(isfinite(L),2) & all(isfinite(S),2);
            tc.verifyGreaterThan(nnz(ok), 0, 'no field scored');
            tc.verifyLessThanOrEqual(S(ok,:), 1 + 1e-12, ...
                'the exact Strehl form must never exceed 1');
            tc.verifyGreaterThanOrEqual(min(S(ok,:),[],'all'), 0, ...
                'Strehl must be non-negative');
        end

        function test_afocal_deck_scores_and_is_collimated(tc)
        %  (h) the metric on a genuinely AFOCAL deck (rodgers2 S3), when the
        %  artifacts are committed.  Two things must hold that cannot hold on
        %  a focal deck: the residual divergence is small (the system IS
        %  afocal), and rung 1 is not dominated by the power term.
            tc.assumeTrue(exist(tc.afocal_deck,'file') == 2, ...
                'rodgers2 S3 deck not committed yet');
            F = macos.design.field_grid(0.25*60, 3, 'units','arcmin');
            [L, info] = afocal_ladder_deck(tc.afocal_deck, F, ...
                                           'lambda', tc.lambda);
            ok = all(isfinite(L),2);
            tc.verifyGreaterThan(nnz(ok), 0, 'no field scored');
            % collimated: the residual marginal-ray angle is arcsecond-class,
            % not the milliradians a focal beam shows.
            tc.verifyLessThan(max(abs(info.divergence_urad(ok))), 500, ...
                'an afocal deck must deliver a nearly collimated beam');
            % and the power rung buys little, i.e. rung 1 is real wavefront
            % error and not a hidden focus term.
            tc.verifyGreaterThan(median(L(ok,3)./L(ok,1)), 0.05, ...
                ['rung 3 collapsed relative to rung 1 -- the afocal ' ...
                 'reference is absorbing a focus term it should not have']);
            % the boresight split, on REAL rays: bounded by the tilt it is a
            % split of.  A split LARGER than the tilt would mean the two
            % estimators point opposite ways, i.e. a sign error; ~0.6 (what
            % these decks show) is the coma signature of gate (e2).
            tc.verifyLessThan(max(info.bore_split_urad(ok) ./ ...
                                  max(info.tilt_urad(ok), realmin)), 1.0, ...
                ['the two boresight estimators disagree by more than the ' ...
                 'tilt itself -- one of them has a sign wrong']);
        end

        function test_scoring_lens_reproduces_the_afocal_ladder(tc)
        %  (i) THE PLANE KERNEL AND THE SPHERE KERNEL MEASURE THE SAME
        %  WAVEFRONT.  AFOCAL_SCORE_PSF appends an ideal lens behind the
        %  interface pupil in a separate deck, which turns the afocal system
        %  into a focal one that the STRICT_* ladder can score.  The two
        %  ladders then have a rung in common: removing the wavefront POWER
        %  (afocal rung 3) is the same freedom as sliding the focus (focal
        %  rung 3), and both then remove least-squares tip/tilt.  So
        %
        %      focal rung 4  ==  afocal rung 3
        %
        %  and it holds to well under a percent on every field with enough
        %  wavefront to measure.  Fields below ~10 nm are excluded: there the
        %  singlet's own residual and the focal-plane fit, not the telescope,
        %  set the number.
            tc.assumeTrue(exist(tc.afocal_deck,'file') == 2, ...
                'rodgers2 S3 deck not committed yet');
            F = macos.design.field_grid(0.25*60, 3, 'units','arcmin');
            o  = afocal_score_psf(tc.afocal_deck, F, 'init', false);
            La = afocal_ladder_deck(tc.afocal_deck, F);
            ok = isfinite(La(:,3)) & isfinite(o.rungs(:,4)) & La(:,3) > 10e-9;
            tc.verifyGreaterThan(nnz(ok), 3, 'too few scorable fields');
            rel = abs(o.rungs(ok,4) - La(ok,3)) ./ La(ok,3);
            tc.verifyLessThan(max(rel), 0.02, ...
                ['the focal ladder through the scoring lens must return the ' ...
                 'afocal ladder''s own power-removed rung -- if it does not, ' ...
                 'one of the two references is wrong or the lens is ' ...
                 'contributing its own aberration']);
            tc.verifyLessThanOrEqual(o.strehl(ok,4), 1 + 1e-12, ...
                'the scoring deck''s Strehl exceeded 1');
        end

        function test_kernel_lives_in_design_src(tc)
        %  one copy, in the shared library, reachable from mmacos_setup alone.
            want = fullfile('design','src');
            for f = {'afocal_plane_opl','afocal_refs','afocal_rungs', ...
                     'afocal_wfe_deck','afocal_ladder_deck','afocal_score_psf'}
                p = which(f{1});
                tc.verifyNotEmpty(p, sprintf('%s is not on the path', f{1}));
                tc.verifyTrue(contains(p, want), sprintf( ...
                    '%s resolves to %s, not the shared kernel in %s', ...
                    f{1}, p, want));
            end
        end
    end
end
