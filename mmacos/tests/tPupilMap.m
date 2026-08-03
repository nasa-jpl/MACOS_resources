classdef tPupilMap < matlab.unittest.TestCase
%TPUPILMAP  Gates for the cone-convergence interface-pupil metric.
%
%   PUPIL_MAP (design/src) measures how well a system images its entrance
%   pupil onto its exit/interface pupil, using the model Dave ruled on
%   2026-08-02: the imaging bundle for an entrance-pupil point is the CONE
%   of rays through it, one per field angle, so the cone's aperture IS the
%   used field set.
%
%   (a) THE XPS IDENTITY -- the gate the plan asks for, and it is exact.
%       The engine's XPS command is the TWO-RAY LIMIT of this model: it
%       traces the grid, rotates the chief about StopPos by 5e-6 rad, and
%       crosses each ray with its own differential partner.  Reduce
%       PUPIL_MAP to the same thing -- anchor on the STOP plane, three
%       collinear field points -- and the two must return the SAME pupil
%       surface.  They agree to 2e-4 relative on defocus and 1e-4 on
%       astigmatism.
%
%       DECODED HERE: XPS's differential is the TANGENTIAL (y) one, not the
%       sagittal.  Its `th = xGrid * 5e-6` is a rotation VECTOR about x,
%       which tilts the chief in Y.  An x-field differential returns a
%       genuinely different surface -- on the rodgers1 pupil deck the
%       astigmatism flips sign, +0.708 vs -0.810 mm -- which is what pupil
%       astigmatism MEANS, and is gate (b).  Anyone comparing a pupil number
%       to XPS must match that axis or they are comparing the sagittal focus
%       with the tangential one.
%
%   (c) ANCHORING.  The cones must sit at fixed points on the curved
%       entrance surface; PUPIL_MAP regrids to get there and estimates its
%       own interpolation error.  That estimate must be negligible against
%       the blur it is measuring.
%
%   (d) AND ANCHORING CHANGES THE ANSWER.  'anchor','index' reproduces the
%       naive same-index grouping deliberately.  The surface-anchored blur
%       must be the SMALLER of the two: the ray's walk across the primary
%       between field angles can only add apparent blur, never remove it.
%
%   (e) THE LADDER.  Magnification, distortion, rim partial cones, and the
%       curved-object reference all have to be present and physical on a
%       deck whose answers are known: the on-axis Rodgers2 afocal, where the
%       pupil magnification is 30 by construction and the primary's own sag
%       must come back imaged at the longitudinal magnification m^2.
%
%   (f) THE MAGNIFICATION FRAME.  Per-field pupil magnification read on the
%       deck's PLACED interface plane carries a 1/cos(incidence) obliquity
%       term as each field's exit chief swings off that plane's normal --
%       ~0.8% on this 30x afocal, i.e. the same order as the pupil-imaging
%       defect.  The chief-normal number must be invariant to how the plane
%       is oriented, and the ratio between the two must BE the areal 1/cos.
%
%   FIXTURES: the committed rodgers1 EPD-4060 exit-pupil deck (which carries
%   an ExitPupil Return, so macos.pupil_quality has something to fit) and the
%   committed rodgers2 S1 afocal deck.

    properties
        pdeck        % rodgers1 stage-4 + exit pupil (focal, has an ExitPupil)
        adeck        % rodgers2 S1 on-axis (afocal)
        nE
        root
    end

    methods (TestClassSetup)
        function setup(tc)
            here = fileparts(mfilename('fullpath'));
            tc.root = fileparts(here);
            run(fullfile(tc.root,'mmacos_setup.m'));
            tc.pdeck = fullfile(tc.root,'design','rodgers1', ...
                                'rodgers1_epd4060_stage4_pupil.in');
            tc.adeck = fullfile(tc.root,'design','rodgers2', ...
                                'rodgers2_S1_onaxis.in');
            tc.assumeTrue(exist(tc.pdeck,'file') == 2, ...
                'rodgers1 exit-pupil deck not present');
            macos.init(256);
            macos.load_rx(tc.pdeck);
            tc.nE = macos.num_elt();
        end
    end

    methods
        function [pq, Rq] = xps_(tc)
            macos.load_rx(tc.pdeck);
            pq = macos.pupil_quality(tc.nE-1, 'quiet', true);
            Rq = max(hypot(pq.uv(:,1), pq.uv(:,2)));
        end

        function [o, s] = cone_(tc, F, Rq)
        %CONE_  PUPIL_MAP reduced to XPS's construction, rescaled to XPS's
        %   own normalisation radius so the Zernike coefficients compare.
            o = pupil_map(tc.pdeck, F, 'anchor','stop', 'img_elt', tc.nE-1, ...
                          'nodes', 25, 'min_fields', 2, 'init', false);
            s = (Rq/o.surface.norm_radius)^2;
        end
    end

    methods (Test)

        function test_xps_is_the_two_ray_limit(tc)
        %  (a) the identity, on the TANGENTIAL differential XPS actually uses.
            [pq, Rq] = tc.xps_();
            [o, s] = tc.cone_([0 0; 0 2e-5; 0 -2e-5], Rq);
            tc.verifyEqual(s*o.surface.defocus, pq.defocus, 'RelTol', 0.01, ...
                ['the cone-convergence surface must BE the XPS surface in ' ...
                 'the two-ray limit -- defocus']);
            tc.verifyEqual(s*o.surface.astig(1), pq.astig(1), 'RelTol', 0.01, ...
                'same, astigmatism');
            tc.verifyGreaterThan(abs(pq.defocus), 1e-6, ...
                'the fixture has no pupil curvature to compare');
        end

        function test_the_xps_differential_is_tangential(tc)
        %  (b) the decode: an x-field differential is a DIFFERENT surface.
        %  If this ever starts agreeing with (a), either the engine changed
        %  its differential axis or the deck stopped being astigmatic -- and
        %  in the second case gate (a) is no longer testing anything.
            [pq, Rq] = tc.xps_();
            [ox, sx] = tc.cone_([0 0; 2e-5 0; -2e-5 0], Rq);
            tc.verifyLessThan(sx*ox.surface.astig(1)*pq.astig(1), 0, ...
                ['the sagittal and tangential pupil astigmatism must have ' ...
                 'OPPOSITE sign -- if they do not, the axis decode is wrong']);
        end

        function test_anchoring_residual_is_negligible(tc)
        %  (c) the regrid's own error, against the blur it is measuring.
            F = macos.design.field_grid(0.25*60, 3, 'units','arcmin');
            o = pupil_map(tc.adeck, F, 'nodes', 15, 'init', false);
            tc.verifyLessThan(o.anchor.blur_ratio, 0.05, ...
                ['the regrid''s interpolation error is not small against ' ...
                 'the blur -- the anchoring is contaminating the metric']);
        end

        function test_index_grouping_only_adds_blur(tc)
        %  (d) anchoring is not free, and it only ever goes one way.
            F = macos.design.field_grid(0.25*60, 3, 'units','arcmin');
            oa = pupil_map(tc.adeck, F, 'anchor','surface', 'nodes',15, 'init',false);
            oi = pupil_map(tc.adeck, F, 'anchor','index',   'nodes',15, 'init',false);
            tc.verifyGreaterThan(oi.blur.rms, oa.blur.rms, ...
                ['grouping rays by source index cannot report LESS blur ' ...
                 'than anchoring them on the primary -- the walk across ' ...
                 'the surface can only add']);
            tc.verifyLessThan(oi.blur.rms/oa.blur.rms, 1.5, ...
                ['index grouping inflated the blur by more than 50% -- ' ...
                 'either the anchoring broke or the deck changed shape']);
        end

        function test_ladder_on_a_known_afocal(tc)
        %  (e) the four parts, on the on-axis Rodgers2 deck.
            F = macos.design.field_grid(0.25*60, 3, 'units','arcmin');
            o = pupil_map(tc.adeck, F, 'nodes', 15, 'init', false);
            % (3) the box-centre field's pupil magnification IS the design 30x
            tc.verifyEqual(o.map.mag_centre, 30, 'RelTol', 1e-3, ...
                'the on-axis afocal must demagnify its pupil by exactly 30');
            % the FIELD-AVERAGED cone magnification is a different number,
            % and its gap from 30 is pupil aberration, not an error
            tc.verifyLessThan(o.map.mag, o.map.mag_centre, ...
                'the field-averaged pupil magnification should sit below the centre-field one here');
            % (2) the primary's own sag comes back imaged at m^2 -- a fast
            % primary's faithfully-imaged sag is not pupil curvature
            tc.verifyEqual(abs(o.surface.ideal.beta_over_m2), 1, 'RelTol', 0.25, ...
                ['the entrance sag is not imaged at the longitudinal ' ...
                 'magnification -- the curved-object reference is wrong']);
            % (1) blur is above the diffraction floor (so geometry is the
            % right treatment) and finite
            tc.verifyGreaterThan(o.blur.rms, o.diffraction.floor_m, ...
                'a blur below the diffraction floor cannot be treated geometrically');
            % rim cones are kept, not repaired: the ONLY nodes dropped are
            % the ones with fewer than min_fields contributions.  (On these
            % decks the field walk across the primary -- 0.44 mm at 0.25 deg
            % -- is far smaller than the node spacing, so genuine partial
            % cones do not arise; the four nodes that drop sit outside every
            % field's convex hull, near the lattice corners of the rim.  The
            % gate is written so it still means something when they do.)
            tc.verifyTrue(all(o.n_fields_per_node(o.good) >= 3), ...
                'a node was scored from fewer rays than min_fields');
            tc.verifyEqual(nnz(~o.good), nnz(o.n_fields_per_node < 3), ...
                ['a node with enough cone rays was discarded -- the rim is ' ...
                 'being repaired away']);
            % (4) wander at the placed plane is real and the best plane is
            % no worse than the placed one
            tc.verifyLessThanOrEqual(o.best_plane.rms, o.wander.rms*(1+1e-9), ...
                'the fitted interface plane is worse than the placed one');
        end

        function test_chief_normal_mag_carries_no_obliquity(tc)
        %  (f) THE FRAME TERM.  A per-field magnification read on the deck's
        %  PLACED plane is a footprint on a surface the field's own exit
        %  chief is oblique to, so it carries a 1/cos(incidence) areal
        %  stretch -- ~0.8% of apparent magnification on a 30x afocal over a
        %  0.5 deg box, which is the same order as the pupil-imaging defect
        %  being hunted.  MAG_PER_FIELD_CHIEF reads the same STATION in the
        %  plane normal to that field's OWN exit chief and must therefore be
        %  invariant to how the interface plane is oriented.
        %
        %  Two ways of saying it, both gated here.  (i) Tilt the evaluation
        %  plane 10 deg: the chief-normal numbers must not move at all, the
        %  placed-plane ones must.  (ii) The on-axis deck is coaxial, so it
        %  is rotationally symmetric and its four box corners are one field
        %  radius -- the chief-normal magnification must be identical across
        %  them even when a tilted plane has broken that symmetry for the
        %  placed-plane read.
            F = macos.design.field_grid(0.25*60, 3, 'units','arcmin');
            o0 = pupil_map(tc.adeck, F, 'nodes', 15, 'init', false);
            n0 = o0.placed.psi(:)/norm(o0.placed.psi);
            t  = [1;0;0];  if abs(n0.'*t) > 0.9, t = [0;1;0]; end
            e1 = t - (n0.'*t)*n0;   e1 = e1/norm(e1);
            phi = 10*pi/180;
            o1 = pupil_map(tc.adeck, F, 'nodes', 15, 'init', false, ...
                 'placed', struct('Vpt', o0.placed.Vpt, ...
                                  'psi', (cos(phi)*n0 + sin(phi)*e1).'));

            % (i) invariance -- the same station, the same per-field normal
            tc.verifyEqual(o1.map.mag_per_field_chief, ...
                           o0.map.mag_per_field_chief, 'RelTol', 1e-9, ...
                ['the chief-normal pupil magnification moved when only the ' ...
                 'interface plane''s ORIENTATION changed -- it has inherited ' ...
                 'the 1/cos obliquity it exists to remove']);
            tc.verifyGreaterThan( ...
                max(abs(o1.map.mag_per_field./o0.map.mag_per_field - 1)), 5e-3, ...
                ['tilting the evaluation plane 10 deg did not move the ' ...
                 'placed-plane magnification, so the invariance gate above ' ...
                 'is not testing anything']);

            % the mechanism, named: the ratio IS 1/sqrt(cos(incidence))
            tc.verifyEqual(o1.map.obliquity_per_field, ...
                           1./sqrt(cosd(o1.map.incidence_deg)), 'RelTol', 0.01, ...
                ['the placed-vs-chief magnification ratio is not the areal ' ...
                 '1/cos stretch, so it is not the frame term it is claimed ' ...
                 'to be']);

            % (ii) rotational symmetry survives in the chief-normal number
            corner = abs(F(:,1)) > 1e-9 & abs(F(:,2)) > 1e-9;
            tc.assumeGreaterThanOrEqual(nnz(corner), 4, 'no box corners in the field set');
            sc = o1.map.mag_per_field_chief(corner);
            sp = o1.map.mag_per_field(corner);
            tc.verifyLessThan(max(sc)/min(sc) - 1, 1e-6, ...
                ['the on-axis deck is coaxial, so its four box corners are ' ...
                 'one field radius and must share one pupil magnification']);
            tc.verifyGreaterThan(max(sp)/min(sp) - 1, 1e-3, ...
                ['the tilted plane did not break the corner symmetry of the ' ...
                 'placed-plane read -- the tilt is not doing what the test assumes']);

            % and on the deck's OWN untilted plane the frame term inflates
            % the apparent breathing: this is the S2 finding, pinned.
            sprd = @(v) (max(v)-min(v))/mean(v);
            tc.verifyGreaterThan(sprd(o0.map.mag_per_field), ...
                                 sprd(o0.map.mag_per_field_chief), ...
                ['on the perfect on-axis afocal the placed-plane read must ' ...
                 'show MORE magnification breathing than the chief-normal ' ...
                 'one -- the obliquity can only add here']);
        end

        function test_simulated_pupil_image_builds(tc)
        %  the incoherent pupil image, and its caveats carried in the struct.
            F = macos.design.field_grid(0.25*60, 5, 'units','arcmin');
            o = pupil_map(tc.adeck, F, 'nodes', 15, 'image', true, ...
                          'zones', 3, 'image_npix', 128, 'init', false);
            tc.verifyNotEmpty(o.image);
            tc.verifyTrue(all(isfinite(o.image.img(:))), ...
                'the simulated pupil image carries non-finite pixels');
            tc.verifyGreaterThan(sum(o.image.img(:)), 0, 'the image is empty');
            tc.verifyEqual(o.image.n_zones, 3);
            tc.verifyEqual(o.image.n_field_samples, size(F,1), ...
                'the kernel must record how coarsely the field was sampled');
        end

        function test_kernel_lives_in_design_src(tc)
            p = which('pupil_map');
            tc.verifyNotEmpty(p, 'pupil_map is not on the path');
            tc.verifyTrue(contains(p, fullfile('design','src')), ...
                sprintf('pupil_map resolves to %s, not the shared library', p));
        end
    end
end
