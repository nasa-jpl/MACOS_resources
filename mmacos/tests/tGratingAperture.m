classdef tGratingAperture < matlab.unittest.TestCase
%TGRATINGAPERTURE  A Grating element VIGNETTES by its declared aperture.
%
%   Found 2026-10-06 on the dyson5 3k end-to-end deck: 251 of 1185 rays hit
%   the grating 154-213 mm from its axis against a declared Circular aperture
%   of 154 mm, and every one of them "passed" (CC, deck_dyson_record slide
%   12: the drawn grating smaller than the beam).  elemsub.F's Grating,
%   TrGrating, FzpTrGrating_ and DoeTrGrating all computed the aperture /
%   obscuration verdict with ChkRayTrans and then OVERWROTE it on the next
%   line with the interpolated-surface check (`LRT = .not.(Interpolated
%   .AND. IERROR)`), so no grating ever vignetted a ray; only the E-field
%   was zeroed.  Reflector / Refractor keep the verdict.  Fixed: the two are
%   ANDed, LRT = .TRUE. when no aperture is declared.
%
%   Fixture: Rx_GratingImmersed.in with a 6 mm Circular aperture written on
%   the grating (a 20 mm collimated beam), by text substitution.  Legs:
%   the engine's pass flags equal the geometric inside-the-circle set ray by
%   ray; the clipped rays stay clipped to the focal plane; the aperture
%   clips SOMETHING (non-vacuous: the pre-fix engine passed all 89 rays);
%   the deck without the aperture passes every ray (the control).  Size
%   128 -> SUITE_FAST.
    properties (Constant)
        Model  = 128
        RxName = 'Rx_GratingImmersed.in'
        ApR    = 0.006          % m, the aperture radius written on the grating
    end
    properties
        tmpdir
    end
    methods (TestClassSetup)
        function setup(tc)
            tc.tmpdir = tempname;  mkdir(tc.tmpdir);
            macos.init(tc.Model);
        end
    end
    methods (TestClassTeardown)
        function teardown(tc)
            if isfolder(tc.tmpdir), rmdir(tc.tmpdir, 's'); end
        end
    end
    methods (Access = private)
        function f = deck_(tc, with_ap)
            txt = fileread(rx_fixture_path(tc.RxName));
            if with_ap
                % only the grating's block (iElt= 2) gets the aperture
                i2 = strfind(txt, 'iElt=  2');  i3 = strfind(txt, 'iElt=  3');
                blk = txt(i2:i3-1);
                blk = strrep(blk, 'ApType=  None', sprintf('ApType=  Circular\n            ApVec=  %.6E  0.0E+00  0.0E+00', tc.ApR));
                txt = [txt(1:i2-1) blk txt(i3:end)];
                f = fullfile(tc.tmpdir, 'grating_ap.in');
            else
                f = fullfile(tc.tmpdir, 'grating_noap.in');
            end
            fid = fopen(f, 'w');  fprintf(fid, '%s', txt);  fclose(fid);
        end
    end
    methods (Test)
        function test_pass_flags_equal_the_geometric_set(tc)
            macos.load_rx(tc.deck_(true));
            s = macos.trace(2);  ri = macos.get_ray_info(s.nRays);
            v = macos.get_elt_vpt(2);  q = ri.pos - v(:);  r = hypot(q(1, :), q(2, :));   % the grating's axis is z
            inside = r(:) <= tc.ApR;
            tc.verifyGreaterThan(nnz(~inside), 0, 'the fixture must put rays outside the aperture (else the gate is vacuous)');
            tc.verifyGreaterThan(nnz(inside), 0, 'and some inside');
            tc.verifyEqual(logical(ri.ok_pass(:)), inside, ...
                sprintf('engine pass flags vs the circle: %d pass of %d, %d inside (pre-fix: every ray passed)', nnz(ri.ok_pass), s.nRays, nnz(inside)));
            st = macos.get_ray_status(s.nRays);
            tc.verifyEqual(nnz(st.status == 1), nnz(~inside), 'the clipped rays carry the Obscured status');
        end
        function test_clipped_rays_stay_clipped_to_the_focal_plane(tc)
            macos.load_rx(tc.deck_(true));
            s2 = macos.trace(2);  r2 = macos.get_ray_info(s2.nRays);
            nE = macos.num_elt();  sE = macos.trace(nE);  rE = macos.get_ray_info(sE.nRays);
            tc.verifyEqual(nnz(rE.ok_pass), nnz(r2.ok_pass), 'the pass count at the focal plane is the grating''s');
        end
        function test_no_aperture_passes_every_ray(tc)
            macos.load_rx(tc.deck_(false));
            nE = macos.num_elt();  s = macos.trace(nE);  ri = macos.get_ray_info(s.nRays);
            tc.verifyEqual(nnz(ri.ok_pass), s.nRays, 'the control: no aperture, every ray passes');
        end
    end
end
