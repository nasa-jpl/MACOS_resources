classdef tAsphHook < matlab.unittest.TestCase
%TASPHHOOK  Telescope.optimize's asphere DOFs ('asph_elts' / 'asph_terms'):
%   CALIB's OptAsph= terms from the design layer (2026-10-04, for dyson5
%   round 4 step 4 -- CCMac's lane).
%
%   What is pinned: on the dyson5 TMA parent (f 330 mm, F/1.8) solved over
%   two off-axis fields, adding h^4 + h^6 on M1 and M3 to the conic solve
%   lowers the WFE at EVERY field and the worst field by more than half
%   (measured [529 484 753] nm -> [55 117 308] nm); the coefficients are
%   read back into spec.elt(k).asph (api elt_asph_get); the emitted design
%   carries Surface=Aspheric + AsphCoef and no OptAsph= line; and the
%   vertex-centred circle declared for the solve (so CALIB's zero-
%   coefficient step is sag-based -- without it a metre deck gets the
%   legacy round-off step) is gone from the spec afterwards.
%   Non-vacuity: the conic-only solve is the control, same fields, same
%   iteration cap.
    properties (Constant)
        ModelSize = 256
    end
    methods (Access = private)
        function tel = parent_(~)
            D = 0.330/1.8;
            [R, t] = macos.design.tma_layout(D, 1.0, 1.8, 'secondary_mag', 3.5, ...
                                              'int_focus_m', -0.125*D, 'm3_behind_m', 0.6*D);
            tel = macos.design.Telescope('family','TMA','aperture_diameter_m',D,'wavelength_m',633e-9,'model_size',256);
            tel.add_mirror('M1','radius_m',R(1),'spacing_after_m',t(1));
            tel.add_mirror('M2','radius_m',R(2),'spacing_after_m',t(2),'convex',true);
            tel.add_mirror('M3','radius_m',R(3),'spacing_after','derive');
            tel.add_focal_plane('FP');
            tel.build();
        end
    end
    methods (TestClassSetup)
        function setupClass(tc)
            macos.init(tc.ModelSize);
        end
    end
    methods (Test)
        function test_aspheres_lower_the_wfe_and_round_trip(tc)
            tel = tc.parent_();
            r0 = tel.optimize('fields_arcmin', [30 60], 'dofs', [0 0 0 0 0 0 0 1], 'max_iters', 60);
            tel = tc.parent_();
            r1 = tel.optimize('fields_arcmin', [30 60], 'dofs', [0 0 0 0 0 0 0 1], ...
                              'asph_elts', [1 3], 'asph_terms', [1 2], 'max_iters', 60);
            tc.verifyTrue(r1.converged, 'the asphere solve must converge');
            tc.verifyTrue(all(r1.wfe_after < r0.wfe_after), sprintf('aspheres must lower the WFE at every field: %s vs %s', ...
                mat2str(r1.wfe_after, 3), mat2str(r0.wfe_after, 3)));
            tc.verifyLessThan(max(r1.wfe_after), 0.5*max(r0.wfe_after), 'the worst field must improve by more than half');
            a1 = tel.spec.elt(1).asph;  a3 = tel.spec.elt(3).asph;
            tc.verifyEqual(numel(a1), 2);  tc.verifyTrue(any(a1 ~= 0) && any(a3 ~= 0), 'solved coefficients must be read back');
            tc.verifyEqual(a1, macos.get_elt_asph(1, 2), 'AbsTol', 0, 'spec must equal the engine''s AsphCoef');
            tc.verifyTrue(isempty(tel.spec.elt(1).ap) && isempty(tel.spec.elt(3).ap), 'the enclosing circle is for the solve only');
            rx = tel.save([tempname '.in']);  txt = fileread(rx);  delete(rx);
            tc.verifyEqual(numel(strfind(txt, 'Surface=  Aspheric')), 2);
            tc.verifyEqual(numel(strfind(txt, 'OptAsph=')), 0, 'the clean design carries no optimizer block');
            tc.verifyTrue(contains(txt, 'AsphCoef='));
        end
        function test_asph_elts_must_be_varied_powered_mirrors(tc)
            tel = tc.parent_();
            tc.verifyError(@() tel.optimize('fields_arcmin', 30, 'asph_elts', 4, 'max_iters', 5), ...
                'macos:design:Telescope:optimize:asphElts');
            tc.verifyError(@() tel.optimize('fields_arcmin', 30, 'asph_elts', 1, 'asph_terms', 10, 'max_iters', 5), ...
                'macos:design:Telescope:optimize:asphTerms');
        end
    end
end
