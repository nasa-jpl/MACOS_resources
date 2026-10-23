classdef tVeneerDefaults < matlab.unittest.TestCase
%TVENEERDEFAULTS  The veneer defaults the 2026-10-23 audit found drifting (PLAN_CONSOLIDATION 1.4a, the
%   defaults audit beyond the OPTIONS table of tDefaults); twin of pymacos test_veneer_defaults.py.
%   Each leg is red on the pre-audit veneer:
%     srs        macos.srs(IS1, IS2) APPLIES the slave (link default true = the CLI's [YES] = pymacos);
%                it was false -> SRS answered NO -> since 1.2 (gg) nothing moved.  CoroExample's
%                near-field pair as in cli_tests/cmd/ORS_SRS.jou: element 7 at z = -7.0 slaved to 5
%                comes back to z = -6.3293 (the journal's 'New z', 5 figures).
%     src_size   macos.set_src_size(ap) keeps the obscuration (it set 0).
%     beam cos   macos.beam('cos') without 'radius'/'power' is refused by name, as its help and
%                pymacos beam() say (it used radius 1 BaseUnit).
%     opd_psf    the model size default is 256 (it was 512; Dave 2026-10-16: 256 everywhere).
    properties (Constant)
        Model = 128
    end
    methods (Static, Access = private)
        function p = example(name)
            home = getenv('MACOS_HOME');
            if isempty(home), home = fullfile(getenv('HOME'), 'dev', 'macos', 'macos_f90'); end
            p = fullfile(fileparts(home), 'docs', 'macos-manual', 'examples', name);
        end
    end
    methods (Test)
        function test_srs_default_applies_the_slave(tc)
            macos.init(tc.Model);  macos.load_rx(tVeneerDefaults.example('CoroExample.in'));
            mmacos('elt_z', 7, -7.0, 1, 1);  macos.modify();
            tc.assertEqual(macos.get_elt_z(7), -7.0);
            macos.srs(7, 5);
            tc.verifyEqual(macos.get_elt_z(7), -6.3293, 'AbsTol', 5e-5);
        end
        function test_srs_link_false_declines(tc)
            macos.init(tc.Model);  macos.load_rx(tVeneerDefaults.example('CoroExample.in'));
            mmacos('elt_z', 7, -7.0, 1, 1);  macos.modify();
            macos.srs(7, 5, 'link', false);
            tc.verifyEqual(macos.get_elt_z(7), -7.0);
        end
        function test_set_src_size_keeps_the_obscuration(tc)
            macos.init(tc.Model);  macos.load_rx(tVeneerDefaults.example('Cassegrain.in'));
            macos.set_src_size(4.0, 1.0);
            macos.set_src_size(3.5);
            s = macos.get_src_size();
            tc.verifyEqual([s.aperture s.obscuration], [3.5 1.0]);
        end
        function test_cos_beam_needs_radius_and_power(tc)
            macos.init(tc.Model);  macos.load_rx(tVeneerDefaults.example('Cassegrain.in'));
            tc.verifyError(@() macos.beam('cos'), 'macos:beam:missingCos');
            tc.verifyError(@() macos.beam('cos', 'radius', 2), 'macos:beam:missingCos');
            macos.beam('cos', 'radius', 2, 'power', 1);
            tc.verifyEqual(macos.beam().type, 'cos');
            macos.beam('uniform');
        end
        function test_opd_psf_model_size_default_is_256(tc)
            out = macos.opd_psf(tVeneerDefaults.example('Cassegrain.in'));
            tc.verifyEqual(out.model_size, 256);
        end
    end
end
