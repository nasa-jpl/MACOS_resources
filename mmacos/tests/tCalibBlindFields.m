classdef tCalibBlindFields < matlab.unittest.TestCase
%TCALIBBLINDFIELDS  calib_blind_fields: which CALIB fields did a solve not see -- gated on RAYS, not on seed WFE.
%   dyson5 TMA stage B (2026-10-04).  A 1 mm threshold on CALIB's BEFORE WFE misfired on a bad SEED whose rays all
%   pass (-4/200: 1.02 mm at the centre, every ray through), and the real blind case -- the pre-125ea9f asphere
%   circle sized on the NOMINAL field (213/91/0 of 253 rays at 1.56/3.13/4.69 deg) -- is a ray-count fact.
%   Fixture: the telecentric Korsch parent (tma_layout 'telecentric', parent F/1.7133), the -4 deg / 180 mm eccentric
%   section, the 3k strip (+-4.69 deg).  MUST-FIRE leg: clear apertures sized on the nominal field only
%   (apply_full_field_apertures at the bias) clip every off-axis field and not the centre -- measured 1.00 / 0.67 /
%   0.21 / 0.00 at 0 / 1.56 / 3.13 / 4.69 deg.  MUST-STAY-QUIET legs: the same section with no apertures, and a
%   seed WFE of 2 mm with every ray passing.  The CALIB failed sentinel (9.9999e36) is honoured on its own.
    properties (Constant)
        ModelSize = 256
    end
    methods (Access = private)
        function [tel, F] = section_(~)
            D = 0.330/1.8;
            [R, t] = macos.design.tma_layout(D, 1.0, 1.7133, 'secondary_mag', 3.5, 'int_focus_m', -0.125*D, 'telecentric', true);
            tel = macos.design.Telescope('family','TMA','aperture_diameter_m',D,'wavelength_m',633e-9,'model_size',256);
            tel.add_mirror('M1','radius_m',R(1),'spacing_after_m',t(1));
            tel.add_mirror('M2','radius_m',R(2),'spacing_after_m',t(2),'convex',true);
            tel.add_mirror('M3','radius_m',R(3),'spacing_after','derive');
            tel.add_focal_plane('FP');  tel.build();
            tel.set_field_bias(-4*60);  tel.set_offaxis('none','dist',0.18);  tel.build();
            fx = linspace(-4.688, 4.688, 7)*pi/180;  F = [0 0; fx(fx ~= 0).' zeros(6, 1)];
        end
    end
    methods (TestClassSetup)
        function setupClass(tc)
            macos.init(tc.ModelSize);
            addpath(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'design', 'src'));
        end
    end
    methods (Test)
        function test_quiet_when_every_ray_passes(tc)
            [tel, F] = tc.section_();
            [b, fp] = calib_blind_fields(tel, F, 2e-3*ones(1, size(F, 1)));   % a 2 mm SEED, every ray through: not blind
            tc.verifyEqual(fp, ones(1, size(F, 1)), 'AbsTol', 1e-12, 'the bare section passes every ray at every field');
            tc.verifyEmpty(b, sprintf('no field is blind on the bare section (got %s)', mat2str(b)));
        end
        function test_fires_on_fields_clipped_by_a_nominal_field_aperture(tc)
            [tel, F] = tc.section_();
            tel.apply_full_field_apertures('fields', [0 -4*pi/180], 'margin', 0.05, 'quiet', true, 'skip', {'FP'});
            [b, fp] = calib_blind_fields(tel, F, []);
            tc.verifyGreaterThanOrEqual(fp(1), 0.99, 'the nominal field (the one the apertures were sized on) is not clipped');
            tc.verifyFalse(ismember(1, b), 'the centre must stay quiet');
            tc.verifyEqual(sort(b), 2:7, sprintf('every off-axis field must be flagged (got %s, pass %s)', mat2str(b), sprintf('%.2f ', fp)));
            % the clipping grows toward the strip edge (fields 2/7 = +-4.69, 4/5 = +-1.56 deg)
            tc.verifyLessThan(fp([2 7]), fp([4 5]), 'the strip edge loses more rays than the inner field');
        end
        function test_the_failed_sentinel_alone_flags_its_field(tc)
            [tel, F] = tc.section_();
            w = 1e-4*ones(1, size(F, 1));  w(3) = 9.9999e36;
            [b, ~] = calib_blind_fields(tel, F, w);
            tc.verifyEqual(b, 3, 'CALIB''s failed sentinel flags exactly its field when every ray passes');
        end
    end
end
