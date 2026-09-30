classdef tPropMedium < matlab.unittest.TestCase
%TPROPMEDIUM  Physical-optics kernels use lambda/n inside a medium (2026-09-30).
%   Twin of pymacos/tests/test_prop_medium.py.  Every propagation kernel in
%   propsub.F used to be handed the VACUUM wavelength whatever medium the
%   leg ran in (right inter-leg phase, wrong Fresnel number by n).  The gate
%   is an identity: a leg of length z in index n equals the same leg of
%   length z/n in vacuum.  Rx_PropMedium_glass.in = Rx_VecChain.in with
%   IndRef 1.5 everywhere; Rx_PropMedium_vac.in = Rx_VecChain.in with every
%   axial distance scaled by 1/1.5.  Intensities are compared (the fields
%   differ by a global piston).  Non-vacuity, measured on the pre-fix engine:
%   glass matched the UNSCALED vacuum deck to 1.1e-15 and missed the scaled
%   one by 74%; post-fix glass == scaled to 1.4e-15, != unscaled by 71-73%.
    properties (Constant)
        ModelSize = 256
        Elts      = [2 4]     % after leg 1 and after leg 2
    end
    properties
        I_glass, I_vac, I_unscaled
    end
    methods (TestClassSetup)
        function setupClass(tc)
            macos.init(tc.ModelSize);
            tc.I_glass    = tc.intens('Rx_PropMedium_glass.in');
            tc.I_vac      = tc.intens('Rx_PropMedium_vac.in');
            tc.I_unscaled = tc.intens('Rx_VecChain.in');
        end
    end
    methods (Access = private)
        function I = intens(tc, name)
            m = macos.Session(tc.ModelSize);
            m.load_rx(rx_fixture_path(name));
            I = cell(1, numel(tc.Elts));
            for k = 1:numel(tc.Elts), I{k} = macos.intensity(tc.Elts(k)); end
        end
    end
    methods (Static, Access = private)
        function r = rel(a, b), r = max(abs(a(:) - b(:))) / max(abs(b(:))); end
    end
    methods (Test)
        function test_glass_leg_equals_scaled_vacuum_leg(tc)
            for k = 1:numel(tc.Elts)
                tc.verifyLessThan(tc.rel(tc.I_glass{k}, tc.I_vac{k}), 1e-12, ...
                    sprintf('elt %d: z in index n must equal z/n in vacuum', tc.Elts(k)));
            end
        end
        function test_glass_leg_differs_from_unscaled_vacuum(tc)
            % the must-fail leg: the pre-fix engine matched THIS deck to 1e-15
            for k = 1:numel(tc.Elts)
                tc.verifyGreaterThan(tc.rel(tc.I_glass{k}, tc.I_unscaled{k}), 0.5, ...
                    sprintf('elt %d: the medium must change the Fresnel number', tc.Elts(k)));
            end
        end
        function test_energy_is_the_same_in_both_media(tc)
            for k = 1:numel(tc.Elts)
                tc.verifyLessThan(abs(sum(tc.I_glass{k}(:)) - sum(tc.I_vac{k}(:))) / sum(tc.I_vac{k}(:)), 1e-10);
            end
        end
    end
end
