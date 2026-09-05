classdef tEltTypeCoverage < matlab.unittest.TestCase
%TELTTYPECOVERAGE  Element-type coverage matrix for the sensitivity tools.
%   Luis's ask 1 (2026-09-05, macos/BRIEF_luis_round3.md): the NSReflector
%   eligibility hole survived because no test drove every Element= type
%   through the discovery layer.  This matrix drives a fixture corpus
%   jointly covering the implemented EltID table through the four channel
%   families' DISCOVERY (no traces -- cheap), against HAND-AUTHORED
%   expectations frozen from reading each deck (never computed from the
%   same engine query the implementation uses -- the circular-gate
%   lesson).
%
%   Per deck: the engine element-type readback, find_powered_elts,
%   find_grid_elts, find_zern_elts, find_freeform_elts, and one
%   explicit-'elts' error-contract sample per family.
%
%   Declared corpus gaps (no fixture exists; extend when one appears):
%   LensArray, TrGrating, RfPolarizer, CGHNullPlate, DoeTrGrating, HOE.
%
%   Recorded question (Dave): kr_max = 1e21 classifies sentinel-VARIANT
%   flats as powered -- e5hex1 elt 10 declares Kr = -1e18 and enters the
%   powered set (harmless: a Kr poke there produces ~zero response, a
%   wasted column not a wrong answer).  The expectations below FREEZE
%   current behavior; tightening the sentinel rule is a separate ruling.

    properties (Constant)
        ModelSize = 256
    end

    methods (TestClassSetup)
        function setupClass(testCase) %#ok<MANU>
            macos.init(256);
        end
    end

    methods (Test)
        function test_cass_farfield(testCase)
            [m, rx] = load_(testCase, 'Rx_Cass_FarField.in');
            chk_types_(testCase, {'Obscuring','Reflector','Reflector', ...
                'Return','Return','FocalPlane'});
            testCase.verifyEqual(pe_(m, rx), [2 3], ...
                'powered = the two conic mirrors; powered Return excluded by TYPE');
        end

        function test_cass_ns_twin(testCase)
            % NSReflector in the powered set -- THE Luis bug's fixture.
            [m, rx] = load_(testCase, 'Rx_Cass_NS.in');
            chk_types_(testCase, {'Obscuring','NSReflector','Reflector', ...
                'Return','Return','FocalPlane'});
            testCase.verifyEqual(pe_(m, rx), [2 3]);
        end

        function test_luneberg_ns_refractors(testCase)
            % 14 powered NSRefractor shells; Returns/FP excluded.
            [m, rx] = load_(testCase, 'Rx_Luneberg.in');
            info = macos.get_elt_info(1);
            testCase.verifyEqual(info.type, 'NSRefractor');
            testCase.verifyEqual(pe_(m, rx), 1:14);
        end

        function test_corner_cube_flats_not_powered(testCase)
            % NSReflector TYPE is powered-capable, but flat faces
            % (|Kr| = 1e22) fail the finite-Kr filter: empty set.
            [m, rx] = load_(testCase, 'Rx_CornerCube.in');
            chk_types_(testCase, {'Reference','NSReflector', ...
                'NSReflector','NSReflector','FocalPlane'});
            testCase.verifyEmpty(pe_(m, rx));
        end

        function test_refract_plate_not_powered(testCase)
            % Refractor TYPE is powered-capable; flat plate surfaces are
            % not powered.  dw_dsurf on this deck must error 'nochan'.
            [m, rx] = load_(testCase, 'Rx_Refract.in');
            chk_types_(testCase, {'Obscuring','Reference','Refractor', ...
                'Refractor','Reference','FocalPlane'});
            testCase.verifyEmpty(pe_(m, rx));
            testCase.verifyError(@() macos.dw_dsurf(m, rx), ...
                'macos:dw_dsurf:nochan');
        end

        function test_pol_elements_readback(testCase)
            % TrPolarizer + WavePlate name readback (WavePlate = EltID 18,
            % missing from the get_elt_info table until 2026-09-05).
            load_(testCase, 'Rx_PolElt.in');
            chk_types_(testCase, {'Obscuring','TrPolarizer','WavePlate', ...
                'WavePlate','TrPolarizer','Reference','FocalPlane'});
        end

        function test_grating_excluded_from_powered(testCase)
            % Dave's ruling 2026-09-05: gratings carry base conics but are
            % excluded from the powered auto-set (revisit on demand).
            [m, rx] = load_(testCase, 'Grating_example_001.in');
            info = macos.get_elt_info(1);
            testCase.verifyEqual(info.type, 'Grating');
            testCase.verifyLessThan(abs(macos.get_elt_kr(1)), 1e21, ...
                'fixture invariant: the grating IS powered in Kr');
            testCase.verifyEmpty(pe_(m, rx), ...
                'grating must NOT enter the powered set (ruling)');
        end

        function test_e5hex1_segment_families(testCase)
            % Segmented FreeForm corpus deck: every family at once.
            [m, rx] = load_(testCase, 'e5hex1.in');
            chk_types_(testCase, {'Segment','Segment','Segment','Segment', ...
                'Segment','Segment','Segment','Reflector','Refractor', ...
                'Refractor','Return','Return','FocalPlane'});
            % elt 10 (Kr = -1e18, sentinel-variant flat) freezes current
            % kr_max behavior -- see the class help.
            testCase.verifyEqual(pe_(m, rx), 1:10);
            g = macos.find_grid_elts();
            testCase.verifyEqual(g(:).', [1:7 9], ...
                'grid-bearing = the 7 segments + the FreeForm refractor');
            ze = m.find_zern_elts(rx);
            testCase.verifyEqual(ze(:).', 8, 'Zernike surface = elt 8');
            ff = m.find_freeform_elts();
            testCase.verifyEqual(ff(:).', [1:7 9]);
        end

        function test_ffsegdemo_families(testCase)
            [m, rx] = load_(testCase, 'FFSegDemoAll.in');
            testCase.verifyEqual(pe_(m, rx), 1:8, ...
                'segments + PM powered; powered Reference 9 excluded by TYPE');
            g = macos.find_grid_elts();
            testCase.verifyEqual(g(:).', 2:7, ...
                'segment 1 is the Conic baseline -- no grid');
            ff = m.find_freeform_elts();
            testCase.verifyEqual(ff(:).', 2:7);
        end

        function test_explicit_elts_error_contract_all_families(testCase)
            % One unserveable explicit request per family: each must raise
            % the named error, never silently drop (Dave's ruling).
            [m, rx] = load_(testCase, 'e5hex1.in');
            eid = 'macos:channels:eltNotEligible';
            testCase.verifyError(@() macos.channels.surf_channels( ...
                m, rx, 'elts', 13), eid);              % FocalPlane
            testCase.verifyError(@() macos.channels.grid_channels( ...
                m, zeros(3,3,1), 'elts', 8), eid);     % Zernike, no grid
            testCase.verifyError(@() macos.channels.zernike_channels( ...
                m, rx, 'elts', 1), eid);               % segment, no Zern srf
            testCase.verifyError(@() macos.channels.freeform_monzern_channels( ...
                m, rx, 'elts', 8), eid);               % not FreeForm-typed
            testCase.verifyError(@() macos.channels.rigid_body_channels( ...
                m, rx, 'elts', 11), eid);              % Return excluded
        end
    end
end

% ==== helpers =========================================================
function [m, rx] = load_(testCase, name)
    m = macos.Session(testCase.ModelSize);
    rx = rx_fixture_path(name);
    m.load_rx(rx);
end

function p = pe_(m, rx)
    p = macos.find_powered_elts(m, rx);
    p = p(:).';
end

function chk_types_(testCase, expected)
    for k = 1:numel(expected)
        info = macos.get_elt_info(k);
        testCase.verifyEqual(info.type, expected{k}, ...
            sprintf('elt %d type readback', k));
    end
end
