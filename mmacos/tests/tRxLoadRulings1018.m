classdef tRxLoadRulings1018 < matlab.unittest.TestCase
%TRXLOADRULINGS1018  The api surface of Dave's three loader / SAVE rulings of 2026-10-18 (the CLI gates are
%   cli_tests/cmd/OUTCORD_missing.jou, PROPTYPE_numeric.jou, CALIB_save_roundtrip.jou; the reader is shared, so
%   each ruling is gated here too -- PLAN_CONSOLIDATION rule 2).
%     1 a deck with no nOutCord=/Tout= block loads through load_rx (it failed silently before): the loader
%       proposes the beam frame at the last element and the api takes it -- the Cassegrain's OPD comes back,
%       save_rx writes the identity Tout the deck used to carry, and the saved deck re-saves byte-identically.
%     2 a numeric PropType= (0..15 in element order) loads; save_rx writes the names, 0 as Geometric.
%     3 the CALIB setup transfers through save_rx: the deck's keys and a cap set by calib_set_iter come back
%       in the saved file, and a reload + re-save is byte-identical.
    properties (Constant)
        ModelSize = 256
    end
    methods (Test)
        function test_no_outcord_deck_loads_with_the_beam_frame(tc)
            wd = tempname;  mkdir(wd);  c = onCleanup(@() rmdir(wd, 's'));
            m = macos.Session(tc.ModelSize);
            n = m.load_rx(rx_fixture_path('tst_no_outcord.in'));
            tc.assertEqual(n, 5, 'the five-element Cassegrain loaded');
            s = m.trace(n);
            tc.verifyEqual(s.rmsWFE, 1.2938e-12, 'RelTol', 2e-3, 'the proposal does not touch the trace');
            f1 = fullfile(wd, 'noc1.in');  m.save_rx(f1);
            L = splitlines(string(fileread(f1)));
            k = find(startsWith(strtrim(L), 'nOutCord='), 1);
            tc.assertNotEmpty(k, 'SAVE writes the proposed block');
            tc.verifyEqual(sscanf(L(k), ' nOutCord= %d'), 5);
            row1 = sscanf(L(k+1), ' Tout= %f %f %f %f %f %f %f');
            row2 = sscanf(L(k+2), ' %f %f %f %f %f %f %f');
            tc.verifyEqual(row1(:)', [1 0 0 0 0 0 0], 'AbsTol', 0, 'x row = global x (the beam is +z)');
            tc.verifyEqual(row2(:)', [0 1 0 0 0 0 0], 'AbsTol', 0, 'y row = global y');
            m.load_rx(f1);  f2 = fullfile(wd, 'noc2.in');  m.save_rx(f2);
            tc.verifyEqual(fileread(f2), fileread(f1), 'the saved deck re-saves byte-identically');
        end
        function test_numeric_proptype_saves_as_names(tc)
            wd = tempname;  mkdir(wd);  c = onCleanup(@() rmdir(wd, 's'));
            m = macos.Session(tc.ModelSize);
            n = m.load_rx(rx_fixture_path('tst_proptype_numeric.in'));
            tc.assertEqual(n, 16);
            f1 = fullfile(wd, 'pt1.in');  m.save_rx(f1);
            L = splitlines(string(fileread(f1)));
            names = strtrim(extractAfter(L(startsWith(strtrim(L), 'PropType=')), 'PropType='));
            want = ["Geometric" "Geometric" "NFS1surf" "FarField" "NFP1surf" "NFSpherical" "NFPlane" ...
                    "SF1surf" "SpatialFilter" "GeomUpdate" "NF1" "NF2" "Fresnel" "NFdftPlane" "NFdftSphere" "FFdft"];
            tc.verifyEqual(names(:)', want, '0..15 resolve to the name table, 0 = Geometric');
            m.load_rx(f1);  f2 = fullfile(wd, 'pt2.in');  m.save_rx(f2);
            tc.verifyEqual(fileread(f2), fileread(f1));
        end
        function test_calib_setup_transfers_through_save(tc)
            wd = tempname;  mkdir(wd);  c = onCleanup(@() rmdir(wd, 's'));
            m = macos.Session(tc.ModelSize);
            m.load_rx(rx_fixture_path('Rx_AsphCalib.in'));   % SPOT target, OptMaxItrs 10, OptAsph on element 1
            m.calib_set_iter(7);                                 % a setting made by the binding
            f1 = fullfile(wd, 'cs1.in');  m.save_rx(f1);
            txt = fileread(f1);
            tc.verifyTrue(contains(txt, 'OptTarget=  SPOT'), 'the target is written');
            tc.verifyTrue(contains(txt, 'OptMaxItrs=   7'), 'the cap set through the api is written');
            tc.verifyTrue(contains(txt, 'OptAsph='), 'the element key is written');
            m.load_rx(f1);  f2 = fullfile(wd, 'cs2.in');  m.save_rx(f2);
            tc.verifyEqual(fileread(f2), fileread(f1), 'reload + re-save is byte-identical');
        end
    end
end
