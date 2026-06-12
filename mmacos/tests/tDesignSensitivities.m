classdef tDesignSensitivities < matlab.unittest.TestCase
%TDESIGNSENSITIVITIES  macos.design.System.sensitivities (Sprint 2A-i, slice 2).
%   The design-layer sensitivities() must HARVEST the Phase 7 drivers,
%   not re-derive them: each returned block must be bitwise-identical to
%   a standalone macos.dw_dx / macos.dw_dz_zernike call with the SAME
%   options.  e5hex1 carries both rigid-body and Zernike-eligible
%   elements, so both families appear.
%
%   Kept lean on purpose: a reduced DOF set (Tx, Tz) and a 2-mode
%   Zernike range cut the FD sweep ~10x vs the full 6-DOF / 12-mode
%   default while still proving the harvest is faithful.  The defaults
%   themselves are exercised by tDwDx / tDwDzZernike.

    properties (Constant)
        ModelSize = 128
        RxName    = 'e5hex1.in'
        DOFS      = [0 5]      % Tx, Tz — reduced rigid sweep
        ZSTART    = 4
        NZ        = 5          % modes 4..5 — reduced Zernike sweep
    end

    properties
        rx_path
    end

    methods (TestClassSetup)
        function setupClass(testCase)
            testCase.rx_path = rx_fixture_path(testCase.RxName);
        end
    end

    methods (Test)
        function test_harvest_is_bitwise(testCase)
            % Core property: both blocks equal standalone driver output
            % under matching options.
            s    = macos.design.System.from_rx(testCase.rx_path, ...
                'model_size', testCase.ModelSize);
            sens = s.sensitivities('dofs', testCase.DOFS, ...
                'zmode_start', testCase.ZSTART, 'n_zcoef', testCase.NZ);

            rbref = macos.dw_dx(macos.Session(testCase.ModelSize), ...
                testCase.rx_path, 'dofs', testCase.DOFS);
            znref = macos.dw_dz_zernike(macos.Session(testCase.ModelSize), ...
                testCase.rx_path, 'zmode_start', testCase.ZSTART, ...
                'n_zcoef', testCase.NZ);

            testCase.verifyEqual(sens.rigid.dwdx, rbref.dwdx);   % bitwise
            testCase.verifyEqual(sens.rigid.dof_idx, rbref.dof_idx);
            testCase.verifyNotEmpty(sens.zern.dwdz);
            testCase.verifyEqual(sens.zern.dwdz, znref.dwdz);    % bitwise
            testCase.verifyEqual(sens.zern.mode, znref.mode);

            % Blocks share the Nw wavefront rows -> horizontally joinable.
            testCase.verifyEqual(size(sens.rigid.dwdx,1), size(sens.zern.dwdz,1));
            J = [sens.rigid.dwdx, sens.zern.dwdz];
            testCase.verifyEqual(size(J,2), ...
                size(sens.rigid.dwdx,2) + size(sens.zern.dwdz,2));
        end

        function test_families_selection(testCase)
            s = macos.design.System.from_rx(testCase.rx_path, ...
                'model_size', testCase.ModelSize);
            sr = s.sensitivities('families', {'rigid'}, 'dofs', testCase.DOFS);
            testCase.verifyNotEmpty(sr.rigid);
            testCase.verifyEmpty(sr.zern);
            testCase.verifyTrue(all(ismember(sr.rigid.dof_idx, testCase.DOFS)));

            sz = s.sensitivities('families', {'zern'}, ...
                'zmode_start', testCase.ZSTART, 'n_zcoef', testCase.NZ);
            testCase.verifyEmpty(sz.rigid);
            testCase.verifyNotEmpty(sz.zern);
        end
    end
end
