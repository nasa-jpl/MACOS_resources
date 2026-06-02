classdef tDwDxGroups < matlab.unittest.TestCase
%TDWDXGROUPS  Regression tests for group channels in dw_dx.

    properties (Constant)
        ModelSize  = 128
        RxName     = 'e5hex1.in'
        TestGroup  = [9; 10; 12; 13]
    end

    properties
        rx_path
    end

    methods (TestClassSetup)
        function setupClass(testCase)
            testCase.rx_path = rx_fixture_path(testCase.RxName);
            macos.init(testCase.ModelSize);
        end
    end

    methods (Test)
        function test_parse_rx_groups_empty(testCase)
            % e5hex1.in has no EltGrp= declarations.
            g = macos.channels.parse_rx_groups(testCase.rx_path);
            testCase.verifyEqual(g.Count, uint64(0), ...
                'parse_rx_groups should return empty Map for e5hex1');
        end

        function test_group_channel_builder(testCase)
            m = macos.Session(testCase.ModelSize);
            m.load_rx(testCase.rx_path);
            groups = containers.Map('KeyType','char','ValueType','any');
            groups('Cam') = testCase.TestGroup;
            chs = macos.channels.grouped_rigid_body_channels(m, ...
                groups, 'rx_path', testCase.rx_path);
            testCase.verifyEqual(numel(chs), 6, ...
                'Should build 6 channels (one per DOF)');
            for k = 1:numel(chs)
                testCase.verifyEqual(chs{k}.kind(), 'Group');
                testCase.verifyEqual(chs{k}.ref_elt, ...
                    testCase.TestGroup(1));
                testCase.verifyEqual(chs{k}.fp_elt, 13, ...
                    'Cam fp_elt should auto-detect to Elt 13 (FocalPlane)');
                testCase.verifyEqual(chs{k}.fp_mode, 'sxp', ...
                    'auto fp_mode -> sxp when FP in group');
            end
        end

        function test_eltgrp_install_restore(testCase)
            m = macos.Session(testCase.ModelSize);
            m.load_rx(testCase.rx_path);
            ref = testCase.TestGroup(1);
            % Initial: no group installed.
            testCase.verifyEmpty(m.get_elt_grp(ref));
            % After apply -> install
            ch = macos.channels.GroupedRigidBodyChannel(m, ...
                testCase.TestGroup, 3, ...
                'fp_elt', 13, 'fp_mode', 'sxp');
            ch.apply(1e-8);
            members = m.get_elt_grp(ref);
            testCase.verifyEqual(sort(members), ...
                sort(testCase.TestGroup), ...
                'apply should install the desired EltGrp');
            % After restore -> uninstall back to empty
            ch.restore();
            testCase.verifyEmpty(m.get_elt_grp(ref), ...
                'restore should release the EltGrp install');
        end

        function test_dw_dx_groups_runs_clean(testCase)
            m = macos.Session(testCase.ModelSize);
            groups = containers.Map('KeyType','char','ValueType','any');
            groups('Cam') = testCase.TestGroup;
            out = macos.dw_dx(m, testCase.rx_path, ...
                'dofs', [3; 4; 5], ...
                'groups', groups, ...
                'delta', 1e-8);
            % 11 actual optics * 3 DOFs = 33 per-element + 3 group DOFs = 36
            testCase.verifyEqual(numel(out.channel_names), 33 + 3);
            % Last three are the group channels.
            for k = 34:36
                testCase.verifyEqual(out.kind{k}, 'Group');
            end
            for k = 1:33
                testCase.verifyEqual(out.kind{k}, 'RigidBody');
            end
            % Group columns should be non-trivial.
            grp_max = max(max(abs(out.dwdx(:, 34:36))));
            testCase.verifyGreaterThan(grp_max, 0, ...
                'group columns should be non-zero');
        end
    end
end
