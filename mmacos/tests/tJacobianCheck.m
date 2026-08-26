classdef tJacobianCheck < matlab.unittest.TestCase
%TJACOBIANCHECK  Gates for design/src/jacobian_check.m.
%
%   The check exists to enforce ONE rule: engine-vs-Jacobian comparisons
%   evaluate the OPD at the harvest's wf_elt.  The e2e6m round-1
%   "slide 11" defect was exactly this rule broken -- the check traced
%   to nElt (a focal plane) against an nElt-1 (exit pupil) Jacobian, and
%   the resulting piston-vs-tilt mismatch masqueraded as a frame
%   clocking error for a whole review cycle (piston/Tz closes at ANY
%   surface, which hid the cause).  See e2e6m_r2_LOG.md R0.1.
%
%   Fixture: e5hex1 (7 off-axis hex segments; elt 12 = exitpupil Return
%   = wf_elt, elt 13 = FocalPlane).  Seg2 is off-axis (pupil radius
%   ~2.66 m), so the wrong-surface failure mode expresses.

    properties (Constant)
        ModelSize = 128
        RxName    = 'e5hex1.in'
        Elt       = 2          % Seg2: ring-1, off-axis
        Tol       = 0.05       % FD-linearity closure
    end

    properties
        rx_path
        ox
    end

    methods (TestClassSetup)
        function setupClass(testCase)
            root = fileparts(fileparts(mfilename('fullpath')));  % mmacos
            addpath(fullfile(root, 'design', 'src'));
            testCase.rx_path = rx_fixture_path(testCase.RxName);
            macos.init(testCase.ModelSize);
            m = macos.Session(testCase.ModelSize);
            testCase.ox = macos.dw_dx(m, testCase.rx_path, ...
                'elts', testCase.Elt, 'dofs', (0:5).');
        end
    end

    methods (Test)
        function test_all_six_dofs_close_at_the_harvest_surface(testCase)
            chk = jacobian_check(testCase.rx_path, testCase.ox, ...
                'elts', testCase.Elt, 'model', testCase.ModelSize);
            testCase.verifyEqual(chk.wf_elt, testCase.ox.wf_elt);
            testCase.verifyEqual(numel(chk.rel), 6, ...
                'expected one row per DOF');
            testCase.verifyLessThanOrEqual(chk.n_null, 1, ...
                'at most the Rz clocking null expected below the floor');
            testCase.verifyLessThan(chk.worst, testCase.Tol, sprintf( ...
                ['engine vs Jacobian must close on ALL six DOFs at ' ...
                 'the harvest surface; worst rel %.3g (%s)'], chk.worst, ...
                chk.tags{find(chk.rel == chk.worst, 1)}));
        end

        function test_wrong_surface_fails_for_rotations(testCase)
            % The tripwire that documents the defect class: the SAME
            % check evaluated at the focal plane (nElt) must FAIL on the
            % rotation DOFs -- if it ever passes there, either the deck
            % lost its pupil/focal distinction or the check went vacuous.
            macos.init(testCase.ModelSize);
            n = macos.load_rx(testCase.rx_path);
            chk = jacobian_check(testCase.rx_path, testCase.ox, ...
                'elts', testCase.Elt, 'model', testCase.ModelSize, ...
                'wf_elt', n);
            rot = chk.rel(chk.dof < 3);
            testCase.verifyGreaterThan(max(rot), 0.5, sprintf( ...
                ['a rotation DOF checked at the WRONG surface (elt %d, ' ...
                 'not wf_elt %d) must blow up; got max rel %.3g'], ...
                n, testCase.ox.wf_elt, max(rot)));
            % ... while piston still closes there -- the property that
            % made the original defect so deceptive.
            pis = chk.rel(chk.dof == 5);
            testCase.verifyLessThan(pis, testCase.Tol, ...
                'piston (Tz) is evaluation-surface-invariant');
        end
    end
end
