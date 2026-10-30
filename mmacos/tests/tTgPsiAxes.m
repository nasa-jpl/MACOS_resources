classdef tTgPsiAxes < matlab.unittest.TestCase
%TTGPSIAXES  tg_psi_dm's pixel-axis convention is the user's to specify.
%   Dave 2026-10-29 (PLAN_CONSOLIDATION 4a): coordinates are the user's to
%   specify, templates allow it, and an unspecified one gets a clear
%   consistent default.  example_tg_psi_dm takes PIXEL_AXES (default
%   'x=+col,y=+row'); its closure scores all eight conventions as a CHECK
%   (pixel_axes_check), never as a chooser.
%
%   Checks:
%     (a) the specified convention is the one used and printed, even when
%         another scores higher
%     (b) a wrong specified convention WARNS (tg_psi_dm:pixelAxes), never
%         errors; the right one does not warn
%     (c) on the shipped run's measured scores (tg_psi_dm.mat), the default
%         ties with its 180-degree partner and the tie is reported: the
%         checker poke pattern is centrosymmetric, the 2026-10-28 flip
%     (d) a name outside the eight is refused (input validation)
%   Measured 2026-10-29 on the shipped run: the default and its partner score
%   0.997210553639742 alike (1e-16 apart), the transposes 2.2e-9 below, the
%   mirror flips -0.997; pixel_axes_check's tolerance is 1e-6.
%
%   See also: pixel_axes_check, example_tg_psi_dm, tTgPol.

    properties (Constant)
        Names = {'x=+col,y=+row','x=+col,y=-row','x=-col,y=+row','x=-col,y=-row', ...
                 'x=+row,y=+col','x=+row,y=-col','x=-row,y=+col','x=-row,y=-col'}
    end

    properties
        dir_
    end

    methods (TestClassSetup)
        function setup(tc)
            here = fileparts(mfilename('fullpath'));
            tc.dir_ = fullfile(fileparts(here), 'templates', '90_polarization', 'tg_psi_dm');
            tc.assumeTrue(isfile(fullfile(tc.dir_, 'pixel_axes_check.m')), ...
                'tg_psi_dm template not present');
            addpath(tc.dir_);
            tc.addTeardown(@() rmpath(tc.dir_));
        end
    end

    methods (Test)
        function test_specified_convention_is_used_and_printed(tc)
            sc = [0.2 0.1 0.1 0.95 0.3 0.1 0.1 0.2];   % another convention wins
            spec = tc.Names{1};
            [txt, chk] = tc.run_check(sc, spec, 1e-6);
            tc.verifyEqual(chk.name, spec, 'the specified convention must be the one used');
            tc.verifyEqual(chk.i, 1);
            tc.verifySubstring(txt, sprintf('%s', spec));
            tc.verifySubstring(txt, '<- specified');
        end

        function test_wrong_convention_warns_and_right_one_does_not(tc)
            sc = [0.95 0.2 0.1 0.30 0.1 0.1 0.1 0.2];
            tc.verifyWarning(@() tc.run_check(sc, tc.Names{4}, 1e-6), 'tg_psi_dm:pixelAxes');
            tc.verifyWarningFree(@() tc.run_check(sc, tc.Names{1}, 1e-6));
            [~, chk] = tc.run_check(sc, tc.Names{4}, 1e-6, true);
            tc.verifyTrue(chk.beaten);
            tc.verifyEqual(chk.best_name, tc.Names{1});
        end

        function test_shipped_run_reports_the_tie(tc)
            S = load(fullfile(tc.dir_, 'tg_psi_dm.mat'), 'results');
            cl = S.results.closure;
            tc.assumeTrue(isfield(cl, 'axes_scores'), 'tg_psi_dm.mat predates the check');
            [txt, chk] = tc.run_check(cl.axes_scores, 'x=+col,y=+row', cl.axes_tol);
            tc.verifyTrue(chk.tie, sprintf( ...
                'the default and its partner must tie on this pattern (%.3e apart)', ...
                abs(chk.partner_score - chk.score)));
            tc.verifyEqual(chk.partner, 'x=-col,y=-row');
            tc.verifyTrue(ismember('x=-col,y=-row', chk.within));
            tc.verifyFalse(chk.beaten, 'no convention beats the default past the tolerance');
            tc.verifySubstring(txt, 'TIE:');
            tc.verifyTrue(cl.axes_tie, 'the shipped run must have recorded the tie');
        end

        function test_unknown_name_is_refused(tc)
            tc.verifyError(@() pixel_axes_check(tc.Names, zeros(1,8), 'x=col', 1e-6), ...
                'tg_psi_dm:pixelAxesName');
        end
    end

    methods
        function [txt, chk] = run_check(tc, sc, spec, tol, quiet)
            if nargin > 4 && quiet
                ws = warning('off', 'tg_psi_dm:pixelAxes');
                c = onCleanup(@() warning(ws));
            end
            txt = evalc('chk = pixel_axes_check(tc.Names, sc, spec, tol);');
        end
    end
end
