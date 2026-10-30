classdef tConvergedEst < matlab.unittest.TestCase
%TCONVERGEDEST  The regularized estimator driven to CONVERGENCE (dm_gauge_lib/
%   dmg_converge; dmg_act_fit opts.converge; tg96_run battery.est_mode).  Pure
%   MATLAB, no engine.  Dave 2026-10-11: the regularization is driven to
%   convergence, not dropped -- the COPHI gain probe found an EXACT reading reads
%   0.995 through the record's one-step Tikhonov matrix estimator (lambda_m 1e-3).
%
%   Gates, on two forward models (a measured-matrix stand-in: Gaussian influence
%   functions sampled on a pupil grid, the record's lambda rule; and the stencil
%   lattice of dmg_act_fit):
%     C1  RED leg: the one-step estimator's gain on an exact reading is < 0.999
%         (single actuator and dense random) -- the bias the probe measured
%     C2  converged: gain 1.0000 +- 1e-4 on the same exact readings
%     C3  the noise trade, MEASURED: white pixel noise, the piston-free floor of
%         the converged estimate against the one-step's at the same noise
%         (asserted finite and reported: <= 3x), and converging to the floor
%         (tol = the one-step floor) stops in fewer steps than to 0
%     C4  dmg_act_fit opts.converge reaches gain 1.0000 +- 1e-4 on a determined
%         stencil problem where its one step reads < 0.999; the default (no opts)
%         is unchanged
%     C5  the converged mode's unknowns = the lit (control) set (ruled
%         2026-10-11): with an unlit border it converges to 1.0000 (it plateaued
%         at 0.962 with every actuator an unknown); off-lit commands are exactly 0
    properties (Constant)
        LIB = fullfile(fileparts(mfilename('fullpath')), '..', 'templates', '40_benches', 'dm_gauge_lib');
    end
    methods (TestClassSetup)
        function addlib(tc)
            addpath(tConvergedEst.LIB);
            tc.addTeardown(@() rmpath(tConvergedEst.LIB));
        end
    end
    methods (Static)
        function S = matrix_problem()
            % 14 x 14 actuators, pitch 4 px, Gaussian influence (1/e at 1.2 pitch), 64 px grid
            na = 14;  p = 4;  N = 64;  w = 1.2*p;
            [X, Y] = meshgrid(1:N);  c0 = (N + 1)/2 - (na - 1)*p/2;
            J = zeros(N*N, na*na);
            for i = 1:na
                for j = 1:na
                    J(:, (j-1)*na + i) = reshape(exp(-((X - (c0+(j-1)*p)).^2 + (Y - (c0+(i-1)*p)).^2)/w^2), [], 1);
                end
            end
            JtJ = J.'*J;  d = diag(JtJ);  lam = 1e-3*median(d);       % the record's rule: lambda_m x median column energy
            R = chol(JtJ + lam*eye(size(JtJ)));
            S.na = na;  S.N = N;  S.J = J;
            S.est = @(h) reshape(R \ (R.' \ (J.'*h(:))), na, na);
            S.fwd = @(a) reshape(J*a(:), N, N);
            S.lit = true(na);
        end
        function g = gain(a, ad)
            g = ad(:) \ a(:);
        end
        function f = floor_pf(a, lit)
            x = a(lit) - mean(a(lit));  f = 1.4826*median(abs(x - median(x)));
        end
    end
    methods (Test)
        function C1_C2_matrix_exact_reading(tc)
            S = tConvergedEst.matrix_problem();
            rng(3);  ar = randn(S.na);  a1 = zeros(S.na);  a1(7, 8) = 1;
            for A = {a1, ar}
                h  = S.fwd(A{1});
                g1 = tConvergedEst.gain(S.est(h), A{1});
                gc = tConvergedEst.gain(dmg_converge(S.est, S.fwd, h, S.lit, 0, 400), A{1});
                tc.verifyLessThan(g1, 0.999, sprintf('one-step gain %.6f should be BIASED', g1));
                tc.verifyEqual(gc, 1, 'AbsTol', 1e-4, sprintf('converged gain %.6f', gc));
            end
        end
        function C3_noise_trade(tc)
            S = tConvergedEst.matrix_problem();
            rng(5);  a0 = zeros(S.na);  sig = 1e-3;  nr = 20;
            f1 = zeros(nr,1);  fc = zeros(nr,1);  it0 = zeros(nr,1);  itf = zeros(nr,1);
            for r = 1:nr
                h = S.fwd(a0) + sig*randn(S.N);
                a1 = S.est(h);  f1(r) = tConvergedEst.floor_pf(a1, S.lit);
                [ac, i0] = dmg_converge(S.est, S.fwd, h, S.lit, 0, 50);  fc(r) = tConvergedEst.floor_pf(ac, S.lit);  it0(r) = i0.iters;
                [~, i1] = dmg_converge(S.est, S.fwd, h, S.lit, median(f1(1:r)), 50);  itf(r) = i1.iters;
            end
            ratio = median(fc)/median(f1);
            fprintf('C3: one-step floor %.4g, converged (50 steps) %.4g -> x%.3f; steps to the floor %g vs to 0: %g\n', ...
                median(f1), median(fc), ratio, median(itf), median(it0));
            tc.verifyTrue(isfinite(ratio) && ratio >= 1 && ratio <= 3, sprintf('noise trade x%.3f', ratio));
            tc.verifyLessThan(median(itf), median(it0));
        end
        function C4_dmg_act_fit_converge(tc)
            % a DETERMINED stencil problem (every actuator lit, measured on the lattice)
            na = 24;  xg = (1:na) - (na+1)/2;  [axg, ayg] = meshgrid(xg);
            [u, v] = meshgrid(-2:2);  stn = exp(-(u.^2 + v.^2)/1.0^2);
            lit = true(na);  rng(9);  A = randn(na);
            hd = conv2(A, stn, 'same');
            [~, a1] = evalc('dmg_act_fit(hd, xg, axg, ayg, stn, lit)');
            [~, a0] = evalc('dmg_act_fit(hd, xg, axg, ayg, stn, lit, [], struct())');
            [~, ac] = evalc('dmg_act_fit(hd, xg, axg, ayg, stn, lit, [], struct(''converge'', true, ''maxit'', 200))');
            g1 = A(lit) \ a1(lit);  gc = A(lit) \ ac(lit);
            tc.verifyEqual(a0, a1, 'default call unchanged');
            tc.verifyLessThan(g1, 0.999, sprintf('one-step %.6f', g1));
            tc.verifyEqual(gc, 1, 'AbsTol', 1e-4, sprintf('converged %.6f', gc));
        end
        function C5_stencil_unknowns_are_the_control_set(tc)
            % RULED (CC/Dave 2026-10-11): the converged mode's unknowns are the lit
            % (control) set.  With an unlit border and ALL actuators as unknowns the
            % converged gain plateaued at 0.9620 (w 1.0) -- the unobservable border
            % absorbing the data; restricted to lit it converges to 1.
            na = 24;  xg = (1:na) - (na+1)/2;  [axg, ayg] = meshgrid(xg);
            [u, v] = meshgrid(-2:2);  stn = exp(-(u.^2 + v.^2)/1.0^2);
            lit = true(na);  lit([1 end], :) = false;  lit(:, [1 end]) = false;
            rng(9);  A = zeros(na);  A(lit) = randn(nnz(lit), 1);
            hd = conv2(A, stn, 'same');
            [~, a1] = evalc('dmg_act_fit(hd, xg, axg, ayg, stn, lit)');
            [~, ac] = evalc('dmg_act_fit(hd, xg, axg, ayg, stn, lit, [], struct(''converge'', true, ''maxit'', 200))');
            g1 = A(lit) \ a1(lit);  gc = A(lit) \ ac(lit);
            fprintf('C5: stencil, unlit border: one-step %.4f, converged (unknowns = lit) %.6f; off-lit max %.2g\n', g1, gc, max(abs(ac(~lit))));
            tc.verifyLessThan(g1, 0.999);
            tc.verifyEqual(gc, 1, 'AbsTol', 1e-4);
            tc.verifyEqual(max(abs(ac(~lit))), 0, 'off-lit unknowns are not unknowns');
        end
    end
end
