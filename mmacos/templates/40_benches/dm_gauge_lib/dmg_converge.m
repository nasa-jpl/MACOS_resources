function [a, info] = dmg_converge(est, fwd, m, lit, tol, maxit)
%DMG_CONVERGE  Drive a regularized estimator to CONVERGENCE on its own residual.
%   a = dmg_converge(est, fwd, m, lit, tol, maxit) is the estimator a servo
%   actually runs (gain 1): the one-step estimate a = est(m), then repeatedly
%   the residual r = m - fwd(a) re-read and added, a = a + est(r), until the
%   update's rms over the lit set is below TOL (the photon floor, in the units of
%   a) or MAXIT steps.  est is the regularized one-step estimator (Tikhonov:
%   (J'J + L)^-1 J'), fwd the forward model it inverts (the measured matrix J,
%   or a stencil convolution), m the measurement.
%
%   Why (Dave 2026-10-11, the COPHI gain probe): a one-step Tikhonov estimator
%   SHRINKS -- an exact reading read 0.995 through the record's matrix at
%   lambda_m 1e-3, and 1.002 at 1e-5.  Iterating on the residual (iterated
%   Tikhonov / preconditioned Landweber) converges to the least-squares solution
%   on the modes the data carry, so the bias goes while the regularization stays
%   in the step: the regularization is driven to convergence, not dropped
%   (dropping lambda trades the bias for noise amplification in ONE step).
%   The form is pupil_blur_demo's iterate_ (tg_psi_dm96_oap), shared here.
%
%   info.iters, info.upd (rms update per step).  Gate tConvergedEst.
if nargin < 5 || isempty(tol), tol = 0; end
if nargin < 6 || isempty(maxit), maxit = 50; end
a = est(m);
upd = nan(1, maxit);  upd(1) = rms_(a, lit);
k = 1;
for k = 2:maxit
    r  = m - fwd(a);
    da = est(r);
    a  = a + da;
    upd(k) = rms_(da, lit);
    if upd(k) <= tol, break; end
end
info = struct('iters', k, 'upd', upd(1:k));
end

function r = rms_(a, lit)
    if isempty(lit), x = a(:); else, x = a(lit); end
    r = sqrt(mean(x.^2));
end
