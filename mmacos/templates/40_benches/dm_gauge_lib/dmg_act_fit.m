function [a, info] = dmg_act_fit(hd, xg, axg, ayg, stn, lit, lam, opts)
%DMG_ACT_FIT  Actuator commands by lattice deconvolution of the stencil.
%   Tikhonov-regularized: the unregularized inverse amplifies modes where
%   the kernel's transfer is small (S2 G4 measured 13 nm on a dense
%   random command -- the raw-pinv-diverges lesson).  LAM is relative to
%   the stencil peak; 0.05 default.  pcg 1e-12/400 (unified at the S3
%   flavor).  Extracted from zwfs_s3/tg96_s3 @ 10cf593.
%   opts.converge true (2026-10-11, Dave: the regularization driven to
%   CONVERGENCE, not dropped): the regularized step iterated on its own residual
%   through the stencil (dmg_converge; pupil_blur_demo's iterate_) until the
%   update's rms over lit <= opts.tol (default 0) or opts.maxit (default 50).
%   Absent / false = the one-step solve, the record's, unchanged.
if nargin < 7 || isempty(lam), lam = 0.05; end
if nargin < 8, opts = struct(); end
hd(isnan(hd)) = 0;
m = interp2(xg, xg.', hd, axg, ayg, 'linear', 0);
m(~lit) = 0;
Wf = double(lit);
l2 = (lam*max(abs(stn(:))))^2;
Cop = @(x) reshape(Wf.*conv2(reshape(x, size(axg)), stn, 'same'), [], 1);
Ct  = @(x) reshape(conv2(Wf.*reshape(x, size(axg)), rot90(stn,2), 'same'), [], 1);
Aop = @(x) Ct(Cop(x)) + l2*x;
solve = @(mm) pcg(Aop, Ct(mm(:)), 1e-12, 400);
info = struct('iters', 1);
if isfield(opts, 'converge') && opts.converge
    tol = 0;  if isfield(opts, 'tol'), tol = opts.tol; end
    mit = 50; if isfield(opts, 'maxit'), mit = opts.maxit; end
    est = @(mm) reshape(solve(mm), size(axg));
    fwd = @(aa) reshape(Cop(aa(:)), size(axg));
    [a, info] = dmg_converge(est, fwd, m, lit, tol, mit);
else
    a = reshape(solve(m), size(axg));
end
end
