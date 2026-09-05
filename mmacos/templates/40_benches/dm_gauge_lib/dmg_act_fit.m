function a = dmg_act_fit(hd, xg, axg, ayg, stn, lit, lam)
%DMG_ACT_FIT  Actuator commands by lattice deconvolution of the stencil.
%   Tikhonov-regularized: the unregularized inverse amplifies modes where
%   the kernel's transfer is small (S2 G4 measured 13 nm on a dense
%   random command -- the raw-pinv-diverges lesson).  LAM is relative to
%   the stencil peak; 0.05 default.  pcg 1e-12/400 (unified at the S3
%   flavor).  Extracted from zwfs_s3/tg96_s3 @ 10cf593.
if nargin < 7, lam = 0.05; end
hd(isnan(hd)) = 0;
m = interp2(xg, xg.', hd, axg, ayg, 'linear', 0);
m(~lit) = 0;
Wf = double(lit);
l2 = (lam*max(abs(stn(:))))^2;
Cop = @(x) reshape(Wf.*conv2(reshape(x, size(axg)), stn, 'same'), [], 1);
Ct  = @(x) reshape(conv2(Wf.*reshape(x, size(axg)), rot90(stn,2), 'same'), [], 1);
Aop = @(x) Ct(Cop(x)) + l2*x;
b = Ct(m(:));
x = pcg(Aop, b, 1e-12, 400);
a = reshape(x, size(axg));
end
