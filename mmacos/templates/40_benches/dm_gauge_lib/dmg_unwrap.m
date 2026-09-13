function [phi, info] = dmg_unwrap(psi, mask, opt)
%DMG_UNWRAP  Two-dimensional least-squares phase unwrapping on a mask.
%   [phi, info] = dmg_unwrap(psi, mask) unwraps the WRAPPED phase map psi
%   (rad, any size) over the true pixels of logical mask, and returns the
%   continuous phase phi (zero off the mask) and a record.
%
%   WHY (Dave / CCL 2026-09-13, after the descent finding): capture is a
%   WRAP problem.  Every phase reading in this campaign returns a WRAPPED
%   differential -- the stepped Zernike reading, the vector pair, both
%   point-diffraction forms, and the interferometer's four-step alike --
%   so a change larger than +-pi of phase (+-158 nm of surface at 632.8
%   nm, double pass) comes back folded whatever the reading's ABSOLUTE
%   range is.  Unwrapping the differential before the estimator moves the
%   limit from the wrap to the PIXEL GRADIENT: adjacent pixels must differ
%   by less than pi, and the DM's surface is smooth at the pixel scale
%   (4 detector px per actuator at 385 rays), so the limit becomes several
%   hundred nm rms instead of 158.
%
%   METHOD.  Ghiglia & Romero, JOSA A 11, 107 (1994): the unwrapped phase
%   is the least-squares solution of a Poisson equation whose source is
%   the divergence of the WRAPPED phase gradients.  Two stages:
%     1. UNWEIGHTED solve, exact and direct.  The Poisson equation with
%        Neumann boundaries is solved by mirroring the source into a 2M x
%        2N even-symmetric array and dividing its FFT by the discrete
%        Laplacian's eigenvalues -- the FFT form of Ghiglia & Romero's DCT
%        solution, chosen because `dct2` is a toolbox function and this
%        tree must run with no external dependency (the release gate).
%     2. MASKED refinement, when a mask excludes anything.  The weighted
%        normal equations Q phi = rho (weights = the mask, so no phase
%        information crosses the boundary) are solved by preconditioned
%        conjugate gradients with stage 1 as the preconditioner -- their
%        section 5.  Without it the unweighted solution lets the region
%        outside the mask, where there is no data, pull on the answer
%        inside it.
%   The work is done on the mask's bounding box, not the whole detector
%   grid: the pupil is ~NGRID px across a MODEL-px frame, so this is a
%   ~200 x 200 solve, not a 1024 x 1024 one.
%
%   RESIDUES.  A wrapped field is consistent only if every 2x2 loop of
%   wrapped differences sums to zero; where it does not, no unwrapper can
%   be right, and least-squares spreads the error rather than failing.
%   info.nres counts those loops inside the mask, so a map that is beyond
%   the pixel-gradient limit is REPORTED, not silently wrong.
%
%   phi is defined up to an additive constant (as every reading here
%   already is -- the estimator is mean-referenced).  The constant is set
%   so that mean(phi - psi) over the mask is the nearest multiple of 2 pi,
%   which makes phi == psi exactly wherever nothing was wrapped.
%
%   opt (optional):
%     .pcg    run the masked refinement          [true when the mask excludes]
%     .tol    PCG relative tolerance                                 [1e-14]
%     .maxit  PCG iterations                                           [400]
%   info: .nres residues inside the mask; .res the residue map (cropped);
%     .maxgrad largest wrapped gradient inside the mask (rad per pixel);
%     .iters, .relres from the refinement (0, 0 when it did not run);
%     .box the bounding box [r0 r1 c0 c1]; .wrapped true when phi differs
%     from psi by more than 1e-9 anywhere on the mask (i.e. it did work).
[M0, N0] = size(psi);
if nargin < 2 || isempty(mask), mask = true(M0, N0); end
if nargin < 3, opt = struct(); end
o = struct('pcg', [], 'tol', 1e-14, 'maxit', 400);
fn = fieldnames(opt);  for i = 1:numel(fn), o.(fn{i}) = opt.(fn{i}); end
mask = logical(mask);
assert(isequal(size(mask), [M0 N0]), 'dmg_unwrap: mask must match psi');
phi = zeros(M0, N0);
info = struct('nres', 0, 'res', [], 'maxgrad', 0, 'iters', 0, 'relres', 0, ...
              'box', [1 M0 1 N0], 'wrapped', false);
if ~any(mask(:)), return; end

% ---- crop to the mask's bounding box (one pixel of margin) --------------
[rr, cc] = find(mask);
r0 = max(1, min(rr)-1);  r1 = min(M0, max(rr)+1);
c0 = max(1, min(cc)-1);  c1 = min(N0, max(cc)+1);
info.box = [r0 r1 c0 c1];
p = psi(r0:r1, c0:c1);  m = mask(r0:r1, c0:c1);
[M, N] = size(p);
if M < 3 || N < 3, phi(mask) = psi(mask); return; end

% ---- wrapped gradients, gated on the mask ------------------------------
% dx(i,j) = wrapped( p(i,j+1) - p(i,j) ), valid only when BOTH ends are in
% the mask; elsewhere zero, so no phase crosses the boundary.
W = @(x) atan2(sin(x), cos(x));
dx = zeros(M, N);  wx = false(M, N);
dx(:, 1:N-1) = W(p(:, 2:N) - p(:, 1:N-1));
wx(:, 1:N-1) = m(:, 2:N) & m(:, 1:N-1);
dx(~wx) = 0;
dy = zeros(M, N);  wy = false(M, N);
dy(1:M-1, :) = W(p(2:M, :) - p(1:M-1, :));
wy(1:M-1, :) = m(2:M, :) & m(1:M-1, :);
dy(~wy) = 0;
info.maxgrad = max([0; abs(dx(wx)); abs(dy(wy))]);

% ---- residues: the 2x2 loops that cannot be made consistent ------------
if M > 1 && N > 1
    r = (dx(1:M-1, 1:N-1) + dy(1:M-1, 2:N) - dx(2:M, 1:N-1) - dy(1:M-1, 1:N-1)) / (2*pi);
    cell_in = m(1:M-1,1:N-1) & m(1:M-1,2:N) & m(2:M,1:N-1) & m(2:M,2:N);
    r = round(r) .* cell_in;
    info.res = r;  info.nres = nnz(r);
end

% ---- stage 1: the unweighted least-squares solve ------------------------
rho = divg_(dx, dy, true(M,N), true(M,N));
u = poisson_(rho);
% ---- stage 2: the masked refinement ------------------------------------
if isempty(o.pcg), o.pcg = ~all(m(:)); end
if o.pcg
    b = divg_(dx, dy, wx, wy);
    [u, info.iters, info.relres] = pcg_(b, wx, wy, u, o.tol, o.maxit);
end

% ---- place, and set the constant ---------------------------------------
q = zeros(M0, N0);  q(r0:r1, c0:c1) = u;
d = mean(q(mask) - psi(mask));
q = q - 2*pi*round(d/(2*pi));
phi(mask) = q(mask);
info.wrapped = max(abs(phi(mask) - psi(mask))) > 1e-9;
end

% =====================================================================
function rho = divg_(dx, dy, wx, wy)
% the (weighted) divergence of the gradient field: the Poisson source.
% rho(i,j) = wx(i,j) dx(i,j) - wx(i,j-1) dx(i,j-1) + the same in y
[M, N] = size(dx);
ax = dx .* wx;  ay = dy .* wy;
rho = ax - [zeros(M,1), ax(:, 1:N-1)] + ay - [zeros(1,N); ay(1:M-1, :)];
end

function y = lap_(phi, wx, wy)
% the weighted discrete Laplacian Q phi, the adjoint of divg_'s gradient
[M, N] = size(phi);
gx = zeros(M, N);  gx(:, 1:N-1) = phi(:, 2:N) - phi(:, 1:N-1);
gy = zeros(M, N);  gy(1:M-1, :) = phi(2:M, :) - phi(1:M-1, :);
y = divg_(gx, gy, wx, wy);
end

function phi = poisson_(rho)
% solve lap(phi) = rho with Neumann boundaries, by mirroring the source
% into an even-symmetric 2M x 2N array (which imposes those boundaries
% exactly) and dividing by the discrete Laplacian's Fourier eigenvalues.
[M, N] = size(rho);
R = [rho, rho(:, end:-1:1); rho(end:-1:1, :), rho(end:-1:1, end:-1:1)];
[jj, ii] = meshgrid(0:2*N-1, 0:2*M-1);
den = 2*cos(pi*ii/M) + 2*cos(pi*jj/N) - 4;
den(1,1) = 1;                                   % the constant mode carries no information
F = fft2(R) ./ den;  F(1,1) = 0;
P = real(ifft2(F));
phi = P(1:M, 1:N);
end

function [x, k, relres] = pcg_(b, wx, wy, x, tol, maxit)
% preconditioned conjugate gradients on Q x = b, Q the weighted Laplacian
% and the UNWEIGHTED Poisson solve as the preconditioner (Ghiglia &
% Romero section 5).  Q is symmetric negative semi-definite; the constant
% mode is in its null space and the preconditioner already removes it.
r = b - lap_(x, wx, wy);
nb = norm(b(:));  if nb == 0, nb = 1; end
relres = norm(r(:)) / nb;  k = 0;  p = [];  rz_old = 0;
while relres > tol && k < maxit
    z = poisson_(r);
    rz = sum(sum(r .* z));
    if k == 0, p = z; else, p = z + (rz/rz_old)*p; end
    Qp = lap_(p, wx, wy);
    den = sum(sum(p .* Qp));
    if den == 0, break; end
    a = rz / den;
    x = x + a*p;  r = r - a*Qp;
    rz_old = rz;  relres = norm(r(:)) / nb;  k = k + 1;
end
end
