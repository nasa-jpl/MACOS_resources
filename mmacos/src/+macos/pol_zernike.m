function pz = pol_zernike(pm, opts)
%MACOS.POL_ZERNIKE  Low-order Zernike expansion of polarization-aberration maps.
%   pz = macos.pol_zernike(PM) expands the diattenuation and retardance
%   maps in PM (the struct returned by macos.pol_maps) onto a Zernike
%   basis over the unvignetted pupil, giving the standard
%   polarization-aberration terms -- piston, tilt, defocus, astigmatism
%   and up -- in each Pauli component.
%
%   This is what makes a MACOS result directly comparable with the
%   published polarization-aberration literature, which is written in
%   aberration terms rather than as maps.  For an on-axis rotationally
%   symmetric two-mirror system the theory predicts a specific answer:
%   diattenuation and retardance vary as rho^2 in magnitude with a 2*theta
%   azimuthal dependence, so the expansion must be dominated by
%   ASTIGMATISM in the two linear Pauli components (mode 6 = astig0 in
%   component s1, mode 4 = astig45 in s2, equal magnitude), with no
%   circular (s3) content and no defocus.  "Polarization astigmatism" is
%   the standard name for exactly that.
%
%   The expansion is a least-squares fit, which is the correct estimator
%   whether or not the basis is orthogonal over the actual mask -- and on
%   an obscured (annular) pupil circular Zernikes are NOT orthogonal.  The
%   conditioning of that fit is reported (.cond); see 'orthonormalize'
%   below if you would rather have independent coefficients than standard
%   ones.
%
%   Name-value options:
%     'modes'      MACOS ANSI mode indices, 1-based, as in MonZernModes=.
%                  Default 1:15 (through secondary astigmatism and
%                  spherical) -- enough to separate the predicted
%                  astigmatism from its rho^4 companion.
%     'center'     [i j] pupil centre in grid indices.  Default: the
%                  centroid of the mask.
%     'radius'     normalization radius in pixels.  Default: the largest
%                  mask-point distance from the centre, so rho <= 1.
%     'orthonormalize'  Gram-Schmidt the basis over the ACTUAL mask before
%                  fitting (default false).  Makes coefficients mutually
%                  independent and the fit perfectly conditioned, at the
%                  cost that they are no longer standard Zernike
%                  coefficients and cannot be compared with a published
%                  table.  Use for energy bookkeeping, not for literature
%                  comparison.
%
%   Returns a struct:
%     .modes      1 x K mode indices
%     .names      1 x K cell, conventional names ('astig0', 'defocus', ...)
%     .nm         K x 2 radial order n and azimuthal order m per mode
%     .D          K x 3 coefficients of Dvec   (Pauli s1, s2, s3)
%     .ret        K x 3 coefficients of retvec (same ordering)
%     .Dmag       K x 1 coefficients of the diattenuation MAGNITUDE map
%     .retmag     K x 1 coefficients of the retardance magnitude map
%     .resid_rms  struct .D (1x3) .ret (1x3) .Dmag .retmag -- RMS of the
%                 fit residual over the mask, same units as the map
%     .frac       same shape -- fraction of each map's mean-square
%                 EXPLAINED by the fit (1 = perfect)
%     .cond       condition number of the design matrix over this mask
%     .recon      struct of reconstructed maps (.Dvec, .retvec, .D, .ret),
%                 NaN off-mask, for residual figures
%     .center .radius .npts .orthonormalized
%
%   Mode 1 (piston) is the pupil MEAN and every other mode is variation
%   about it -- the same separation macos.pol_maps reports as .mean and
%   .var_rms, and it must be kept: a uniform diattenuation or retardance
%   is a state change, not an aberration, and only the variation drives a
%   contrast floor or a phase-shifting-interferometry systematic.
%
%   The Pauli convention is inherited from macos.pol_maps: s1 = 0/90
%   linear, s2 = +/-45 linear, s3 = circular.  Because a physical rotation
%   by theta rotates (s1, s2) by 2*theta, an azimuthal cos(2*theta)
%   pattern in the MAP is what a physically radial/tangential
%   diattenuation axis looks like -- the doubling is in the
%   representation, not in the optics.
%
%   Retardance caveat: retvec is only meaningful where the branch is
%   unambiguous.  Points flagged PM.ambiguous (retardance within 0.2 rad
%   of pi) are EXCLUDED from the retardance fits and counted in
%   .npts_ret; if that count differs from .npts, treat the retardance
%   expansion as a fit over a punctured pupil.
%
%   Example -- the two-mirror literature form:
%       pm = macos.pol_maps(macos.jones_pupil(6));
%       pz = macos.pol_zernike(pm);
%       [pz.names(:), num2cell(pz.D)]     % astig dominates s1 and s2
%
%   See also: macos.pol_maps, macos.jones_pupil, macos.zernike_grid_basis.
arguments
    pm struct
    opts.modes  (1,:) double {mustBeInteger, mustBePositive} = 1:15
    opts.center (1,:) double = []
    opts.radius (1,:) double = []
    opts.orthonormalize (1,1) logical = false
end

mask = logical(pm.mask);
if ~any(mask(:))
    error('macos:pol_zernike:mask', 'pupil mask is empty');
end
N = size(mask, 1);
modes = opts.modes;
K = numel(modes);

% ---- pupil geometry ---------------------------------------------------
% Index convention matches the engine ray grid and macos.zernike_grid_basis:
% the FIRST array index is +x, the SECOND is +y.
[II, JJ] = ndgrid(1:N, 1:size(mask, 2));
if isempty(opts.center)
    ctr = [mean(II(mask)), mean(JJ(mask))];
else
    ctr = opts.center(:).';
end
dx = II - ctr(1);  dy = JJ - ctr(2);
R  = hypot(dx, dy);
if isempty(opts.radius)
    rad = max(R(mask));
else
    rad = opts.radius;
end
if rad <= 0
    error('macos:pol_zernike:radius', 'pupil radius must be positive');
end
rho = R / rad;
th  = atan2(dy, dx);

% ---- basis over the mask ---------------------------------------------
idx = find(mask);
A = zeros(numel(idx), K);
for k = 1:K
    Z = ansi_zernike_eval(modes(k), rho, th);
    A(:, k) = Z(idx);
end
pz.cond = cond(A);
if opts.orthonormalize
    [A, ~] = qr(A, 0);                 % orthonormal columns over the mask
    A = A * sqrt(size(A,1));           % unit RMS over the mask, not unit norm
end

% ---- fits -------------------------------------------------------------
retmask = mask;
if isfield(pm, 'ambiguous')
    amb = logical(pm.ambiguous);
    amb(isnan(amb)) = false;
    retmask = mask & ~amb;
end
ridx = find(retmask);
Ar = A;
if ~isequal(ridx, idx)
    Ar = zeros(numel(ridx), K);
    for k = 1:K
        Z = ansi_zernike_eval(modes(k), rho, th);
        Ar(:, k) = Z(ridx);
    end
    if opts.orthonormalize
        [Ar, ~] = qr(Ar, 0);  Ar = Ar * sqrt(size(Ar,1));
    end
end

pz.D      = zeros(K, 3);   pz.ret    = zeros(K, 3);
rD        = zeros(1, 3);   rR        = zeros(1, 3);
fD        = zeros(1, 3);   fR        = zeros(1, 3);
reconD    = nan(N, size(mask,2), 3);
reconR    = nan(N, size(mask,2), 3);
for c = 1:3
    [pz.D(:,c),   rD(c), fD(c), reconD(:,:,c)] = ...
        fit_(A,  idx,  pm.Dvec(:,:,c),   size(mask));
    [pz.ret(:,c), rR(c), fR(c), reconR(:,:,c)] = ...
        fit_(Ar, ridx, pm.retvec(:,:,c), size(mask));
end
[pz.Dmag,   rDm, fDm, reconDm] = fit_(A,  idx,  pm.D,   size(mask));
[pz.retmag, rRm, fRm, reconRm] = fit_(Ar, ridx, pm.ret, size(mask));

pz.modes = modes;
[pz.names, pz.nm] = mode_names_(modes);
pz.resid_rms = struct('D', rD, 'ret', rR, 'Dmag', rDm, 'retmag', rRm);
pz.frac      = struct('D', fD, 'ret', fR, 'Dmag', fDm, 'retmag', fRm);
pz.recon     = struct('Dvec', reconD, 'retvec', reconR, ...
                      'D', reconDm, 'ret', reconRm);
pz.center = ctr;  pz.radius = rad;
pz.npts = numel(idx);  pz.npts_ret = numel(ridx);
pz.orthonormalized = opts.orthonormalize;
pz.mask = mask;
end

% ---------------------------------------------------------------------------
function [c, resid_rms, frac, recon] = fit_(A, idx, Map, sz)
%FIT_  Least-squares expansion of one map over the masked points.
y = Map(idx);
ok = isfinite(y);
if ~all(ok)
    % NaNs inside the nominal mask would poison the whole fit; drop them
    % and keep going rather than returning a silently all-NaN answer.
    A = A(ok, :);  y = y(ok);  idx = idx(ok);
end
c = A \ y;
yhat = A * c;
r = y - yhat;
resid_rms = sqrt(mean(r.^2));
den = mean((y - mean(y)).^2);
if den > 0
    frac = 1 - mean((r - mean(r)).^2) / den;
else
    frac = 1;
end
recon = nan(sz);
recon(idx) = yhat;
end

% ---------------------------------------------------------------------------
function [names, nm] = mode_names_(modes)
%MODE_NAMES_  Conventional labels + (n,m) for MACOS ANSI 1-based indices.
lut = {'piston', 'tilt-y', 'tilt-x', 'astig45', 'defocus', 'astig0', ...
       'trefoil-y', 'coma-y', 'coma-x', 'trefoil-x', 'quadrafoil-y', ...
       'astig45-2', 'spherical', 'astig0-2', 'quadrafoil-x'};
names = cell(1, numel(modes));
nm    = zeros(numel(modes), 2);
for k = 1:numel(modes)
    j = modes(k);
    jj = j - 1;
    n = ceil((-3 + sqrt(9 + 8*jj)) / 2);
    m = 2*jj - n*(n + 2);
    nm(k, :) = [n, m];
    if j <= numel(lut)
        names{k} = lut{j};
    else
        names{k} = sprintf('n%d m%+d', n, m);
    end
end
end
