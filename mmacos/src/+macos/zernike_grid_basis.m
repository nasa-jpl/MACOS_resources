function B = zernike_grid_basis(N, modes, ap_frac, convention)
%MACOS.ZERNIKE_GRID_BASIS  Engine-exact Zernike maps sampled on an N×N grid.
%   B = macos.zernike_grid_basis(N, MODES) returns an [N×N×K] array of K
%   Zernike-mode shapes sampled on a surface's N×N grid, using MACOS's OWN
%   convention (ZerntoMon1, ANSI ordering) so that a grid poke and the matching
%   MonZern coefficient produce the IDENTICAL sag -- left/right pairs overlay.
%
%   MODES are 1-based MACOS Zernike indices (the same numbers you write in
%   MonZernModes=).  In ANSI ordering, index k <-> OSA single-index k-1:
%     4=astig45, 5=defocus, 6=astig0, 7=trefoil-y, 8=coma-y, 9=coma-x,
%     10=trefoil-x, 13=spherical, ...
%   The RMS normalisation factors (elt_mod.F NORM_RMS_PARAM_ANSI: sqrt(6),
%   sqrt(3), sqrt(6), sqrt(8), ...) are baked in, matching MonZernType=NormANSI.
%
%   ORIENTATION matches the engine's GridMat(i,j): the FIRST array index is +x
%   and the SECOND is +y (ndgrid layout, per surfsub.F NGSrf), so a map handed
%   to macos.elt_grid_add lands on the surface WITHOUT a transpose.  This is
%   load-bearing for the ODD (coma / trefoil) modes: a meshgrid array is x<->y
%   transposed relative to the analytic MonZern, which leaves the EVEN modes
%   (focus/astig) matching but flips coma-x <-> coma-y into orthogonality.
%
%   B = macos.zernike_grid_basis(N, MODES, AP_FRAC) confines each Zernike to the
%   aperture: rho = 1 at R = AP_FRAC * (grid half-width), 0 outside.  Set
%   AP_FRAC = lMon / (((N-1)/2)*GridSrfdx) to match a MonZern of normalisation
%   radius lMon -- the grid-data and MonZern surfaces then carry the same sag.
%
%   These are the default INFLUENCE basis for macos.dw_dgrid.
%
%   See also: macos.dw_dgrid, macos.elt_grid_add.
arguments
    N          (1,1) double {mustBeInteger, mustBePositive}
    modes      (1,:) double {mustBeInteger, mustBePositive} = [4 5 6 7 8 9]
    ap_frac    (1,1) double {mustBePositive} = 1.0
    convention (1,:) char {mustBeMember(convention,{'ansi'})} = 'ansi'
end
t = linspace(-1, 1, N);
[X, Y] = ndgrid(t, t);                 % X(i,j)=t(i) -> first index = +x ; second = +y
rho = sqrt(X.^2 + Y.^2) / ap_frac;     % rho = 1 at the aperture edge (R = ap_frac)
th  = atan2(Y, X);
inAp = rho <= 1;
B = zeros(N, N, numel(modes));
for k = 1:numel(modes)
    Z = ansi_zernike_(modes(k), rho, th);
    Z(~inAp) = 0;
    B(:,:,k) = Z;
end
end

% ---------------------------------------------------------------------------
function Z = ansi_zernike_(j, rho, th)
%ANSI_ZERNIKE_  MACOS ANSI Zernike (ZerntoMon1 convention), index j (1-based).
%   zc(j) <-> OSA single index jj = j-1; m<0 -> sin(|m|th), m>=0 -> cos(m th);
%   RMS-normalised (NORM_RMS_PARAM_ANSI) to match MonZernType=NormANSI.
jj = j - 1;
n  = ceil((-3 + sqrt(9 + 8*jj)) / 2);
m  = 2*jj - n*(n + 2);
am = abs(m);
R  = zeros(size(rho));
for s = 0:((n - am)/2)
    c = (-1)^s * factorial(n - s) / ...
        (factorial(s) * factorial((n + am)/2 - s) * factorial((n - am)/2 - s));
    R = R + c * rho.^(n - 2*s);
end
if m >= 0, ang = cos(m*th); else, ang = sin(am*th); end
Z = norm_rms_ansi_(j) .* R .* ang;
end

% ---------------------------------------------------------------------------
function v = norm_rms_ansi_(j)
%NORM_RMS_ANSI_  MACOS NORM_RMS_PARAM_ANSI(1:15) (elt_mod.F lines 288-299).
P = [1, 2, 2, sqrt(6), sqrt(3), sqrt(6), sqrt(8), sqrt(8), sqrt(8), sqrt(8), ...
     sqrt(10), sqrt(10), sqrt(5), sqrt(10), sqrt(10)];
if j > numel(P)
    error('macos:zernike_grid_basis:mode', ...
        'NORM_RMS_PARAM_ANSI tabulated to mode 15 here (got %d); extend P.', j);
end
v = P(j);
end
