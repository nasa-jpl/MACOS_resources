function Z = ansi_zernike_eval(j, rho, th)
%ANSI_ZERNIKE_EVAL  MACOS ANSI Zernike (ZerntoMon1 convention), 1-based index.
%   Z = ansi_zernike_eval(J, RHO, TH) evaluates MACOS Zernike mode J -- the
%   same number you write in MonZernModes= -- on the normalized polar
%   coordinates RHO, TH (any matching array shapes).  RHO = 1 is the
%   normalization radius; the caller decides where that is and masks
%   outside it if wanted (nothing is clipped here).
%
%   Convention: zc(j) <-> OSA single index jj = j-1; m < 0 -> sin(|m|*TH),
%   m >= 0 -> cos(m*TH); RMS-normalized by NORM_RMS_PARAM_ANSI
%   (elt_mod.F:288-299), matching MonZernType=NormANSI.  So mode 1 = piston,
%   2 = tilt-y, 3 = tilt-x, 4 = astig45, 5 = defocus, 6 = astig0,
%   7 = trefoil-y, 8 = coma-y, 9 = coma-x, 10 = trefoil-x, 13 = spherical.
%
%   Shared by macos.zernike_grid_basis (grid-poke influence bases, where
%   the normalization radius is a fraction of the grid half-width) and
%   macos.pol_zernike (polarization-aberration expansion, where it is the
%   pupil radius inferred from the vignetting mask).  Kept in one place
%   because the two must agree exactly -- a mode index that means
%   different things in an influence basis and in an aberration report is
%   a silent cross-language trap.
%
%   See also: macos.zernike_grid_basis, macos.pol_zernike.
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
    error('macos:zernike:mode', ...
        'NORM_RMS_PARAM_ANSI tabulated to mode 15 here (got %d); extend P.', j);
end
v = P(j);
end
