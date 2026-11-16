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
%NORM_RMS_ANSI_  MACOS NORM_RMS_PARAM_ANSI, the RMS normalization factor.
%   Computed ANALYTICALLY (sqrt(n+1) for m=0, sqrt(2(n+1)) for m~=0) rather
%   than from a short table: this reproduces elt_mod.F's NORM_RMS_PARAM_ANSI
%   (lines 288-299) BIT-IDENTICALLY for modes 1..15 and extends it to the
%   higher radial orders the asphere->Zernike fold needs (e.g. mode 25 =
%   secondary spherical).  OSA single index jj = j-1 -> (n, m) as above.
jj = j - 1;
n  = ceil((-3 + sqrt(9 + 8*jj)) / 2);
m  = 2*jj - n*(n + 2);
if m == 0, v = sqrt(n + 1); else, v = sqrt(2*(n + 1)); end
end
