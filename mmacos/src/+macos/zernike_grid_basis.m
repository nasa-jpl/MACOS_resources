function B = zernike_grid_basis(N, modes, ap_frac, convention)
%MACOS.ZERNIKE_GRID_BASIS  Engine-exact Zernike maps sampled on an N×N grid.
%   B = macos.zernike_grid_basis(N, MODES) returns an [N×N×K] array of K
%   Zernike-mode shapes sampled on a surface's N×N grid, using MACOS's OWN
%   convention so that a grid poke and the matching MonZern coefficient produce
%   the IDENTICAL sag -- left/right pairs overlay.
%
%   MODES are 1-based MACOS Zernike indices (the same numbers you write in
%   MonZernModes=), interpreted in the chosen CONVENTION.  In ANSI ordering,
%   index k <-> OSA single-index k-1:
%     4=astig45, 5=defocus, 6=astig0, 7=trefoil-y, 8=coma-y, 9=coma-x,
%     10=trefoil-x, 13=spherical, ...
%   The RMS normalisation factors (elt_mod.F NORM_RMS_PARAM_ANSI: sqrt(6),
%   sqrt(3), sqrt(6), sqrt(8), ...) are baked in, matching the NORMALISED
%   MonZernType (NormANSI / NormNoll / NormBornWolf for the respective
%   conventions).
%
%   B = macos.zernike_grid_basis(N, MODES, AP_FRAC, CONVENTION) selects the
%   Zernike ordering:
%     'ansi'      (default)  ANSI / OSA ordering        -> MonZernType=NormANSI
%     'noll'                 Noll 1976 ordering         -> MonZernType=NormNoll
%     'bornwolf'             Born & Wolf ordering       -> MonZernType=NormBornWolf
%   Each non-ANSI mode is realised by REMAPPING its index to the equivalent
%   ANSI index and evaluating the ANSI polynomial -- exactly what the engine
%   does (surfsub.F ZerntoMon6 / ZerntoMon2 are pure ordering permutations that
%   relay to ZerntoMon1), so the SAME analytic evaluator and the SAME ndgrid
%   orientation serve every convention.  Fringe, NormHex and NormAnnularNoll
%   are NOT yet available here (Fringe's CODE-V ordering is partial + carries a
%   special radial-12 mode; Hex/AnnularNoll use different radial polynomials) --
%   requesting them errors rather than returning a wrong basis.
%
%   ORIENTATION matches the engine's GridMat(i,j): the FIRST array index is +x
%   and the SECOND is +y (ndgrid layout, per surfsub.F NGSrf), so a map handed
%   to macos.elt_grid_add lands on the surface WITHOUT a transpose.  This is
%   load-bearing for the ODD (coma / trefoil) modes: a meshgrid array is x<->y
%   transposed relative to the analytic MonZern, which leaves the EVEN modes
%   (focus/astig) matching but flips coma-x <-> coma-y into orthogonality.
%   Because the non-ANSI conventions reuse this same evaluator, they inherit
%   the correct orientation too.
%
%   AP_FRAC confines each Zernike to the aperture: rho = 1 at R = AP_FRAC *
%   (grid half-width), 0 outside.  Set AP_FRAC = lMon / (((N-1)/2)*GridSrfdx)
%   to match a MonZern of normalisation radius lMon.
%
%   These are the default INFLUENCE basis for macos.dw_dgrid.
%
%   See also: macos.dw_dgrid, macos.elt_grid_add, macos.segment_grid_basis.
arguments
    N          (1,1) double {mustBeInteger, mustBePositive}
    modes      (1,:) double {mustBeInteger, mustBePositive} = [4 5 6 7 8 9]
    ap_frac    (1,1) double {mustBePositive} = 1.0
    convention (1,:) char {mustBeMember(convention, ...
                     {'ansi','noll','bornwolf'})} = 'ansi'
end
t = linspace(-1, 1, N);
[X, Y] = ndgrid(t, t);                 % X(i,j)=t(i) -> first index = +x ; second = +y
rho = sqrt(X.^2 + Y.^2) / ap_frac;     % rho = 1 at the aperture edge (R = ap_frac)
th  = atan2(Y, X);
inAp = rho <= 1;
B = zeros(N, N, numel(modes));
for k = 1:numel(modes)
    a = conv_to_ansi_(convention, modes(k));   % engine-exact ordering remap
    Z = ansi_zernike_eval(a, rho, th);
    Z(~inAp) = 0;
    B(:,:,k) = Z;
end
end

% ---------------------------------------------------------------------------
function a = conv_to_ansi_(convention, j)
%CONV_TO_ANSI_  Map a 1-based convention index to its ANSI index.
%   The engine's ZerntoMon6 (Noll) and ZerntoMon2 (Born&Wolf) fill the ANSI
%   coefficient slot A from convention slot PERM(A) then relay to ZerntoMon1,
%   so convention mode j has the ANSI shape at index A where PERM(A) = j.
%   PERM is transcribed VERBATIM from surfsub.F (the "Conv Idx" column).
if strcmpi(convention, 'ansi')
    a = j;
    return
end
switch lower(convention)
    case 'noll'      % surfsub.F ZerntoMon6, ANSI slot -> Noll idx
        perm = [ 1  3  2  5  4  6  9  7  8 10 15 13 11 12 14 21 19 17 16 18 ...
                20 27 25 23 22 24 26 28 35 33 31 29 30 32 34 36 45 43 41 39 ...
                37 38 40 42 44 55 53 51 49 47 46 48 50 52 54 65 63 61 59 57 ...
                56 58 60 62 64 66];
    case 'bornwolf'  % surfsub.F ZerntoMon2, ANSI slot -> B&W idx
        perm = [ 1  3  2  6  5  4 10  9  8  7 15 14 13 12 11 21 20 19 18 17 ...
                16 28 27 26 25 24 23 22 36 35 34 33 32 31 30 29 45 44 43 42 ...
                41 40 39 38 37 55 54 53 52 51 50 49 48 47 46 66 65 64 63 62 ...
                61 60 59 58 57 56];
end
a = find(perm == j, 1);
if isempty(a)
    error('macos:zernike_grid_basis:mode', ...
        ['%s Zernike index %d is out of the tabulated range (1..%d); ' ...
         'extend the permutation table transcribed from surfsub.F.'], ...
        convention, j, numel(perm));
end
end

% Mode evaluation lives in private/ansi_zernike_eval.m -- shared with
% macos.pol_zernike so an influence-basis mode index and an
% aberration-report mode index cannot drift apart.
