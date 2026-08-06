function M = ctb_mask_bandlimited(N, dx_f, lamD_m, epsilon, order, form)
%CTB_MASK_BANDLIMITED  Band-limited Lyot focal-plane AMPLITUDE mask.
%   M = CTB_MASK_BANDLIMITED(N, DX_F, LAMD_M, EPSILON, ORDER, FORM) returns
%   an N-by-N REAL amplitude transmission mask M-hat (0..1) for a
%   band-limited Lyot coronagraph, centred on the beam pixel floor(N/2)
%   (0-based) = 1-based N/2+1 -- the FFT DC pixel where MACOS's FarField/NF2
%   focus lands (the half-pixel centering rule, matching ctb_mask_disk).
%
%   The mask is applied to the COMPLEX FIELD (via macos.apodize, which
%   multiplies WFElt), so M is the AMPLITUDE transmission M-hat; the
%   intensity transmission is |M-hat|^2.
%
%   FORMULAE (verified verbatim from the ar5iv LaTeX of the papers):
%
%   ORDER = 4  -- Kuchner & Traub 2002, ApJ 570, 900 (arXiv astro-ph/0203455)
%      Eq. 7 (AMPLITUDE):   M-hat(X) = N4 * ( 1 - sinc(pi*eps*X) )
%      Eq. 8 (INTENSITY): |M-hat(X)|^2 = N4^2 * ( 1 - sinc(pi*eps*X) )^2
%      where sinc(z) = sin(z)/z (UNNORMALIZED; NOT MATLAB's sinc), X is the
%      focal separation in lambda/D, eps is the mask bandwidth, and N4 =
%      1/1.21723 normalises 0 <= M-hat <= 1.  This is a 4th-order mask
%      because (1 - sinc(z))^2 ~ z^4 near the origin.  IMPORTANT: the
%      canonical K&T mask is "1 - sinc" in AMPLITUDE; the loosely-quoted
%      "1 - sinc^2" is the INTENSITY |M-hat|^2 of THIS mask -- do not code
%      amplitude = 1 - sinc^2 (a different mask).  Half-power (|M-hat|^2 =
%      1/2) inner working angle ~ 1.448/eps (lambda/D).
%
%   ORDER = 8  -- Kuchner, Crepp & Ge 2005, ApJ 628, 466 (astro-ph/0411077),
%      Eq. 13 with the recommended (m=1, l=3) instance (less ringing than
%      the Eq. 12 sinc^n+cos form):
%        M-hat(X) = N8 * ( 2/3 - sinc(pi*eps*X/3)^3 + (1/3)*sinc(pi*eps*X) )
%      8th-order null; half-power IWA ~ 1.788/eps (lambda/D).  Needs no Lyot
%      apodisation (K&T's 4th-order does, for full performance).
%
%   FORM (2-D construction from the 1-D M-hat; KCG offer all three):
%     'separable' (DEFAULT)  M(x,y) = M-hat(X_x)*M-hat(X_y).  Rigorous
%                            band-limited null (the null theorem is
%                            per-Cartesian-axis); darkens the full 2-D field
%                            (darkest on the axes).  What K&T's TPF used.
%     'radial'               M(r) = M-hat(X_r).  Matches a circular pupil +
%                            annular dark zone, BUT a purely radial mask does
%                            NOT guarantee the exact Lyot null -- use for the
%                            annulus-shaped dark hole, expect a shallower floor.
%     'linear'               M(x,y) = M-hat(X_x).  Banded in x only (dark
%                            vertical strip); the cleanest single-axis null.
%
%   Args:
%     N        grid size.
%     dx_f     focal-plane pixel pitch (m) -- the deterministic Fraunhofer
%              pitch lambda*R/(N*dx_sphere), NOT dx_at at an NF2 plane.
%     lamD_m   lambda/D at the mask plane, in METRES (lambda*R/D_beam).
%     epsilon  mask bandwidth (= per-axis Lyot trim fraction; see the
%              companion ctb_bandlimited driver's Lyot rule).
%     order    4 (default) or 8.
%     form     'separable' (default) | 'radial' | 'linear'.
%
%   See also: ctb_bandlimited, ctb_mask_disk, macos.apodize.
    if nargin < 5 || isempty(order), order = 4;          end
    if nargin < 6 || isempty(form),  form  = 'separable'; end
    mustBeMember(order,[4 8]);
    mustBeMember(form,{'separable','radial','linear'});

    c  = floor(N/2);                                     % 0-based beam pixel
    ax = ((0:N-1) - c) * dx_f / lamD_m;                  % coord in lambda/D
    [Xx, Xy] = meshgrid(ax, ax);

    Nn = norm_(order, epsilon);                          % amplitude normaliser
    switch form
        case 'radial'
            Xr = hypot(Xx, Xy);
            M  = Nn * profile_(Xr, epsilon, order);
        case 'linear'
            M  = Nn * profile_(Xx, epsilon, order);
        otherwise % 'separable'
            M  = (Nn * profile_(Xx, epsilon, order)) .* ...
                 (Nn * profile_(Xy, epsilon, order));
    end
    M = max(min(M,1),0);                                 % clamp rounding
end

% ----------------------------------------------------------------------
function p = profile_(X, eps, order)
%PROFILE_  UN-normalised 1-D band-limited amplitude profile at X (lambda/D).
    z = pi * eps * X;
    if order == 4
        p = 1 - sinc1_(z);                               % K&T Eq. 7
    else % 8
        p = 2/3 - sinc1_(z/3).^3 + (1/3)*sinc1_(z);      % KCG Eq. 13 (m=1,l=3)
    end
end

function Nn = norm_(order, eps)
%NORM_  Amplitude normaliser so max(profile) = 1 (K&T's N ~ 1/1.21723 at
%   order 4).  Computed numerically over a dense X grid -> exact, general,
%   independent of eps (eps only rescales X).
    if eps <= 0, Nn = 1; return; end
    Xg = linspace(0, 40/eps, 200000);                    % covers the global max
    Nn = 1 / max(profile_(Xg, eps, order));
end

function s = sinc1_(z)
%SINC1_  Unnormalised sinc sin(z)/z with the removable singularity at 0.
    s = ones(size(z));
    nz = z ~= 0;
    s(nz) = sin(z(nz)) ./ z(nz);
end
