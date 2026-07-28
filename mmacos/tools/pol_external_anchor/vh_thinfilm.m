function [rp, rs] = vh_thinfilm(layers, n_sub, cthi, lam)
%VH_THINFILM  Amplitude reflection coefficients of a multilayer stack.
%
%   [RP, RS] = VH_THINFILM(LAYERS, N_SUB, CTHI, LAM)
%
%   Independent implementation of the Macleod / Born & Wolf characteristic-
%   matrix (Abeles) formulation, written from Macleod ch. 2 and from the
%   published equations of
%
%     G. van Harten, F. Snik & C. U. Keller, "Polarization properties of
%     real aluminum mirrors I. Influence of the aluminum oxide layer",
%     PASP 121, 377 (2009); arXiv:0903.2740, Eqs (1)-(6).
%
%   *** DO NOT TRANSCRIBE THIS FROM elemsub.F. ***  An "analytic" reference
%   copied out of the engine is circular in exactly the coefficient it is
%   supposed to check -- that is how the 2022 r_p sign defect survived every
%   gate for four years (REVIEW_POL_SP_SIGN_2026-07-27.md).  Everything
%   below comes from the textbook/publication form.
%
%   INPUTS
%     LAYERS : L-by-2 array, OUTERMOST layer first.
%                col 1 : complex index N = n - 1i*k  (k >= 0 is loss)
%                col 2 : PHYSICAL thickness, same length units as LAM
%              May be empty (0-by-2) for a bare substrate.
%     N_SUB  : complex index of the semi-infinite substrate.
%     CTHI   : cos(theta) in the incident medium (air, n = 1).  May be a
%              vector -- one entry per ray -- and RP/RS come back the same
%              shape.
%     LAM    : wavelength, SAME length units as the thicknesses.
%
%   OUTPUT CONVENTIONS -- both stated, neither assumed:
%     * Index sign: N = n - 1i*k with k >= 0 = loss.  This is the
%       publication's own convention (their Sec. 3) AND the MACOS
%       convention (elemsub.F stores DCMPLX(n,-kappa)), so no index-sign
%       translation is needed anywhere in this comparison.  The paper notes
%       that the opposite sign made their fitted oxide come out at ~50 nm
%       instead of ~4 nm, so this is a load-bearing agreement, not a
%       cosmetic one.
%     * p-hat: FIXED transverse frame (Macleod).  In this convention
%       r_p -> r_s as theta -> 0, so the retardance eps_p - eps_s vanishes
%       at normal incidence and the reflection Mueller matrix reduces to
%       diag(1,1,-1,-1).  The MACOS engine uses the RAY-FOLLOWING p-hat
%       (perfect conductor: RS = -1, RP = +1 at normal incidence), so
%              Delta_engine = Delta_paper + pi.
%       Callers must apply that bridge explicitly.  tPolExternal MEASURES
%       it rather than trusting this comment.
%
%   Admittance assignment (Macleod "tilted optical admittance"):
%       eta_s = N cos(theta),   eta_p = N / cos(theta).
%   The paper prints these as its Eqs (5)-(6) in an order that PDF text
%   extraction scrambles; the assignment above is the one that makes the
%   [1,2] Mueller element POSITIVE for a metal (R_s > R_p), which is what
%   their Fig. 1a shows (its [1,2] axis runs 0.00 .. 0.15, and this form
%   gives 0.087 .. 0.115 at 70 deg over their wavelength range).
%   tPolExternal pins that sign so the choice cannot silently rot.

    cthi = cthi(:);
    rp = stack_r(layers, n_sub, cthi, lam, 'p');
    rs = stack_r(layers, n_sub, cthi, lam, 's');
end

% -------------------------------------------------------------------------
function r = stack_r(layers, n_sub, cthi, lam, pol)
    % Snell from the incident medium (air, n = 1), per ray.
    sin0sq = 1 - cthi.^2;

    eta0 = admit(ones(size(cthi)), cthi, pol);

    cth_sub = sqrt(1 - sin0sq ./ n_sub.^2);
    B = ones(size(cthi));
    C = admit(n_sub * ones(size(cthi)), cth_sub, pol);

    % innermost -> outermost
    for j = size(layers, 1):-1:1
        N = layers(j, 1);
        d = layers(j, 2);
        cth   = sqrt(1 - sin0sq ./ N.^2);
        eta   = admit(N * ones(size(cthi)), cth, pol);
        delta = 2 * pi * N .* d .* cth ./ lam;
        cd = cos(delta);
        sd = sin(delta);
        Bn = cd .* B + 1i * sd ./ eta .* C;
        Cn = 1i * eta .* sd .* B + cd .* C;
        B = Bn;
        C = Cn;
    end

    Y = C ./ B;
    r = (eta0 - Y) ./ (eta0 + Y);
end

% -------------------------------------------------------------------------
function e = admit(N, cth, pol)
    if strcmp(pol, 's')
        e = N .* cth;
    else
        e = N ./ cth;
    end
end
