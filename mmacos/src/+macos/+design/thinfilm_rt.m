function out = thinfilm_rt(layers, n_inc, n_sub, aoi_deg, lambda)
%MACOS.DESIGN.THINFILM_RT  Amplitude/power r and t of a multilayer stack.
%
%   OUT = MACOS.DESIGN.THINFILM_RT(LAYERS, N_INC, N_SUB, AOI_DEG, LAMBDA)
%
%   Characteristic-matrix (Abeles) solution for a stratified stack, written
%   from Macleod, "Thin-Film Optical Filters", ch. 2 -- NOT transcribed from
%   elemsub.F.  An "analytic" copied out of the engine is circular in exactly
%   the coefficient it is supposed to check; that is how the 2022 r_p sign
%   defect survived every gate for four years
%   (REVIEW_POL_SP_SIGN_2026-07-27.md).
%
%   This is the GENERAL sibling of tools/pol_external_anchor/vh_thinfilm.m,
%   which is pinned to reflection from air.  Here the incident medium is
%   arbitrary (a cemented PBS cube is glass -> stack -> glass) and the
%   TRANSMITTED coefficients come back too.
%
%   INPUTS
%     LAYERS  : L-by-2, OUTERMOST layer first (the one the light meets).
%                 col 1 : complex index N = n - 1i*k, k >= 0 = loss
%                 col 2 : PHYSICAL thickness, same length units as LAMBDA
%               May be 0-by-2 (bare interface).
%     N_INC   : index of the incident medium (the medium the ray is in).
%     N_SUB   : index of the semi-infinite substrate behind the stack.
%     AOI_DEG : angle of incidence IN THE INCIDENT MEDIUM, degrees.
%     LAMBDA  : wavelength, same length units as the thicknesses.
%
%   OUTPUT struct OUT, s and p each:
%     .rs .rp   amplitude reflection, Macleod's FIXED-TRANSVERSE p-hat.
%     .ts .tp   Macleod's TANGENTIAL amplitude transmission, t = 2*eta0/(eta0*B+C).
%     .Rs .Rp   power reflectance |r|^2.
%     .Ts .Tp   power transmittance 4*eta0*Re(eta_sub)/|eta0*B+C|^2.
%     .eta0s .eta0p .etasubs .etasubp   tilted admittances (diagnostics).
%
%   THREE CONVENTION TRAPS, all paid for once already:
%
%   1. .ts/.tp are TANGENTIAL amplitudes.  For p the tangential component is
%      E*cos(theta), so Macleod's t_p exceeds the ordinary Fresnel t_p by
%      cos_sub/cos_inc (1.2472 at 45 deg into n=1.5) -- a discrepancy exactly
%      the size of a plausible radiometric error, so it reads as a failed gate
%      rather than a units slip (REVIEW_POL_RADIOMETRIC_2026-07-28).  .Ts/.Tp
%      are unaffected: COMPARE POWERS unless you know you want amplitudes.
%      The engine's TP/TS are power-amplitudes (|TP|^2 == T), so
%        |TP_engine| = sqrt(.Tp)   and   |TS_engine| = sqrt(.Ts).
%      When N_INC == N_SUB (a cemented cube) every one of these factors is
%      identically 1 and the three conventions coincide.
%
%   2. p-hat.  Macleod's is the FIXED TRANSVERSE frame, where r_p -> r_s as
%      theta -> 0; MACOS assembles reflection on prhat = shat x rhat, which
%      FOLLOWS the outgoing ray (perfect conductor: RS = -1, RP = +1 at normal
%      incidence).  The bridge is a sign on the reflected p amplitude.  MEASURE
%      it, never assume it: tPolExternal found the bridge to be ZERO on the
%      perfect-conductor idiom where the ray-following doctrine predicted pi.
%      Magnitudes (.Rs/.Rp) and the s/p power ratio are bridge-free.
%
%   3. Index sign N = n - 1i*k with k >= 0 = loss -- the same convention the
%      engine stores (DCMPLX(n,-kappa)), so nothing translates.
%
%   See also: macos.design.pbs_macneille, macos.design.Bench/add_pbs_pass.

    if isempty(layers), layers = zeros(0,2); end
    assert(size(layers,2) == 2, 'thinfilm_rt: LAYERS must be L-by-2 [N, thickness].');
    assert(lambda > 0, 'thinfilm_rt: LAMBDA must be positive.');

    % Snell invariant, carried complex so absorbing layers stay well defined.
    sin_i  = sind(aoi_deg);
    ninc_s = n_inc * sin_i;                       % N*sin(theta), conserved

    out = struct();
    for pol = {'s','p'}
        p = pol{1};
        cth_i   = sqrt(1 - (ninc_s/n_inc)^2);
        cth_sub = sqrt(1 - (ninc_s/n_sub)^2);
        eta0    = admit(n_inc, cth_i,   p);
        etasub  = admit(n_sub, cth_sub, p);

        B = 1;  C = etasub;                       % substrate boundary
        for j = size(layers,1):-1:1               % innermost -> outermost
            N   = layers(j,1);
            d   = layers(j,2);
            cth = sqrt(1 - (ninc_s/N)^2);
            eta = admit(N, cth, p);
            dl  = 2*pi*N*d*cth/lambda;
            Bn  = cos(dl)*B + 1i*sin(dl)/eta*C;
            Cn  = 1i*eta*sin(dl)*B + cos(dl)*C;
            B = Bn;  C = Cn;
        end

        Y = C/B;
        r = (eta0 - Y)/(eta0 + Y);
        t = 2*eta0/(eta0*B + C);
        R = abs(r)^2;
        T = 4*eta0*real(etasub)/abs(eta0*B + C)^2;

        out.(['r' p])    = r;      out.(['t' p])    = t;
        out.(['R' p])    = R;      out.(['T' p])    = T;
        out.(['eta0' p]) = eta0;   out.(['etasub' p]) = etasub;
    end
end

% -------------------------------------------------------------------------
function e = admit(N, cth, pol)
%ADMIT  Macleod's tilted optical admittance (free-space units).
    if strcmp(pol, 's'), e = N*cth; else, e = N/cth; end
end
