function [Phi, info] = ctb_apod_prolate(N, r_pup_px, r_occ_lamD, opts)
%CTB_APOD_PROLATE  Prolate-spheroidal apodizer for an APLC (Soummer 2005).
%   [PHI, INFO] = CTB_APOD_PROLATE(N, R_PUP_PX, R_OCC_LAMD) returns an
%   N-by-N REAL pupil-plane AMPLITUDE apodizer Phi (0..1) for an Apodized
%   Pupil Lyot Coronagraph, computed as the dominant prolate-spheroidal
%   eigenfunction of the APLC operator (Soummer 2005, ApJ 618, L161,
%   Eq. 3; Soummer, Aime & Falloon 2003, A&A 397, 1161).
%
%   THE APERTURE apodizer is the eigenfunction of the largest eigenvalue
%   Lambda0 of the operator (Soummer 2005 Eq. 3):
%       integral_P  Phi(t) * Mhat(r - t) dt  =  Lambda0 * Phi(r)
%   where P is the pupil support and Mhat is the Fourier transform of the
%   focal-plane occulter indicator M.  That operator is exactly:
%       [restrict to pupil P] o FT^-1 o [restrict to occulter M] o FT
%   so the dominant, node-less prolate is found by POWER ITERATION of that
%   operator (the "subtract the residual Lyot amplitude" loop of Guyon &
%   Roddier 2000, realised as a power iteration):
%       Phi <- P .* real( FT^-1( M .* FT( P .* Phi ) ) ),  renormalise.
%   The growth factor per apply converges to Lambda0 = the fraction of
%   energy the occulter passes (the encircled energy behind the mask),
%   Lambda0 in (0,1).
%
%   The internal FFT grid sets lambda/D = N / (2*R_pup_px) pixels, so the
%   occulter radius R_OCC_LAMD (in lambda/D, Soummer's alpha in units of
%   lambda/D) maps to R_OCC_LAMD * lambda/D pixels in the focal array.
%
%   Args:
%     N          apodizer array size (match the beam grid).
%     r_pup_px   pupil beam radius in pixels on that grid.
%     r_occ_lamD hard-occulter radius in lambda/D (Soummer 2011 GPI: 2.8;
%                occulter DIAMETER 5.6 lambda/D).
%   Name-value:
%     'n_iter'   max power iterations (default 200).
%     'tol'      convergence tol on max|dPhi| (default 1e-7).
%     'supersample' occulter edge supersample K (default 8).
%
%   INFO fields: .lambda0 (dominant eigenvalue = throughput-behind-mask),
%     .n_iter_used, .converged, .r_pup_px, .r_occ_px, .lamD_px.
%
%   See also: ctb_aplc, ctb_mask_disk, macos.apodize.
    arguments
        N          (1,1) double {mustBeInteger,mustBePositive}
        r_pup_px   (1,1) double {mustBePositive}
        r_occ_lamD (1,1) double {mustBePositive}
        opts.n_iter      (1,1) double = 200
        opts.tol         (1,1) double = 1e-7
        opts.supersample (1,1) double = 8
    end
    c = floor(N/2)+1;                                    % 1-based beam pixel
    [X,Y] = meshgrid((1:N)-c,(1:N)-c);
    rr = hypot(X,Y);

    % pupil support (supersampled soft edge for a clean operator)
    P = ctb_mask_disk(N, 1, r_pup_px, opts.supersample);  % dx=1 -> px units
    lamD_px = N / (2*r_pup_px);                            % FFT lambda/D
    r_occ_px = r_occ_lamD * lamD_px;
    M = ctb_mask_disk(N, 1, r_occ_px, opts.supersample);  % occulter support

    % power iteration of the APLC operator
    Phi = double(P > 0.5);                                % init: flat on pupil
    lam = NaN; converged = false; used = opts.n_iter;
    for it = 1:opts.n_iter
        PsiA   = P .* Phi;
        foc    = fftshift(fft2(ifftshift(PsiA)));
        masked = M .* foc;
        lyot   = fftshift(ifft2(ifftshift(masked)));
        Phi_new= P .* real(lyot);
        pk     = max(abs(Phi_new(:)));
        prev   = max(abs(Phi(:)));
        lam    = pk / max(prev,eps);                      % growth -> Lambda0
        Phi_new= Phi_new / max(pk,eps);                   % renormalise peak=1
        d      = max(abs(Phi_new(:) - Phi(:)));
        Phi    = Phi_new;
        if d < opts.tol, converged = true; used = it; break; end
    end
    Phi = max(min(real(Phi),1),0) .* (P > 0);             % clamp, confine to pupil

    info = struct('lambda0',lam,'n_iter_used',used,'converged',converged, ...
        'r_pup_px',r_pup_px,'r_occ_px',r_occ_px,'lamD_px',lamD_px);
end
