function [E, cond2] = ctb_probe_solve(dI, dE)
%CTB_PROBE_SOLVE  Per-pixel pairwise-probe least squares for the field.
%   [E, COND2] = CTB_PROBE_SOLVE(DI, DE) solves, independently at every
%   dark-zone pixel, the K equations
%       DI(:,k) = I_k+ - I_k-  =  4 Re( conj(DE(:,k)) .* E )
%   for the two unknowns [Re E; Im E] (Give'on, Kern & Shaklan 2011).
%     DI   [npix x K] real   probe difference images (same units as |E|^2)
%     DE   [npix x K] complex  model probe fields (the Jacobian's G a_k)
%     E    [npix x 1] complex  the estimated field
%     COND2 [npix x 1]        condition number of each pixel's 2x2 normal
%                             matrix (large = the probes do not give that
%                             pixel two independent phases)
%   K >= 2 is required for a determined solve; each row is one pixel.
%
%   See also: ctb_probe_estimate.
    assert(isequal(size(dI), size(dE)) && size(dI, 2) >= 2, ...
        'ctb_probe_solve: dI and dE must both be [npix x K], K >= 2');
    a = 4 * real(dE);                 % coefficient of Re E
    b = 4 * imag(dE);                 % coefficient of Im E
    % normal equations [saa sab; sab sbb] [x; y] = [sa; sb], per pixel
    saa = sum(a.^2, 2);  sbb = sum(b.^2, 2);  sab = sum(a.*b, 2);
    sa  = sum(a.*dI, 2); sb  = sum(b.*dI, 2);
    det = saa .* sbb - sab.^2;
    x = ( sbb .* sa - sab .* sb) ./ det;
    y = (-sab .* sa + saa .* sb) ./ det;
    E = complex(x, y);
    if nargout > 1
        tr = saa + sbb;
        disc = sqrt(max(tr.^2 / 4 - det, 0));
        lmax = tr/2 + disc;  lmin = tr/2 - disc;
        cond2 = lmax ./ max(lmin, realmin);
    end
end
