function [E, cond2, w] = ctb_probe_solve(dI, dE, opts)
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
%   Name-value (the regularized solve, e2e stage 11's follow-up -- MEASURED
%   on the saved cycle-4 state 2026-10-31: no gain, 17.3 % -> 16.2 % at reg
%   1e-3 and worse beyond; the error is in the probe FIELDS (the C0 grid-DM
%   steps, notes/surfaces.md), not in the solve.  Kept for the A/B.):
%     'reg'    Tikhonov weight, dimensionless (0 = the plain solve).  The
%              pixel's 2x2 normal matrix N gains reg * m on its diagonal,
%              m = the median over pixels of trace(N)/2 (the typical probe
%              energy), so a pixel whose probes carry energy e is shrunk
%              by about e / (e + reg m): the sinc probes' near-null pixels
%              (e << m), where the plain solve divides by almost nothing,
%              go toward zero and the bulk of the zone is untouched.
%     'energy_floor'  fraction f (0 = none): a pixel whose trace(N)/2 is
%              below f m gets E = 0 outright (the controller leaves it).
%   [E, COND2, W] = ... also returns W [npix x 1], each pixel's weight
%   trace(N)/2 / m (1 = typical; the diagnostic for both options).
%
%   See also: ctb_probe_estimate.
    arguments
        dI   double
        dE   double
        opts.reg          (1, 1) double {mustBeNonnegative} = 0
        opts.energy_floor (1, 1) double {mustBeNonnegative} = 0
    end
    assert(isequal(size(dI), size(dE)) && size(dI, 2) >= 2, ...
        'ctb_probe_solve: dI and dE must both be [npix x K], K >= 2');
    a = 4 * real(dE);                 % coefficient of Re E
    b = 4 * imag(dE);                 % coefficient of Im E
    % normal equations [saa sab; sab sbb] [x; y] = [sa; sb], per pixel
    saa = sum(a.^2, 2);  sbb = sum(b.^2, 2);  sab = sum(a.*b, 2);
    sa  = sum(a.*dI, 2); sb  = sum(b.*dI, 2);
    m = median((saa + sbb) / 2);
    w = (saa + sbb) / 2 / max(m, realmin);
    if opts.reg > 0
        saa = saa + opts.reg * m;  sbb = sbb + opts.reg * m;
    end
    det = saa .* sbb - sab.^2;
    x = ( sbb .* sa - sab .* sb) ./ det;
    y = (-sab .* sa + saa .* sb) ./ det;
    E = complex(x, y);
    if opts.energy_floor > 0, E(w < opts.energy_floor) = 0; end
    if nargout > 1
        tr = saa + sbb;
        disc = sqrt(max(tr.^2 / 4 - det, 0));
        lmax = tr/2 + disc;  lmin = tr/2 - disc;
        cond2 = lmax ./ max(lmin, realmin);
    end
end
