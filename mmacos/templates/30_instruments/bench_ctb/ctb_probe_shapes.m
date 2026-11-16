function S = ctb_probe_shapes(d, outer_lamD, K)
%CTB_PROBE_SHAPES  The default pairwise-probe shapes (ctb_probe_estimate), [nact^2 x K], peak 1.
%   S = CTB_PROBE_SHAPES(D, OUTER_LAMD, K) for the DM geometry D (a ctb_dm
%   object or a Jacobian's .dm(k) struct: acx, acy, beam_d_mm, active), the
%   dark zone's outer radius in lambda/D and K probes: sinc-sine probes after
%   Give'on, a narrow row in y, along x a sinc whose focal-plane lobe spans
%   (outer+2) lambda/D centred at +/-(outer+1)/2 lambda/D, phases
%   theta_k = (k-1) pi / K; even-numbered probes are the same shape turned
%   90 degrees (see ctb_probe_estimate 'probes').  Public so a diagnostic
%   (ctb_dm_linearity) can apply the estimator's own probe.

    % actuator centres in aperture units (D = the controlled beam)
    x = d.acx(:) / d.beam_d_mm;  y = d.acy(:) / d.beam_d_mm;
    xi0 = (outer_lamD + 1) / 2;               % lobe centre, lambda/D
    Wx  = outer_lamD + 2;                     % lobe width in x, lambda/D
    Wy  = 2 * (outer_lamD + 1);               % full annulus height
    S = zeros(numel(x), K);
    for k = 1:K
        th = (k - 1) * pi / K;
        if mod(k, 2) == 1      % lobes at +/- xi0 along x
            S(:, k) = sinc_(Wx * x) .* sinc_(Wy * y) .* cos(2*pi*xi0*x + th);
        else                   % the same probe turned 90 degrees
            S(:, k) = sinc_(Wx * y) .* sinc_(Wy * x) .* cos(2*pi*xi0*y + th);
        end
    end
    S(~d.active(:), :) = 0;
    S = S ./ max(abs(S), [], 1);
end

function s = sinc_(t)
    s = ones(size(t));
    nz = t ~= 0;
    s(nz) = sin(pi * t(nz)) ./ (pi * t(nz));
end

