function S = dyson_scaling(n, fno, h_max, blur_max, opts)
%DYSON_SCALING  The block radius a concentric Dyson needs for a given field.
%   S = dyson_scaling(n, fno, h_max, blur_max) sweeps the block radius r of
%   the classical concentric Dyson (dyson_layout, at the Dyson condition)
%   and returns the smallest r whose exact-trace transverse rms blur at the
%   field corner h_max is below blur_max, in the length unit of h_max and
%   blur_max.  The pre-registered dyson5 question: can a SINGLE-BLOCK Dyson
%   meet F/1.8 at a 54 mm slit?  The answer is the scaling law
%
%       blur ~ h^4 / r^3     (fifth-order residual of the concentric form)
%
%   so r must grow as h^(4/3) for a fixed blur -- report this BEFORE adding
%   elements (an asphere on the block face, a meniscus, an air gap at the
%   slit: what EMIT/CWIS/Carbon-I carry).
%
%   Options: 'r_grid' (default h_max*[2 3 4 6 8 10 12 16]), 'nring' (4).
%   Returns S.r_grid, S.blur (per r), S.r_required (interpolated on the
%   log-log law between the bracketing grid points; NaN if the largest r
%   still fails), S.R_g_required, S.law_exponent (fitted r-exponent).
    arguments
        n (1,1) double
        fno (1,1) double
        h_max (1,1) double {mustBePositive}
        blur_max (1,1) double {mustBePositive}
        opts.r_grid (1,:) double = h_max*[2 3 4 6 8 10 12 16]
        opts.nring (1,1) double = 4
    end
    rg = sort(opts.r_grid);  blur = nan(size(rg));
    for k = 1:numel(rg)
        D = dyson_layout(rg(k), n, 'fno', fno, 'h', h_max, 'sweep', false, 'nring', opts.nring);
        blur(k) = D.blur_rms;
    end
    S.r_grid = rg;  S.blur = blur;  S.h_max = h_max;  S.blur_max = blur_max;
    okk = ~isnan(blur);
    if nnz(okk) >= 2
        p = polyfit(log(rg(okk)), log(blur(okk)), 1);
        S.law_exponent = p(1);                 % expect ~ -3
    else
        S.law_exponent = NaN;
    end
    S.r_required = NaN;
    i = find(okk & blur <= blur_max, 1, 'first');
    if ~isempty(i)
        if i == 1
            S.r_required = rg(1);
        else
            % log-log interpolation between the bracketing points
            lr = interp1(log(blur([i-1 i])), log(rg([i-1 i])), log(blur_max));
            S.r_required = exp(lr);
        end
    end
    S.R_g_required = n/(n-1)*S.r_required;
end
