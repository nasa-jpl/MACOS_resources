function [ac, Geff, Gs] = dmg_color_comb(A, mode, pk, gk, beta, NACT, dc)
%DMG_COLOR_COMB  Multi-channel (multi-COLOR) Wiener combination on the
%   actuator lattice by FFT.
%   ac = dmg_color_comb(A, MODE, PK, GK, BETA, NACT) takes K raw
%   actuator-space estimates A{k} (one per color, each through its own
%   calibrated estimator) and the K per-color modal gains GK{k} measured
%   on the SAME probes PK (MODE 'radial': PK = fk per probe -- IFO;
%   'separable': PK = the (p,0) row indices -- ZWFS; exactly the
%   dmg_modal_corr conventions), builds each color's lattice transfer
%   G_k(f) by dmg_modal_corr's interpolation, and returns
%       a_hat(f) = sum_k G_k(f) A_k(f) / ( sum_k G_k(f)^2 + beta^2 )
%   -- the least-squares estimate of a from the K channels A_k = G_k a
%   (equal noise weights; the noiseless campaign has no per-color noise
%   to weight by), damped by BETA.  Where one color's transfer crosses
%   zero another's need not, so the denominator never collapses where
%   ANY color still carries the mode -- the "fill the nulls" mechanism.
%   With K = 1 in 'radial' mode this is dmg_modal_corr to round-off (the
%   caller asserts it).  In 'separable' mode the record's dmg_modal_corr
%   is the PER-AXIS product w1(fx) w1(fy); the joint form G/(G^2+beta^2)
%   with G = g1(fx) g1(fy) is used here for every K (a K=1 call is the
%   apples-to-apples single-color baseline for the combination).
%   Geff = sum_k G_k^2 / (sum_k G_k^2 + beta^2): the combination's own
%   transfer on the lattice; its minimum over the band is the
%   worst-region metric.  Gs = the per-color transfers.
%   DC: 'unit' (default) pins g(0) = 1 -- the separable model's own
%   normalization (the record's separability checks imply g1(0) ~ 0.99);
%   'record' extends g(0) as the lowest probe's gain, as dmg_modal_corr
%   does (use it to reproduce dmg_modal_corr with K = 1 in 'radial' mode).
if nargin < 7, dc = 'unit'; end
K = numel(A);
Gs = cell(1, K);
for k = 1:K
    Gs{k} = transfer_(mode, pk, gk{k}, NACT, dc);
end
den = beta^2;
for k = 1:K, den = den + Gs{k}.^2; end
num = 0;
for k = 1:K, num = num + Gs{k} .* fft2(A{k}); end
ac = real(ifft2(num ./ den));
Geff = (den - beta^2) ./ den;
end

function G = transfer_(mode, pk, gk, NACT, dc)
switch mode
    case 'radial'
        [uu, vv] = meshgrid(0:NACT-1);
        fu = min(uu, NACT-uu);  fv = min(vv, NACT-vv);
        fr = hypot(fu, fv)/2;
        [fks, si] = sort(pk(:));  gks = gk(si);
        g0 = gks(1);  if strcmp(dc, 'unit'), g0 = 1; end
        G = interp1([0; fks], [g0; gks], min(fr, max(fks)), 'linear');
    case 'separable'
        [pks, si] = sort(pk(:));  gks = gk(si);
        g0 = gks(1);  if strcmp(dc, 'unit'), g0 = 1; end
        fu = min(0:NACT-1, NACT-(0:NACT-1)).';
        g1 = interp1([0; pks], [g0; gks], min(fu, max(pks)), 'linear');
        G = g1 * g1.';
    otherwise
        error('dmg_color_comb: mode must be radial|separable');
end
end
