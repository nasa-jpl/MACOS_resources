function ac = dmg_modal_corr(a, mode, pk, gk, beta, NACT)
%DMG_MODAL_CORR  Wiener modal correction on the actuator lattice by FFT.
%   ac = dmg_modal_corr(A, 'radial',    fk,  gk, BETA, NACT)  -- IFO: the
%       transfer is near-isotropic (measured (48,0) 0.900 vs (32,32)
%       0.893); fk = hypot(p,q)/2 per probe, gk the measured gains.
%   ac = dmg_modal_corr(A, 'separable', pk1, gk1, BETA, NACT) -- ZWFS:
%       the kernel is separable (validated ~2%); pk1/gk1 from the (p,0)
%       probes, G(fx,fy) ~ g1(fx) g1(fy).  NOTE the ZWFS fine-scale
%       transfer is OSCILLATORY (sign lobes) -- interpolated corrections
%       destabilize there; dense per-frequency calibration is the
%       recorded open path (S3).
%   W = G/(G^2+beta^2).  Extracted verbatim from tg96_s3/zwfs_s3 @ 10cf593.
switch mode
    case 'radial'
        [uu, vv] = meshgrid(0:NACT-1);
        fu = min(uu, NACT-uu);  fv = min(vv, NACT-vv);
        fr = hypot(fu, fv)/2;
        [fks, si] = sort(pk(:));  gks = gk(si);
        Gf = interp1([0; fks], [gks(1); gks], min(fr, max(fks)), 'linear');
        W = Gf ./ (Gf.^2 + beta^2);
    case 'separable'
        [pks, si] = sort(pk(:));  gks = gk(si);
        fu = min(0:NACT-1, NACT-(0:NACT-1)).';
        g1 = interp1([0; pks], [gks(1); gks], min(fu, max(pks)), 'linear');
        w1 = g1 ./ (g1.^2 + beta^2);
        W = w1 * w1.';
    otherwise
        error('dmg_modal_corr: mode must be radial|separable');
end
ac = real(ifft2(fft2(a) .* W));
end
