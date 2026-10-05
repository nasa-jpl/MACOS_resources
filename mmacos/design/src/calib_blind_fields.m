function [blind, fp] = calib_blind_fields(tel, F, wfe_before, minpass)
%CALIB_BLIND_FIELDS  Which CALIB fields did the solve not see?  Gated on RAYS, not on the size of the seed's WFE.
%   [BLIND, FP] = calib_blind_fields(TEL, F, WFE_BEFORE, MINPASS) traces the built Telescope TEL at each row of F
%   ([thx thy] offsets about the bias, rad, in CALIB field order -- row 1 = [0 0], the nominal) and returns FP, the
%   fraction of the source grid that passes the whole train per field, and BLIND, the CALIB field indices that are
%   either marked failed by CALIB (WFE_BEFORE > 1e30, its 9.9999e36 sentinel) or pass fewer than MINPASS (default
%   0.9) of their rays at the seed.  A seed whose WFE is large while every ray passes is a bad SEED, not a blind
%   solve (dyson5 TMA stage B, 2026-10-04: a 1 mm WFE threshold misfired on such seeds); a field whose rays are
%   clipped (the pre-125ea9f asphere circle: 213/91/0 of 253 at 1.56/3.13/4.69 deg) is blind.  WFE_BEFORE may be
%   [] (rays only).  Gate: tCalibBlindFields.
%
%   See also TELESCOPE/OPTIMIZE.
    arguments
        tel
        F (:,2) double
        wfe_before (1,:) double = []
        minpass (1,1) double = 0.9
    end
    nE = numel(tel.spec.elt);  fp = nan(1, size(F, 1));
    for k = 1:size(F, 1)
        tel.trace_at_field(F(k, :));  s = macos.trace(nE);  ri = macos.get_ray_info(s.nRays);
        fp(k) = nnz(ri.ok_trace(:) & ri.ok_pass(:))/numel(ri.ok_trace);
    end
    tel.trace_at_field([]);
    failed = false(1, size(F, 1));
    if ~isempty(wfe_before), failed(1:numel(wfe_before)) = wfe_before(:)' > 1e30; end
    blind = find(failed | fp < minpass);
end
