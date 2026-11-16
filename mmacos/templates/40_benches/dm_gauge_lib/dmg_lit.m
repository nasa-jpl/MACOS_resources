function [lit, R_ILL] = dmg_lit(msk, dxd_mm, mag, axg, ayg)
%DMG_LIT  Illuminated-actuator mask from the measured detector support.
%   The source cone fills ~74% of the aperture on these benches: count
%   ILLUMINATED actuators, never the full lattice (the illuminated-fill
%   doctrine).  R_ILL = 98th-percentile support radius mapped to DM mm;
%   lit = actuators inside 0.85 R_ILL.  Extracted verbatim from
%   zwfs_s3/tg96_s3 @ 10cf593.
[syy, sxx] = find(msk);
rpx = hypot(sxx-mean(sxx), syy-mean(syy));
rs = sort(rpx);  R_ILL = rs(round(0.98*numel(rs))) * dxd_mm * mag;
lit = hypot(axg, ayg) < 0.85*R_ILL;
end
