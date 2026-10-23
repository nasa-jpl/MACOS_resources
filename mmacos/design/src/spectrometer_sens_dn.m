function [dn, lam] = spectrometer_sens_dn(q, lam)
%SPECTROMETER_SENS_DN  The index change the lens_index row actually applies, over wavelength: the Sellmeier B
%   coefficients scaled by (1 + e) with e set so dn = q.amount at q.lambda0_m.  Its spread over the band is the
%   scaling's residual DISPERSION (a uniform dn would have none).  lam in m (default the 0.38-2.5 um band, 50 pts).
if nargin < 2, lam = linspace(0.38e-6, 2.5e-6, 50); end
B = q.sellmeier(1:3);  C = q.sellmeier(4:6);
n = @(L2, Bx) sqrt(1 + sum(Bx.*L2./(L2 - C)));
L0 = (q.lambda0_m*1e6)^2;  n20 = n(L0, B)^2;  e = 2*sqrt(n20)*q.amount/(n20 - 1);
dn = arrayfun(@(l) n((l*1e6)^2, B*(1 + e)) - n((l*1e6)^2, B), lam);
end
