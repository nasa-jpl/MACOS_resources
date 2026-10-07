function X = tls_design(P, FO)
%TLS_DESIGN  The design vector of a long-slit TMA section, seeded from the first order.
%
%   X = TLS_DESIGN(P, FO) returns the struct TLS_SECTION builds a deck from.
%   Every quantity the figure stage varies lives here, in GENERATOR terms --
%   so a solve never edits deck text, the chief always runs through the three
%   poles, and the stop is always exactly at the M2 pole:
%     legs   [M1->M2 M2->M3 M3->slit] chief path lengths, m
%     aoi    chief-ray AOI at M1..M3, deg (the layout; P.turn the senses)
%     Rt, Rs local radii at the pole, fold plane / strip direction, m
%            (signed: + concave).  The section's parent conic follows:
%            R = sqrt(Rs^3/Rt), K = (R^2 - Rs^2)/h^2 at h = Rs sin(theta)
%     theta  angle between the pole normal and the parent axis, deg (the
%            off-axis-ness: theta = aoi makes K = -1, the exact first-order
%            paraboloid SEED -- the solve decides)
%     asph   3x2 h^4, h^6 departures as SAG in um at the lit radius hlit
%            (the AsphCoef scaling: AsphCoef(j) = asph(k,j)*1e-6/hlit^(2j+2))
%     hlit   per-mirror lit radius from the parent axis, m (fixed at the
%            seed: it only scales the asphere DOFs)
%     slit_dz focus: the slit moved along the chief, m (+ = away from M3)
%     mon    3 x nterm pole-frame freeform departures (um of sag at lmon),
%            terms mon_terms = [degree i, x-power j] -> x^j y^(i-j) in the
%            section frame normalized by lmon (Surface= Monomial, MonCoef):
%            degree 3..6, even j (mirror symmetry about the fold plane)
%     lmon   per-mirror normalization radius, m (the seed footprint half-size)
%     closure true: R_t, R_s are re-derived from the first order of (legs,
%            aoi) at every solve evaluation (TLS_FIGURE), never free
%   Seeded from FO (TLS_FIRST_ORDER) and P (aoi, legs, axis_theta_deg).
%
%   See also TLS_SECTION, TLS_FIGURE.
th = P.axis_theta_deg;  if isempty(th), th = P.aoi_deg; end
X = struct('legs', P.legs_m(:)', 'aoi', P.aoi_deg(:)', 'Rt', FO.Rt(:)', 'Rs', FO.Rs(:)', 'theta', th(:)', ...
           'asph', zeros(3, 2), 'hlit', nan(1, 3), 'slit_dz', 0, 'declare_asph', false, 'closure', true, ...
           'mon_terms', zeros(0, 2), 'mon', zeros(3, 0), 'lmon', nan(1, 3));
% the pole-frame freeform basis: x^j y^(i-j), degree i = 3..6, j EVEN (the section is mirror-symmetric about the fold
% plane, x -> -x); degree >= 3 adds nothing to value, slope or curvature at the pole, so the first-order closure holds
T = zeros(0, 2);
for i = 3:6, for j = 0:2:i, T(end+1, :) = [i j]; end, end   %#ok<AGROW>
X.mon_terms = T;  X.mon = zeros(3, size(T, 1));
% the lit radius from the parent axis: pole height h plus the footprint radius
for k = 1:3
    h = abs(X.Rs(k))*sind(X.theta(k));
    X.hlit(k) = h + max(FO.foot_x(k), FO.foot_y_surf(k))/2;
    X.lmon(k) = max(FO.foot_x(k), FO.foot_y_surf(k))/2;       % the freeform normalization: the footprint half-size
end
end
