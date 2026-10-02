function S = telescope_seed(f, D, L_xp, t1, y2, opts)
%TELESCOPE_SEED  First-order PNP three-mirror layout + Seidel conics, closed form.
%   S = telescope_seed(f, D, L_xp, t1, y2) lays out a coaxial positive-
%   negative-positive three-mirror telescope (concave M1, convex M2, concave
%   M3 -- the Cook / Korsch anastigmat family of Mouroulis & Green 2018
%   Sec. 5.2) from the two knobs a designer chooses, the M1->M2 spacing t1
%   and the beam COMPRESSION at M2, y2 = (beam height at M2)/(beam height at
%   M1) = 1 - t1/f1, and three first-order conditions:
%     (1) effective focal length f (the marginal ray from infinity);
%     (2) a FLAT field: Petzval sum phi1 + phi2 + phi3 = 0 (mirror powers
%         phi = 2/R, concave positive, convex negative);
%     (3) the EXIT PUPIL at L_xp beyond the image along the beam, the stop
%         being M2's vertex -- so the telescope images its stop onto the
%         spectrometer's stop (the Dyson's grating, which the slit sees
%         through the block and the meniscus at L_xp; R4 is telecentric to
%         0.09 deg, L_xp = 16.8 m).
%   In the telecentric limit (3) pins phi3 = 1/t2 and (1) then reads
%   t2 = f * y2 whatever the front pair does (the EFL and the pupil condition
%   collapse into one), so the family is one-dimensional in y2 at a given
%   t1: that degeneracy is why a general three-equation Newton solve has no
%   basin here.  The finite L_xp is carried as a small correction.
%   Unfolded paraxial model (mirrors as thin lenses, spacings positive).
%   Conics: macos.design.seidel_seed (nulls the coaxial parent's 3rd-order
%   spherical, coma and astigmatism) -- the exact-chain ladder refines.
%
%   Returns S: .R [R1 R2 R3] (|radii|), .phi, .t [t1 t2 t3], .t3 (M3 ->
%   image), .y [y1 y2 y3] (marginal heights), .K (Seidel conics), .EFL_check
%   / .t3_check (seidel_seed's paraxial numbers), .z_int (real intermediate
%   image after M2 along the beam, NaN if the front pair diverges), .L3
%   (the chief's axis crossing after M3), .chief_y3 (the chief's height at
%   M3 per radian of field bias: f), .ok.
    arguments
        f (1,1) double {mustBePositive}
        D (1,1) double {mustBePositive}
        L_xp (1,1) double
        t1 (1,1) double {mustBePositive}
        y2 (1,1) double {mustBePositive}
        opts.quiet (1,1) logical = true
    end
    phi1 = (1 - y2)/t1;                           % M1: y2 = 1 - t1 phi1
    t2 = f*y2;  t3 = f;
    % finite L_xp: phi3 = (1 + t2/(t3 + L_xp))/t2 (the chief's crossing t3 +
    % L_xp after M3); t2 re-solved for the EFL by the scaling f ~ t2 of the
    % telecentric limit (geometric convergence); the powers are recomputed
    % from the FINAL t2 so Petzval holds exactly
    for it = 1:80
        [phi2, phi3] = powers_(phi1, t2, t3, L_xp);
        [t3, ~, ~, u3p] = marginal_([phi1 phi2 phi3], t1, t2);
        f_now = -1/u3p;
        if abs(f_now - f) < 1e-13*f, break; end
        t2 = t2*f/f_now;
    end
    [phi2, phi3] = powers_(phi1, t2, t3, L_xp);
    phi = [phi1 phi2 phi3];
    [t3, L3, z_int, u3p, y] = marginal_(phi, t1, t2);
    S.phi = phi;  S.R = 2./abs(phi);  S.t3 = t3;  S.t = [t1 t2 t3];  S.y = y;
    S.L3 = L3;  S.z_int = z_int;  S.f = -1/u3p;  S.D = D;  S.L_xp = L_xp;  S.y2 = y2;
    S.chief_y3 = f;
    S.ok = all(isfinite(phi)) && phi(1) > 0 && phi(2) < 0 && phi(3) > 0 && t3 > 0 && abs(S.f - f) < 1e-8*f;
    S.K = [NaN NaN NaN];  S.EFL_check = NaN;  S.t3_check = NaN;
    if S.ok
        try
            [K, tf, EFL] = macos.design.seidel_seed(S.R, [t1 t2], D);
            S.K = K;  S.t3_check = tf;  S.EFL_check = EFL;
        catch ME
            S.seidel_error = ME.message;
        end
    end
    if ~opts.quiet
        fprintf('telescope_seed: f %.1f mm, D %.1f mm, L_xp %.2f m, t1 %.1f y2 %.2f -> R [%.1f %.1f %.1f] mm, t2 %.1f t3 %.1f mm, K %s, z_int %.1f mm, %s\n', ...
            f*1e3, D*1e3, L_xp, t1*1e3, y2, S.R*1e3, t2*1e3, t3*1e3, mat2str(S.K, 4), z_int*1e3, tern_(S.ok, 'OK', 'NO SOLUTION'));
    end
end

function [phi2, phi3] = powers_(phi1, t2, t3, L_xp)
    if isfinite(L_xp), phi3 = (1 + t2/(t3 + L_xp))/t2; else, phi3 = 1/t2; end
    phi2 = -phi1 - phi3;                         % Petzval flat
end

function [t3, L3, z_int, u3p, y] = marginal_(phi, t1, t2)
%MARGINAL_  Unfolded paraxial marginal + chief (stop at M2) rays.
    y1 = 1;  u1p = -phi(1)*y1;
    y2 = y1 + t1*u1p;  u2p = u1p - phi(2)*y2;
    y3 = y2 + t2*u2p;  u3p = u2p - phi(3)*y3;
    t3 = -y3/u3p;                               % M3 -> image
    L3 = t2/(phi(3)*t2 - 1);                    % chief from M2's vertex: axis crossing after M3
    if u2p < 0, z_int = -y2/u2p; else, z_int = NaN; end   % M1+M2 front: real image after M2?
    y = [y1 y2 y3];
end

function t = tern_(c, a, b), if c, t = a; else, t = b; end, end
