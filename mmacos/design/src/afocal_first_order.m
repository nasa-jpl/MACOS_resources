function out = afocal_first_order(R, t, convex, opts)
%AFOCAL_FIRST_ORDER  Paraxial marginal AND chief trace of an N-mirror train.
%
%   out = AFOCAL_FIRST_ORDER(R, T, CONVEX, 'D',D, 'stop_ahead',S) traces the
%   two paraxial rays that between them carry every first-order property an
%   afocal telescope is specified by:
%
%     the MARGINAL ray -- collimated in at full aperture -- gives the AFOCAL
%       CONDITION (.u_out, exactly 0 when the train recollimates), the exit
%       beam diameter, and hence the angular magnification .mag = D/exit_dia
%       by the Lagrange invariant;
%     the CHIEF ray -- through the stop at unit field angle -- gives the
%       PUPIL: .pupil_dist past the last mirror, and .pupil_dia there.
%
%   Those are the two conditions a 4-mirror afocal form study has to close
%   (PLAN_AFOCAL4 S3): recollimate at M, and land the image of the stop on
%   the interface plane.  Everything else is aberration.
%
%   MODEL.  Unfolded thin-lens equivalent, mirror f = R/2, a CONVEX mirror a
%   negative lens -- the same paraxial model TELESCOPE.PARAXIAL_FOCUS_ and
%   MACOS.DESIGN.SEIDEL_SEED use, so a layout that closes here builds there.
%   R is 1xN MAGNITUDES, T the N-1 inter-mirror spacings, CONVEX a 1xN
%   logical.  Any consistent length unit; metres by convention here.
%
%   'stop_ahead' is the stop's distance AHEAD of the first mirror (positive =
%   upstream, the Rodgers2 geometry: 50 mm ahead of M1).  It is not a
%   detail: at 0.6 deg it walks the beam half a millimetre across a 1 m
%   primary, and it is what makes the entrance pupil not simply be M1.
%
%   Returns .u_out .y_out .mag .exit_dia .pupil_dist .pupil_dia
%   .y_marginal / .u_marginal / .y_chief / .u_chief (per mirror, AFTER that
%   mirror's power) and .f (the signed focal lengths).
%
%   See also TELESCOPE/ADD_EXIT_REFERENCE, PUPIL_MAP, AFOCAL_LADDER_DECK.

    arguments
        R (1,:) double {mustBePositive}
        t (1,:) double
        convex (1,:) logical
        opts.D          (1,1) double {mustBePositive} = 1.0
        opts.stop_ahead (1,1) double = 0.0
    end
    N = numel(R);
    if numel(t) ~= N-1
        error('macos:design:afocal_first_order:t', ...
            'need %d inter-mirror spacings for %d mirrors (got %d).', ...
            N-1, N, numel(t));
    end
    if numel(convex) ~= N
        error('macos:design:afocal_first_order:convex', ...
            'convex must be 1x%d.', N);
    end

    f = R/2;   f(convex) = -f(convex);
    phi = 1./f;

    % marginal: collimated, full aperture, at M1
    ym = opts.D/2;   um = 0;
    % chief: through the stop at unit field angle, propagated to M1
    yc = opts.stop_ahead;   uc = 1;

    YM = zeros(1,N);  UM = zeros(1,N);  YC = zeros(1,N);  UC = zeros(1,N);
    YMi = zeros(1,N); YCi = zeros(1,N);
    for k = 1:N
        YMi(k) = ym;   YCi(k) = yc;          % heights ON the mirror
        um = um - ym*phi(k);
        uc = uc - yc*phi(k);
        YM(k) = ym;  UM(k) = um;  YC(k) = yc;  UC(k) = uc;
        if k < N
            ym = ym + t(k)*um;
            yc = yc + t(k)*uc;
        end
    end

    % The angular magnification is |uc_out| for a unit input field angle;
    % the Lagrange invariant makes that identical to D/exit_dia, and the two
    % are returned separately BECAUSE their disagreement is the check that
    % the train is really afocal (they part company as u_out leaves zero).
    out = struct();
    out.f          = f;
    out.u_out      = um;
    out.y_out      = ym;
    out.exit_dia   = 2*abs(ym);
    out.mag        = abs(uc);
    out.mag_lagrange = opts.D/max(2*abs(ym), realmin);
    out.pupil_dist = -yc/uc;                  % past the last mirror
    out.pupil_dia  = 2*abs(ym + out.pupil_dist*um);
    out.y_marginal = YMi;   out.u_marginal = UM;
    out.y_chief    = YCi;   out.u_chief    = UC;
    out.D          = opts.D;
    out.stop_ahead = opts.stop_ahead;
end
