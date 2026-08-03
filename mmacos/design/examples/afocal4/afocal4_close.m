function C = afocal4_close(P, form, opts)
%AFOCAL4_CLOSE  First-order closure of a 4-mirror afocal candidate.
%
%   C = AFOCAL4_CLOSE(P, FORM) returns the paraxial layout of one candidate
%   4-mirror form, with the two first-order conditions of PLAN_AFOCAL4 S3
%   closed in CLOSED FORM -- recollimate at M = P.M, and land the image of
%   the stop on the interface plane P.iface_dist past the last mirror.  No
%   optimization: this is layout, and everything it does is algebra on the
%   paraxial marginal and chief rays that AFOCAL_FIRST_ORDER carries.
%
%   FORM is one of:
%
%   'field'   his TMA with a FIELD MIRROR near the intermediate image
%             (between M2 and M3).  A mirror AT an image does not touch the
%             marginal ray -- so it cannot change the afocal condition or M
%             -- and sits where the chief ray is highest, so it moves the
%             exit pupil at maximum leverage.  That is the whole idea, and
%             it is also the whole difficulty: AT the image the beam
%             footprint is zero.  'standoff' backs it off by s toward M2,
%             which buys a footprint of 2*s*|u| and costs a re-solve of the
%             collimator (M3) power, both closed here.  With the standoff
%             fixed, the interface condition determines the field mirror's
%             POWER, and the collimator's power and station follow --
%             P.iface_dist in, R4 out.
%
%   'relay'   his M1/M2 with the back end RE-SPLIT: M3 no longer
%             recollimates but forms a SECOND image, and M4 collimates that.
%             This is what "a 4th mirror downstream of M3" has to mean --
%             the exit space of an afocal train is COLLIMATED, and a
%             collimated space admits no powered element without breaking
%             the afocality it is the definition of.  The pupil condition
%             determines M3's new power; M4's power and station follow.
%
%   'mersenne'  two confocal-parabola pairs, M = m1*m2.  Each pair is
%             afocal and anastigmatic by construction, so the afocal
%             condition and M are satisfied identically and the ONLY free
%             closure is the inter-stage gap, which the pupil condition
%             fixes.  'm1' splits the compression, 'stage2_fnum' sizes the
%             second pair.
%
%   Name-value:
%     'standoff'     'field': field mirror before the image, m (0.20)
%     'phi4'         'field': impose the power instead of closing it, in
%                    which case the interface distance FLOATS to wherever
%                    the pupil goes.  This is the useful mode: the closed
%                    power on this benchmark is ~0 (a flat), because the
%                    3-mirror already closes both conditions, so the design
%                    freedom is the family and not the root.
%     'phi3'         'relay': impose M3's new power (NaN = close it)
%     'c'            'relay': M2-image -> M3 distance (NaN = parent's)
%     'm1'           'mersenne': stage-1 magnification (6)
%     'stage2_fnum'  'mersenne': f3 / (stage-1 beam diameter) (2.0)
%     'stage2_type'  'mersenne': 'cassegrain' (convex M4, no internal focus)
%                    or 'gregorian' (concave M4, real internal focus).  The
%                    two put the exit pupil on OPPOSITE SIDES of M4 and the
%                    choice decides whether the form has an exit pupil to
%                    interface with at all.
%     'gap'          'mersenne': inter-stage spacing, m (NaN = close it)
%
%   Returns C with .R .t .convex .K .names (ready for the builder), the
%   paraxial verification .fo (an AFOCAL_FIRST_ORDER struct on the closed
%   layout), .footprint_m on the new mirror, and .closure -- the residuals
%   of the two conditions, which are the check that the algebra closed.

    arguments
        P (1,1) struct
        form (1,:) char {mustBeMember(form,{'field','relay','mersenne'})}
        opts.standoff    (1,1) double = 0.20
        opts.phi4        (1,1) double = NaN
        opts.phi3        (1,1) double = NaN
        opts.c           (1,1) double = NaN
        opts.m1          (1,1) double = 6.0
        opts.stage2_fnum (1,1) double = 2.0
        opts.stage2_type (1,:) char {mustBeMember(opts.stage2_type, ...
                          {'cassegrain','gregorian'})} = 'gregorian'
        opts.gap         (1,1) double = NaN
    end

    D = P.D;  SA = P.stop_ahead;  Mt = P.M;  dsp = P.iface_dist;
    R0 = P.parent.R;  t0 = P.parent.t;  c0 = P.parent.convex;
    K0 = P.parent.K;

    switch form
    case {'field','relay'}
        % state leaving M2 of the parent front end, and the intermediate
        % image it forms.  The front end is UNTOUCHED in both forms.
        p   = afocal_first_order(R0, t0, c0, 'D',D, 'stop_ahead',SA);
        ym  = p.y_marginal(2);   um = p.u_marginal(2);
        yc  = p.y_chief(2);      uc = p.u_chief(2);
        a0  = -ym/um;                          % M2 -> intermediate image
        % exit marginal height: one intermediate image = one axis crossing,
        % so the sign is negative.  |y| = (D/2)/M is the Lagrange statement
        % of M, and imposing it IS imposing the magnification.
        yout = -(D/2)/Mt;
    end

    switch form
    % ------------------------------------------------------------------
    case 'field'
        s = opts.standoff;
        a = a0 - s;                             % M2 -> field mirror
        if a <= 0
            error('macos:design:afocal4_close:standoff', ...
                'standoff %.3f m is past the intermediate image at %.3f m.', s, a0);
        end
        if isnan(opts.phi4)
            g = @(x) field_d_(x, a, ym, um, yc, uc, yout) - dsp;
            % d(phi4) is a RATIONAL function with poles, so a wide bracket
            % straddles a pole rather than a root and fzero returns the
            % pole.  [-1,1] per metre is the physical window (a field mirror
            % of |R| < 2 m in a 1.4 m relay space is not a field mirror) and
            % d is monotone across it.
            phi4 = fzero(g, [-1 1]);
        else
            phi4 = opts.phi4;
        end
        [d, b, phi3, y4] = field_d_(phi4, a, ym, um, yc, uc, yout);
        C.names   = {'M1','M2','FM','M3'};
        C.R       = [R0(1) R0(2) abs(2/phi4) abs(2/phi3)];
        C.convex  = [c0(1) c0(2) phi4 < 0, phi3 < 0];
        C.t       = [t0(1) a b];
        C.K       = [K0(1) K0(2) 0 K0(3)];     % new mirror seeds at K=0
        C.new     = 3;
        C.footprint_m = 2*abs(y4);
        C.note    = sprintf('field mirror %.0f mm before the image', s*1e3);
        C.standoff = s;
        C.d_pred  = d;

    % ------------------------------------------------------------------
    case 'relay'
        cm = opts.c;   if isnan(cm), cm = t0(2); end     % M2 -> M3
        % M3 must be given MORE power than the parent's collimating value or
        % there is no second image for M4 to collimate: below it the beam
        % leaves M3 still diverging and never returns to the axis.  The
        % second image also means a SECOND axis crossing, so the exit
        % marginal height changes sign relative to the 3-mirror -- that sign
        % is taken from the requirement that M4 sit at a positive distance,
        % not assumed.
        phi_col = -u_after_(ym, um, cm, 0) / (ym + cm*um);  %#ok<NASGU>
        if isnan(opts.phi3)
            g = @(x) relay_d_(x, cm, ym, um, yc, uc, yout) - dsp;
            lo = 3.6;  hi = 40;
            if sign(g(lo)) == sign(g(hi))
                error('macos:design:afocal4_close:relayUnreachable', ...
                    ['the relay form cannot put the exit pupil at %.4f m: ' ...
                     'the station runs %.4f m to %.4f m over phi3 in [%g, %g] ' ...
                     'and never crosses.  Impose phi3 and read the station ' ...
                     'that results.'], dsp, g(lo)+dsp, g(hi)+dsp, lo, hi);
            end
            phi3 = fzero(g, [lo hi]);
        else
            phi3 = opts.phi3;
        end
        [d, e, phi4, ~] = relay_d_(phi3, cm, ym, um, yc, uc, yout);
        if e <= 0.05
            error('macos:design:afocal4_close:relayDegenerate', ...
                ['phi3 = %.4f puts the collimator %.4f m from M3, which is ' ...
                 'not a relay -- below the parent''s collimating power the ' ...
                 'beam never returns to the axis, and just above it the two ' ...
                 'mirrors stack into one.  M3 needs MORE power than the ' ...
                 'parent''s.'], phi3, e);
        end
        C.names  = {'M1','M2','M3','M4'};
        C.R      = [R0(1) R0(2) abs(2/phi3) abs(2/phi4)];
        C.convex = [c0(1) c0(2) phi3 < 0, phi4 < 0];
        C.t      = [t0(1) cm e];
        C.K      = [K0(1) K0(2) K0(3) 0];
        C.new    = 4;
        C.footprint_m = D/Mt;                  % M4 works in the exit beam
        C.note   = sprintf('M3 refocuses (R %.4f m), M4 collimates %.0f mm later', ...
                           abs(2/phi3), e*1e3);
        C.d_pred = d;

    % ------------------------------------------------------------------
    case 'mersenne'
        m1 = opts.m1;   m2 = Mt/m1;
        f1 = R0(1)/2;                            % keep HIS primary
        f2 = -f1/m1;                             % confocal convex secondary
        d1 = D/m1;                               % stage-1 exit beam
        f3 = opts.stage2_fnum*d1;
        % The second pair's TYPE is the whole question.  A Cassegrain pair
        % (convex M4, no internal focus) puts its exit pupil BEHIND M4 --
        % virtual, and nothing can be interfaced to it.  A Gregorian pair
        % (concave M4, a real focus between M3 and M4) puts it in front.
        if strcmp(opts.stage2_type,'cassegrain'), f4 = -f3/m2; else, f4 = f3/m2; end
        RR = [2*f1, 2*abs(f2), 2*f3, 2*abs(f4)];
        cc = [false, true, false, f4 < 0];
        gap = opts.gap;
        if isnan(gap)
            g = @(x) pupil_of_(RR, [f1+f2, x, f3+f4], cc, D, SA) - dsp;
            lo = 0.02;  hi = 20;
            if sign(g(lo)) == sign(g(hi))
                error('macos:design:afocal4_close:mersenneUnreachable', ...
                    ['the %s double Mersenne cannot put the exit pupil at ' ...
                     '%.4f m: it runs %.4f m (gap %.2f) to %.4f m (gap %.2f) ' ...
                     'and never crosses.  Impose the gap and read the ' ...
                     'station that results.'], opts.stage2_type, dsp, ...
                     g(lo)+dsp, lo, g(hi)+dsp, hi);
            end
            gap = fzero(g, [lo hi]);
        end
        C.names  = {'M1','M2','M3','M4'};
        C.R      = RR;   C.convex = cc;
        C.t      = [f1+f2, gap, f3+f4];
        C.K      = [-1 -1 -1 -1];                % confocal PARABOLAS
        C.new    = 4;
        C.footprint_m = D/Mt;
        C.note   = sprintf('%s 2nd pair, m1 = %.3f, m2 = %.3f, gap %.0f mm', ...
                           opts.stage2_type, m1, m2, gap*1e3);
        C.m1 = m1;  C.m2 = m2;  C.gap = gap;  C.stage2_type = opts.stage2_type;
        C.d_pred = pupil_of_(RR, C.t, cc, D, SA);
    end

    % ---- verify the closure on the assembled layout ---------------------
    C.form = form;
    C.fo   = afocal_first_order(C.R, C.t, C.convex, 'D',D, 'stop_ahead',SA);
    C.closure = struct( ...
        'u_out',        C.fo.u_out, ...
        'mag_err',      C.fo.mag/Mt - 1, ...
        'pupil_err_m',  C.fo.pupil_dist - dsp, ...
        'exit_dia_err', C.fo.exit_dia/(D/Mt) - 1);
    C.train_len = sum(C.t) + dsp;
end

% =====================================================================
function d = pupil_of_(R, t, cvx, D, SA)
    q = afocal_first_order(R, t, cvx, 'D',D, 'stop_ahead',SA);
    d = q.pupil_dist;
end

function [d, b, phi3, y4] = field_d_(phi4, a, ym, um, yc, uc, yout)
%FIELD_D_  Where the exit pupil lands for a field mirror of power PHI4.
%   The marginal ray fixes everything but the pupil: the collimator's
%   station B and power PHI3 are whatever makes the beam leave at height
%   YOUT (= M) and angle zero (= afocal), so imposing the two first-order
%   conditions is not a solve at all -- it is substitution.  What is left
%   over is the chief ray, and D is the one number PHI4 actually buys.
    y4   = ym + a*um;                   % marginal height on the field mirror
    u4   = um - y4*phi4;
    b    = (yout - y4)/u4;              % field mirror -> collimator
    phi3 = u4/yout;                     % collimator power (afocal)
    yc4  = yc + a*uc;                   % chief height on the field mirror
    uc4  = uc - yc4*phi4;
    yc3  = yc4 + b*uc4;
    uc3  = uc4 - yc3*phi3;
    d    = -yc3/uc3;
end

function u = u_after_(y, u0, c, phi)
    u = u0 - (y + c*u0)*phi;
end

function [d, e, phi4, y4] = relay_d_(phi3, c, ym, um, yc, uc, yout)
%RELAY_D_  Where the exit pupil lands when M3 refocuses at power PHI3 and
%   M4 collimates.  Same substitution: M4's station and power are forced by
%   the two first-order conditions, and the chief ray is the residue.
%   C is M2 -> M3.  The exit marginal height is +-|YOUT|; the sign is the
%   one that puts M4 DOWNSTREAM of M3, because a second image means a
%   second axis crossing and assuming the 3-mirror's sign gives a
%   collimator at a negative distance.
    y3   = ym + c*um;
    u3   = um - y3*phi3;
    e1   = ( abs(yout) - y3)/u3;
    e2   = (-abs(yout) - y3)/u3;
    if e1 > 0 && (e2 <= 0 || e1 < e2), y4 = abs(yout);  e = e1;
    else,                              y4 = -abs(yout); e = e2;
    end
    phi4 = u3/y4;
    yc3  = yc + c*uc;
    uc3  = uc - yc3*phi3;
    yc4  = yc3 + e*uc3;
    uc4  = uc3 - yc4*phi4;
    d    = -yc4/uc4;
end
