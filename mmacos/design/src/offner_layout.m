function [legs, tilts, geom] = offner_layout(R, h, opts)
%OFFNER_LAYOUT  Chief-path chain parameters for a concentric Offner relay.
%   [legs, tilts, geom] = offner_layout(R, h) lays out the classic
%   1:1 OFFNER relay -- concave sphere (radius R) used twice + convex
%   sphere (radius R/2) at the stop, ALL CONCENTRIC about one center C
%   -- for an object at RING RADIUS h off the Offner axis, and returns
%   it in the Bauer-chain form the Telescope builder consumes (vertex
%   spacings ALONG THE CHIEF + signed tilt of each mirror normal from
%   normal incidence):
%
%     legs  = [ |O->P1|  |P1->P2|  |P2->P3|  |P3->I| ]   (m, chief path)
%     tilts = [ a1 a2 a3 ]  (deg; add_mirror 'tilt_deg' for the
%             concave / convex / concave passes -- the resolver's
%             psi = rotx(tilt)*(-d_in) convention)
%
%   Concentricity + 1:1 symmetry zero the Seidel sums over the RING
%   FIELD at radius h: no tilted-powered-surface astigmatism, no coma,
%   no distortion -- the relay for a biased-field (ring-arc) instrument
%   bench, with several small-field pickoffs living on the same ring.
%
%   The chief is defined through the STOP center (the convex vertex):
%   the launch angle from O = (h, 0) is solved so the once-reflected
%   ray passes through (0, -R/2).  Closure invariants are ASSERTED:
%   the image lands at (-h, 0) (1:1, inverted across C) and the path
%   is symmetric (|O->P1| = |P3->I|, a1 = a3).
%
%   geom carries the 2-D construction for plotting/clearance checks:
%   .P1 .P2 .P3 .O .I (y,z about C), .inc_deg (incidence angles),
%   .conv_clear_m = lateral daylight between the O->P1 leg and the
%   convex vertex region at the z = -R/2 plane (the classic Offner
%   vignetting check -- size the convex body under this).
%
%   Options: 'fno' (beam f/# for the clearance number, default 18).
%
%   See also: macos.design.Telescope/add_mirror, tma_layout.
    arguments
        R (1,1) double {mustBePositive}
        h (1,1) double {mustBePositive}
        opts.fno (1,1) double {mustBePositive} = 18
    end
    if h >= R/2
        error('offner_layout:ring', ...
            'ring radius h=%.3g must be well inside R/2=%.3g.', h, R/2);
    end
    O  = [h; 0];                     % object point (y; z), C at the origin
    V2 = [0; -R/2];                  % stop center = convex vertex

    % -- chief launch angle: once-reflected ray passes through V2 --------
    miss = @(a) miss_(O, a, R, V2);
    alo = -0.2;  ahi = 0.2;          % the chief is near axis-parallel
    flo = miss(alo);
    fhi = miss(ahi);
    if sign(flo) == sign(fhi)
        error('offner_layout:bracket', ...
            'chief bracket failed (h/R = %.3g too large?)', h/R);
    end
    for it = 1:80
        am = 0.5*(alo+ahi);  fm = miss(am);
        if sign(fm) == sign(flo), alo = am;  flo = fm;
        else,                     ahi = am;
        end
    end
    a0 = 0.5*(alo+ahi);

    % -- trace the chief through the three reflections -------------------
    d0 = [sin(a0); -cos(a0)];
    [P1, d1] = bounce_(O,  d0, R,   'far');     % concave, first pass
    [P2, d2] = bounce_(P1, d1, R/2, 'near');    % convex at the stop
    [P3, d3] = bounce_(P2, d2, R,   'far');     % concave, second pass
    sI = -P3(2) / d3(2);                        % back to the plane z = 0
    I  = P3 + sI*d3;

    legs  = [norm(P1-O), norm(P2-P1), norm(P3-P2), sI];
    tilts = [tilt_(d0,d1), tilt_(d1,d2), tilt_(d2,d3)];

    % -- closure invariants (the concentric geometry must close) ---------
    assert(abs(I(1) + h) < 1e-9*max(1,R), ...
        'offner_layout: image at y=%.3e, expected %.3e (1:1 inversion)', I(1), -h);
    assert(abs(legs(1) - legs(4)) < 1e-9*max(1,R), ...
        'offner_layout: path asymmetry |O-P1|=%.6f vs |P3-I|=%.6f', legs(1), legs(4));
    assert(abs(tilts(1) - tilts(3)) < 1e-9, ...
        'offner_layout: tilt asymmetry a1=%.6f vs a3=%.6f', tilts(1), tilts(3));

    % -- convex-vignetting clearance at the z = -R/2 plane ---------------
    sV = (-R/2 - O(2)) / d0(2);
    yV = O(1) + sV*d0(1);                       % O->P1 leg at the stop plane
    beam_r = legs(1) / (2*opts.fno);            % cone radius there ~ full leg
    geom = struct('O',O, 'P1',P1, 'P2',P2, 'P3',P3, 'I',I, 'R',R, 'h',h, ...
        'inc_deg', abs(tilts), ...
        'conv_clear_m', abs(yV) - beam_r);      % minus the convex BODY radius
    tilts = tilts * 180/pi;
end

% =====================================================================
function m = miss_(O, a, R, Q)
%MISS_  Signed lateral miss of the once-reflected chief from point Q.
    d0 = [sin(a); -cos(a)];
    [P1, d1] = bounce_(O, d0, R, 'far');
    v = Q - P1;
    m = d1(1)*v(2) - d1(2)*v(1);                % 2-D cross(d1, v)
end

function [P, dout] = bounce_(p, d, r, which)
%BOUNCE_  Intersect ray p+s*d with the circle |x|=r about the origin and
%   reflect.  'far' takes the larger s (the concave surface across the
%   center), 'near' the smaller positive s (the convex at the stop).
    b = p.'*d;  c = p.'*p - r^2;
    disc = b^2 - c;
    assert(disc > 0, 'offner_layout: chief misses the r=%.3g sphere', r);
    s1 = -b - sqrt(disc);  s2 = -b + sqrt(disc);
    if strcmp(which,'far'), s = s2; else, s = min(s1(s1>1e-12), s2); end
    assert(~isempty(s) && s > 0, 'offner_layout: no forward intersection');
    P = p + s*d;
    n = P / norm(P);                             % radial (sign-invariant)
    dout = d - 2*(d.'*n)*n;
end

function a = tilt_(din, dout)
%TILT_  Signed mirror tilt (rad) from normal incidence that folds din
%   into dout: the deviation din->dout equals pi + 2a (a = 0 is retro).
    dev = atan2(din(1)*dout(2) - din(2)*dout(1), din.'*dout);
    a = wrap_(dev - pi) / 2;
end

function w = wrap_(x)
    w = mod(x + pi, 2*pi) - pi;
end
