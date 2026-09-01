function S = offaxis_seed(P, form, opts)
%OFFAXIS_SEED  An OFF-AXIS afocal seed, exact by construction, not by solving.
%
%   S = OFFAXIS_SEED(P, FORM) returns a DESCENT_BUILD spec for an off-axis
%   afocal telescope seeded from a MERSENNE -- a confocal parabola pair whose
%   focal lengths stand in the ratio P.M.  Such a pair is EXACTLY afocal and
%   EXACTLY P.M x for a beam entering anywhere on it, on axis or off, because
%   a parabola takes a collimated beam to its focus from any part of its
%   surface.  So the seed's two hardest properties are identities of the
%   geometry rather than residuals of a solve.
%
%   THIS IS WHAT MAKES THE SLICE ANSWERABLE.  The rigid-body probe measured
%   only that the coaxial point is a local optimum under perturbation -- it
%   could not reach the off-axis family because it started inside the coaxial
%   basin and 15 deg of tilt did not leave it.  A Mersenne is not a perturbed
%   coaxial design; it is a different member of the family, arrived at
%   directly.  Whether that member is BETTER is what the floor measures; that
%   it is genuinely off-axis is guaranteed here.
%
%   TWO FORMS, and they are different geometries and different basins, not
%   two parameterizations of one:
%
%     'cass'  CASSEGRAIN Mersenne, separation f1 - f2, secondary CONVEX,
%             beam never crosses a focus.  Compact (2.42 m for f1 = 2.5),
%             the secondary is small and the package is short.  No real
%             intermediate image, so nothing can be field-stopped there.
%     'greg'  GREGORIAN Mersenne, separation f1 + f2, secondary CONCAVE,
%             REAL intermediate focus between the mirrors.  Longer (2.58 m
%             for the same f1) but the focus is a place to put a field stop
%             or a scattered-light baffle, and the sign of the secondary's
%             contribution to field curvature is opposite -- which is why it
%             is a separate seed and not a tweak of the first.
%
%   BOTH ARE USED OFF AXIS BY DECENTERING THE PUPIL, not by tilting a
%   mirror.  Paraxially the decentered train is IDENTICAL to the coaxial one
%   -- same powers, same spacings, same afocal/magnification/pupil closures
%   -- which is exactly why DESCENT_CLOSE and the whole merit apply to it
%   unchanged.  OFFAXIS_DECENTER does the displacement and re-fits the
%   apertures; see its header for why the aperture fit is not optional.
%
%   THE DECENTER IS SIZED TO CLEAR, NOT GUESSED.  The requirement is that the
%   secondary's BODY stand clear of the entering beam:
%       h >= P.D/2 + r_body(M2) + clearance
%   with r_body from the union gate's own declared allowance (body_k x
%   footprint + body_pad) so the seed and the gate that judges it use one
%   definition.  DEFAULT_H_ returns that number; 'h' overrides it, and a
%   value below it is passed through with a warning rather than silently
%   corrected -- an obscured "off-axis" design is a finding, not an error.
%
%   THE EXTRA MIRRORS.  A two-mirror Mersenne is afocal at P.M and has NO
%   free parameter left: both powers are consumed by the magnification and
%   the collimation, so the exit pupil lands where the geometry puts it and
%   the field is uncorrected.  N > 2 adds mirrors whose powers DESCENT_CLOSE
%   then re-solves so that the exit pupil reaches the interface plane -- the
%   Mersenne pair seeds the front end, the closure owns the back end.  That
%   is the same division of labour the descent used to climb to seven.
%
%   THE MIRROR COUNT IS SET BY PACKAGING PARITY, NOT BY ABERRATIONS -- and
%   this is the first thing the off-axis family says that the coaxial one did
%   not.  DESCENT_CLOSE's last spacing ABSORBS the free tail spacing exactly:
%   scanned over t2 from 0.3 to 6.0 m, the N = 4 closure's last powered mirror
%   sits at behind_m1 = -1388.7 mm for EVERY t2 (t3 tracks t2 with a constant
%   offset of 0.1803 m).  So with an N = 4 Mersenne front end the packaging
%   station is not a knob at all -- it is a constant of the front end, and it
%   is on the wrong side.  The parity law z_N = sum (-1)^k t_k says why, and
%   says what to do: one more reflection flips the sign of the whole back end.
%   Measured across form x f1 x N (2 x 3 x 3, both forms, f1 in {0.75, 1.25,
%   2.5}): N = 5 COMPLIES over essentially the whole (t2, t3) grid, N = 4 and
%   N = 6 comply NOWHERE on it.  Hence the default below.
%
%   That is a statement about the LAYOUT and not about image quality, and it
%   is worth keeping the two apart: N = 4 off-axis is not ruled out because it
%   images badly, it is ruled out because its back end lands in front of its
%   own primary.  The coaxial study reached N = 4 comfortably because its
%   front end is not a Mersenne -- t1 there is a fraction of f1, where a
%   Mersenne's is f1 - f2, essentially the whole focal length.
%
%   Name-value:
%     'N'      total powered mirrors (default 5 -- see the parity note).  N = 2 is the bare
%              Mersenne and CANNOT meet the interface-pupil condition; it is
%              built anyway when asked, for the collimation/M identity check.
%     'f1'     primary focal length, m (default 2.5 -- P.D/2/f1 = f/2.5,
%              the committed study's own front-end speed class).
%     'h'      pupil decenter, m (default: DEFAULT_H_, the clearing value).
%     'iface'  interface standoff, m (default P.iface).
%     'tail_R' radius magnitudes for the mirrors past the Mersenne pair
%              (default: a mild concave/convex alternation scaled off f1;
%              only the FREE ones matter, the closure sets the last two).
%
%   Returns S with the DESCENT_BUILD D-fields (.N .R .convex .t .K .iface
%   .tilt_deg .decenter) plus .form .f1 .f2 .sep .h .why.
%
%   See also OFFAXIS_DECENTER, OFFAXIS_BUILD, DESCENT_CLOSE, DESCENT_BUILD.

    arguments
        P (1,1) struct
        form (1,:) char {mustBeMember(form,{'cass','greg'})} = 'cass'
        opts.N      (1,1) double = 5
        opts.f1     (1,1) double = 2.5
        opts.h      (1,1) double = NaN
        opts.iface  (1,1) double = NaN
        opts.tail_R (1,:) double = []
    end

    f1 = opts.f1;   f2 = f1/P.M;                 % the Mersenne ratio, exactly
    switch form
    case 'cass', sep = f1 - f2;   cvx2 = true;   % convex secondary
    case 'greg', sep = f1 + f2;   cvx2 = false;  % concave, real focus between
    end
    R1 = 2*f1;   R2 = 2*f2;

    h = opts.h;
    if isnan(h), h = default_h_(P, f2); end

    iface = opts.iface;
    if isnan(iface), iface = getf_(P, 'iface', 0.343); end

    N = opts.N;
    if N < 2
        error('macos:design:offaxis_seed:N', ...
              'a Mersenne seed needs at least its two mirrors (N >= 2).');
    end

    % ---- the free front end: the Mersenne pair, then the tail ------------
    % DESCENT_CLOSE takes the FREE mirrors (1 .. N-2) and solves the last
    % two.  For N = 4 that is exactly the Mersenne pair free and the back end
    % closed -- the division this seed is built around.
    nfree = N - 2;
    Rf = zeros(1,nfree);   Cf = false(1,nfree);   tf = zeros(1,nfree);
    if nfree >= 1, Rf(1) = R1;  Cf(1) = false; tf(1) = sep;  end
    if nfree >= 2, Rf(2) = R2;  Cf(2) = cvx2;  tf(2) = 0.60;  end
    tailR = opts.tail_R;
    for k = 3:nfree
        if k-2 <= numel(tailR), Rf(k) = tailR(k-2); else, Rf(k) = f1*(1 + 0.3*(k-2)); end
        Cf(k) = mod(k,2)==1;                     % alternate, mildly
        tf(k) = 1.0;                             % on the compliant plateau
    end

    S = struct('N',N, 'R',Rf, 'convex',Cf, 't',tf, 'iface',iface, ...
               'K',-ones(1,N), ...               % parabolas: the seed's point
               'tilt_deg',zeros(1,N), 'decenter',h, ...
               'form',form, 'f1',f1, 'f2',f2, 'sep',sep, 'h',h);
    S.why = sprintf(['%s Mersenne f1 %.4f / f2 %.6f m (ratio %.4f), ' ...
                     'separation %.4f m, pupil decentered %.4f m'], ...
                    upper(form), f1, f2, f1/f2, sep, h);
end

% =====================================================================
function h = default_h_(P, f2)
%DEFAULT_H_  The decenter that clears the secondary's BODY, using the union
%   gate's own allowance so the seed and its judge speak one language.
    rbeam = P.D/2;
    r2    = (P.D/P.M)/2;                         % the compressed beam on M2
    bk    = getf_(getf_(P,'pack',struct()), 'union_body_k',   1.15);
    bp    = getf_(getf_(P,'pack',struct()), 'union_body_pad', 0.015);
    rbody = bk*r2 + bp;
    h     = rbeam + rbody + bp;                  % one more pad of daylight
    h     = ceil(h*1e3/10)*10/1e3;               % round up to 10 mm
end

function v = getf_(s, f, d)
    if isstruct(s) && isfield(s,f), v = s.(f); else, v = d; end
end
