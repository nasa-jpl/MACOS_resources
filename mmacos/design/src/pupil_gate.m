function g = pupil_gate(opts)
%PUPIL_GATE  Is the engine tracing the pupil the prescription declares?
%
%   g = PUPIL_GATE() measures the traced entrance pupil of the CURRENTLY
%   LOADED prescription and compares it with the declared `Aperture=`.
%   Run it after EVERY source or aperture edit.
%
%   THE MEASUREMENT, and why it is not a span.  A ColSource ray grid is a
%   disc CLIPPED by the lattice square |x|,|y| <= Aperture/2.  Measured
%   along either axis it reads EXACTLY the declared aperture whatever the
%   acceptance radius is; only the DIAGONAL exposes an oversize circle.
%   So this gate reads:
%     .r_max      outermost ray radius at the stop, about the chief
%                 intercept, transverse to the chief   -> vs Aperture/2
%     .chord      greatest point-to-point distance in that plane
%                                                       -> vs Aperture
%     .n_outside  rays beyond the declared radius       -> must be 0
%   never a span.  A +5% pupil read exactly right on both axes for
%   decades (macos PR #70; design/rodgers1/PACKET.md Addendum 10), and the
%   extra rays sat at the OUTER pupil where the high-order field
%   aberration is largest, so the error was amplified rather than
%   averaged away.
%
%   The engine has been correct since PR #70.  The gate stays because the
%   failure mode is silent and the cost is one trace.
%
%   Name-value:
%     'elt'      element to measure at (default 1 = the stop for a
%                stop-at-M1 design).  Give the stop element for others.
%     'aperture' declared aperture, base units.  Default: read back from
%                the engine via macos.get_src_size (collimated sources
%                report a DIAMETER there; point sources report an N.A.,
%                so pass it explicitly for those).
%     'rtol'     relative bar on r_max/(Aperture/2) and chord/Aperture
%                (default 1e-3).  The engine's own acceptance radius is
%                Aperture*0.50001 plus a pre-existing absolute 1e-9 slack
%                at the r2<=A2 test, so the floor is ~2e-5 relative at
%                metre apertures and larger at small ones.
%     'nmax'     subsample cap for the O(N^2) chord (default 3000)
%     'quiet'    suppress the printed line (default false)
%     'assert'   throw on failure instead of returning ok=false
%                (default false)
%
%   Returns g with .ok .r_max .r_ratio .chord .chord_ratio .n_outside
%   .n_trace .aperture .elt, and .msg (the one-line verdict).
%
%   NOTE it reads ok_trace, NOT ok_pass: element apertures and
%   obscurations clear ok_pass downstream of the source grid and would
%   mask the very thing under test.
%
%   See also tests/tPupilAperture.m (the regression class this shares its
%   measurement with), design/rodgers1/pupil_audit.m (the five-way
%   diagnostic).

    arguments
        opts.elt      (1,1) double {mustBeInteger,mustBePositive} = 1
        opts.aperture (1,1) double = NaN
        opts.rtol     (1,1) double = 1e-3
        opts.nmax     (1,1) double = 3000
        opts.quiet    (1,1) logical = false
        opts.assert   (1,1) logical = false
    end

    Ap = opts.aperture;
    if isnan(Ap), Ap = declared_aperture_(); end

    tr = macos.trace(macos.num_elt());
    n  = tr.nRays;
    macos.trace(opts.elt);
    rm = macos.get_ray_info(n);

    % transverse to the chief ray, about the chief's own intercept
    pm   = rm.pos - rm.pos(:,1);
    d    = rm.dir(:,1);   d = d/norm(d);
    perp = pm - d*(d.'*pm);
    ok   = logical(rm.ok_trace(:));
    rho  = sqrt(sum(perp.^2,1)).';
    rho  = rho(ok);
    Pp   = perp(:,ok);

    g = struct();
    g.elt      = opts.elt;
    g.aperture = Ap;
    g.n_trace  = nnz(ok);
    g.r_max    = max(rho);
    g.r_ratio  = g.r_max/(Ap/2);
    g.n_outside= nnz(rho > (Ap/2)*(1+opts.rtol));

    % greatest chord -- the diagonal a span check cannot see
    Q = Pp;
    if size(Q,2) > opts.nmax
        Q = Q(:, round(linspace(1, size(Q,2), opts.nmax)));
    end
    C = 0;
    for a = 1:size(Q,2)
        C = max(C, max(sqrt(sum((Q - Q(:,a)).^2, 1))));
    end
    g.chord       = C;
    g.chord_ratio = C/Ap;

    g.ok = (g.r_ratio <= 1 + opts.rtol) && ...
           (g.chord_ratio <= 1 + opts.rtol) && (g.n_outside == 0);

    g.msg = sprintf(['pupil gate @elt %d: r_max %.6f (%.6f x semi), chord ' ...
                     '%.6f (%.6f x Aperture), %d rays outside, %d traced -> %s'], ...
                    g.elt, g.r_max, g.r_ratio, g.chord, g.chord_ratio, ...
                    g.n_outside, g.n_trace, tern_(g.ok,'PASS','** FAIL **'));
    if ~opts.quiet, fprintf('    %s\n', g.msg); end
    if opts.assert && ~g.ok
        error('macos:design:pupil_gate:fail', '%s', g.msg);
    end
end

% ---------------------------------------------------------------------
function Ap = declared_aperture_()
%DECLARED_APERTURE_  The source Aperture the loaded Rx declares.
%   macos.get_src_size reports a collimated source's aperture as a
%   DIAMETER in BaseUnits -- which is what `Aperture=` carries.  A point
%   source reports a numerical aperture there instead, and the caller must
%   pass the pupil diameter explicitly.
    s = macos.get_src_size();
    if ~(isfinite(s.aperture) && s.aperture > 0)
        error('macos:design:pupil_gate:aperture', ...
              ['the engine reports no usable source aperture -- pass one ' ...
               'explicitly with ''aperture''.']);
    end
    Ap = s.aperture;
end

function s = tern_(c,a,b), if c, s = a; else, s = b; end, end
