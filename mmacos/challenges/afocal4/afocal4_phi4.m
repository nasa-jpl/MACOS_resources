function [phi4, C, found] = afocal4_phi4(P, s, iface)
%AFOCAL4_PHI4  The field-mirror power that puts the exit pupil at IFACE.
%
%   [phi4, C, found] = AFOCAL4_PHI4(P, S, IFACE) closes the 'field' form at
%   field-mirror standoff S and returns the power the interface condition
%   demands, the closed layout C, and whether a real one was found.  This
%   is the inner root of AFOCAL4_BUILD, factored out so the SEEDER
%   (AFOCAL4_PACK_SEED) and the study scripts use exactly the same
%   selection rule the build does -- a seeder that picked roots by a
%   different rule would hand the solver designs the builder would reject.
%
%   A SIGN CHANGE IS NOT A ROOT.  d(phi4) is a rational function of the
%   power, so it changes sign across its POLES as well as its zeros, and
%   fzero converges onto a pole quite happily -- returning a "layout" with
%   a collimator 1e14 m away.  Under the S4b packaging constraint the
%   useful roots moved from phi4 ~ 2 /m to ~ 4 /m, past a pole, and taking
%   the first sign change started failing outright.  So every candidate is
%   CLOSED and CHECKED -- finite spacings, the pupil where it was asked
%   for, and the two first-order identities intact -- and the first one
%   that is actually a telescope wins.  First, i.e. LOWEST power, because
%   the weakest field mirror is the one closest to the three-mirror parent.
%
%   The original 119-point window over [-0.9, 5] /m is scanned first and
%   unchanged, so every S4 design re-derives to its committed value; the
%   wider [-0.9, 9] window is a fallback, not a redefinition.
%
%   FOUND = false rather than an error: the callers differ on whether "no
%   root here" is a failure (the builder) or a datum (the seeder scanning
%   a plane of standoffs, most of which have none).
%
%   See also AFOCAL4_BUILD, AFOCAL4_CLOSE, AFOCAL4_PACK_SEED.

    arguments
        P (1,1) struct
        s (1,1) double
        iface (1,1) double
    end
    Q = P;   Q.iface_dist = iface;
    d = @(x) pupil_of_(Q, s, x);
    for xs = {linspace(-0.9, 5.0, 119), linspace(-0.9, 9.0, 401)}
        [phi4, C, found] = scan_(Q, s, iface, d, xs{1});
        if found, return; end
    end
end

% =====================================================================
function [phi4, C, found] = scan_(Q, s, iface, d, xs)
    phi4 = NaN;  C = [];  found = false;
    g = arrayfun(@(x) safe_(d, x) - iface, xs);
    k = find(isfinite(g(1:end-1)) & isfinite(g(2:end)) & ...
             sign(g(1:end-1)) ~= sign(g(2:end)));
    for j = k(:).'
        try
            x  = fzero(@(y) d(y) - iface, [xs(j) xs(j+1)]);
            Ct = afocal4_close(Q, 'field', 'standoff',s, 'phi4',x);
        catch
            continue;
        end
        if any(~isfinite(Ct.t)) || any(Ct.t < 0.02) || any(Ct.t > 8), continue; end
        if abs(Ct.fo.pupil_dist - iface) > 1e-8, continue; end
        if abs(Ct.fo.mag/Q.M - 1) > 1e-9 || abs(Ct.fo.u_out) > 1e-9, continue; end
        phi4 = x;   C = Ct;   found = true;   return;
    end
end

function y = pupil_of_(Q, s, phi4)
%   MATLAB will not index into a function-call result, so the closure's
%   pupil station needs a named intermediate.  Cheap: pure algebra.
    C = afocal4_close(Q, 'field', 'standoff',s, 'phi4',phi4);
    y = C.fo.pupil_dist;
end

function y = safe_(f, x)
%   A standoff past the intermediate image makes the closure error rather
%   than return a number; on a SCAN that is a gap, not a stop.
    try, y = f(x); catch, y = NaN; end
end
