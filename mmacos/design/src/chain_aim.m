function [p0, ok] = chain_aim(S, plane, d, iStop, lam, G, p_guess)
%CHAIN_AIM  Launch POINT on a plane whose ray along d hits surface iStop at its vertex.
%   p0 = chain_aim(S, plane, d, iStop, lam, G) is the point on the launch plane
%   (plane.C, plane.N) from which a ray of FIXED direction d (a collimated
%   field) reaches surface iStop of the chain S at S(iStop).vpt -- the
%   collimated-source twin of spectrometer_geom's aim_ (which turns the
%   DIRECTION of a point source).  Newton on the two in-plane coordinates.
%   p_guess seeds the search (default: the plane point under the vertex
%   along -d).  ok = false when Newton does not converge (the field is lost).
    N = plane.N(:)/norm(plane.N);  d = d(:)/norm(d);
    ex = cross([0;1;0], N);  if norm(ex) < 1e-9, ex = cross([1;0;0], N); end
    ex = ex/norm(ex);  ey = cross(N, ex);
    tgt = S(iStop).vpt(:);
    % the miss is measured in the TARGET's tangent plane (two axes normal to
    % its psi), not the launch plane's: a grating whose surface is normal to
    % the Dyson axis leaves a miss ALONG that axis invisible to the launch
    % plane's axes (0.2 mm at the field ends, 2026-10-01)
    nt = S(iStop).psi(:)/norm(S(iStop).psi);
    et1 = cross([0;1;0], nt);  if norm(et1) < 1e-9, et1 = cross([1;0;0], nt); end
    et1 = et1/norm(et1);  et2 = cross(nt, et1);
    if nargin < 7 || isempty(p_guess)
        % back along d from the target to the plane
        t = ((plane.C(:) - tgt)'*N)/(d'*N);  p_guess = tgt + t*d;
    end
    uv = [(p_guess(:) - plane.C(:))'*ex; (p_guess(:) - plane.C(:))'*ey];
    pt_of = @(uv) plane.C(:) + uv(1)*ex + uv(2)*ey;
    miss = @(uv) miss_(S, pt_of(uv), d, lam, G, iStop, tgt, et1, et2);
    ok = false;
    for it = 1:40
        m0 = miss(uv);
        if any(isnan(m0)), break; end
        if norm(m0) < 1e-13, ok = true; break; end
        h = 1e-7;  J = zeros(2);
        J(:,1) = (miss(uv + [h;0]) - m0)/h;  J(:,2) = (miss(uv + [0;h]) - m0)/h;
        if any(isnan(J(:))) || rcond(J) < 1e-14, break; end
        uv = uv - J\m0;
    end
    if ~ok && norm(miss(uv)) < 1e-10, ok = true; end
    p0 = pt_of(uv);
end

function m = miss_(S, p0, d, lam, G, iStop, tgt, ex, ey)
    [pts, ~, ok] = chain_trace(S(1:iStop), p0, d, lam, G);
    if ~ok, m = [NaN; NaN]; return; end
    v = pts(:, iStop) - tgt;
    m = [v'*ex; v'*ey];
end
