function F = chain_footprints(G, B)
%CHAIN_FOOTPRINTS  Per-surface beam footprint of a bundle, in the APERTURE frame.
%   F = chain_footprints(G, B) is spectrometer_geom's footprints_ on an
%   arbitrary bundle B (chain_bundle or the Dyson's): for every surface the
%   hits are expressed about the VERTEX in the engine's aperture frame (x_ap =
%   global x projected into the vertex tangent plane, y_ap = psi x x_ap --
%   xObs/yObs with xObs = x written) and reduced to a centre (xc, yc), the
%   enclosing radius about it, the extents and the frame axes.  The emitter
%   declares ApType Circular, ApVec = (radius + margin, xc, yc); the
%   clearance gate sizes the bodies from it.
    nS = numel(G.surf);
    assert(~isempty(B.P) && size(B.P, 2) >= 3, 'chain_footprints: %d rays reach the end of the chain', size(B.P, 2));
    F = struct('xc', {}, 'yc', {}, 'radius', {}, 'xlim', {}, 'ylim', {}, 'xap', {}, 'yap', {}, 'n', {});
    for k = 1:nS
        S = G.surf(k);  psi = S.psi(:)/norm(S.psi);
        xap = [1;0;0] - ([1;0;0]'*psi)*psi;  xap = xap/norm(xap);  yap = cross(psi, xap);
        H = squeeze(B.P(:, :, k+1));  rho = H - S.vpt(:);
        px = (rho'*xap)';  py = (rho'*yap)';
        xc = 0.5*(min(px) + max(px));  yc = 0.5*(min(py) + max(py));
        F(k) = struct('xc', xc, 'yc', yc, 'radius', max(hypot(px - xc, py - yc)), ...
                      'xlim', [min(px) max(px)], 'ylim', [min(py) max(py)], 'xap', xap, 'yap', yap, 'n', numel(px));
    end
end
