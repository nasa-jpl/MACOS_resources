function out = ray_hist(arg)
%MACOS.RAY_HIST  Per-trace ray-position history (engine RayPosHist).
%   The engine can record EVERY ray's global 3-D crossing at EVERY
%   element during a trace (traceutil_mod RayPosHist/LRayOKHist -- the
%   Vis3D substrate).  Capture is off by default:
%
%     macos.ray_hist('on')      enable capture (then run macos.trace)
%     macos.ray_hist('off')     disable
%     h = macos.ray_hist(nRays) read the LAST trace's history:
%         h.P   3 x nRays x (nElt+1) global positions, BaseUnits;
%               slot 1 = the source plane, slot k+1 = element k
%         h.ok  nRays x (nElt+1) logical -- true where the ray reached
%               that element (positions elsewhere are stale)
%
%   nRays is the source ray count of the trace (macos.trace().nRays).
%   Unlike the DRAW-fan harvest (macos.draw_rays3d), this covers the
%   FULL ray grid, so callers can pick arbitrary sparse bundles
%   (rings/spokes/rim) for layout drawing -- see macos.view_rx.
%
%   See also: macos.view_rx, macos.draw_rays3d, macos.trace.
arguments
    arg (1,:)
end
if ischar(arg) || isstring(arg)
    st = validatestring(arg, {'on', 'off'});
    mmacos('ray_hist_set', double(strcmp(st, 'on')));
    % dirty the cached trace so the NEXT macos.trace actually re-runs
    % CTRACE with the new capture state (same rule as the grid setters:
    % the engine skips the retrace when nothing is marked modified)
    macos.modify();
    return
end
nRays = arg;
validateattributes(nRays, {'numeric'}, {'scalar', 'integer', 'positive'});
nE1 = macos.num_elt() + 1;
z = zeros(nRays, nE1);
[Px, Py, Pz, lok] = mmacos('ray_pos_hist_get', z, z, z, z, ...
                           double(nRays), double(nE1));
P = zeros(3, nRays, nE1);
P(1,:,:) = Px;  P(2,:,:) = Py;  P(3,:,:) = Pz;
out = struct('P', P, 'ok', logical(lok));
end
