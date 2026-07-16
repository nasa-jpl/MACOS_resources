function b = draw_rays3d(plane, iStart, iEnd)
%MACOS.DRAW_RAYS3D  Real DRAW ray fan as 3-D data (no graphics device).
%   b = macos.draw_rays3d(PLANE, ISTART, IEND) runs the engine DRAW command
%   in DATA-ONLY mode and returns the traced fan's surface crossings in
%   GLOBAL 3-D coordinates (BaseUnits) — the unprojected companion of
%   macos.draw_rays.  PLANE ('YZ' | 'XZ') selects WHICH meridian fan the
%   engine traces (DRAW draws the middle fan of that plane); the returned
%   positions are true (x,y,z) either way, so two calls give two orthogonal
%   3-D fans with no projection assumption — correct for folded /
%   off-axis systems.  Crossings are recorded for EVERY element type the
%   trace crosses (Segment and non-sequential elements included), which
%   per-element macos.trace(k) harvesting cannot do (OPD refuses
%   NSRefractor/Segment/NSReflector targets).
%
%   Returns a struct:
%     b.P     (3 x nMaxElt x nRay) global position of each crossing
%     b.elt   (nMaxElt x nRay)     element index of each crossing
%     b.nper  (1 x nRay)           number of crossings for each ray
%     b.nray, b.plane
%   Ray r is the 3-D polyline b.P(:, 1:b.nper(r), r).
%
%   Backs macos.view_rx (general prescription visualizer).
%   See also: macos.draw_rays, macos.view_rx.
    arguments
        plane  (1,:) char   = 'YZ'
        iStart (1,1) double  = 0
        iEnd   (1,1) double  = 0
    end
    key = upper(plane);
    pmap = struct('YZ',1, 'XZ',2, 'XY',3);
    if ~isfield(pmap, key)
        error('macos:draw_rays3d:plane', ...
              'plane must be YZ, XZ or XY (got %s)', plane);
    end
    if iEnd <= 0, iEnd = macos.num_elt(); end

    [nDE, nRay]     = mmacos('draw_rays_cmd', pmap.(key), iStart, iEnd);
    [~, ~, E, nper] = mmacos('draw_rays_get', nDE, nRay);
    [Px, Py, Pz]    = mmacos('draw_rays3d_get', nDE, nRay);

    P = zeros(3, nDE, nRay);
    P(1,:,:) = Px;  P(2,:,:) = Py;  P(3,:,:) = Pz;
    b = struct('P', P, 'elt', double(E), 'nper', double(nper(:).'), ...
               'nray', double(nRay), 'plane', key);
end
