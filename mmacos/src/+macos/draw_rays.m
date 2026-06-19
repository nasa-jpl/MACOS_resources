function b = draw_rays(plane, iStart, iEnd)
%MACOS.DRAW_RAYS  Real DRAW ray bundle as data (no graphics device).
%   b = macos.draw_rays(PLANE, ISTART, IEND) runs the engine DRAW command
%   in DATA-ONLY mode and returns its traced ray fan projected onto PLANE
%   ('YZ' | 'XZ' | 'XY', source coords), for elements ISTART..IEND
%   (ISTART = 0 => from the source).  No window is opened, nothing renders.
%
%   Returns a struct:
%     b.U, b.V  (nMaxElt x nRay) projected coords at each surface crossing
%     b.elt     (nMaxElt x nRay) element index of each crossing
%     b.nper    (1 x nRay)       number of crossings for each ray
%     b.nray, b.plane
%   Ray r is the polyline ( b.U(1:b.nper(r),r), b.V(1:b.nper(r),r) ).
%
%   Backs macos.design.Telescope.view_layout (real-ray layout + clipping).
%   See also: macos.design.Telescope/view_layout.
    arguments
        plane  (1,:) char   = 'YZ'
        iStart (1,1) double  = 0
        iEnd   (1,1) double  = 0
    end
    key = upper(plane);
    pmap = struct('YZ',1, 'XZ',2, 'XY',3);
    if ~isfield(pmap, key)
        error('macos:draw_rays:plane', 'plane must be YZ, XZ or XY (got %s)', plane);
    end
    if iEnd <= 0, iEnd = macos.num_elt(); end

    [nDE, nRay]       = mmacos('draw_rays_cmd', pmap.(key), iStart, iEnd);
    [U, V, E, nper]   = mmacos('draw_rays_get', nDE, nRay);

    b = struct('U', U, 'V', V, 'elt', double(E), 'nper', double(nper(:).'), ...
               'nray', double(nRay), 'plane', key);
end
