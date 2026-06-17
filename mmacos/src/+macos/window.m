function window(frame, sizPix, opts)
%MACOS.WINDOW  Place diffraction images at their TRUE offset (pixel-loc ON).
%   macos.window(FRAME, SIZPIX) turns on the WINDOW pixel-location option
%   so PIX / COMPOSE place each source's image at its real sky offset
%   (via the chief-ray projection) instead of re-centred on the grid --
%   required to COMPOSE an off-axis source (e.g. a planet) at its true
%   position relative to an on-axis star.  Needs a prior trace.
%
%   FRAME selects the output coordinate frame the placement references:
%     'tout'  the prescription's output coordinate frame (Tout)
%     'beam'  the local beam frame at the output element
%   (ENTER / custom xOut,yOut frames are not exposed.)
%
%   SIZPIX is the window pixel size in BaseUnits -- match the COMPOSE
%   pixel pitch (dxpix).
%
%   Name-value:
%     'elt_pix' [x y] element reference pixel (default [0 0])
%     'win_cen' [x y] window centre pixel    (default [0 0])
%   Equal (or both [0 0]) means no fixed offset; placement comes purely
%   from each source's chief-ray offset.
%
%   See also: macos.window_off, macos.compose, macos.ffp, macos.pfp.
arguments
    frame        (1,:) char {mustBeMember(frame, {'tout','beam'})}
    sizPix       (1,1) double {mustBePositive}
    opts.elt_pix (1,2) double = [0 0]
    opts.win_cen (1,2) double = [0 0]
end
codes = struct('tout', 1, 'beam', 2);
mmacos('window_set', codes.(frame), sizPix, ...
       opts.elt_pix(1), opts.elt_pix(2), opts.win_cen(1), opts.win_cen(2));
end
