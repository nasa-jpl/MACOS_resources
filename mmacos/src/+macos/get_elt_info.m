function info = get_elt_info(k)
%MACOS.GET_ELT_INFO  Element metadata for layout drawing.
%   info = macos.get_elt_info(K) queries element K of the loaded Rx:
%     .elt_id     engine EltID code (see .type for the name)
%     .type       element type name ('Reflector', 'Refractor', ...)
%     .ap_type    aperture type code (0 None, 1 Circular, 2 Elliptical,
%                 6 Hexagonal, 7 Polygonal, 8 Tapered_Polygonal, ...)
%     .ap_vec     6x1 ApVec as stored (Polygonal: (1:2) = polygon
%                 centroid in the aperture frame, (3) = vertex count)
%     .x_obs      3x1 aperture-frame x axis (global)
%     .lmon       surface half-size lMon (0 if unset)
%     .poly       2xN aperture polygon vertices when ApType is
%                 Polygonal/Tapered -- APERTURE-FRAME, CENTROID-RELATIVE
%                 (the engine's projected PolyApVtx); [] otherwise
%
%   See also: macos.view_rx, macos.get_elt_kc, macos.get_elt_kr.
arguments
    k (1,1) double {mustBeInteger, mustBePositive}
end
ms = 128;                                  % engine mPolySide
[eid, apt, apv, xo, lm, np, pv] = mmacos('elt_info_get', ...
    zeros(6,1), zeros(3,1), zeros(2,ms), double(k), double(ms));
names = {'Reflector','FocalPlane','Reference','HOE','Grating', ...
         'Refractor','Obscuring','Return','NSRefractor','LensArray', ...
         'Segment','NSReflector','TrGrating','RfPolarizer', ...
         'TrPolarizer','CGHNullPlate','DoeTrGrating','WavePlate'};
nm = 'Unknown';
if eid >= 1 && eid <= numel(names), nm = names{eid}; end
info = struct('elt_id', eid, 'type', nm, 'ap_type', apt, ...
              'ap_vec', apv(:), 'x_obs', xo(:), 'lmon', lm, ...
              'poly', pv(:, 1:np));
end
