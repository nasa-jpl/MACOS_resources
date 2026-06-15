function p = first_order_properties(srf)
%MACOS.FIRST_ORDER_PROPERTIES  First-order system properties (SYSPROP).
%   p = macos.first_order_properties() returns a struct of the system's
%   first-order / diffraction properties at the focal plane (default =
%   last element).  p = macos.first_order_properties(SRF) uses element
%   SRF.  This is the SAME engine computation the interactive SYSPROP
%   command prints and the design layer can target (e.g. an RC with
%   system f/D = 22.5 -> minimise p.fno - 22.5).
%
%   Runs the engine EFL analysis (chief + marginal ray; the source must
%   be at infinity).  The pixel-based fields (lamD_px, plate_arcsec_px,
%   plate_px_rad, dx_focal_baseunits) require a prior propagation to SRF
%   (e.g. macos.intensity(SRF)); they are 0 otherwise.
%
%   Struct fields:
%     efl_baseunits      effective focal length (BaseUnits)
%     fno                F-number (EFL / entrance-pupil diameter)
%     dpup_m             entrance-pupil diameter (metres)
%     obscuration        central obscuration ratio
%     lambda_m           wavelength (metres)
%     lamD_rad           lambda/D (radians); also the source-tilt offset
%                        for 1 lambda/D ("planet" placement)
%     lamD_arcsec        lambda/D (arcsec)
%     lamD_px            lambda/D (detector pixels; 0 pre-INT)
%     plate_arcsec_px    plate scale (arcsec/pixel; 0 pre-INT)
%     plate_px_rad       source tilt -> focal shift (px/rad; 0 pre-INT)
%     nyquist_baseunits  Nyquist focal sampling (BaseUnits)
%     dx_focal_baseunits detector pixel pitch (BaseUnits; 0 pre-INT)
arguments
    srf double = []
end
if isempty(srf), srf = macos.num_elt(); end

[efl, fno, dpup_m, obsc, lambda_m, lamD_rad, lamD_arcsec, lamD_px, ...
 plate_arcsec_px, plate_px_rad, nyquist_bu, dx_focal_bu] = ...
    mmacos('sysprop', double(srf));

p = struct( ...
    'efl_baseunits',      efl, ...
    'fno',                fno, ...
    'dpup_m',             dpup_m, ...
    'obscuration',        obsc, ...
    'lambda_m',           lambda_m, ...
    'lamD_rad',           lamD_rad, ...
    'lamD_arcsec',        lamD_arcsec, ...
    'lamD_px',            lamD_px, ...
    'plate_arcsec_px',    plate_arcsec_px, ...
    'plate_px_rad',       plate_px_rad, ...
    'nyquist_baseunits',  nyquist_bu, ...
    'dx_focal_baseunits', dx_focal_bu);
end
