function I = compose(srf, wavelengths, npix, dx, opts)
%MACOS.COMPOSE  Multi-wavelength PSF on a fixed pixel grid (COMPOSE+ADD).
%   I = macos.compose(SRF, WAVELENGTHS, NPIX, DX) assembles a composite
%   image at element SRF: for each wavelength in WAVELENGTHS the source
%   wavelength is set, the field is propagated to SRF, and its intensity
%   is accumulated onto a single NPIX x NPIX detector grid of pitch DX.
%   The result is the INCOHERENT sum -- a broadband PSF.
%
%   Each wavelength has a different native diffraction sampling (focal dx
%   is proportional to lambda); COMPOSE resamples each onto the common
%   pixel grid, so this is the right primitive for broadband scoring
%   rather than summing raw intensity() arrays.
%
%   Name-value:
%     'dx_unit'  'm' (default) | 'mm' | 'cm' | 'um' | 'native'
%                units of DX ('native' = prescription BaseUnits).
%
%   Only incoherent intensity composition is supported (the engine's
%   coherent 'CADD' path is unimplemented).
%
%   Example:
%     macos.load_rx('Rx_Coro_FPM.in');
%     psf = macos.compose(21, [0.80 0.85 0.90], 128, 5e-6);  % 5 um pixels
arguments
    srf          (1,1) double {mustBeInteger, mustBePositive}
    wavelengths  (1,:) double {mustBePositive}
    npix         (1,1) double {mustBeInteger, mustBePositive}
    dx           (1,1) double {mustBePositive}
    opts.dx_unit (1,:) char {mustBeMember(opts.dx_unit, ...
        {'m','mm','cm','um','native'})} = 'm'
end

% Engine COMPOSE wants the pixel size in BaseUnits; convert from the
% requested unit via CBM (metres-per-BaseUnit).
switch opts.dx_unit
    case 'native'
        dxpix = dx;
    otherwise
        to_m = struct('m',1.0,'mm',1e-3,'cm',1e-2,'um',1e-6);
        c = mmacos('base_unit_to_metres');
        if c == 0.0
            error('macos:compose:noCBM', 'CBM unavailable.');
        end
        dxpix = dx * to_m.(opts.dx_unit) / c;
end

mmacos('compose_start', double(srf), double(npix), double(dxpix));
for k = 1:numel(wavelengths)
    mmacos('src_wvl', double(wavelengths(k)), 1);  % set wavelength
    mmacos('intensity', double(srf), 1.0);         % MODIFY + propagate to srf
    mmacos('compose_add', 0.0);                     % accumulate, no plot
end
I = mmacos('compose_get', double(npix));
end
