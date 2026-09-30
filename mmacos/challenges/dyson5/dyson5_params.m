function P = dyson5_params(over)
%DYSON5_PARAMS  Single source of truth for the dyson5 challenge runner.
%
%   P = DYSON5_PARAMS() returns the default parameter set: Joe's EMIT-class
%   VSWIR imaging-spectrometer spec (dyson5_guidance.txt), the fixed problem
%   this challenge is scored against.  P = DYSON5_PARAMS(OVER) applies the
%   fields of struct OVER on top (the offset_imager pattern) -- how another
%   instrument is run without touching this file:
%
%       P = dyson5_params(struct('Fno',2.2,'band_m',[2040 2380]*1e-9));
%
%   Every stage of DYSON5_RUN reads ONLY this struct.  Fields:
%
%   The spec (Joe, 2026-09; recorded verbatim, EMIT to the digit)
%     Fno          image-space (air-equivalent) F-number           1.8
%     npix         [spatial spectral] detector format             [3000 500]
%     pixel_m      pixel pitch, m                                 18e-6
%     band_m       [lambda_min lambda_max], m                     [380 2500]e-9
%     smile_px     smile requirement, px (0.2 "might be ok")      0.1
%     keystone_px  keystone requirement, px                       0.1
%     srf_px       SRF FWHM requirement band, px                  [1.5 2.0]
%     xrf_px       XRF FWHM requirement, px                       1.5
%     slit_px      slit width in pixels (Jim: 2-px slits common)  2
%   Jim's realism, recorded alongside the spec (not scored against):
%     as-built SRF 2.5-3 px; photon-limited, not diffraction-limited;
%     smile/keystone are the drivers; the GRATING is the stop.
%
%   Derived (computed by the stages, never typed): slit length = npix(1) *
%   pixel_m = 54 mm; FPA spectral height = npix(2)*pixel_m = 9 mm; spectral
%   sampling = (band span)/npix(2) = 4.24 nm/px.
%
%   Form
%     glass        block material (engine GlassElt name)          'Silica'
%     lambda_ref_m index evaluation wavelength for the layout, m  1.0e-6
%     y_offset_m   slit centre offset from the Dyson axis along
%                  the dispersion direction, m (the FPA sits at
%                  -y; the two must clear each other physically)  8e-3
%     blur_px      layout blur budget: transverse rms spot at the
%                  slit CORNER, in pixels, that the concentric seed
%                  must meet before any element is added           0.25
%
%   Numerics / bookkeeping
%     r_grid_m     block radii swept by the s0 scaling stage, m
%     model        MACOS model size (engine stages)               128
%     tag, outdir  artifact naming; outdir '' = this directory
%     stages       which stages to run (default {'s0'})
    arguments
        over struct = struct()
    end
    P.Fno         = 1.8;
    P.npix        = [3000 500];
    P.pixel_m     = 18e-6;
    P.band_m      = [380e-9 2500e-9];
    P.smile_px    = 0.1;
    P.keystone_px = 0.1;
    P.srf_px      = [1.5 2.0];
    P.xrf_px      = 1.5;
    P.slit_px     = 2;

    P.glass        = 'Silica';
    P.lambda_ref_m = 1.0e-6;
    P.y_offset_m   = 8e-3;
    P.blur_px      = 0.25;

    P.r_grid_m = [0.05 0.075 0.10 0.15 0.20 0.25 0.30 0.40 0.50];
    P.model    = 128;
    P.tag      = 'dyson5';
    P.outdir   = '';
    P.stages   = {'s0'};

    f = fieldnames(over);
    for k = 1:numel(f)
        if ~isfield(P, f{k})
            error('dyson5_params:unknown', 'unknown parameter "%s"', f{k});
        end
        P.(f{k}) = over.(f{k});
    end
end
