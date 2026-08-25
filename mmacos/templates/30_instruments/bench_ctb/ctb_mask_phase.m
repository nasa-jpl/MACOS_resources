function V = ctb_mask_phase(N, dx_f, lamD_m, kind, p)
%CTB_MASK_PHASE  Complex focal-plane PHASE mask (Roddier & Roddier / dual-zone).
%   V = CTB_MASK_PHASE(N, DX_F, LAMD_M, KIND, P) returns an N-by-N COMPLEX
%   focal-plane mask (transmission exp(i*phi(r)), zone edges gray) for the
%   phase-mask coronagraph families, centred on the beam pixel floor(N/2)
%   (0-based) = 1-based N/2+1 (the FFT DC pixel where the focus lands).
%   Applied to the complex field via macos.apodize_complex.
%
%   KIND = 'roddier'  -- Roddier & Roddier 1997 phase mask (PASP 109, 815):
%     a circular focal spot of radius rho0 (lambda/D) that imposes a PI
%     phase shift; unit transmission everywhere.
%         phi(r) = pi  for r <  rho0*lamD_m ,   0  otherwise
%     OPTIMAL rho0 = 0.53 lambda/D (verbatim R&R: "encircles 50% of the
%     energy of the Airy pattern"), which flux-balances the pi-shifted core
%     against the outer field for on-axis destructive interference at the
%     Lyot plane.  R&R use NO pupil apodizer -- the extinction is by flux
%     balance (the prolate-apodized RRPM/ARPM is the LATER Aime/Soummer
%     work, not R&R 1997).  P.rho0_lamD (default 0.53).
%
%   KIND = 'dualzone' -- achromatic dual-zone phase mask (N'Diaye et al.
%     2012, A&A 538, A55 = arXiv:1111.3194; Soummer, Dohlen & Aime 2003):
%     an inner phase disk (diameter d1) and an outer phase ring (diameter
%     d2), BOTH PURE PHASE (unit transmission), neither pi:
%         phi(r) = phi1   for r <  d1/2
%                = phi2   for d1/2 <= r < d2/2
%                = 0      for r >= d2/2
%     The phase steps are specified as OPDs that SLIDE with wavelength
%     (phi = 2*pi*OPD*lambda0/lambda) -- the achromatization mechanism.
%     N'Diaye 25%-band reference (Table 1): d1=0.874, d2=1.445 (DIAMETERS
%     in lambda0/D); OPD1=0.309*lambda0, OPD2=0.678*lambda0 -> at lambda0
%     phi1=1.94 rad (0.62 pi), phi2=4.26 rad (1.36 pi).  For a full DZPM the
%     amplitude apodization lives in the ENTRANCE PUPIL (N'Diaye Eqs. 8-9),
%     not the mask; this builder emits the MASK only (the pupil apodizer is
%     a separate factor a driver may add).  P fields: d1_lamD (0.874),
%     d2_lamD (1.445), phi1 (1.94), phi2 (4.26); at lambda0 pass phi
%     directly, or pass opd1/opd2 (lambda0 units) + lam_ratio=lambda0/lambda
%     for the chromatic slide.
%
%   Args:
%     N       grid size.
%     dx_f    focal-plane pixel pitch (m) -- deterministic Fraunhofer pitch.
%     lamD_m  lambda/D at the mask plane, in METRES.
%     kind    'roddier' | 'dualzone'.
%     p       parameter struct (see per-kind fields above); [] for defaults.
%
%   Zone-edge pixels carry the pixel-averaged complex transmittance
%   (8x supersampled area fractions), not a hard phase step -- the same
%   generate-high-and-bin rule as ctb_mask_vortex.
%
%   See also: ctb_phase_masks, macos.apodize_complex, ctb_vortex.
    if nargin < 5 || isempty(p), p = struct(); end
    % Zone edges are GRAY: each phase disk is the K=8 supersampled
    % area-fraction disk (ctb_mask_disk), and the mask is composed as
    %   V = 1 + (e^{i*phi}-1) * D(r)
    % per zone -- the pixel-averaged COMPLEX transmittance of the hard
    % phase step (generate-at-8x-and-bin, without the 8x grid).
    % Interior pixels have D=1, so the zone phases are exact.

    switch lower(kind)
        case 'roddier'
            rho0 = getdef_(p,'rho0_lamD',0.53);
            D = ctb_mask_disk(N, dx_f, rho0*lamD_m, 8);
            V = 1 + (exp(1i*pi) - 1) * D;
            return
        case 'dualzone'
            d1 = getdef_(p,'d1_lamD',0.874);             % DIAMETERS (lambda0/D)
            d2 = getdef_(p,'d2_lamD',1.445);
            lam_ratio = getdef_(p,'lam_ratio',1.0);      % lambda0/lambda
            if isfield(p,'opd1') || isfield(p,'opd2')
                phi1 = 2*pi*getdef_(p,'opd1',0.309)*lam_ratio;   % OPD slide
                phi2 = 2*pi*getdef_(p,'opd2',0.678)*lam_ratio;
            else
                phi1 = getdef_(p,'phi1',1.94) * lam_ratio;
                phi2 = getdef_(p,'phi2',4.26) * lam_ratio;
            end
            D1 = ctb_mask_disk(N, dx_f, (d1/2)*lamD_m, 8);
            D2 = ctb_mask_disk(N, dx_f, (d2/2)*lamD_m, 8);
            V = 1 + (exp(1i*phi2) - 1) * D2 + (exp(1i*phi1) - exp(1i*phi2)) * D1;
            return
        otherwise
            error('ctb_mask_phase:kind','unknown kind ''%s''',kind);
    end
end

function v = getdef_(s, f, d)
    if isfield(s,f) && ~isempty(s.(f)), v = s.(f); else, v = d; end
end
