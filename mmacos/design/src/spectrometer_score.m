function R = spectrometer_score(G, M, P, opts)
%SPECTROMETER_SCORE  Ray-side spectrometer metrics on an emitted deck.
%   R = spectrometer_score(G, M, P) traces the loaded deck (spectrometer_rx
%   map M of the spectrometer_geom chain G) over a (slit position x
%   wavelength) grid and scores it in PIXELS.  Every convention is stated
%   here and printed by the caller before any number:
%
%   FPA frame: origin G.fpa.center (the lambda_c image of the slit centre),
%   u = (p - c).xhat along the SLIT (spatial, +x), v = (p - c).yhat along the
%   DISPERSION (spectral, the sign the trace gives), both in pixels of
%   P.pixel_m.  Per (s, lambda): the engine's ray positions at the FPA
%   (ok_trace & ok_pass), their centroid (u_c, v_c), rms widths, and the
%   geometric ensquared fraction in one pixel centred on the centroid.
%
%   Field-angle map   U(s, lambda) = u_c ;   wavelength map  V(s, lambda) = v_c.
%   SMILE(lambda)     = max_s v_c - min_s v_c  (spectral centroid drift ALONG
%                       the slit at fixed lambda); headline = max over lambda.
%   KEYSTONE(s)       = max_lambda u_c - min_lambda u_c  (spatial centroid
%                       drift ACROSS the band at fixed s); headline = max over s.
%   SRF(s, lambda)    = FWHM of  rect(slit image, P.slit_px) (x) LSF_v (x)
%                       rect(1 px) (x) Airy-LSF(lambda) ; LSF_v = the ray
%                       distribution along v (Mouroulis & Green 2018 Sec. 4.1
%                       form; the Airy term is the incoherent approximation,
%                       legitimate while 2.44 lambda F < pixel -- 11 um vs
%                       18 um at F/1.8, 2.5 um).
%   CRF(s, lambda)    = FWHM of  LSF_u (x) rect(1 px) (x) Airy-LSF  -- the
%                       SPECTROMETER's cross-track function (the system LSF
%                       of the paper includes the telescope, not modelled).
%   RADIOMETRIC chain (closed form, per lambda): Fresnel transmission of
%   the uncoated refractive faces, scalar blaze efficiency
%   sinc^2(m (lambda_B/lambda - 1)) at P.blaze_m, QE from P.qe (a
%   placeholder table unless the caller supplies one).  The SLIT LOSS is
%   the measured term of the propagation twin and is NOT in this chain.
%
%   Options: 'nx' slit samples (7), 'nlam' wavelengths (7), 'xs' explicit
%   slit positions (m), 'lams' explicit wavelengths (m), 'quiet'.
    arguments
        G struct
        M struct
        P struct
        opts.nx (1,1) double = 7
        opts.nlam (1,1) double = 7
        opts.xs (1,:) double = []
        opts.lams (1,:) double = []
        opts.quiet (1,1) logical = false
    end
    W = P.npix(1)*P.pixel_m;
    xs   = opts.xs;    if isempty(xs),   xs   = linspace(-W/2, W/2, opts.nx); end
    lams = opts.lams;  if isempty(lams), lams = linspace(P.band_m(1), P.band_m(2), opts.nlam); end
    nx = numel(xs);  nl = numel(lams);  px = P.pixel_m;
    c  = G.fpa.center(:);  xh = G.fpa.xhat(:);  yh = G.fpa.yhat(:);
    Fno = P.Fno;

    U = nan(nx, nl);  V = U;  SU = U;  SV = U;  EE = U;  SRF = U;  CRF = U;  NR = U;
    raysU = cell(nx, nl);  raysV = cell(nx, nl);
    for i = 1:nx
        slit = G.slit + [xs(i); 0; 0];
        for j = 1:nl
            lam = lams(j);
            da = G.aim(slit, lam);
            % ChfRayPos IS the physical source point once a deck is loaded
            macos.stop(M.iG);                 % stop FIRST (its own aim is one pass short)
            macos.set_src_fov('src_pos', slit, 'src_dir', da, 'zSrc', -G.src.zsrc_gap);
            macos.set_src_wvl(lam);  macos.modify();
            s  = macos.trace(M.nElt);  ri = macos.get_ray_info(s.nRays);
            ok = ri.ok_trace & ri.ok_pass;
            p  = ri.pos(:, ok);
            u  = ((p - c)'*xh)'/px;  v = ((p - c)'*yh)'/px;
            NR(i,j) = nnz(ok);
            if nnz(ok) < 10, continue; end
            U(i,j) = mean(u);  V(i,j) = mean(v);
            SU(i,j) = std(u);  SV(i,j) = std(v);
            EE(i,j) = mean(abs(u - U(i,j)) <= 0.5 & abs(v - V(i,j)) <= 0.5);
            a_px = lam*Fno/px;                       % Airy scale lambda F in pixels
            SRF(i,j) = spectrometer_score_fwhm(v - V(i,j), P.slit_px, a_px);
            CRF(i,j) = spectrometer_score_fwhm(u - U(i,j), 0, a_px);
            raysU{i,j} = u;  raysV{i,j} = v;
        end
        if ~opts.quiet, fprintf('  spectrometer_score: slit x = %+6.2f mm done (%d lambdas)\n', xs(i)*1e3, nl); end
    end
    R.xs = xs;  R.lams = lams;  R.U = U;  R.V = V;  R.SU = SU;  R.SV = SV;  R.EE = EE;
    R.SRF = SRF;  R.CRF = CRF;  R.nrays = NR;  R.raysU = raysU;  R.raysV = raysV;
    R.smile_px    = max(V, [], 1) - min(V, [], 1);          % per lambda
    R.keystone_px = max(U, [], 2) - min(U, [], 2);          % per s
    R.smile_max = max(R.smile_px);  R.keystone_max = max(R.keystone_px);
    R.srf_max = max(SRF(:));  R.crf_max = max(CRF(:));  R.ee_min = min(EE(:));
    R.srf_var_field = max((max(SRF,[],1) - min(SRF,[],1)) ./ mean(SRF,1));   % per lambda, max
    R.crf_var_lambda = max((max(CRF,[],2) - min(CRF,[],2)) ./ mean(CRF,2));  % per s, max
    R.dispersion_px_per_nm = (V(ceil(nx/2), end) - V(ceil(nx/2), 1)) / ((lams(end) - lams(1))*1e9);

    % -- radiometric chain (closed form) ------------------------------------
    nface = nnz(strcmp({G.surf.act}, 'refract'));
    T_fres = ones(1, nl);  eta = ones(1, nl);  qe = ones(1, nl);
    for j = 1:nl
        n = G.n(lams(j));
        T_fres(j) = (1 - ((n-1)/(n+1))^2)^nface;
    end
    if isfield(P, 'blaze_m') && ~isempty(P.blaze_m)
        x = G.grating.m*(P.blaze_m./lams - 1);
        eta = (sin(pi*x)./(pi*x)).^2;  eta(x == 0) = 1;
    end
    if isfield(P, 'qe') && ~isempty(P.qe)
        qe = interp1(P.qe(:,1), P.qe(:,2), lams, 'linear', 'extrap');
    end
    R.rad = struct('lams', lams, 'T_fresnel', T_fres, 'nface', nface, 'eta_blaze', eta, 'qe', qe, ...
                   'gain', T_fres.*eta.*qe, 'note', 'slit loss = the propagation twin''s measured term, not included');
end

