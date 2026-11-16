function R = spectrometer_score_chain(G, P, opts)
%SPECTROMETER_SCORE_CHAIN  The same spectrometer metrics from the exact chain.
%   R = spectrometer_score_chain(G, P) scores the spectrometer_geom chain G
%   by its OWN exact tracer (no engine), with the same FPA frame, grid and
%   definitions as spectrometer_score, over a ring-sampled cone
%   (opts.nring, default 6 -> 169 rays).  Its purpose is the groove-model
%   comparison: build G with P.grating_model = 'planes' (straight-ruled,
%   constant period along the chord -- the classical concave grating) and
%   this is the design's prediction; build it with 'surface' and it must
%   reproduce the engine's numbers (the engine holds the period constant
%   along the curved surface).  Returns the same fields as
%   spectrometer_score (U, V, SU, SV, EE, SRF, CRF, smile/keystone ...)
%   minus the per-ray cell arrays and the radiometric chain.
    arguments
        G struct
        P struct
        opts.nx (1,1) double = 7
        opts.nlam (1,1) double = 7
        opts.nring (1,1) double = 6
    end
    W = P.npix(1)*P.pixel_m;
    xs = linspace(-W/2, W/2, opts.nx);  lams = linspace(P.band_m(1), P.band_m(2), opts.nlam);
    nx = numel(xs);  nl = numel(lams);  px = P.pixel_m;
    c = G.fpa.center(:);  xh = G.fpa.xhat(:);  yh = G.fpa.yhat(:);
    dirs0 = G.cone(G.src.u, opts.nring);
    U = nan(nx, nl);  V = U;  SU = U;  SV = U;  EE = U;  SRF = U;  CRF = U;
    for i = 1:nx
        slit = G.slit + [xs(i); 0; 0];
        for j = 1:nl
            lam = lams(j);  d0 = G.aim(slit, lam);
            ez = d0;  ex = cross([0;1;0], ez);  ex = ex/norm(ex);  ey = cross(ez, ex);
            dd = ex*dirs0(1,:) + ey*dirs0(2,:) + ez*dirs0(3,:);
            q = nan(3, size(dd,2));
            for k = 1:size(dd,2)
                [pk, ~, ok] = G.trace(slit, dd(:,k), lam);
                if ok, q(:,k) = pk(:,end); end
            end
            q = q(:, all(~isnan(q), 1));
            if size(q,2) < 10, continue; end
            u = ((q - c)'*xh)'/px;  v = ((q - c)'*yh)'/px;
            U(i,j) = mean(u);  V(i,j) = mean(v);  SU(i,j) = std(u);  SV(i,j) = std(v);
            EE(i,j) = mean(abs(u - U(i,j)) <= 0.5 & abs(v - V(i,j)) <= 0.5);
            a_px = lam*P.Fno/px;
            SRF(i,j) = spectrometer_score_fwhm(v - V(i,j), P.slit_px, a_px);
            CRF(i,j) = spectrometer_score_fwhm(u - U(i,j), 0, a_px);
        end
    end
    R.xs = xs;  R.lams = lams;  R.U = U;  R.V = V;  R.SU = SU;  R.SV = SV;  R.EE = EE;
    R.SRF = SRF;  R.CRF = CRF;
    R.smile_px = max(V, [], 1) - min(V, [], 1);  R.keystone_px = max(U, [], 2) - min(U, [], 2);
    R.smile_max = max(R.smile_px);  R.keystone_max = max(R.keystone_px);
    R.srf_max = max(SRF(:));  R.crf_max = max(CRF(:));  R.ee_min = min(EE(:));
    R.srf_var_field = max((max(SRF,[],1) - min(SRF,[],1)) ./ mean(SRF,1));
    R.crf_var_lambda = max((max(CRF,[],2) - min(CRF,[],2)) ./ mean(CRF,2));
    R.model = G.grating.model;
end
