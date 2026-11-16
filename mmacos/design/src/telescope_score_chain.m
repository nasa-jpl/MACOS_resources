function R = telescope_score_chain(G, opts)
%TELESCOPE_SCORE_CHAIN  The telescope scored at the slit on the exact chain.
%   R = telescope_score_chain(G, 'nfield', 9, 'nring', 4, 'pixel_m', 18e-6,
%   'slit_px', 2) launches, per field angle across the cross-track field, a
%   collimated disc of G.src.D_src through the chain (chief through the
%   stop's vertex) and reads the rays on the SLIT plane in the slit frame:
%   u = (p - slit).x / px along the slit, v = (p - slit).y / px across it.
%   Per field: centroid (u_c, v_c), rms widths (su, sv), the fraction inside
%   one pixel about the centroid (ee1) and inside the SLIT (|v| <= slit_px/2
%   about the slit line, |u - u_c| <= 0.5: what the spectrometer admits),
%   the chief's arrival direction, its angle to the slit normal (the
%   TELECENTRICITY) and to the spectrometer's own aim at that slit point
%   (the PUPIL MATCH), the latter times the apparent pupil distance = the
%   chief's walk on the grating in mm; the best-focus offset along the slit
%   normal (field flatness: a quadratic fit of the spot variance in the
%   defocus, closed form) and the rms there; and the mapping u_c(theta) --
%   the local IFOV per pixel between neighbouring fields against the design
%   IFOV, and the end-field landing against the slit's ends.
    arguments
        G struct
        opts.nfield (1,1) double = 9
        opts.nring (1,1) double = 4
        opts.pixel_m (1,1) double = 18e-6
        opts.slit_px (1,1) double = 2
        opts.fields (1,:) double = []
        opts.ifov (1,1) double = NaN          % design IFOV (rad / pixel); default fov / (slit length / pixel)
        opts.slit_len (1,1) double = NaN      % slit length (m); default from G.dyson
    end
    px = opts.pixel_m;
    fields = opts.fields;  if isempty(fields), fields = linspace(-G.src.fov/2, G.src.fov/2, opts.nfield); end
    nf = numel(fields);
    W = opts.slit_len;
    if isnan(W)
        if isfield(G, 'dyson'), W = G.dyson.P.npix(1)*G.dyson.P.pixel_m; else, W = G.src.fov*G.P.f; end
    end
    ifov = opts.ifov;  if isnan(ifov), ifov = G.src.fov/(W/px); end
    B = G.bundle('fields', fields, 'nlam', 1, 'nring', opts.nring);
    kS = numel(G.surf);                     % the slit is the last surface of the telescope chain
    if isfield(G, 'iSlit'), kS = G.iSlit; end
    xh = [1;0;0];  yh = [0;1;0];  zh = [0;0;1];
    c = G.slit(:);
    [U, V, SU, SV, EE1, SLIT, TEL, ERR, WALK, ZBF, SBF, NR] = deal(nan(1, nf));
    CHD = nan(3, nf);
    for i = 1:nf
        m = B.meta(:, 1) == fields(i);
        p = squeeze(B.P(:, m, kS+1));        % hits on the slit plane (station kS)
        d = squeeze(B.D(:, m, kS));          % B.D(:, :, k+1) is the direction AFTER surface k, so (:, :, kS) arrives at the slit
        if size(p, 2) < 3, continue; end
        u = ((p - c)'*xh)'/px;  v = ((p - c)'*yh)'/px;
        NR(i) = numel(u);
        U(i) = mean(u);  V(i) = mean(v);  SU(i) = std(u);  SV(i) = std(v);
        EE1(i) = mean(abs(u - U(i)) <= 0.5 & abs(v - V(i)) <= 0.5);
        SLIT(i) = mean(abs(v) <= opts.slit_px/2 & abs(u - U(i)) <= 0.5);
        % the chief = the first ray of the field (chain_bundle launches the disc centre first)
        ich = find(B.meta(m, 3) == 1, 1);
        if ~isempty(ich)
            dch = d(:, ich)/norm(d(:, ich));  CHD(:, i) = dch;
            TEL(i) = acos(abs(dch'*zh));                              % to the slit normal
            if abs(U(i)*px) <= 0.6*W                                 % the Dyson's aim is defined along its slit
                dreq = G.req_dir(U(i)*px);  dreq = dreq(:)/norm(dreq);
                ERR(i) = acos(min(1, max(-1, dch'*dreq)));            % to the spectrometer's aim
            end
            % the pupil-match NUMBER: send the telescope's chief from where it
            % lands into the spectrometer and read how far from the grating's
            % vertex it strikes (the Dyson's own exact chain)
            WALK(i) = NaN;
            if isfield(G, 'dyson') && isfield(G.dyson, 'trace')
                pch = p(:, ich);
                [pd, ~, okd] = G.dyson.trace(pch, dch, G.dyson.lambda_c);
                if okd, WALK(i) = norm(pd(:, G.dyson.iG) - G.dyson.vptG(:)); end
            end
        end
        % best focus along the slit normal: q(dz) = p + dz * d/dz_component; variance quadratic in dz
        mxy = [d(1, :)./d(3, :); d(2, :)./d(3, :)];  pxy = [u; v]*px;
        pm = mean(pxy, 2);  mm = mean(mxy, 2);
        C1 = sum(sum((pxy - pm).*(mxy - mm)));  C2 = sum(sum((mxy - mm).^2));
        if C2 > 0
            dz = -C1/C2;  ZBF(i) = dz;
            q = pxy + dz*mxy;  SBF(i) = sqrt(mean(sum((q - mean(q, 2)).^2, 1)))/px;
        end
    end
    R.B = B;                                               % the bundle (the clearance wall reuses it)
    R.fields = fields;  R.U = U;  R.V = V;  R.SU = SU;  R.SV = SV;  R.EE1 = EE1;  R.SLIT = SLIT;
    R.tel_rad = TEL;  R.err_rad = ERR;  R.walk_m = WALK;  R.zbf_m = ZBF;  R.sbf_px = SBF;  R.nrays = NR;
    R.chief_dir = CHD;  R.pixel_m = px;  R.slit_px = opts.slit_px;
    R.S = hypot(SU, SV);                                   % rms radius, px
    % mapping: u_c vs theta; local IFOV between neighbours vs the design IFOV
    du = diff(U);  dth = diff(fields);
    R.ifov_local = dth./du;                                % rad per pixel between neighbouring fields
    R.ifov = ifov;  R.ifov_ratio = R.ifov_local/ifov;
    pf = polyfit(fields, U, 1);  R.efl_fit_m = pf(1)*px;   % px per rad -> m per rad
    R.efl_target_m = W/G.src.fov;
    R.map_lin = (U - polyval(pf, fields))*px;              % departure from the linear (f-theta) map, m
    R.end_err_m = [U(1), U(end)]*px - [-W/2, W/2];          % end fields vs the slit ends
    R.W = W;
    R.headline = struct('s_max_px', max(R.S), 'su_max', max(SU), 'sv_max', max(SV), 'ee1_min', min(EE1), 'slit_min', min(SLIT), ...
                        'tel_max_deg', max(TEL)*180/pi, 'err_max_deg', max(ERR)*180/pi, 'walk_max_mm', max(WALK)*1e3, ...
                        'flat_pv_um', (max(ZBF) - min(ZBF))*1e6, 'ifov_ratio_range', [min(R.ifov_ratio) max(R.ifov_ratio)], ...
                        'end_err_px', R.end_err_m/px, 'map_lin_pv_px', (max(R.map_lin) - min(R.map_lin))/px, 'nrays_min', min(NR));
end
