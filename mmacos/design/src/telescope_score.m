function R = telescope_score(G, M, P, opts)
%TELESCOPE_SCORE  The telescope scored at the slit in the ENGINE.
%   R = telescope_score(G, M, P) drives the loaded telescope deck (map M of
%   spectrometer_rx on the telescope_geom chain G) over the cross-track
%   field and scores it at the slit plane (the terminal FocalPlane) with the
%   SAME definitions as telescope_score_chain: per field angle the engine's
%   rays at the slit (ok_trace & ok_pass) in the slit frame -- u along the
%   slit, v across it, pixels of P.pixel_m -- centroid, rms widths, the
%   fraction in one pixel about the centroid and the fraction the SLIT
%   admits (|v| <= slit_px/2, |u - u_c| <= 0.5), the chief's direction
%   against the slit normal (telecentricity) and against the spectrometer's
%   own aim at that slit point (pupil match), the chief's miss of the
%   grating vertex when sent on through the Dyson's exact chain (mm), the
%   best-focus offset along the slit normal (closed form on the rays'
%   positions and directions) and the mapping.  Launch per field: the stop
%   declared FIRST (macos.stop(M.iStop), M2's vertex), then the chain's own
%   aimed chief written as the source (set_src_fov: collimated, the chief
%   through the stop) so the engine's re-aim is a no-op -- the Dyson
%   gate's pattern.  Options: 'nfield' (9), 'fields' (rad), 'quiet'.
    arguments
        G struct
        M struct
        P struct
        opts.nfield (1,1) double = 9
        opts.fields (1,:) double = []
        opts.quiet (1,1) logical = false
    end
    px = P.pixel_m;  slit_px = P.slit_px;
    fields = opts.fields;  if isempty(fields), fields = linspace(-G.src.fov/2, G.src.fov/2, opts.nfield); end
    nf = numel(fields);
    W = P.npix(1)*px;  ifov = G.src.fov/(W/px);
    xh = [1;0;0];  yh = [0;1;0];  zh = [0;0;1];  c = G.slit(:);
    lam = G.src.lambda_c;
    [U, V, SU, SV, EE1, SLIT, TEL, ERR, WALK, ZBF, SBF, NR] = deal(nan(1, nf));
    CHD = nan(3, nf);  raysU = cell(1, nf);  raysV = cell(1, nf);
    for i = 1:nf
        d = G.field_dir(fields(i));  [p0, okA] = G.aim_pt(d, lam);
        if ~okA, continue; end
        macos.stop(M.iStop);                        % stop FIRST, then the chain's exact chief
        macos.set_src_fov('src_pos', p0, 'src_dir', d, 'zSrc', 1e22);
        macos.modify();
        s = macos.trace(M.nElt);  ri = macos.get_ray_info(s.nRays);
        ok = ri.ok_trace & ri.ok_pass;
        p = ri.pos(:, ok);  dd = ri.dir(:, ok);
        NR(i) = nnz(ok);
        if nnz(ok) < 10, continue; end
        u = ((p - c)'*xh)'/px;  v = ((p - c)'*yh)'/px;
        U(i) = mean(u);  V(i) = mean(v);  SU(i) = std(u);  SV(i) = std(v);
        EE1(i) = mean(abs(u - U(i)) <= 0.5 & abs(v - V(i)) <= 0.5);
        SLIT(i) = mean(abs(v) <= slit_px/2 & abs(u - U(i)) <= 0.5);
        raysU{i} = u;  raysV{i} = v;
        if ri.ok_trace(1) && ri.ok_pass(1)
            dch = ri.dir(:, 1)/norm(ri.dir(:, 1));  CHD(:, i) = dch;
            TEL(i) = acos(abs(dch'*zh));
            if abs(U(i)*px) <= 0.6*W
                dreq = G.req_dir(U(i)*px);  dreq = dreq(:)/norm(dreq);
                ERR(i) = acos(min(1, max(-1, dch'*dreq)));
            end
            if isfield(G, 'dyson') && isfield(G.dyson, 'trace')
                [pd, ~, okd] = G.dyson.trace(ri.pos(:, 1), dch, G.dyson.lambda_c);
                if okd, WALK(i) = norm(pd(:, G.dyson.iG) - G.dyson.vptG(:)); end
            end
        end
        mxy = [dd(1, :)./dd(3, :); dd(2, :)./dd(3, :)];  pxy = [u; v]*px;
        pm = mean(pxy, 2);  mm = mean(mxy, 2);
        C1 = sum(sum((pxy - pm).*(mxy - mm)));  C2 = sum(sum((mxy - mm).^2));
        if C2 > 0
            dz = -C1/C2;  ZBF(i) = dz;
            q = pxy + dz*mxy;  SBF(i) = sqrt(mean(sum((q - mean(q, 2)).^2, 1)))/px;
        end
        if ~opts.quiet, fprintf('  telescope_score: field %+6.2f deg: %d rays, spot %.3f px, slit %.3f, walk %.2f mm\n', fields(i)*180/pi, NR(i), hypot(SU(i), SV(i)), SLIT(i), WALK(i)*1e3); end
    end
    R.fields = fields;  R.U = U;  R.V = V;  R.SU = SU;  R.SV = SV;  R.EE1 = EE1;  R.SLIT = SLIT;
    R.tel_rad = TEL;  R.err_rad = ERR;  R.walk_m = WALK;  R.zbf_m = ZBF;  R.sbf_px = SBF;  R.nrays = NR;
    R.chief_dir = CHD;  R.pixel_m = px;  R.slit_px = slit_px;  R.raysU = raysU;  R.raysV = raysV;
    R.S = hypot(SU, SV);
    du = diff(U);  dth = diff(fields);
    R.ifov_local = dth./du;  R.ifov = ifov;  R.ifov_ratio = R.ifov_local/ifov;
    pf = polyfit(fields, U, 1);  R.efl_fit_m = pf(1)*px;  R.efl_target_m = W/G.src.fov;
    R.map_lin = (U - polyval(pf, fields))*px;
    R.end_err_m = [U(1), U(end)]*px - [-W/2, W/2];  R.W = W;
    R.headline = struct('s_max_px', max(R.S), 'su_max', max(SU), 'sv_max', max(SV), 'ee1_min', min(EE1), 'slit_min', min(SLIT), ...
                        'tel_max_deg', max(TEL)*180/pi, 'err_max_deg', max(ERR)*180/pi, 'walk_max_mm', max(WALK)*1e3, ...
                        'flat_pv_um', (max(ZBF) - min(ZBF))*1e6, 'ifov_ratio_range', [min(R.ifov_ratio) max(R.ifov_ratio)], ...
                        'end_err_px', R.end_err_m/px, 'map_lin_pv_px', (max(R.map_lin) - min(R.map_lin))/px, 'nrays_min', min(NR));
end
