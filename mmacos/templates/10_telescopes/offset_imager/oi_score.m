function out = oi_score(txt0, G, fields_deg, opts)
%OI_SCORE  Strict RMS WFE of an offset_imager deck over a field set.
%
%   OUT = OI_SCORE(TXT0, G, FIELDS_DEG) scores the deck text TXT0 (from
%   OI_DECK) at the K x 2 field list FIELDS_DEG = [XAN YAN] (deg, CODE V
%   tangent-composed).  G carries the geometry the scorer needs:
%     .stopC (3x1)  stop centre     .z_m1          M1 vertex z
%     .fpa          struct Vpt/psi  (the FROZEN reference plane)
%
%   Per field: the chief is aimed through the stop centre, the bundle is
%   traced, and the strict WFE kernel (design/src strict_refs) evaluates
%   a reference sphere centred on the SPOT CENTROID on the frozen FP,
%   anchored at the exit pupil (crossing of two aimed chiefs), piston-
%   only removal -- the rodgers3 Stage-0 metric, stated wherever a number
%   is quoted.  Chief-referenced values are returned alongside.
%
%   Name-value:
%     'aiming'   'native' (DEFAULT) -- macos.stop(2,[0 0]) drives the
%                engine's ChiefRayAiming at the Stop Reference element.
%                'newton' -- the Stage-0 hand aiming (kept as the A/B
%                reference; see challenges/rodgers3/probe_native_stop.m).
%     'anchor'   'perfield' (DEFAULT) -- the exit-pupil anchor is found
%                per field from two aimed chiefs (the Stage-0 metric;
%                use for every REPORTED number).  'center' -- one anchor
%                from the box-centre field, shared by all fields (the
%                exit pupil is the stop image, nearly field-independent;
%                ~2.5x faster).  SOLVE-loop use only.
%     'rays'     false (default) | true: also return per-field ray states
%                at every element (for layout/clearance gates).
%     'resid'    false (default) | true: also return .resid -- the
%                stacked per-ray centroid-rung residual wavefronts (nm,
%                piston removed per field, FIXED length = K*(nRays-1)
%                with zeros at dead rays) -- the true Gauss-Newton
%                residual for the solve loop (per-ray OPD is nearly
%                LINEAR in surface coefficients where a per-field RMS
%                is not; strict_rungs column 2).
%
%   OUT fields: .wfe_cen_nm, .wfe_chf_nm (K x 1), .nrays (K x 1),
%   .aim_miss (K x 1, m), .cFP (3 x K chief FP landings), .rays (K x 1
%   cell of per-element states when requested), .resid (see above).
%
%   See also OI_DECK, STRICT_REFS, OFFSET_IMAGER.

    arguments
        txt0 (1,:) char
        G struct
        fields_deg (:,2) double
        opts.aiming (1,:) char {mustBeMember(opts.aiming,{'native','newton'})} = 'native'
        opts.anchor (1,:) char {mustBeMember(opts.anchor,{'perfield','center'})} = 'perfield'
        opts.rays (1,1) logical = false
        opts.resid (1,1) logical = false
    end

    tmp = [tempname '.in'];
    cu  = onCleanup(@() delete_if_(tmp));

    Nd = G.fpa.psi(:)/norm(G.fpa.psi);  Vd = G.fpa.Vpt(:);
    K  = size(fields_deg,1);
    out = struct('wfe_cen_nm',nan(K,1), 'wfe_chf_nm',nan(K,1), ...
                 'nrays',zeros(K,1), 'aim_miss',nan(K,1), ...
                 'cFP',nan(3,K), 'rays',{cell(K,1)}, 'fields_deg',fields_deg, ...
                 'resid',[], 'chief_dir',nan(3,K));
    rcell = cell(K,1);

    % shared exit-pupil anchor (solve-loop fast path)
    Xshared = [];
    if strcmp(opts.anchor,'center')
        fc = mean([min(fields_deg,[],1); max(fields_deg,[],1)], 1);
        dc = tancomp_(fc(1), fc(2));
        sc = aimed_trace_(txt0, tmp, G, dc, opts.aiming);
        bx = asin(dc(1));  by = asin(dc(2));
        dp = [sin(bx+1e-5); sin(by); sqrt(1-sin(bx+1e-5)^2-sin(by)^2)];
        sp = aimed_trace_(txt0, tmp, G, dp, opts.aiming);
        if ~isempty(sc) && ~isempty(sp)
            Xshared = fex_cross_(sc.pos(:,1), sc.dir(:,1), sp.pos(:,1), sp.dir(:,1));
        end
    end

    for q = 1:K
        dq = tancomp_(fields_deg(q,1), fields_deg(q,2));
        [sq, missq] = aimed_trace_(txt0, tmp, G, dq, opts.aiming, ...
                                   strcmp(opts.anchor,'perfield'));
        if isempty(sq), continue; end
        ok = sq.ok;  ok(1) = false;
        if nnz(ok) < 10, continue; end

        if isempty(Xshared)
            % probe chief (x-angle offset) for the exit-pupil anchor
            bx = asin(dq(1));  by = asin(dq(2));
            dp = [sin(bx+1e-5); sin(by); sqrt(1-sin(bx+1e-5)^2-sin(by)^2)];
            sp = aimed_trace_(txt0, tmp, G, dp, opts.aiming);
            if isempty(sp), continue; end
            X  = fex_cross_(sq.pos(:,1), sq.dir(:,1), sp.pos(:,1), sp.dir(:,1));
        else
            X = Xshared;
        end

        if opts.resid
            [vr, W] = strict_rungs(sq.pos(:,ok), sq.dir(:,ok), sq.opl(ok), ...
                                   sq.pos(:,1), sq.dir(:,1), Vd, Nd, X);
            rf = struct('wfe_centroid', vr(2), 'wfe_chief', vr(1));
            w2 = W(:,2) - mean(W(:,2));            % piston out, metres
            rfull = zeros(numel(ok)-1, 1);
            rfull(ok(2:end)) = w2*1e9;             % nm, by ray id
            rcell{q} = rfull;
        else
            rf = strict_refs(sq.pos(:,ok), sq.dir(:,ok), sq.opl(ok), ...
                             sq.pos(:,1), sq.dir(:,1), Vd, Nd, X);
        end
        out.wfe_cen_nm(q) = rf.wfe_centroid*1e9;
        out.wfe_chf_nm(q) = rf.wfe_chief*1e9;
        out.nrays(q)   = nnz(ok);
        out.aim_miss(q) = missq;
        p1 = sq.pos(:,1);  d1 = sq.dir(:,1);
        out.cFP(:,q) = p1 + d1*(dot(Nd, Vd - p1)/dot(Nd, d1));
        out.chief_dir(:,q) = d1;         % post-M3 chief (exit) direction

        if opts.rays
            out.rays{q} = per_elt_states_(txt0, tmp, G, dq, opts.aiming);
        end
    end
    if opts.resid
        nper = cellfun(@numel, rcell);
        len = max([nper; 0]);
        for q = 1:K                       % a lost FIELD becomes a wall row
            if isempty(rcell{q}), rcell{q} = 1e9*ones(max(len,1),1); end
        end
        out.resid = vertcat(rcell{:});
    end
end

% =========================================================================
function [st, miss] = aimed_trace_(txt0, tmp, G, cdir, aiming, want_miss)
    if nargin < 6, want_miss = false; end
    st = [];  miss = nan;
    seed = seed_pos_(G, cdir);
    switch aiming
        case 'native'
            emit_src_(txt0, tmp, seed, cdir);
            macos.load_rx(tmp);
            if ~macos.has_rx(), return; end
            macos.stop(2, [0 0]);
            nE = macos.num_elt();
            tr = macos.trace(nE);
            ri = macos.get_ray_info(tr.nRays);
            st = struct('pos',ri.pos,'dir',ri.dir,'opl',ri.opl, ...
                        'ok', ri.ok_trace(:) & ri.ok_pass(:));
            if want_miss
                tr2 = macos.trace(2);
                r2  = macos.get_ray_info(tr2.nRays);
                miss = norm(r2.pos(1:2,1) - G.stopC(1:2));
            end
        case 'newton'
            [p0, aim] = aim_newton_(txt0, tmp, G, cdir, seed);
            emit_src_(txt0, tmp, p0, cdir);
            macos.load_rx(tmp);
            if ~macos.has_rx(), return; end
            nE = macos.num_elt();
            tr = macos.trace(nE);
            ri = macos.get_ray_info(tr.nRays);
            st = struct('pos',ri.pos,'dir',ri.dir,'opl',ri.opl, ...
                        'ok', ri.ok_trace(:) & ri.ok_pass(:));
            miss = aim.miss;
    end
end

function E = per_elt_states_(txt0, tmp, G, cdir, aiming)
%PER_ELT_STATES_  Ray states at each element (layout/clearance gates).
    seed = seed_pos_(G, cdir);
    emit_src_(txt0, tmp, seed, cdir);
    macos.load_rx(tmp);
    if strcmp(aiming,'native'), macos.stop(2,[0 0]); end
    nE = macos.num_elt();
    E = cell(nE,1);
    for ie = 1:nE
        tr = macos.trace(ie);
        ri = macos.get_ray_info(tr.nRays);
        E{ie} = struct('pos',ri.pos,'dir',ri.dir, ...
                       'ok', ri.ok_trace(:) & ri.ok_pass(:));
    end
end

function p = seed_pos_(G, cdir)
%SEED_POS_  Crude geometric chief seed (the Stage-0 constructor): image
%   the stop centre back through a flat-M1 approximation, then stand off.
    cdR = [cdir(1); cdir(2); -cdir(3)];
    tq  = (G.z_m1 - G.stopC(3))/cdir(3);
    q   = G.stopC - tq*cdR;
    p   = q - (0.75/cdir(3))*cdir;
end

function [p0, aim] = aim_newton_(txt0, tmp, G, cdir, seed)
    p0 = seed;
    h = 1e-4;  tol = 1e-9;  aim = struct('niter',0,'miss',inf);
    r0 = stop_miss_(txt0, tmp, G, p0, cdir);
    if norm(r0) >= tol
        rx = stop_miss_(txt0, tmp, G, p0+[h;0;0], cdir);
        ry = stop_miss_(txt0, tmp, G, p0+[0;h;0], cdir);
        J  = [(rx-r0)/h, (ry-r0)/h];
        for it = 1:8
            dp = -J\r0;
            p0 = p0 + [dp(1); dp(2); 0];
            r0 = stop_miss_(txt0, tmp, G, p0, cdir);
            aim.niter = it;
            if norm(r0) < tol, break; end
        end
    end
    aim.miss = norm(r0);
end

function r = stop_miss_(txt0, tmp, G, p0, cdir)
%STOP_MISS_  Chief crossing at the stop plane vs the stop centre, from
%   the post-M1 state (no powered optics between M1 and the stop plane).
    emit_src_(txt0, tmp, p0, cdir);
    macos.load_rx(tmp);
    tr = macos.trace(1);
    ri = macos.get_ray_info(tr.nRays);
    p = ri.pos(:,1);  d = ri.dir(:,1);
    t = (G.stopC(3) - p(3))/d(3);
    q = p + d*t;
    r = q(1:2) - G.stopC(1:2);
end

function emit_src_(txt0, tmp, p0, cdir)
    v3 = @(v) sprintf('%.16E  %.16E  %.16E', v(1), v(2), v(3));
    s = regexprep(txt0, '(ChfRayDir=\s*)[^\n]*', ['$1' v3(cdir)]);
    s = regexprep(s,    '(ChfRayPos=\s*)[^\n]*', ['$1' v3(p0)]);
    fid = fopen(tmp,'w');  fprintf(fid,'%s',s);  fclose(fid);
end

function d = tancomp_(xan_deg, yan_deg)
    d = [tand(xan_deg); tand(yan_deg); 1];
    d = d/norm(d);
end

function X = fex_cross_(p1,d1,p2,d2)
    d1 = d1/norm(d1);  d2 = d2/norm(d2);
    w0 = p1 - p2;  b = dot(d1,d2);  den = 1 - b^2;
    if abs(den) < 1e-14, X = p1; return; end
    s1 = ( b*dot(d2,w0) - dot(d1,w0)) / den;
    s2 = ( dot(d2,w0) - b*dot(d1,w0)) / den;
    X  = 0.5*((p1 + d1*s1) + (p2 + d2*s2));
end

function delete_if_(p), if exist(p,'file'), delete(p); end, end
