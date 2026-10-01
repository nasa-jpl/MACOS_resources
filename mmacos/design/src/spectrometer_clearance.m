function C = spectrometer_clearance(G, P, opts)
%SPECTROMETER_CLEARANCE  Does every beam leg clear every body it does not traverse?
%   C = spectrometer_clearance(G, P) traces the multi-field, multi-lambda
%   bundle through the chain G (slit centre + ends, band centre + edges,
%   chief + marginal rings) and, for every LEG (the straight segments between
%   consecutive surface hits, the slit as the first station and the FPA as
%   the last) against every BODY the leg is not an endpoint of, returns the
%   minimum clearance in mm: the smallest distance from any ray segment of
%   the leg to the body's sampled surface, minus the body's mount margin.
%   Negative = the leg passes through the body.  A leg INSIDE a box body
%   (slit mask, FPA package) is a penetration however far it sits from the
%   box's faces (-depth), and the two boxes are ALSO scored against every
%   other body (C.body_table: a package standing inside the block is an
%   interference no ray can show).  The record is a table, worst first; the
%   caller FAILS its stage on any negative entry (BRIEF_to_dyson5 addendum 6:
%   clearance is a number, not a picture).  C.min_mm is the worst of both
%   tables and the worst body pair leads C.table when it is the worst of all.
%
%   Bodies: every optical surface as its aperture disc (footprint radius +
%   P.mount_margin_m) lifted onto the real surface (the block's sphere cap,
%   the grating cap, the meniscus caps, the flat face), sampled at ~2 mm;
%   plus two MECHANICAL bodies at the Dyson face: the SLIT MASK (a plate
%   P.slit_mask_m = [length, height, thickness] centred on the slit, in the
%   plane z = slit) and the FPA PACKAGE (the active area 54 x 9 mm centred
%   on the band's image, grown by P.pkg_margin_m all round, P.pkg_depth_m
%   deep on the far side of the face, and P.pkg_shield_m tall toward the
%   block -- a cold shield / window stack).  Surfaces that are the same
%   physical part (the block's two face passes, the meniscus's two passes)
%   are grouped by name stem and never scored against their own legs.
%   Options: 'nx','nlam','nring' (bundle), 'sample_m' (2e-3), 'quiet'.
    arguments
        G struct
        P struct
        opts.nx (1,1) double = 3
        opts.nlam (1,1) double = 3
        opts.nring (1,1) double = 2
        opts.sample_m (1,1) double = 2e-3
        opts.quiet (1,1) logical = false
    end
    mount = field_(P, 'mount_margin_m', 5e-3);
    B = G.bundle('nx', opts.nx, 'nlam', opts.nlam, 'nring', opts.nring);
    F = G.footprints('nx', opts.nx, 'nlam', opts.nlam, 'nring', opts.nring);
    nS = numel(G.surf);  nR = size(B.P, 2);
    % ---- stations and legs: station 1 = slit, k+1 = surface k; the FPA is the last
    legs = {};
    for k = 1:nS
        legs{end+1} = struct('name', sprintf('%s -> %s', stname_(G, k-1), stname_(G, k)), ...
                             'a', squeeze(B.P(:, :, k)), 'b', squeeze(B.P(:, :, k+1)), 'ends', [k-1, k]);  %#ok<AGROW>
    end
    % ---- bodies
    bodies = {};
    for k = 1:nS
        S = G.surf(k);
        if strcmp(S.act, 'stop'), continue; end
        stem = stem_(S.name);
        pts = body_pts_(S, F(k), mount, opts.sample_m);
        bodies{end+1} = struct('name', S.name, 'stem', stem, 'pts', pts, 'surfs', k, 'mount', mount, 'box', []);  %#ok<AGROW>
    end
    if strcmp(G.form, 'dyson')
        sm = field_(P, 'slit_mask_m', [0.064 0.004 0.001]);
        bx = struct('c', G.slit(:), 'ex', [1;0;0], 'ey', [0;1;0], 'ez', [0;0;1], 'Lx', sm(1), 'Ly', sm(2), 'zr', [-sm(3) 0]);
        bodies{end+1} = struct('name', 'SlitMask', 'stem', 'SlitMask', 'surfs', 0, 'mount', 0, 'box', bx, ...
            'pts', box_pts_(G.slit, [1 0 0], [0 1 0], [0 0 1], sm(1), sm(2), [-sm(3) 0], opts.sample_m));
        pm = field_(P, 'pkg_margin_m', 5e-3);  pd = field_(P, 'pkg_depth_m', 10e-3);  ps = field_(P, 'pkg_shield_m', 0);
        c = G.fpa.center;
        bx = struct('c', c(:), 'ex', G.fpa.xhat(:), 'ey', G.fpa.yhat(:), 'ez', G.fpa.normal(:), 'Lx', G.fpa.W + 2*pm, 'Ly', G.fpa.H + 2*pm, 'zr', [-pd ps]);
        bodies{end+1} = struct('name', 'FPApackage', 'stem', 'FPApackage', 'surfs', nS, 'mount', 0, 'box', bx, ...
            'pts', box_pts_(c, G.fpa.xhat, G.fpa.yhat, G.fpa.normal, G.fpa.W + 2*pm, G.fpa.H + 2*pm, [-pd ps], opts.sample_m));
        % (the package box lives in the FPA's own frame: with the fold prism the
        % FPA normal is +y and its dispersion axis +z; the shield grows along
        % the normal toward the beam -- toward the prism's exit face)
    end
    % ---- the table
    rows = {};
    for L = legs
        L = L{1};
        for Bd = bodies
            Bd = Bd{1};
            % skip bodies the leg starts or ends on, and the same physical part
            skip = any(ismember(Bd.surfs, L.ends));
            for e = L.ends
                if e >= 1 && e <= nS && strcmp(stem_(G.surf(e).name), Bd.stem), skip = true; end
            end
            if strcmp(Bd.stem, 'SlitMask') && L.ends(1) == 0, skip = true; end
            if strcmp(Bd.stem, 'FPApackage') && L.ends(2) == nS, skip = true; end
            if skip, continue; end
            d = segs_to_pts_(L.a, L.b, Bd.pts) - Bd.mount;
            if ~isempty(Bd.box)                 % a leg INSIDE a box is a penetration, however far from the box's faces
                dep = inside_depth_(seg_samples_(L.a, L.b, 21), Bd.box);
                if dep > 0, d = -dep; end
            end
            rows(end+1, :) = {L.name, Bd.name, d*1e3};                     %#ok<AGROW>
        end
    end
    C.table = cell2table(rows, 'VariableNames', {'leg', 'body', 'clearance_mm'});
    C.table = sortrows(C.table, 'clearance_mm');
    % ---- the mechanical boxes against every other body (BODY vs BODY): a
    % package or mask that sits inside the glass, or against it, is an
    % interference no ray can show (the 2026-10-01 R5 package corner stood
    % 1.5 mm inside the block above the face plane while every leg cleared)
    brows = {};
    for i = 1:numel(bodies)
        Bi = bodies{i};  if isempty(Bi.box), continue; end
        for j = 1:numel(bodies)
            Bj = bodies{j};  if j == i || strcmp(Bi.stem, Bj.stem), continue; end
            % pure geometry here, no mount margin: a mask or a package is DESIGNED
            % to sit against the block's face (R4: 0.85 mm), and the mount
            % margin is a rule for rays, not for parts that touch by design
            d = pts_to_pts_(Bi.pts, Bj.pts);
            dep = inside_depth_(Bj.pts, Bi.box);                    % the other body's samples inside this box
            if ~isempty(Bj.box), dep = max(dep, inside_depth_(Bi.pts, Bj.box)); end
            if dep > 0, d = -dep; end
            brows(end+1, :) = {Bi.name, Bj.name, d*1e3};   %#ok<AGROW>
        end
    end
    C.body_table = cell2table(brows, 'VariableNames', {'body', 'other', 'clearance_mm'});
    if ~isempty(brows), C.body_table = sortrows(C.body_table, 'clearance_mm'); end
    C.min_leg_mm = min(C.table.clearance_mm);
    C.min_body_mm = min([Inf; C.body_table.clearance_mm]);
    C.min_mm = min(C.min_leg_mm, C.min_body_mm);  C.pass = C.min_mm >= 0;
    if C.min_body_mm < C.min_leg_mm                 % the worst pair, whichever table holds it, leads C.table
        C.table = [cell2table({sprintf('BODY %s', C.body_table.body{1}), C.body_table.other{1}, C.body_table.clearance_mm(1)}, ...
                              'VariableNames', {'leg', 'body', 'clearance_mm'}); C.table];
    end
    C.footprints = F;  C.bodies = bodies;  C.bundle = B;
    if ~opts.quiet
        fprintf('spectrometer_clearance (%s): %d legs x %d bodies, mount %.0f mm; worst first\n', G.form, numel(legs), numel(bodies), mount*1e3);
        for i = 1:min(12, height(C.table))
            fprintf('  %-34s vs %-16s %+9.2f mm %s\n', C.table.leg{i}, C.table.body{i}, C.table.clearance_mm(i), tern_(C.table.clearance_mm(i) < 0, '  <-- BLOCKED', ''));
        end
    end
end

function st = stem_(name)
%STEM_  One physical part behind several surface records: the block's two
%   face passes and two sphere passes (one block), the meniscus's two passes,
%   the Offner concave mirror's two zones (M1/M3 = one sphere).
    st = regexprep(name, '_(out|in)$|In$|Out$', '');
    if any(strcmp(st, {'M1', 'M3'})), st = 'ConcaveMirror'; end
    if any(strcmp(st, {'BlockFace', 'BlockSphere'})), st = 'Block'; end
    if any(strcmp(st, {'MenA', 'MenB'})), st = 'Meniscus'; end     % one plate, two faces, two passes
    if any(strcmp(st, {'Plate', 'FoldMirror', 'PrismExit'})), st = 'Block'; end   % R5: plate and prism CEMENTED to the block
end

function n = stname_(G, k)
    if k == 0, n = 'Slit'; else, n = G.surf(k).name; end
end

function v = field_(P, f, d)
    if isfield(P, f) && ~isempty(P.(f)), v = P.(f); else, v = d; end
end

function t = tern_(c, a, b), if c, t = a; else, t = b; end, end

function pts = body_pts_(S, Fk, mount, h)
%BODY_PTS_  Surface samples inside the aperture disc (footprint + mount) in
%   the aperture frame, lifted onto the surface (plane or sphere).
    if any(strcmp(S.name, {'PlateIn', 'FoldMirror', 'PrismExit'}))
        % R5's plate and prism are RECTANGULAR parts (a slab under the slit, a
        % prism under the image): the footprint's extents + mount, not the
        % enclosing disc, whose 64 mm diameter under a 54 x 0.5 mm slit strip
        % reaches 27 mm sideways into the detector package's place
        xl = Fk.xlim + [-mount mount];  yl = Fk.ylim + [-mount mount];
        [u, v] = meshgrid(linspace(xl(1), xl(2), max(8, ceil(diff(xl)/h))), linspace(yl(1), yl(2), max(8, ceil(diff(yl)/h))));
        u = u(:);  v = v(:);
    else
        R = Fk.radius + mount;  n = max(8, ceil(2*R/h));
        [u, v] = meshgrid(linspace(-R, R, n));  m = u.^2 + v.^2 <= R^2;
        u = u(m) + Fk.xc;  v = v(m) + Fk.yc;
    end
    psi = S.psi(:)/norm(S.psi);  xap = Fk.xap(:);  yap = Fk.yap(:);
    pts = S.vpt(:) + xap*u' + yap*v';
    if any(strcmp(S.kind, {'sphere', 'asph'}))
        % lift onto the sphere about C (sag along psi, toward the CoC)
        for i = 1:size(pts, 2)
            q = pts(:, i) - S.C(:);  qt = q - (q'*psi)*psi;
            s2 = S.R^2 - qt'*qt;
            if s2 > 0, pts(:, i) = S.C(:) + qt - sign((S.vpt(:) - S.C(:))'*psi)*sqrt(s2)*psi*(-1); end
        end
    end
end

function pts = box_pts_(c, ex, ey, ez, Lx, Ly, zr, h)
%BOX_PTS_  Surface samples of a box: centre c in the (ex,ey) plane, full
%   sizes Lx, Ly, extending from zr(1) to zr(2) along ez.
    nx = max(2, ceil(Lx/h));  ny = max(2, ceil(Ly/h));  nz = max(2, ceil(abs(diff(zr))/h));
    [u, v] = meshgrid(linspace(-Lx/2, Lx/2, nx), linspace(-Ly/2, Ly/2, ny));
    w = linspace(zr(1), zr(2), nz);
    pts = [];
    for wi = [zr(1) zr(2)]                                  % the two faces
        pts = [pts, c(:) + ex(:)*u(:)' + ey(:)*v(:)' + ez(:)*wi];  %#ok<AGROW>
    end
    for ui = [-Lx/2 Lx/2]                                   % the two x-sides
        [vv, ww] = meshgrid(linspace(-Ly/2, Ly/2, ny), w);
        pts = [pts, c(:) + ex(:)*ui + ey(:)*vv(:)' + ez(:)*ww(:)'];  %#ok<AGROW>
    end
    for vi = [-Ly/2 Ly/2]                                   % the two y-sides
        [uu, ww] = meshgrid(linspace(-Lx/2, Lx/2, nx), w);
        pts = [pts, c(:) + ex(:)*uu(:)' + ey(:)*vi + ez(:)*ww(:)'];  %#ok<AGROW>
    end
end

function Q = seg_samples_(A, Bp, n)
%SEG_SAMPLES_  n points along every segment A(:,i)->Bp(:,i).
    t = linspace(0, 1, n);  Q = zeros(3, size(A, 2)*n);
    for i = 1:size(A, 2), Q(:, (i-1)*n + (1:n)) = A(:, i) + (Bp(:, i) - A(:, i))*t; end
end

function dep = inside_depth_(Q, bx)
%INSIDE_DEPTH_  Deepest penetration (m) of the points Q into the box bx (0 = none).
    R = Q - bx.c;  u = bx.ex'*R;  v = bx.ey'*R;  w = bx.ez'*R;
    m = min([bx.Lx/2 - abs(u); bx.Ly/2 - abs(v); w - bx.zr(1); bx.zr(2) - w], [], 1);
    dep = max([0, m(m > 0)]);
end

function dmin = pts_to_pts_(A, B)
%PTS_TO_PTS_  Minimum distance between two point clouds (blocked).
    dmin = Inf;
    for i = 1:500:size(A, 2)
        a = A(:, i:min(i+499, end));
        d2 = sum(a.^2, 1)' + sum(B.^2, 1) - 2*(a'*B);
        dmin = min(dmin, sqrt(max(0, min(d2(:)))));
    end
end

function dmin = segs_to_pts_(A, Bp, Q)
%SEGS_TO_PTS_  Minimum distance between any segment A(:,i)->Bp(:,i) and any point Q(:,j).
    dmin = Inf;
    for i = 1:size(A, 2)
        a = A(:, i);  d = Bp(:, i) - a;  L2 = d'*d;
        if L2 <= 0, continue; end
        t = ((Q - a)'*d)/L2;  t = min(max(t, 0), 1);
        proj = a + d*t';
        dmin = min(dmin, min(vecnorm(Q - proj)));
    end
end
