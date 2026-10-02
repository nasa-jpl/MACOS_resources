function [cmin, worst, cvec, pairs] = telescope_clear_quick(G, B, F, cloud, mount)
%TELESCOPE_CLEAR_QUICK  A cheap clearance wall for the telescope solve.
%   [cmin, worst] = telescope_clear_quick(G, B, F, cloud, mount) is the
%   minimum distance (m) from any ray segment of any telescope LEG (the
%   bundle B's stations: sky -> M1 -> M2 -> M3 [-> fold] -> slit) to any
%   BODY the leg is not an endpoint of -- the telescope's own mirrors and
%   fold as their footprint discs (F, radius + mount, sampled on the rim and
%   the centre) and the spectrometer's bodies as a point cloud (cloud:
%   3 x N, built once by telescope_cloud from the Dyson chain's footprints,
%   its slit mask and its detector package) -- minus the mount margin.  The
%   record's clearance is spectrometer_clearance on the end-to-end chain
%   (surfaces sampled at 2 mm, box penetration depths); this is the wall the
%   optimizer feels at every iterate (Dave's rule: walls on iterates, the
%   gate on the record).  worst names the pair; cvec / pairs list EVERY
%   leg-body clearance (the solver's wall is a hinge per pair, smoother than
%   a hinge on the minimum alone).
    nS = numel(G.surf);  nleg = nS;
    bodies = {};
    for k = 1:nS
        if any(strcmp(G.surf(k).act, {'stop', 'pass'})), continue; end
        bodies{end+1} = struct('name', G.surf(k).name, 'surf', k, 'pts', disc_pts_(G.surf(k), F(k), mount));  %#ok<AGROW>
    end
    if ~isempty(cloud)
        bodies{end+1} = struct('name', 'Spectrometer', 'surf', 0, 'pts', cloud);
    end
    cmin = Inf;  worst = '';  cvec = [];  pairs = {};
    for L = 1:nleg
        a = squeeze(B.P(:, :, L));  b = squeeze(B.P(:, :, L+1));
        if size(a, 1) ~= 3, a = a';  b = b'; end
        ends = [L-1, L];
        for i = 1:numel(bodies)
            Bd = bodies{i};
            if any(Bd.surf == ends), continue; end
            if Bd.surf == 0 && L == nleg, continue; end      % the leg INTO the slit meets the spectrometer by design
            d = segs_to_pts_(a, b, Bd.pts) - mount*(Bd.surf > 0);
            cvec(end+1) = d;  pairs{end+1} = sprintf('%s -> %s vs %s', stname_(G, L-1), stname_(G, L), Bd.name);   %#ok<AGROW>
            if d < cmin, cmin = d;  worst = pairs{end}; end
        end
    end
end

function pts = disc_pts_(S, Fk, mount)
%DISC_PTS_  The body as the footprint's RECTANGLE (extents + mount) in the
%   aperture frame: a telescope mirror carries a one-dimensional field, so
%   its footprint is long along the slit (x) and narrow across it, and a
%   disc enclosing it overstates the body in the y-z plane -- where the
%   folded beams pass -- by tens of millimetres (every layout read -4 mm,
%   2026-10-01).  spectrometer_clearance uses the same rectangle for the
%   telescope's surfaces.
    xl = Fk.xlim + [-mount mount];  yl = Fk.ylim + [-mount mount];
    [u, v] = meshgrid(linspace(xl(1), xl(2), 9), linspace(yl(1), yl(2), 7));
    pts = S.vpt(:) + Fk.xap(:)*u(:)' + Fk.yap(:)*v(:)';
end

function n = stname_(G, k)
    if k == 0, n = 'Sky'; else, n = G.surf(k).name; end
end

function dmin = segs_to_pts_(A, Bp, Q)
    dmin = Inf;
    for i = 1:size(A, 2)
        a = A(:, i);  d = Bp(:, i) - a;  L2 = d'*d;
        if L2 <= 0, continue; end
        t = ((Q - a)'*d)/L2;  t = min(max(t, 0), 1);
        proj = a + d*t';
        dmin = min(dmin, min(vecnorm(Q - proj)));
    end
end
