function [dmin, d] = oi_clear(X, G, P, offset_deg)
%OI_CLEAR  Beam-leg vs element clearances of the box-centre field.
%
%   [DMIN, D] = OI_CLEAR(X, G, P, OFFSET) traces THREE field bundles
%   (box centre + the two YAN extremes -- the y-z blockers) through
%   every element and measures, for each of the NINE ordered
%   leg/obstacle pairs, the minimum distance from any field's leg
%   segments to the obstacle's GLASS: the union of all three fields'
%   patch disks (footprint centre + 1.15x footprint radius per field --
%   the hardware must carry every field).  The FP counts as an obstacle
%   for legs that do not end on it (an image plane sitting in the
%   incoming corridor is exactly as unbuildable as a mirror there).
%   Centre-field-only measurement is NOT enough: the rodgers3 S4
%   blockage that motivated this (M3->FP beam through M2) is carried by
%   the EDGE fields while the centre field clears by 48 mm.
%
%     legs:   in->M1, M1->M2, M2->M3, M3->FP
%     pairs:  in->M1 x {M2,M3,FP};  M1->M2 x {M3,FP};
%             M2->M3 x {M1,FP};     M3->FP x {M1,M2}
%
%   D is the fixed-length 9-vector (m, order above); DMIN = min(D).
%   The stop (a Reference aperture, not glass) is not an obstacle.
%
%   This is the SOLVE-side model backing the S4/S5 clearance hinge rows
%   (per-element patches about their own centres -- much tighter than
%   OI_GATES' report-side union disks).  Failure to trace returns
%   DMIN = 0 (a candidate that cannot trace has no clearance).
%
%   See also OI_SOLVE, OI_GATES, OFFSET_IMAGER.

    d = zeros(9,1);  dmin = 0;
    tmp = [tempname '.in'];
    cu  = onCleanup(@() delete_if_(tmp));

    D = X;
    D.EPD_m = P.EPD_m;  D.WL_m = P.lambda_m;
    D.sampling = P.sampling;  D.name = P.name;
    if isfield(P,'solve_sampling') && ~isempty(P.solve_sampling)
        D.sampling = P.solve_sampling;
    end
    txt = oi_deck(D);
    by = P.box_deg(2)/2;
    yans = offset_deg + [0, -by, +by];

    ie = [1 3 4 5];                       % M1 M2 M3 FP element ids
    S = cell(3,4);  cds = cell(1,3);
    for q = 1:3
        cdir = tancomp_(0, yans(q));  cds{q} = cdir;
        emit_src_(txt, tmp, seed_pos_(G, cdir), cdir);
        macos.load_rx(tmp);
        if ~macos.has_rx(), return; end
        macos.stop(2, [0 0]);
        % one traced pass records every station (engine RayPosHist)
        macos.ray_hist('on');
        tr = macos.trace(macos.num_elt());
        h = macos.ray_hist(tr.nRays);
        macos.ray_hist('off');
        for k = 1:4
            ok = h.ok(:, ie(k)+1);  ok(1) = false;
            if nnz(ok) < 5, return; end
            S{q,k} = h.P(:, ok, ie(k)+1);
        end
    end

    % glass per element: one patch disk PER FIELD (centre + 1.15x radius)
    nrm = { [0;0;1], [0;0;1], [0;0;1], G.fpa.psi(:)/norm(G.fpa.psi) };
    pat = cell(1,4);
    for k = 1:4
        pk = struct('C',{},'n',{},'r',{});
        for q = 1:3
            C = mean(S{q,k}, 2);
            r = 1.15 * max(vecnorm(S{q,k} - C, 2, 1));
            pk(q) = struct('C',C, 'n',nrm{k}, 'r',r);
        end
        pat{k} = pk;
    end

    % nine pairs; each = min over (leg field) x (obstacle field disks)
    span = 1.2 * max(1e-3, abs(X.spacings(1)) + abs(X.spacings(3)));
    obst = { [2 3 4], [3 4], [1 4], [1 2] };
    j = 0;
    for L = 1:4
        for o = obst{L}
            j = j + 1;
            dm = inf;
            for q = 1:3
                if L == 1
                    A = S{q,1} - span*cds{q};  B = S{q,1};
                else
                    A = S{q,L-1};  B = S{q,L};
                end
                for pq = 1:3
                    dm = min(dm, seg_disk_min_(A, B, pat{o}(pq)));
                end
            end
            d(j) = dm;
        end
    end
    dmin = min(d);
end

% =========================================================================
function dm = seg_disk_min_(A, B, dk)
    ns = 25;
    dm = inf;
    for s = linspace(0, 1, ns)
        Q = A + s*(B - A);
        v = Q - dk.C;
        h = dk.n.'*v;
        Pp = v - dk.n*h;
        rr = vecnorm(Pp, 2, 1);
        ro = max(rr - dk.r, 0);
        dm = min(dm, min(hypot(ro, abs(h))));
    end
end

function p = seed_pos_(G, cdir)
    cdR = [cdir(1); cdir(2); -cdir(3)];
    tq  = (G.z_m1 - G.stopC(3))/cdir(3);
    q   = G.stopC - tq*cdR;
    p   = q - (0.75/cdir(3))*cdir;
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

function delete_if_(p), if exist(p,'file'), delete(p); end, end
