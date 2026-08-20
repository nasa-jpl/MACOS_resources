function gt = oi_gates(X, G, P, offset_deg)
%OI_GATES  Constraint gates: exit-beam direction + beam/mirror clearances.
%
%   GT = OI_GATES(X, G, P, OFFSET_DEG) traces the box-centre and the four
%   box-corner fields, then evaluates the template's constraint set:
%
%   EXIT BEAM -- the box-centre exit chief direction (post-M3).  If
%   P.exit_dir is set, the gate is the angle to it vs P.exit_tol_deg;
%   otherwise report-only.  Either way the angle in the Y-Z plane is
%   reported (atan2d(-dy,-dz) style, the rodgers2 convention).
%
%   CLEARANCES -- for every beam LEG (incoming, M1->stop/M2, M2->M3,
%   M3->FP) and every MIRROR not an endpoint of that leg: the minimum
%   distance from the traced ray segments (all pass rays, all five
%   fields) to that mirror's footprint disk.  GATE: every pair >=
%   min(P.clear_m); pairs below max(P.clear_m) are flagged WARN.  (The
%   split interpretation -- Mike states two clearance values without
%   naming their pairs -- is recorded in the challenge PACKET.)
%
%   GT: .exit_dir (3x1), .exit_ang_deg, .exit_err_deg (NaN if unpinned),
%   .exit_pass, .clear_table (struct array leg/elt/min_m), .clear_min_m,
%   .clear_pass, .clear_warn.
%
%   See also OI_SCORE, OFFSET_IMAGER.

    % ---- fields: centre + corners -----------------------------------------
    bx = P.box_deg(1)/2;  by = P.box_deg(2)/2;
    F = [0 offset_deg;
         -bx offset_deg-by; -bx offset_deg+by;
          bx offset_deg-by;  bx offset_deg+by];

    txt = oi_deck(fill_(X, P));
    sc = oi_score(txt, G, F, 'rays', true);

    % ---- element geometry ---------------------------------------------------
    z_stop = X.z_m1 + X.spacings(1);
    zEl = [X.z_m1, z_stop, z_stop + X.spacings(2), ...
           z_stop + X.spacings(2) + X.spacings(3)];
    mirror_ie = [1 3 4];              % element ids of M1 M2 M3 in the train

    % footprint disks of the mirrors (centre + radius from ALL fields)
    disk = struct('C',{},'n',{},'r',{});
    for m = 1:3
        Q = [];
        for q = 1:size(F,1)
            E = sc.rays{q};  if isempty(E), continue; end
            e = E{mirror_ie(m)};
            ok = e.ok;  ok(1) = false;
            Q = [Q, e.pos(:,ok)]; %#ok<AGROW>
        end
        C = mean(Q,2);
        al = X.ade(m);
        n = [0; -sind(al); cosd(al)];
        r = max(vecnorm(Q - C, 2, 1));
        disk(m) = struct('C',C,'n',n,'r',r);
    end

    % ---- exit chief -----------------------------------------------------------
    Ec = sc.rays{1};
    ex_p = Ec{4}.pos(:,1);  ex_d = Ec{4}.dir(:,1);   %#ok<NASGU> % post-M3 chief
    gt.exit_dir = ex_d;
    gt.exit_ang_deg = atan2d(ex_d(2), ex_d(3));
    if ~isempty(P.exit_dir)
        ed = P.exit_dir(:)/norm(P.exit_dir);
        gt.exit_err_deg = acosd(min(1, dot(ed, ex_d)));
        gt.exit_pass = gt.exit_err_deg <= P.exit_tol_deg;
    else
        gt.exit_err_deg = NaN;
        gt.exit_pass = true;          % report-only
    end

    % ---- clearances --------------------------------------------------------------
    % legs by element-state pairs: 0 = incoming (reconstructed), else (ie->ie+1)
    legs = {'in->M1',[0 1]; 'M1->M2',[1 3]; 'M2->M3',[3 4]; 'M3->FP',[4 5]};
    T = struct('leg',{},'mirror',{},'min_m',{});
    for L = 1:size(legs,1)
        pr = legs{L,2};
        for m = 1:3
            if any(mirror_ie(m) == pr), continue; end
            dmin = inf;
            for q = 1:size(F,1)
                E = sc.rays{q};  if isempty(E), continue; end
                if pr(1) == 0
                    e1 = E{1};  ok = e1.ok;  ok(1) = false;
                    B = e1.pos(:,ok);
                    cd = tancomp_(F(q,1), F(q,2));
                    A = B - 1.5*norm(zEl(4)-zEl(1))*cd;   % incoming segment
                else
                    e1 = E{pr(1)};  e2 = E{pr(2)};
                    ok = e1.ok & e2.ok;  ok(1) = false;
                    A = e1.pos(:,ok);  B = e2.pos(:,ok);
                end
                dmin = min(dmin, seg_disk_min_(A, B, disk(m)));
            end
            T(end+1) = struct('leg',legs{L,1},'mirror',sprintf('M%d',m), ...
                              'min_m',dmin); %#ok<AGROW>
        end
    end
    gt.clear_table = T;
    gt.clear_min_m = min([T.min_m]);
    gt.clear_pass  = gt.clear_min_m >= min(P.clear_m);
    gt.clear_warn  = gt.clear_min_m <  max(P.clear_m);
end

% =========================================================================
function d = seg_disk_min_(A, B, dk)
%SEG_DISK_MIN_  Min distance from segments A(:,i)->B(:,i) to a disk.
    ns = 21;
    d = inf;
    for s = linspace(0,1,ns)
        Q = A + s*(B - A);
        v = Q - dk.C;
        h = dk.n.'*v;                          % height over the disk plane
        Pp = v - dk.n*h;                       % in-plane component
        rr = vecnorm(Pp, 2, 1);
        ro = max(rr - dk.r, 0);                % radial excess outside the rim
        d = min(d, min(hypot(ro, abs(h))));
    end
end

function D = fill_(X, P)
    D = X;
    D.EPD_m = P.EPD_m;  D.WL_m = P.lambda_m;
    D.sampling = P.sampling;  D.name = P.name;
end

function d = tancomp_(xan_deg, yan_deg)
    d = [tand(xan_deg); tand(yan_deg); 1];
    d = d/norm(d);
end
