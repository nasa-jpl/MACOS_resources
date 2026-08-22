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
%   CLEARANCES -- the OI_CLEAR model, IDENTICAL to the S4/S5 solve
%   hinge (nine ordered leg/obstacle pairs, per-field footprint disks
%   x1.15 over the box centre + YAN extremes, the FP counted as an
%   obstacle), so the report grades exactly what the solve paid for.
%   GATE: min over pairs >= min(P.clear_m) less a 1.5 mm hinge knee;
%   below max(P.clear_m) flags WARN.  (The split interpretation -- Mike
%   states two clearance values without naming their pairs -- is
%   recorded in the challenge PACKET.)
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

    % ---- exit chief -----------------------------------------------------------
    % (guarded, F6 family: a box-centre field that loses every ray hands
    % back an empty double -- the exit direction is then UNMEASURABLE and
    % the gate FAILS with a message, it does not crash)
    Ec = sc.rays{1};
    if ~iscell(Ec) || numel(Ec) < 4 || isempty(Ec{4}.pos)
        fprintf(['  oi_gates: box-centre field lost every ray -- exit ' ...
                 'direction unmeasurable (gate FAIL)\n']);
        gt.exit_dir = nan(3,1);
        gt.exit_ang_deg = NaN;
        gt.exit_err_deg = NaN;
        gt.exit_pass = false;
    else
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
    end

    % ---- clearances (the OI_CLEAR model -- identical to the S4/S5 solve
    % hinge: nine ordered leg/obstacle pairs, per-field patch disks x1.15
    % over the box centre + YAN extremes, FP counted as an obstacle) ----
    pairs = {'in->M1 x M2','in->M1 x M3','in->M1 x FP', ...
             'M1->M2 x M3','M1->M2 x FP','M2->M3 x M1','M2->M3 x FP', ...
             'M3->FP x M1','M3->FP x M2'};
    [dmin, dvec] = oi_clear(X, G, P, offset_deg);
    T = struct('leg',{},'mirror',{},'min_m',{});
    for j = 1:9
        T(j) = struct('leg',pairs{j},'mirror','','min_m',dvec(j));
    end
    gt.clear_table = T;
    gt.clear_min_m = dmin;
    gt.clear_pass  = gt.clear_min_m >= min(P.clear_m) - 1.5e-3;  % hinge knee
    gt.clear_warn  = gt.clear_min_m <  max(P.clear_m);
end

% =========================================================================
function D = fill_(X, P)
    D = X;
    D.EPD_m = P.EPD_m;  D.WL_m = P.lambda_m;
    D.sampling = P.sampling;  D.name = P.name;
end

function d = tancomp_(xan_deg, yan_deg)
    d = [tand(xan_deg); tand(yan_deg); 1];
    d = d/norm(d);
end
