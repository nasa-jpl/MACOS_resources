function [D, info] = afocal4_pack_seed(P, iface, opts)
%AFOCAL4_PACK_SEED  A BUILDABLE starting design at a given operating point.
%
%   [D, info] = AFOCAL4_PACK_SEED(P, IFACE) searches the (field-mirror
%   standoff, M2 radius) plane for a closure that satisfies the S4b
%   packaging constraint at interface standoff IFACE, and returns it as an
%   AFOCAL4_BUILD design struct.
%
%   WHY THIS EXISTS.  The constraint is a WALL, and a wall needs a
%   compliant seed to be a wall rather than a cage: dropped into a solver
%   at a non-compliant point, every finite-difference direction is an
%   error, the Jacobian is walls all the way round, and lsqnonlin returns
%   the seed it was given.  That is not a design failure and must not be
%   reported as one -- it is a seeding failure.  The S4 trade seeded every
%   operating point by warm-starting from the last one, which is fine
%   inside a feasible region and useless for entering one.
%
%   WHAT IT PREFERS, in order:
%     1  HIS FRONT END.  M2's radius and the M1-M2 spacing are searched
%        only if his own values cannot close the point -- the anchor of the
%        S4b brief, and the thing the study is entitled to change last.
%     2  THE WEAKEST FIELD MIRROR that clears the constraint with margin.
%        phi4 is what the image quality pays for (RESULTS section 3), so
%        the seed spends as little of it as the packaging allows.
%     3  MARGIN over the bound, not the bound itself: a seed sitting
%        exactly on the wall has half its finite-difference stencil
%        outside it.
%
%   Name-value:
%     'margin'   metres of clearance over P.pack.m3_behind_min (0.03)
%     'standoff' standoff grid to search (default -0.75:0.025:0.30)
%     'R2'       M2 radii to try IN ORDER (default: his, then 0.470 down
%                to 0.430 -- a slower secondary pushes the intermediate
%                image, and everything behind it, further back)
%     'quiet'    (true)
%
%   Returns D (ready for AFOCAL4_BUILD) and INFO with .ok .phi4 .behind_m1
%   .standoff .R2 .used_his_front_end .n_compliant.
%
%   See also AFOCAL4_PHI4, AFOCAL4_BUILD, AFOCAL4_PACK, AFOCAL4_SEED.

    arguments
        P (1,1) struct
        iface (1,1) double
        opts.margin   (1,1) double = 0.03
        opts.standoff (1,:) double = -0.75:0.025:0.30
        opts.R2       (1,:) double = []
        opts.quiet    (1,1) logical = true
    end
    R2s = opts.R2;
    if isempty(R2s), R2s = [P.parent.R(2), 0.470, 0.460, 0.450, 0.440, 0.430]; end
    need = P.pack.m3_behind_min + opts.margin;

    D = afocal4_seed(P, 'iface', iface);
    info = struct('ok',false, 'phi4',NaN, 'behind_m1',NaN, 'standoff',NaN, ...
                  'R2',NaN, 'used_his_front_end',false, 'n_compliant',0);

    for R2 = R2s
        Q = P;   Q.parent.R(2) = R2;
        best = [];   n = 0;
        for s = opts.standoff
            [phi4, C, found] = afocal4_phi4(Q, s, iface);
            if ~found || C.behind_m1 < need, continue; end
            n = n + 1;
            if isempty(best) || abs(phi4) < abs(best.phi4)
                best = struct('phi4',phi4, 'C',C, 's',s);
            end
        end
        if isempty(best), continue; end
        D.fm_standoff = best.s;
        D.R2 = R2;
        info = struct('ok',true, 'phi4',best.phi4, 'behind_m1',best.C.behind_m1, ...
                      'standoff',best.s, 'R2',R2, ...
                      'used_his_front_end', R2 == P.parent.R(2), 'n_compliant',n);
        if ~opts.quiet
            fprintf(['  seed at iface %.0f mm: s %+.0f mm, R_M2 %.1f mm%s, ' ...
                     'phi4 %+.3f /m, %s %.0f mm behind M1 (%d compliant ' ...
                     'standoffs)\n'], iface*1e3, best.s*1e3, R2*1e3, ...
                    tag_(info.used_his_front_end), best.phi4, ...
                    best.C.names{end}, best.C.behind_m1*1e3, n);
        end
        return;
    end
    if ~opts.quiet
        fprintf(['  seed at iface %.0f mm: NO COMPLIANT CLOSURE over ' ...
                 's in [%.0f %.0f] mm and R_M2 in [%.0f %.0f] mm\n'], ...
                iface*1e3, min(opts.standoff)*1e3, max(opts.standoff)*1e3, ...
                min(R2s)*1e3, max(R2s)*1e3);
    end
end

function s = tag_(his),  if his, s = ' (his)'; else, s = ''; end,  end
