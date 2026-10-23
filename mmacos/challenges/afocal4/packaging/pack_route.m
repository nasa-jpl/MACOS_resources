function [folds, plan] = pack_route(deck, opts)
%PACK_ROUTE  A four-fold trombone that wraps an afocal4 back end into a
%   stated envelope behind the primary.
%
%   [FOLDS, PLAN] = PACK_ROUTE(DECK) returns the FOLDS list PACK_FOLD wants,
%   plus the station table the route implies, for the committed deck's own
%   geometry.  Nothing is searched: the route is DETERMINED by four stated
%   quantities and the deck's own spacings, and the two clearances the
%   topology can violate are MEASURED from the parent's traced rays and
%   asserted before a deck is written.
%
%   THE SHAPE, AND WHY IT IS THIS SHAPE.  The depth to be removed sits in
%   the M2 -> field-mirror leg: on the 343 mm family-2 design the field
%   mirror is the deepest element, and it sits there because basin 2 buys
%   its pupil control by pushing the intermediate image ~900 mm behind the
%   primary so the field mirror can sit ON it.  So the folds go THERE, not
%   after the last mirror where the committed S4b demonstration puts its
%   single flat (that one moves the pupil out of the beam and leaves the
%   depth exactly as it was -- measured, see the README).
%
%   Four folds, normals in the x-z plane so the field bias in y maps to y
%   (the fold rule from the centered-TMA extraction work):
%
%       +z --F1--> +x --F2--> -z --F3--> -x --F4--> +z
%
%   Two facts set the count.  A -z leg is the only thing that buys depth
%   back, and reaching one costs a fold; and the train has to LEAVE heading
%   +z, because the instrument is a metre of volume that must run AFT inside
%   the shroud rather than radially out of it -- the committed demonstration
%   runs it radially, out to 1.39 m off the axis.  A 2-fold dog-leg exits -z
%   (instrument into the primary) and a 3-fold exits +x (instrument radial);
%   4 is the minimum that exits aft.  The FOURTH leg returns toward the axis
%   (-x, not +x) so the lateral step that buys clearance between the two
%   axial legs is not also paid as instrument girth.
%
%   THE FOUR STATED QUANTITIES (opts):
%     'x_step'   how far out F2 sits, m (0.300).  Half of it is the
%                clearance between the outbound and return axial legs, and
%                it must clear the plane-intersection bound below.
%     'x_out'    where the train ends up, m (0.150).  Sets the instrument's
%                girth, and must clear the outbound axial leg.
%     'z_front'  front plane of the optics slab behind M1, m (0.300).  The
%                primary's own mount ring owns everything shallower.
%     'm3_gap'   how far the last powered mirror sits BEYOND the return
%                fold's station, m (0.100).  Without it the return leg from
%                the field mirror runs back into the fold that launched it.
%
%   THE BOUND THE NULL TEST FOUND, and it is a real hardware statement.  A
%   +90/-90 fold PAIR is two flats whose planes are perpendicular, so they
%   always intersect -- for F1/F2 at x = x_step/2, for F3/F4 at
%   x_step - r/2 with r the return step.  A ray landing on the first flat
%   BEYOND that line has to reflect toward a point behind it: the engine
%   rejects the negative path length and loses the ray, and physically the
%   two flats would be cut into each other.  So each pair's step must exceed
%   TWICE the beam's own half-extent on its first flat, measured over the
%   WHOLE field box (the box corners walk the beam further out than the
%   centre field ever shows).  This is what the null caught: at a 125 mm
%   step the corner fields lost rays and AFOCAL4_SCORE moved 674 nm on a
%   deck an isometry cannot have changed, while at 175 mm and beyond it
%   reproduced the parent exactly.  The bound is now measured and asserted
%   here rather than discovered downstream.
%
%   Everything else follows.  With S the leg's vertex spacing, z0 the leg's
%   starting station, L the NEXT leg's spacing, e = L + m3_gap and
%   r = x_step - x_out,
%
%       zA = (S + z0 + z_front - x_step - r - e)/2
%
%   is the station of the first fold.
%
%   Name-value also: 'leg' (default: the leg carrying the deepest element),
%   'margin' extra clearance demanded beyond the measured beam (0.010 m),
%   'fields' K x 2 field offsets for the extent measurement (default: the
%   deck's own box corners + centre), 'check' (true), 'init', 'quiet'.
%
%   See also PACK_FOLD, PACK_LEGS, PACK_CLEAR.

    arguments
        deck (1,:) char
        opts.x_step  (1,1) double = 0.300
        opts.x_out   (1,1) double = 0.150
        opts.z_front (1,1) double = 0.300
        opts.m3_gap  (1,1) double = 0.100
        opts.leg     (1,1) double = 0
        opts.margin  (1,1) double = 0.010
        opts.fields  (:,2) double = []
        opts.check   (1,1) logical = true
        opts.init    (1,1) logical = true
        opts.quiet   (1,1) logical = false
    end

    if opts.init, macos.load_rx(deck); end
    nE = macos.num_elt();
    V  = zeros(3,nE);
    for k = 1:nE, V(:,k) = macos.get_elt_vpt(k); end

    k = opts.leg;
    if k == 0
        [~, kd] = max(V(3,:));              % deepest element behind M1
        k = kd - 1;                         % the leg that reaches it
    end
    if k < 1 || k > nE-2
        error('pack_route:leg', ...
              'leg %d has no following leg to place the field mirror on.', k);
    end

    S   = norm(V(:,k+1) - V(:,k));
    z0  = V(3,k);
    L   = norm(V(:,k+2) - V(:,k+1));
    p   = opts.x_step;   xo = opts.x_out;   r = p - xo;
    zf  = opts.z_front;
    e   = L + opts.m3_gap;
    zA  = (S + z0 + zf - p - r - e)/2;
    q   = zA - zf;
    d1  = zA - z0;

    if r <= 0
        error('pack_route:xout', ...
              'x_out %.3f m must be less than x_step %.3f m.', xo, p);
    end
    if q <= 0
        error('pack_route:depth', ...
            ['no room for the return leg: the first fold would land at z ' ...
             '%.3f m, in front of the stated slab front %.3f m.'], zA, zf);
    end
    if d1 <= 0
        error('pack_route:budget', ...
            'the route does not fit the %.4f m leg.', S);
    end

    din = (V(:,k+1) - V(:,k))/S;            % +z on these coaxial trains
    ex  = [1;0;0];
    if abs(dot(din, ex)) > 0.9, ex = [0;1;0]; end
    ax  = din;

    folds = struct( ...
      'name', {'F1','F2','F3','F4'}, ...
      'after',{ k,   'F1', 'F2', 'F3'}, ...
      'dist', { d1,   p,    q,    r  }, ...
      'to',   {ex.', (-ax).', (-ex).', ax.'});

    zFM = zf + e;
    plan = struct('leg',k, 'S',S, 'z0',z0, 'L_next',L, 'x_step',p, ...
        'x_out',xo, 'r_return',r, 'z_front',zf, 'm3_gap',opts.m3_gap, ...
        'zA',zA, 'q',q, 'd1',d1, 'e',S - d1 - p - q - r, ...
        'z_F1',zA, 'z_F3',zf, 'x_F2',p, 'x_F4',xo, ...
        'z_next',zFM, 'z_last_mirror',zf + opts.m3_gap, ...
        'slab',[zf, max(zA, zFM)], 'margin',opts.margin);

    % ---- the plane-intersection bound, MEASURED --------------------------
    if opts.check
        F = opts.fields;
        if isempty(F)
            g = [-1 0 1]*0.25*pi/180;       % the afocal4 field box corners
            [gx,gy] = meshgrid(g,g);
            F = [gx(:) gy(:)];
        end
        P1 = V(:,k) + d1*din;               % F1 vertex
        n1 = (din - ex);   n1 = n1/norm(n1);
        % Measure the offset along the TURN direction, which is the
        % coordinate the bound is written in: a ray crossing F1's plane at
        % transverse offset a lands at x = a, and the F1/F2 planes meet at
        % x = x_step/2, so the bound is a_max < x_step/2 exactly.
        w1 = plane_extent_(deck, k, P1, n1, ex, F);
        % F3 sits on the RETURN leg; its plane is reached by the same rays
        % after two reflections, but the bound is on the same bundle's
        % half-extent, which shrinks along the leg -- so measuring it at F1
        % and applying it at F3 is conservative and needs no folded deck.
        plan.w_F1 = w1;
        plan.need_step   = 2*(w1 + opts.margin);
        plan.need_return = 2*(w1 + opts.margin);
        plan.ok_step   = p > plan.need_step;
        plan.ok_return = r > plan.need_return;
        plan.ok_xout   = xo > w1 + opts.margin;
        if ~(plan.ok_step && plan.ok_return && plan.ok_xout)
            error('pack_route:planes', ...
                ['the fold pair would cut into itself: beam half-extent on ' ...
                 'F1 is %.1f mm over the field box, so x_step and the return ' ...
                 'step must each exceed %.1f mm (have %.1f, %.1f) and x_out ' ...
                 'must exceed %.1f mm (have %.1f).'], w1*1e3, ...
                plan.need_step*1e3, p*1e3, r*1e3, ...
                (w1+opts.margin)*1e3, xo*1e3);
        end
    end

    if ~opts.quiet
        fprintf(['\n  ROUTE  leg %d (%.4f m); stated: x_step %.0f, x_out %.0f, ' ...
                 'z_front %.0f, m3_gap %.0f mm\n'], k, S, p*1e3, xo*1e3, ...
                zf*1e3, opts.m3_gap*1e3);
        fprintf('    F1 at z %+.4f (dist %.4f from elt %d) -> +x\n', zA, d1, k);
        fprintf('    F2 at x %+.4f (dist %.4f)              -> -z\n', p, p);
        fprintf('    F3 at z %+.4f (dist %.4f)              -> -x\n', zf, q);
        fprintf('    F4 at x %+.4f (dist %.4f)              -> +z\n', xo, r);
        if opts.check
            fprintf(['    plane-intersection bound: beam half-extent on F1 ' ...
                     '%.1f mm over the field box\n' ...
                     '      => each V-pair step must exceed %.1f mm; ' ...
                     'x_step %.0f OK, return %.0f OK, x_out %.0f OK\n'], ...
                    plan.w_F1*1e3, plan.need_step*1e3, p*1e3, r*1e3, xo*1e3);
        end
        fprintf('    predicted: elt %d at z %+.4f, elt %d at z %+.4f; slab %+.3f..%+.3f m\n', ...
                k+1, zFM, k+2, zf + opts.m3_gap, plan.slab);
    end
end

% =====================================================================
function w = plane_extent_(deck, k, P, n, u, F)
%PLANE_EXTENT_  Half-extent, along the fold plane's in-plane direction U, of
%   the leg-K bundle where it crosses the plane (P,N) -- over EVERY field in
%   F.  Measured on the UNFOLDED deck, so no ray has been lost yet and the
%   answer is the true bundle, not a truncated one.  Re-points the source
%   the way AFOCAL_LADDER_DECK does (chief direction + yGrid, pivoting about
%   the stop) so the field set means the same thing here as in the scoring.
    txt = fileread(deck);
    txt = regexprep(txt, '(ApType=\s*)\S+', '$1None');
    cd0 = grab3_(txt,'ChfRayDir');   cp0 = grab3_(txt,'ChfRayPos');
    apst = grab3_(txt,'ApStop');
    stand = dot(apst - cp0, cd0);
    bx0 = asin(cd0(1));   by0 = asin(cd0(2));
    tmp = [tempname '.in'];
    cu  = onCleanup(@() del_(tmp)); %#ok<NASGU>
    w = 0;
    for i = 1:size(F,1)
        bx = bx0 + F(i,1);   by = by0 + F(i,2);
        cdir = [sin(bx); sin(by); sqrt(max(0,1-sin(bx)^2-sin(by)^2))];
        cpos = apst - stand*cdir;
        s = regexprep(txt, '(ChfRayDir=\s*)[^\n]*', ['$1' v3_(cdir)]);
        s = regexprep(s,   '(ChfRayPos=\s*)[^\n]*', ['$1' v3_(cpos)]);
        s = regexprep(s,   '(yGrid=\s*)[^\n]*', ['$1' v3_([0;cos(by);-sin(by)])]);
        fid = fopen(tmp,'w');  fprintf(fid,'%s',s);  fclose(fid);
        macos.load_rx(tmp);
        nE = macos.num_elt();
        macos.ray_hist('on');   t = macos.trace();   h = macos.ray_hist(t.nRays);
        macos.ray_hist('off');
        off = size(h.P,3) - nE;
        m = h.ok(:,k+off) & h.ok(:,k+1+off);
        A = squeeze(h.P(:,m,k+off));   B = squeeze(h.P(:,m,k+1+off));
        den = n.'*(B - A);
        tt  = (n.'*(P - A))./den;
        X   = A + tt.*(B - A);
        w   = max(w, max(abs(u.'*(X - P))));
    end
    macos.load_rx(deck);
end

function v = grab3_(txt, key)
    t = regexp(txt, ['(?m)^\s*' key '=\s*([^\n]*)'], 'tokens', 'once');
    v = sscanf(strrep(t{1},'D','E'), '%f', 3);
end

function s = v3_(v),  s = sprintf('%.16E  %.16E  %.16E', v(1), v(2), v(3));  end
function del_(p),  if exist(p,'file'), delete(p); end,  end
