function L = clear_law(deck, opts)
%CLEAR_LAW  Why a body stands in a beam: the field-walk ratio, measured.
%
%   L = CLEAR_LAW(DECK, 'fields',F, 'leg',K, 'elt',E) measures, field by
%   field, where the obstructing bundle (leg K) and the obstructed part
%   (element E) actually sit in element E's own plane, and reduces the two
%   to the single number that decides whether they can EVER come apart.
%
%   THE LAW.  On a coaxial train every chief ray height is proportional to
%   its field angle, so a part's UNION footprint and a beam's union
%   footprint at the same station are two scaled copies of the SAME field
%   box -- scales c_body and c_beam, both anchored at the axis.  A field box
%   biased by B with half-width A occupies angles [B-A, B+A] in the bias
%   direction, so the two copies occupy [c*(B-A), c*(B+A)] and they are
%   disjoint ONLY if
%
%           c_beam / c_body  >  (B + A) / (B - A)
%
%   -- a pure property of the field specification, 0.85/0.35 = 2.4286 for
%   this benchmark.  Radii make it harder still: the real condition is
%
%           c_beam*(B-A) - r_beam  >  c_body*(B+A) + r_body.
%
%   WHAT IT RULES OUT, AND THIS IS THE POINT.  Both sides scale with the
%   field, so NO field-proportional remedy can move the ratio far: a
%   different collimator station moves c_beam by the chief slope alone, a
%   different interface standoff moves c_body = M * iface, and a FLAT FOLD
%   moves neither -- an isometry carries both copies together.  The
%   quantities that DO break it are the ones that are not proportional to
%   field: a tilt (CLEAR_TILT separates by 2*alpha*d), or an element the
%   train does not currently have.
%
%   AND ONE OF THE TWO SCALES IS PINNED BY THE SPECIFICATION.  For the LAST
%   powered mirror, the chief ray converges to the exit pupil a distance
%   IFACE away at the exit angular magnification M, so
%
%           c_body(collimator) = M * iface,  exactly,
%
%   which this routine checks against the traced fit rather than asserting.
%   That is why the collimator, and not some other part, is the one standing
%   in the beam, and why the interference gets worse as the interface
%   standoff grows -- the same knob the S4 ruling carries as the operating
%   point, now pulling a third way.
%
%   Name-value:
%     'fields'  K x 2 field offsets, rad (required for a meaningful fit --
%               one field cannot show a walk)
%     'leg'     the obstructing leg index k (bundle from element k to k+1)
%     'elt'     the obstructed element index
%     'axis'    'bias' (default) -- report the ratio along the bias
%               direction, which is the one the law is about; 'norm' uses
%               the centroid magnitudes instead
%     'init' / 'quiet'
%
%   Returns L with .c_beam .c_body (m per rad, bias direction), .ratio,
%   .need (the field box's own (B+A)/(B-A)), .clears_centres, .gap_m (the
%   signed centre-set gap), .r_beam .r_body, .gap_full_m (gap minus radii),
%   .per_field (the daylight ONE field sees -- the number that makes an
%   interference look like a 10 mm margin), .fit_resid_m and .M_iface
%   (the pinned prediction for c_body, and how far the fit is from it).
%
%   See also CLEAR_SCAN, AFOCAL4_UNION, CLEAR_TILT.

    arguments
        deck (1,:) char
        opts.fields (:,2) double
        opts.leg    (1,1) double
        opts.elt    (1,1) double
        opts.axis   (1,:) char {mustBeMember(opts.axis,{'bias','norm'})} = 'bias'
        opts.M      (1,1) double = NaN
        opts.init   (1,1) logical = true
        opts.quiet  (1,1) logical = false
    end
    F = opts.fields;
    if size(F,1) < 2
        error('macos:design:clear_law:fields', ...
              'the law is about a WALK: give at least two fields.');
    end

    if opts.init, macos.load_rx(deck); end
    nE = macos.num_elt();
    e  = opts.elt;   k = opts.leg;
    if k >= nE || k < 1, error('macos:design:clear_law:leg','leg %d out of range.', k); end
    psi = macos.get_elt_psi(e);   psi = psi(:)/norm(psi);
    Vp  = macos.get_elt_vpt(e);   Vp = Vp(:);

    % In-plane basis of the obstructed element.  The bias is in global +y,
    % so the in-plane direction that carries it is +y orthogonalised.
    yb = [0;1;0] - psi*(psi.'*[0;1;0]);
    if norm(yb) < 1e-9, yb = [1;0;0] - psi*(psi.'*[1;0;0]); end
    yb = yb/norm(yb);
    xb = cross(yb, psi);

    nF = size(F,1);
    cb = zeros(2,nF);  cf = zeros(2,nF);
    rb = zeros(1,nF);  rf = zeros(1,nF);
    day = nan(1,nF);
    txt = fileread(deck);
    tmp = [tempname '.in'];
    cu  = onCleanup(@() del_(tmp)); %#ok<NASGU>

    for i = 1:nF
        write_field_(txt, F(i,:), tmp);
        macos.load_rx(tmp);
        macos.ray_hist('on');   t = macos.trace();   h = macos.ray_hist(t.nRays);
        macos.ray_hist('off');
        off = size(h.P,3) - nE;
        % the part's own footprint
        m = h.ok(:,e+off);
        Q = squeeze(h.P(:,m,e+off));   if size(Q,1) ~= 3, Q = Q(:); end
        c = mean(Q,2);
        cb(:,i) = [xb.'*(c - Vp); yb.'*(c - Vp)];
        rb(i)   = max(vecnorm(Q - c));
        % where the obstructing leg crosses the part's plane
        ma = h.ok(:,k+off) & h.ok(:,k+1+off);
        A  = squeeze(h.P(:,ma,k+off));   B = squeeze(h.P(:,ma,k+1+off));
        if size(A,1) ~= 3, A = A(:);  B = B(:); end
        den = psi.'*(B - A);
        tt  = (psi.'*(Vp - A))./den;
        hit = abs(den) > 1e-14 & tt > 0 & tt < 1;
        if ~any(hit)
            error('macos:design:clear_law:nocross', ...
                  ['leg %d-%d never crosses element %d''s plane at field ' ...
                   '%d -- the pair this law is about does not exist here.'], ...
                  k, k+1, e, i);
        end
        X  = A(:,hit) + tt(hit).*(B(:,hit) - A(:,hit));
        cx = mean(X,2);
        cf(:,i) = [xb.'*(cx - Vp); yb.'*(cx - Vp)];
        rf(i)   = max(vecnorm(X - cx));
        % what ONE field sees: the daylight between the two patches
        day(i) = min(vecnorm(X - c)) - rb(i);
    end
    macos.load_rx(deck);

    % ---- WALK plus OFFSET, least squares in the ABSOLUTE field angle -----
    % centroid(theta) = c*theta + o.  The two terms are the whole story:
    %   c  the FIELD-PROPORTIONAL walk.  On a coaxial train this is all
    %      there is, and it is what the ratio law is about.
    %   o  the FIELD-INDEPENDENT offset.  Identically zero on a coaxial
    %      train -- and the only quantity that can defeat the law, because
    %      it does not shrink with the field angle the way c*theta does.
    %      A tilt puts it there (2*alpha*d); a flat fold does not (an
    %      isometry moves BOTH bundles by the same o and the difference is
    %      unchanged).
    % Fitting the intercept rather than forcing the origin is what makes
    % this routine report a tilted design honestly instead of returning a
    % meaningless "walk" that is absorbing an offset.
    by = asin(grab3_(txt,'ChfRayDir'));
    L.bias_rad = by(2);
    ay = max(abs(F(:,2)));
    L.half_rad = ay;
    L.tmin = L.bias_rad - ay;   L.tmax = L.bias_rad + ay;
    L.need = L.tmax / L.tmin;

    ta = L.bias_rad + F(:,2);                 % ABSOLUTE bias-direction angle
    A  = [ta, ones(size(ta))];
    if strcmp(opts.axis,'norm')
        pb = A\vecnorm(cb).';   pf = A\vecnorm(cf).';
        rb_fit = vecnorm(cb).' - A*pb;   rf_fit = vecnorm(cf).' - A*pf;
    else
        pb = A\cb(2,:).';       pf = A\cf(2,:).';
        rb_fit = cb(2,:).' - A*pb;       rf_fit = cf(2,:).' - A*pf;
    end
    L.c_body_abs = pb(1);   L.o_body = pb(2);
    L.c_beam_abs = pf(1);   L.o_beam = pf(2);
    L.c_body = L.c_body_abs;   L.c_beam = L.c_beam_abs;
    L.fit_resid_m = max(abs([rb_fit; rf_fit]));
    L.offset_m = L.o_beam - L.o_body;          % what is NOT proportional
    L.ratio = L.c_beam_abs / L.c_body_abs;
    L.clears_centres = L.ratio > L.need;

    L.r_body = max(rb);   L.r_beam = max(rf);
    % the centre-set gap, with the offset carried explicitly: the two sets
    % are c*[tmin,tmax] + o, so their gap is the proportional part plus the
    % offset difference.
    L.gap_prop_m = L.c_beam_abs*L.tmin - L.c_body_abs*L.tmax;
    L.gap_m      = L.gap_prop_m + L.offset_m;
    L.gap_full_m = L.gap_m - L.r_beam - L.r_body;
    L.per_field  = day;
    L.per_field_min = min(day);
    L.cb = cb;  L.cf = cf;  L.rb = rb;  L.rf = rf;  L.fields = F;
    L.elt = e;  L.leg = k;  L.deck = deck;

    % ---- the pinned prediction, for the LAST powered mirror only ---------
    % c_body(last mirror) = M * iface, because the chief converges from that
    % mirror to the exit pupil IFACE away at the exit angular magnification.
    % Checked, not asserted: IFACE is the vertex-to-interface distance read
    % off the deck and M is passed in by the caller (it is a specification
    % number, not something to be re-derived from a footprint).
    L.iface_m = NaN;   L.M_iface = NaN;   L.M_iface_err = NaN;
    if e == nE-1
        % The DECLARED standoff, from the last mirror's zElt -- not the
        % vertex-to-interface distance.  The builder poses the interface
        % plane on the traced chief, so on the committed 343 mm deck those
        % two are 343 and 359 mm; taking the vertex distance flatters the
        % pin by 5 % and is measuring a different quantity from the one the
        % specification names.
        z = grab1_(txt,'zElt');
        L.iface_m = z(nE-1);
        if ~isfinite(L.iface_m) || L.iface_m <= 0
            Vi = macos.get_elt_vpt(nE);
            L.iface_m = norm(Vi(:) - Vp);
        end
        if ~isnan(opts.M)
            L.M_iface     = opts.M * L.iface_m;
            L.M_iface_err = L.c_body_abs/L.M_iface - 1;
        end
    end

    if ~opts.quiet, report_(L); end
end

% =====================================================================
function report_(L)
    fprintf('\n  FIELD-WALK LAW  %s\n', L.deck);
    fprintf('    obstructed part: element %d;  obstructing bundle: leg %d-%d\n', ...
            L.elt, L.leg, L.leg+1);
    fprintf(['    field box in the bias direction: %.4f .. %.4f deg ' ...
             '-> required ratio %.4f\n'], rad2deg(L.tmin), rad2deg(L.tmax), L.need);
    fprintf(['    measured WALK  : body %8.4f m/rad, beam %8.4f m/rad ' ...
             '-> ratio %.4f  %s\n'], L.c_body_abs, L.c_beam_abs, L.ratio, ...
            tern_(L.clears_centres,'(proportional part clears)', ...
                                   '(proportional part OVERLAPS)'));
    fprintf(['    measured OFFSET: body %+8.4f m, beam %+8.4f m ' ...
             '-> field-independent separation %+8.2f mm\n'], ...
            L.o_body, L.o_beam, L.offset_m*1e3);
    fprintf('    residual of the walk+offset fit: %.3e m\n', L.fit_resid_m);
    fprintf(['    centre-set gap: proportional %+8.2f + offset %+8.2f = ' ...
             '%+8.2f mm;  radii body %.1f + beam %.1f -> gap %+8.2f mm\n'], ...
            L.gap_prop_m*1e3, L.offset_m*1e3, L.gap_m*1e3, L.r_body*1e3, ...
            L.r_beam*1e3, L.gap_full_m*1e3);
    fprintf('    what ONE field sees (daylight per field): %.1f .. %.1f mm\n', ...
            min(L.per_field)*1e3, max(L.per_field)*1e3);
    if ~isnan(L.M_iface)
        fprintf(['    pinned prediction  c_body = M * iface = %.4f m/rad; ' ...
                 'measured is %+.3e of it\n'], L.M_iface, L.M_iface_err);
    end
end

function write_field_(txt, f, out)
    cd0  = grab3_(txt,'ChfRayDir');   cp0 = grab3_(txt,'ChfRayPos');
    apst = grab3_(txt,'ApStop');
    stand = dot(apst - cp0, cd0);
    bx = asin(cd0(1)) + f(1);   by = asin(cd0(2)) + f(2);
    cdir = [sin(bx); sin(by); sqrt(max(0,1-sin(bx)^2-sin(by)^2))];
    cpos = apst - stand*cdir;
    s = regexprep(txt, '(ChfRayDir=\s*)[^\n]*', ['$1' v3_(cdir)]);
    s = regexprep(s,   '(ChfRayPos=\s*)[^\n]*', ['$1' v3_(cpos)]);
    s = regexprep(s,   '(yGrid=\s*)[^\n]*', ['$1' v3_([0;cos(by);-sin(by)])]);
    fid = fopen(out,'w');  fprintf(fid,'%s',s);  fclose(fid);
end

function v = grab3_(txt, key)
    t = regexp(txt, ['(?m)^\s*' key '=\s*([^\n]*)'], 'tokens', 'once');
    v = sscanf(strrep(t{1},'D','E'), '%f', 3);
end
function v = grab1_(txt, key)
    t = regexp(txt, ['(?m)^\s*' key '=\s*([^\n]*)'], 'tokens');
    v = zeros(1, numel(t));
    for i = 1:numel(t), v(i) = sscanf(strrep(t{i}{1},'D','E'), '%f', 1); end
end
function s = v3_(v),  s = sprintf('%.16E  %.16E  %.16E', v(1), v(2), v(3));  end
function del_(p),  if exist(p,'file'), delete(p); end,  end
function s = tern_(c,a,b), if c, s = a; else, s = b; end, end
