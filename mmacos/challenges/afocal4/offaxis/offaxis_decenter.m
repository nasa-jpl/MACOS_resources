function info = offaxis_decenter(deck, h, opts)
%OFFAXIS_DECENTER  Turn a coaxial afocal deck into an OFF-AXIS SECTION one.
%
%   INFO = OFFAXIS_DECENTER(DECK, H) displaces the entering beam by H metres
%   off the parent axis, in place, and re-fits every mirror's clear aperture
%   to the footprint the decentered beam actually makes.  The parent conics,
%   the vertex stations and every power are UNTOUCHED.
%
%   WHY THIS IS THE OFF-AXIS SEED AND NOT A PERTURBATION OF ONE.  An off-axis
%   section is not a different optical system from its parent -- it is the
%   SAME system used away from its axis.  Paraxially the powers, the spacings
%   and therefore the afocal condition, the magnification and the pupil
%   conjugate are all IDENTICAL to the coaxial train's; what changes is which
%   part of each surface the light uses, and hence the aberrations, the
%   obscuration and the parts you have to build.  That is why the whole
%   descent machinery -- DESCENT_CLOSE's three exact closures, the merit, the
%   walls -- applies to an off-axis design with no change whatsoever, and why
%   this routine is a deck edit rather than a new closure.
%
%   It is also why the seed is EXACT where a cold N-mirror closure is not.
%   Measured on the confocal parabola pair (f1/f2 = 30) at decenters of 0,
%   0.6, 0.8 and 1.0 m: collimation 0.00 urad at EVERY decenter, to the last
%   bit -- a parabola images a parallel beam to its focus from any part of
%   its surface, so the Mersenne stays afocal off-axis by construction, not
%   by convergence.  (The descent's warning stands for cold closures: one
%   probe traced M = 40.45 against a paraxial 30.  It does not apply here.)
%
%   THE APERTURES ARE MEASURED, NOT ASSUMED -- and this is the step that
%   makes the deck traceable at all.  The emitted coaxial deck centres each
%   ApVec on the element VERTEX, so a beam displaced by 0.6 m walks straight
%   off a 0.55 m primary: the same probe lost 777 of 1185 rays at H = 0.6 and
%   1173 of 1185 at H = 1.0, and reported the SURVIVORS' diameter as a
%   magnification of 37.65x and 106.36x.  A clipped beam does not announce
%   itself as clipped; it announces itself as a different telescope.  So the
%   fit runs BEFORE any number is taken, and the ray count is returned.
%
%   FRAME.  ApVec offsets are element-LOCAL, read by the engine as
%       rho = intersection - VptElt,   px = xObs.rho,   py = yObs.rho
%   with the triad zObs = psiElt, yObs = unit(zObs x xObs), xObs = yObs x
%   zObs, seeded by the ChkDf2 default xObs = (psi_z, psi_x, psi_y) when the
%   deck declares none (these decks never do).  Written out in full because
%   handing back a GLOBAL offset as a LOCAL one is a defect this codebase has
%   shipped twice -- it put a stop 3.37 m off the beam on a tilted 6 m design
%   and is correct only while an element sits at the origin with
%   psi = (0,0,-1).  The general rule is implemented here; the special case
%   is not assumed.
%
%   Name-value:
%     'axis'    decenter direction, global unit 3-vector (default [0 1 0] --
%               +Y, the field-bias plane, so the decenter and the field box
%               share a plane of symmetry and the design stays plane-
%               symmetric.  Decentering across the bias plane would break
%               that symmetry for nothing).
%     'fields'  Kx2 field points (rad) the apertures must span (default: the
%               deck's own bias point alone; pass P.Fsolve to size for the
%               full field box, which is what a QUOTED design needs).
%     'margin'  fractional radius margin on the fitted apertures (0.05).
%   THE MEASURING PASS REMOVES THE APERTURES (ApType=None); IT DOES NOT WIDEN
%   THEM.  Widening looks equivalent and is not.  A clear aperture is read by
%   the engine against the SURFACE it sits on, and these trains carry mirrors
%   whose radius of curvature is smaller than the entering beam: the Mersenne
%   secondary at M = 30 has Kr = -0.083 m and carries a 33 mm beam.  Opening
%   its aperture to a radius that would comfortably contain a decentered 1 m
%   pupil asks the engine to intersect rays with a parabola 75 radii from its
%   own vertex, and it answers -- correctly -- with a surface MISS.  Measured
%   on the N = 5 cass seed at h = 0.55 m: the widen-then-measure pass lost all
%   1185 rays (522 surface miss, 663 obscured) and the same deck with
%   ApType=None traces 1185 of 1185, every ray both geometrically valid and
%   unvignetted.  The failure is not subtle in hindsight and was not visible
%   in prospect; it is recorded here so the next person does not re-derive it.
%
%   Name-value continued:
%     'fit'     re-fit the apertures (true).  FALSE LEAVES THE DECK WITH NO
%               APERTURES AT ALL (ApType=None on every element) -- useful for
%               isolating whether a ray loss is aperture or geometry, and
%               never a quoted design: an unapertured train has no stop, so
%               nothing in it is vignetted and nothing in it is realizable.
%     'quiet'   (true)
%
%   Returns INFO with .h .axis .ap (per element: name, r_m, xc_m, yc_m,
%   r_open_m), .nrays, .nmiss (geometric misses during the aperture-free
%   measuring pass), .nlost (rays that fail to TRACE **or** to PASS on the
%   FITTED deck -- the count that decides whether a number means anything;
%   see step 5 for why the two are not the same), and .traced.
%
%   See also OFFAXIS_BUILD, DESCENT_BUILD, AFOCAL4_UNION.

    arguments
        deck (1,:) char
        h    (1,1) double
        opts.axis   (1,3) double  = [0 1 0]
        opts.fields (:,2) double  = []
        opts.margin (1,1) double  = 0.05
        opts.open   (1,1) double  = 0
        opts.fit    (1,1) logical = true
        opts.quiet  (1,1) logical = true
    end

    ax = opts.axis(:).';   ax = ax / norm(ax);
    F  = opts.fields;
    rop = opts.open;   if rop == 0, rop = 4*(abs(h) + 1); end

    % ---- 1. displace the entering beam ----------------------------------
    txt = fileread(deck);
    cp  = grab3_(txt, 'ChfRayPos');
    st  = grab3_(txt, 'ApStop');
    txt = put3_(txt, 'ChfRayPos', cp + h*ax(:));
    txt = put3_(txt, 'ApStop',    st + h*ax(:));
    write_(deck, txt);

    % ---- 2. REMOVE every aperture, so the MEASUREMENT is not clipped -----
    txt = regexprep(fileread(deck), '(?m)(^\s*ApType=\s*)\S+', '$1None');
    write_(deck, txt);

    info = struct('h',h, 'axis',ax, 'ap',[], 'nrays',0, 'nlost',0, ...
                  'traced',[], 'fitted',opts.fit, 'r_open',rop);

    % ---- 3. measure each footprint in its OWN element frame --------------
    [C, R, nm, nray, nmiss] = footprints_(deck, F, opts.margin);
    info.nrays = nray;   info.nmiss = nmiss;   info.nlost = nmiss;

    ap = struct('name',{},'r_m',{},'xc_m',{},'yc_m',{},'r_open_m',{});
    for k = 1:numel(R)
        ap(end+1) = struct('name',nm{k}, 'r_m',R(k), ...
                           'xc_m',C(1,k), 'yc_m',C(2,k), 'r_open_m',rop); %#ok<AGROW>
    end
    info.ap = ap;

    % ---- 4. write the fitted sections back -------------------------------
    if opts.fit
        txt = fileread(deck);
        for k = 1:numel(R)
            if R(k) <= 0, continue; end          % nothing landed: leave open
            txt = set_apvec_(txt, k, R(k), C(1,k), C(2,k));
        end
        write_(deck, txt);
    end

    % ---- 5. THROUGHPUT, measured on the FITTED deck ----------------------
    % RAY_HIST's `ok` is GEOMETRIC validity: an obscured ray keeps a valid
    % intersection (elemsub.F sets the flux flag, not RayPos), so a count
    % taken from it reports a fully VIGNETTED beam as lossless.  That is the
    % wrong notion for the guard this routine advertises -- the fitted
    % apertures are sized to the measured footprint plus a margin, so
    % vignetting SHOULD be nil, and the whole point of the guard is to
    % measure that rather than assume it.  So the loss reported to callers is
    % taken from RAY_INFO's ok_trace AND ok_pass, over the same field set.
    if opts.fit
        [info.nlost, info.nrays] = throughput_(deck, F);
    end

    info.traced = traced_(deck);

    if ~opts.quiet, report_(info); end
end

% =====================================================================
function [C, R, nm, nray, nlost] = footprints_(deck, F, margin)
%FOOTPRINTS_  Union footprint per element, in the element's own ApVec frame.
%   Engine truth: the ray HISTORY, not a meridian fan and not the .in text.
    macos.load_rx(deck);
    nE  = macos.num_elt();
    psi = zeros(3,nE);   V = zeros(3,nE);   nm = cell(1,nE);
    for k = 1:nE
        psi(:,k) = macos.get_elt_psi(k);
        V(:,k)   = macos.get_elt_vpt(k);
        nm{k}    = sprintf('elt%d', k);
    end
    if isempty(F), F = [0 0]; end                % the deck's own bias point

    % Field points are posed the way AFOCAL4_UNION poses them, deliberately:
    % ChfRayDir/yGrid rewritten and the chief position RE-DERIVED from ApStop
    % at fixed standoff.  That is what keeps a DECENTERED pupil decentered as
    % the field swings -- the beam pivots about the stop, which is where the
    % decenter was applied.  Setting a field any other way would walk the
    % off-axis section back toward the axis with field angle and fit the
    % apertures to a bundle the scorer never traces.
    txt   = fileread(deck);
    cd0   = grab3_(txt,'ChfRayDir');   cp0 = grab3_(txt,'ChfRayPos');
    apst  = grab3_(txt,'ApStop');
    stand = dot(apst - cp0, cd0);
    bx0   = asin(cd0(1));   by0 = asin(cd0(2));
    tmp   = [tempname '.in'];
    cu    = onCleanup(@() del_(tmp)); %#ok<NASGU>

    U = cell(1,nE);   W = cell(1,nE);   nray = 0;   nlost = 0;
    for i = 1:size(F,1)
        bx = bx0 + F(i,1);   by = by0 + F(i,2);
        cdir = [sin(bx); sin(by); sqrt(max(0,1-sin(bx)^2-sin(by)^2))];
        cpos = apst - stand*cdir;
        s = regexprep(txt, '(ChfRayDir=\s*)[^\n]*', ['$1' v3_(cdir)]);
        s = regexprep(s,   '(ChfRayPos=\s*)[^\n]*', ['$1' v3_(cpos)]);
        s = regexprep(s,   '(yGrid=\s*)[^\n]*',     ['$1' v3_([0;cos(by);-sin(by)])]);
        fid = fopen(tmp,'w');  fprintf(fid,'%s',s);  fclose(fid);
        macos.load_rx(tmp);
        macos.ray_hist('on');
        t = macos.trace();
        H = macos.ray_hist(t.nRays);
        macos.ray_hist('off');
        ok = logical(H.ok(:,end));
        nray  = nray  + numel(ok);
        nlost = nlost + nnz(~ok);
        off = size(H.P,3) - nE;                  % history carries the source
        for k = 1:nE
            [xo, yo] = obs_frame_(psi(:,k));
            d = squeeze(H.P(:,ok,k+off)) - V(:,k);
            if isempty(d), continue; end
            U{k} = [U{k}, xo.'*d];   W{k} = [W{k}, yo.'*d];
        end
    end
    macos.load_rx(deck);

    C = zeros(2,nE);   R = zeros(1,nE);
    for k = 1:nE
        u = U{k};  v = W{k};
        if isempty(u), continue; end
        % centre on the footprint bounding box, size to the farthest sample.
        % The half-DIAGONAL circumscribes the box and oversizes a round beam
        % by sqrt(2) -- and an oversized clear aperture is not free, it is
        % the part somebody has to polish.
        C(:,k) = [(min(u)+max(u))/2; (min(v)+max(v))/2];
        R(k)   = max(hypot(u - C(1,k), v - C(2,k)))*(1 + margin);
    end
end

function [nlost, nray] = throughput_(deck, F)
%THROUGHPUT_  Rays that both TRACE and PASS, over the field set -- the count
%   that decides whether a number taken off this deck means anything.
    if isempty(F), F = [0 0]; end
    txt   = fileread(deck);
    cd0   = grab3_(txt,'ChfRayDir');   cp0 = grab3_(txt,'ChfRayPos');
    apst  = grab3_(txt,'ApStop');
    stand = dot(apst - cp0, cd0);
    bx0   = asin(cd0(1));   by0 = asin(cd0(2));
    tmp   = [tempname '.in'];
    cu    = onCleanup(@() del_(tmp)); %#ok<NASGU>
    nlost = 0;   nray = 0;
    for i = 1:size(F,1)
        bx = bx0 + F(i,1);   by = by0 + F(i,2);
        cdir = [sin(bx); sin(by); sqrt(max(0,1-sin(bx)^2-sin(by)^2))];
        cpos = apst - stand*cdir;
        s = regexprep(txt, '(ChfRayDir=\s*)[^\n]*', ['$1' v3_(cdir)]);
        s = regexprep(s,   '(ChfRayPos=\s*)[^\n]*', ['$1' v3_(cpos)]);
        s = regexprep(s,   '(yGrid=\s*)[^\n]*',     ['$1' v3_([0;cos(by);-sin(by)])]);
        fid = fopen(tmp,'w');  fprintf(fid,'%s',s);  fclose(fid);
        macos.load_rx(tmp);
        t  = macos.trace(macos.num_elt());
        ri = macos.get_ray_info(t.nRays);
        ok = ri.ok_trace(:) & ri.ok_pass(:);
        nray  = nray  + numel(ok);
        nlost = nlost + nnz(~ok);
    end
    macos.load_rx(deck);
end

function [xo, yo] = obs_frame_(psi)
%OBS_FRAME_  The engine's element aperture triad (tracesub.F/propsub.F) with
%   the ChkDf2 default seed xObs = (psi_z, psi_x, psi_y) (iosub.inc).
    z  = psi(:)/norm(psi);
    xs = [z(3); z(1); z(2)];
    yo = cross(z, xs);
    if norm(yo) < 1e-12                          % seed parallel to psi
        xs = [1;0;0];   yo = cross(z, xs);
        if norm(yo) < 1e-12, xs = [0;1;0]; yo = cross(z, xs); end
    end
    yo = yo/norm(yo);
    xo = cross(yo, z);   xo = xo/norm(xo);
end

function s = traced_(deck)
    % Entrance diameter from the deck's own Aperture=, so the magnification
    % reported here is the deck's, not a caller's idea of it.
    tk = regexp(fileread(deck), '(?m)^\s*Aperture=\s*([^\n]*)', 'tokens', 'once');
    Dap = NaN;
    if ~isempty(tk), Dap = sscanf(strrep(tk{1},'D','E'), '%f', 1); end
    macos.load_rx(deck);
    tr = macos.trace(macos.num_elt());   ri = macos.get_ray_info(tr.nRays);
    ok = ri.ok_trace(:) & ri.ok_pass(:);   ok(1) = false;
    if ~any(ok)
        s = struct('exit_dia',NaN, 'collimation_urad',NaN, 'nrays',0, ...
                   'mag',NaN, 'entrance_dia',Dap);
        return;
    end
    dd = ri.dir(:,ok);   dd = dd ./ vecnorm(dd);
    dm = mean(dd,2);     dm = dm/norm(dm);
    q  = ri.pos(:,ok) - mean(ri.pos(:,ok),2);
    q  = q - dm*(dm.'*q);
    dia = 2*max(vecnorm(q));
    s = struct('exit_dia',dia, ...
               'collimation_urad', max(acos(min(1, dm.'*dd)))*1e6, ...
               'nrays', nnz(ok), 'mag', Dap/max(dia,realmin), ...
               'entrance_dia', Dap);
end

% ---- deck surgery ----------------------------------------------------
function v = grab3_(txt, key)
    tk = regexp(txt, ['(?m)^\s*' key '=\s*([^\n]*)'], 'tokens', 'once');
    if isempty(tk)
        error('macos:design:offaxis_decenter:key', ...
              'the deck declares no %s=.', key);
    end
    v = sscanf(strrep(tk{1},'D','E'), '%f', 3);   v = v(:);
end

function txt = put3_(txt, key, v)
    txt = regexprep(txt, ['(?m)(^\s*' key '=\s*)[^\n]*'], ...
                    ['$1' sprintf('%.16E  %.16E  %.16E', v(1), v(2), v(3))], ...
                    'once');
end

function txt = set_apvec_(txt, k, r, xc, yc)
%SET_APVEC_  Rewrite element K's ApVec.  Elements are located by counting
%   Element= blocks, never by line number.
    ix = regexp(txt, '(?m)^\s*Element=');
    if k > numel(ix), return; end
    lo = ix(k);
    if k < numel(ix), hi = ix(k+1) - 1; else, hi = numel(txt); end
    blk = txt(lo:hi);
    val = sprintf('%.16E  %.16E  %.16E', r, xc, yc);
    if isempty(regexp(blk, '(?m)^\s*ApVec=', 'once'))
        return;                                  % element carries no aperture
    end
    blk = regexprep(blk, '(?m)(^\s*ApVec=\s*)[^\n]*', ['$1' val], 'once');
    blk = regexprep(blk, '(?m)(^\s*ApType=\s*)\S+', '$1  Circular', 'once');
    txt = [txt(1:lo-1), blk, txt(hi+1:end)];
end

function s = v3_(v), s = sprintf('%.16E  %.16E  %.16E', v(1), v(2), v(3)); end
function del_(f), if exist(f,'file'), delete(f); end, end

function write_(deck, txt)
    fid = fopen(deck, 'w');
    if fid < 0, error('macos:design:offaxis_decenter:write', ...
                      'cannot write %s', deck); end
    fprintf(fid, '%s', txt);   fclose(fid);
end

function report_(I)
    fprintf('\n  OFF-AXIS SECTION  decenter %.4f m along [%g %g %g]\n', ...
            I.h, I.axis);
    fprintf('    %-6s %12s %12s %12s\n', 'elt','ap r mm','xc mm','yc mm');
    for k = 1:numel(I.ap)
        fprintf('    %-6s %12.4f %12.4f %12.4f\n', I.ap(k).name, ...
                I.ap(k).r_m*1e3, I.ap(k).xc_m*1e3, I.ap(k).yc_m*1e3);
    end
    fprintf(['    rays %d, lost %d;  exit %.4f mm, M %.6f, ' ...
             'collimation %.3f urad\n'], I.nrays, I.nlost, ...
            I.traced.exit_dia*1e3, I.traced.mag, ...
            I.traced.collimation_urad);
end
