function C = deck_clearance(rx, opts)
%DECK_CLEARANCE  Leg-vs-body clearances and launch-shroud extent of a prescription.
%   C = DECK_CLEARANCE(RX) loads the deck RX, traces it with the engine's
%   ray-position history (macos.ray_hist) for the field points requested,
%   and scores every BEAM LEG (the straight segment each ray travels
%   between consecutive surface hits) against every BODY the leg does not
%   start or end on.  A body is an element's illuminated patch modelled
%   as a DISC in the element's own plane: centre = the footprint centroid
%   over every traced field, normal = the element's psiElt, radius =
%   opts.margin x the farthest footprint point (every field at once, the
%   hardware must carry them all) + opts.mount.  The distance is the EXACT
%   segment-to-disc minimum (tms_clear's measure, addendum 34), SIGNED: a
%   leg crossing the disc inside its rim returns minus the in-plane depth.
%   Positive = clearance.  The hard minimum over the leg's rays and fields
%   is the leg-body number; C.table lists every pair, worst first.
%
%   Why a deck-level tool: `Telescope.check_clipping` scores a design spec
%   and `dmg_bench_clearance` a Bench rig; a SPLICED train (a telescope
%   with an instrument appended by macos.design.append_rx) is text with no
%   design spec, and its clearance question -- does the instrument's
%   accordion clear the telescope's relay, and the relay's beams clear the
%   instrument -- is the fold beat's gate (PLAN_CONSOLIDATION 3a, the CTB
%   folded into the shroud, 2026-10-29).  Clearance is a NUMBER, not a
%   picture (BRIEF_to_dyson5 addendum 6).
%
%   Also measured, about the launch axis (x = y = 0, +z):
%     C.shroud   per-body radial extent hypot(cx,cy) + r_body (the
%                shroud_deck measure, with the mount), its max over the
%                bodies, and the diameter C.shroud.D = 2*max;
%     C.zmin     the smallest z of any body (how close to the primary's
%                back the hardware comes -- Dave's keep-out behind the
%                PM vertex is a design driver, gate it with opts.keepout_z).
%
%   Name-value:
%     'model'      engine model size (default: the session as it stands,
%                  or 256 when none is open)
%     'fields'     2 x F chief-ray offsets (rad) about the deck's own
%                  chief direction (default [0;0]: the deck's field only);
%                  every field's legs and footprints enter the measure
%     'margin'     footprint radius scale (1.15)
%     'mount'      added to every body radius, BaseUnits (0.008 = 8 mm in m)
%     'bodies'     struct name -> radius override (BaseUnits): a part
%                  larger than its footprint (a detector package, a DM
%                  with its cell), scored at max(footprint, override)
%     'skip'       cellstr of element names that are NOT bodies (markers
%                  the deck keeps for bookkeeping: an exit-pupil sphere
%                  or a bare focus marker); Element=Return is never a body
%     'elts'       element indices whose LEGS are scored (default all);
%                  bodies are always all of them
%     'holes'      struct name -> through-hole radius (BaseUnits): a leg
%                  crossing the body inside that radius of its centre is
%                  the design (a perforated primary), not a piercing, and
%                  the pair is not scored (Telescope.check_clipping's rule)
%     'addon_from' element index where an APPENDED instrument begins (the
%                  splice's n_base): pairs with the leg or the body in the
%                  add-on and the other in the base are the gated table
%                  (C.table); pairs internal to the base go to C.table_base
%                  (the telescope's own clearances) and pairs internal to
%                  the add-on to C.table_addon (the instrument's own) --
%                  separate verdicts, each with its .min_*
%     'keepout_z'  z (BaseUnits) every body must stay beyond (default -Inf)
%     'nleg_rays'  rays per field whose LEGS are scored (33): the chief
%                  plus the rim of the pupil (the farthest ray in each of
%                  (n-1)/2 angular bins) plus a half-radius ring -- the
%                  extremal rays bound a leg's approach to a body; every
%                  ray still enters the footprint discs
%     'quiet'      no printing (false)
%
%   C fields: .table (struct array, worst first: .leg (elt index the leg
%   leaves from), .leg_name, .body, .body_name, .d (signed min distance),
%   .field (which field gave it)), .min, .worst (its row), .bodies
%   (per body: .name .C .n .r .ext .z), .shroud (.ext per body, .max,
%   .D), .zmin, .nrays, .legs_scored, .rx.
%
%   See also SHROUD_DECK, TMS_CLEAR, SPECTROMETER_CLEARANCE, DMG_BENCH_CLEARANCE.

    arguments
        rx (1,:) char
        opts.model     (1,1) double = 0
        opts.fields    (2,:) double = [0;0]
        opts.margin    (1,1) double = 1.15
        opts.mount     (1,1) double = 0.008
        opts.bodies    struct = struct()
        opts.skip      = {}
        opts.elts      = []
        opts.keepout_z (1,1) double = -Inf
        opts.nleg_rays (1,1) double = 33
        opts.holes     struct = struct()
        opts.addon_from (1,1) double = 0
        opts.quiet     (1,1) logical = false
    end
    % model: an explicit size re-inits; otherwise an existing session is
    % used as it is (a size transition is the PLAN.md section-0 heap hazard,
    % never induced silently), and a fresh session opens at 256
    if opts.model > 0
        macos.init(opts.model);
    elseif ~macos.has_rx()
        macos.init(256);
    end
    txt = fileread(rx);
    nE = macos.load_rx(rx);
    kinds = tok_(txt, 'Element');  names = tok_(txt, 'EltName');
    psiT = regexp(txt, '(?m)^\s*psiElt=\s*(\S+)\s+(\S+)\s+(\S+)', 'tokens');
    PSI = zeros(3, nE);
    for k = 1:min(nE, numel(psiT)), PSI(:,k) = str2double(psiT{k}).'; end
    dB = str2double(regexp(txt, 'ChfRayDir=\s*(\S+)\s+(\S+)\s+(\S+)', 'tokens', 'once')).';
    pB = str2double(regexp(txt, 'ChfRayPos=\s*(\S+)\s+(\S+)\s+(\S+)', 'tokens', 'once')).';
    dB = dB/norm(dB);  L0 = norm(pB);
    skip = cellstr(opts.skip);
    isBody = true(1, nE);
    for k = 1:nE
        if k <= numel(kinds) && strcmpi(kinds{k}, 'Return'), isBody(k) = false; end
        if k <= numel(names) && any(strcmp(skip, names{k})), isBody(k) = false; end
    end

    % ---- trace every field, keep every ray's hit at every element ----
    F = opts.fields;  nF = size(F, 2);
    H = cell(1, nF);  nR = 0;
    for f = 1:nF
        if any(F(:,f) ~= 0)
            d = dB + [F(1,f); F(2,f); 0];  d = d/norm(d);
            t2 = regexprep(txt, '(?m)^(\s*ChfRayDir=)[^\n]*', sprintf('$1  %.16E  %.16E  %.16E', d));
            t2 = regexprep(t2, '(?m)^(\s*ChfRayPos=)[^\n]*', sprintf('$1  %.16E  %.16E  %.16E', -L0*d));
            tmp = [tempname '.in'];  fid = fopen(tmp, 'w');  fwrite(fid, t2);  fclose(fid);
            macos.load_rx(tmp);  delete(tmp);
        elseif f > 1
            macos.load_rx(rx);
        end
        macos.ray_hist('on');  s = macos.trace(nE);
        h = macos.ray_hist(s.nRays);  macos.ray_hist('off');
        H{f} = h;  nR = s.nRays;
    end
    % the leg-scoring subset: chief + rim + half-radius ring, from the source-plane pupil
    Ps = H{1}.P(:, :, 1);  okS = H{1}.ok(:, 1).';
    pc = Ps(:, 1);  q = Ps - pc;  d0 = dB;  q = q - d0*(d0.'*q);
    rr = vecnorm(q, 2, 1);  rr(~okS) = -Inf;  rr(1) = -Inf;
    ang = atan2(q(2,:), q(1,:));
    nb = max(1, floor((opts.nleg_rays - 1)/2));  sel = 1;
    edges = linspace(-pi, pi, nb + 1);
    for bq = 1:nb
        m = ang >= edges(bq) & ang < edges(bq+1) & isfinite(rr);
        if ~any(m), continue; end
        [~, i1] = max(rr .* m - ~m);  sel(end+1) = i1;                      %#ok<AGROW>
        rh = abs(rr - 0.5*max(rr)) + ~m*1e9;  [~, i2] = min(rh);  sel(end+1) = i2;  %#ok<AGROW>
    end
    sel = unique(sel);

    % ---- bodies: discs over the union of every field's footprint ----
    B = struct('name', {}, 'elt', {}, 'C', {}, 'n', {}, 'r', {}, 'hole', {}, 'ext', {}, 'z', {});
    for k = 1:nE
        if ~isBody(k), continue; end
        Q = [];
        for f = 1:nF
            m = H{f}.ok(:, k+1);
            if nnz(m) >= 3, Q = [Q, H{f}.P(:, m, k+1)]; end                %#ok<AGROW>
        end
        if size(Q, 2) < 3, continue; end
        n = PSI(:,k);  if norm(n) == 0, continue; end
        n = n/norm(n);  c = mean(Q, 2);
        rr = vecnorm((Q - c) - n*(n.'*(Q - c)), 2, 1);
        r = opts.margin*max(rr);
        nm = '';  if k <= numel(names), nm = names{k}; end
        if ~isempty(nm) && isfield(opts.bodies, nm), r = max(r, opts.bodies.(nm)); end
        r = r + opts.mount;
        hole = 0;
        if ~isempty(nm) && isfield(opts.holes, nm), hole = opts.holes.(nm); end
        B(end+1) = struct('name', nm, 'elt', k, 'C', c, 'n', n, 'r', r, 'hole', hole, ...
                          'ext', hypot(c(1), c(2)) + r, 'z', c(3));        %#ok<AGROW>
    end

    % ---- legs vs bodies ----
    legs = 0:nE-1;                                % the leg LEAVING element i (0 = the source plane)
    if ~isempty(opts.elts), legs = intersect(legs, opts.elts(:).'); end
    rows = struct('leg', {}, 'leg_name', {}, 'body', {}, 'body_name', {}, 'd', {}, 'field', {});
    for i = legs
        for b = 1:numel(B)
            kb = B(b).elt;
            if kb == i || kb == i+1, continue; end
            dmin = Inf;  fmin = 0;
            for f = 1:nF
                h = H{f};
                m = h.ok(:, i+1) & h.ok(:, i+2);
                if i == 0, m = h.ok(:, 2); end
                idx = sel(m(sel));
                for q = idx
                    A = h.P(:, q, i+1);  Bp = h.P(:, q, i+2);
                    dq = seg_disc_(A, Bp, B(b));
                    if dq < dmin, dmin = dq; fmin = f; end
                end
            end
            if ~isfinite(dmin), continue; end
            ln = sprintf('%d', i);  if i >= 1 && i <= numel(names), ln = names{i}; elseif i == 0, ln = 'source'; end
            rows(end+1) = struct('leg', i, 'leg_name', ln, 'body', kb, 'body_name', B(b).name, ...
                                 'd', dmin, 'field', fmin);                %#ok<AGROW>
        end
    end
    [~, o] = sort([rows.d]);  rows = rows(o);
    rows_base = rows([]);  rows_addon = rows([]);
    if opts.addon_from > 0 && ~isempty(rows)
        inL = [rows.leg] >= opts.addon_from;  inB = [rows.body] > opts.addon_from;
        cross = xor(inL, inB);                     % one side in the add-on, one in the base: the gated table
        rows_base = rows(~inL & ~inB);  rows_addon = rows(inL & inB);  rows = rows(cross);
    end
    C = struct('rx', rx, 'table', rows, 'table_base', rows_base, 'table_addon', rows_addon, 'min', Inf, 'worst', [], 'bodies', B, ...
               'shroud', struct('ext', [B.ext], 'max', max([B.ext]), 'D', 2*max([B.ext])), ...
               'zmin', min([B.z]), 'nrays', nR, 'legs_scored', numel(legs), ...
               'keepout_z', opts.keepout_z, 'keepout_margin', min([B.z]) - opts.keepout_z);
    if ~isempty(rows), C.min = rows(1).d;  C.worst = rows(1); end
    C.min_base = Inf;  if ~isempty(rows_base), C.min_base = rows_base(1).d;  C.worst_base = rows_base(1); end
    C.min_addon = Inf;  if ~isempty(rows_addon), C.min_addon = rows_addon(1).d;  C.worst_addon = rows_addon(1); end

    if ~opts.quiet
        fprintf('deck_clearance: %s\n  %d bodies, %d legs, %d fields, %d rays; worst first (%s):\n', ...
                rx, numel(B), numel(legs), nF, nR, 'BaseUnits');
        for q = 1:min(12, numel(rows))
            fprintf('   leg from %-10s vs body %-10s %+9.4f  (field %d)\n', rows(q).leg_name, rows(q).body_name, rows(q).d, rows(q).field);
        end
        fprintf('  min clearance %+.4f; shroud extent max %.4f (D %.4f); zmin %.4f\n', C.min, C.shroud.max, C.shroud.D, C.zmin);
    end
end

% ---------------------------------------------------------------------
function t = tok_(txt, key)
    c = regexp(txt, ['(?m)^\s*' key '=\s*(\S+)'], 'tokens');
    t = cellfun(@(x) x{1}, c, 'UniformOutput', false);
end

function dm = seg_disc_(A, Bp, b)
%SEG_DISC_  tms_clear's exact signed segment-to-disc distance (design/src/tms_clear.m, addendum 34).
    hA = b.n.'*(A - b.C);  hB = b.n.'*(Bp - b.C);
    if hA*hB < 0
        q = A + (Bp - A)*(hA/(hA - hB));
        rad = norm((q - b.C) - b.n*(b.n.'*(q - b.C)));
        if rad < b.hole, dm = Inf; return, end            % through the hole: the design
        if rad < b.r, dm = rad - b.r; return, end
    end
    f = @(s) d2_(A + s*(Bp - A), b);
    ss = linspace(0, 1, 21);  fs = arrayfun(f, ss);  [~, k] = min(fs);
    % a coarse scan brackets the minimum; refine only when it could matter
    % (the exact minimum lies within one scan step of the coarse one)
    if sqrt(fs(k)) > 0.05 + norm(Bp - A)/20, dm = sqrt(fs(k)); return, end
    lo = ss(max(k - 1, 1));  hi = ss(min(k + 1, numel(ss)));
    [~, fm] = fminbnd(f, lo, hi, optimset('TolX', 1e-12, 'Display', 'off'));
    dm = sqrt(min([fm, fs(k)]));
end

function v = d2_(q, b)
    v3 = q - b.C;  h = b.n.'*v3;  rad = norm(v3 - b.n*h);
    v = h^2 + max(rad - b.r, 0)^2;
end
