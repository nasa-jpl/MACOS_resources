function R = pack_legs(deck, opts)
%PACK_LEGS  Engine-truth leg table + packaging envelope of one afocal4 deck.
%
%   R = PACK_LEGS(DECK) loads a committed prescription, traces it, and
%   reports EVERY leg of the chief ray by length, the vertex stations, the
%   beam footprint on each optic, and the envelope the whole train occupies
%   -- all from the traced rays and the engine's own element getters.  The
%   element COUNT and every geometric quantity come from the engine; only
%   the display NAMES are read from the file, and only after asserting that
%   the file's EltName count matches the engine's element count (the
%   corpus-indexing lesson: several decks in this corpus declare an nElt
%   that disagrees with their Element= block count, and taking the declared
%   value attributes the wrong element's numbers).
%
%   THE QUESTION IT ANSWERS.  Sky is at -z, so BEHIND M1 is +z.  A telescope
%   packages when the structure behind the primary fits inside the envelope
%   the telescope already needs in FRONT of it -- practically, inside the
%   M1-M2 spacing.  R reports:
%
%     .span_front_m   |z(M2)|            the M1-M2 spacing (the yardstick)
%     .span_back_m    max z over optics  how deep the back end actually runs
%     .overhang_m     span_back - span_front       the gap, in one number
%     .path_back_m    chief-ray arclength from the M1 plane to the interface
%                     -- the back focal path that has to be folded down
%     .reach_m        that plus the stated instrument length
%     .r_env_m        radial extent of every body about the telescope axis
%
%   Name-value:
%     'instr'   instrument envelope past the interface, m (default 1.000,
%               afocal4 P.pack.instr_len)
%     'init'    load the deck (true);  'quiet' (false)
%
%   Returns R with .names .type .z .vpt .psi .leg (.from .to .len_m .d .z1
%   .z2) .foot_r .foot_c .body_r .span_* .overhang_m .path_back_m .reach_m.
%
%   See also PACK_CLEAR, PACK_FOLD, PACK_ROUTE, AFOCAL4_PACK.

    arguments
        deck (1,:) char
        opts.instr (1,1) double = 1.000
        opts.init  (1,1) logical = true
        opts.quiet (1,1) logical = false
    end

    if opts.init, macos.load_rx(deck); end
    nE = macos.num_elt();

    % ---- display names, count-checked against the engine ---------------
    nm = regexp(fileread(deck), '(?m)^\s*EltName=\s*(\S*)', 'tokens');
    names = cellfun(@(c) c{1}, nm, 'UniformOutput', false);
    if numel(names) ~= nE
        warning('pack_legs:names', ...
            ['%s: %d EltName lines but the engine reports %d elements -- ' ...
             'falling back to indices.'], deck, numel(names), nE);
        names = arrayfun(@(k) sprintf('e%d',k), 1:nE, 'UniformOutput', false);
    end

    % ---- stations + orientations, from the engine ----------------------
    vpt = zeros(3,nE);  psi = zeros(3,nE);  typ = cell(1,nE);
    for k = 1:nE
        vpt(:,k) = macos.get_elt_vpt(k);
        psi(:,k) = macos.get_elt_psi(k);
        I = macos.get_elt_info(k);   typ{k} = I.type;
    end

    % ---- the traced ray history ----------------------------------------
    macos.ray_hist('on');
    t = macos.trace();
    h = macos.ray_hist(t.nRays);
    macos.ray_hist('off');
    off = size(h.P,3) - nE;                 % 1: a source plane leads

    R = struct('deck',deck, 'names',{names}, 'type',{typ}, 'z',vpt(3,:), ...
               'vpt',vpt, 'psi',psi, 'nElt',nE, 'nRays',t.nRays, ...
               'nLost', t.nRays - sum(h.ok(:,end)), 'rmsWFE_m', t.rmsWFE);

    P  = squeeze(h.P(:,1,:));               % the chief ray
    ok = h.ok(1,:);
    R.chief = P;   R.chief_ok = ok;

    % ---- footprint per element, over every passing ray -----------------
    foot_r = zeros(1,nE);  foot_c = zeros(3,nE);
    for k = 1:nE
        j = k + off;   m = h.ok(:,j);
        if ~any(m), continue; end
        Q = squeeze(h.P(:,m,j));  if size(Q,1) ~= 3, Q = Q(:); end
        foot_c(:,k) = mean(Q,2);
        foot_r(k)   = max(vecnorm(Q - foot_c(:,k)));
    end
    R.foot_r = foot_r;  R.foot_c = foot_c;
    R.body_r = vecnorm(foot_c([1 2],:)) + foot_r;      % about the telescope axis

    % ---- the legs, chief ray -------------------------------------------
    L = struct('from',{},'to',{},'len_m',{},'d',{},'z1',{},'z2',{},'r1',{},'r2',{});
    for k = 1:nE-1
        j = k + off;
        if ~(ok(j) && ok(j+1)), continue; end
        v = P(:,j+1) - P(:,j);
        L(end+1) = struct('from',names{k}, 'to',names{k+1}, ...
            'len_m',norm(v), 'd',v/norm(v), 'z1',P(3,j), 'z2',P(3,j+1), ...
            'r1',foot_r(k), 'r2',foot_r(k+1)); %#ok<AGROW>
    end
    R.leg = L;

    % ---- the envelope ---------------------------------------------------
    zb = vpt(3,:);
    R.span_front_m = abs(min(zb));                 % |z(M2)|; M2 is the -z end
    [R.span_back_m, kb] = max(zb);
    R.deepest      = names{kb};
    R.overhang_m   = R.span_back_m - R.span_front_m;
    R.z_slab       = [min(zb(zb > 0)), max(zb)];   % optics behind the primary
    R.r_env_m      = max(R.body_r);

    % the back focal path: chief arclength from the M1 PLANE (z=0) to the
    % interface.  The leg that crosses z=0 contributes only its z>0 piece.
    pb = 0;
    for k = 1:numel(L)
        z1 = L(k).z1;  z2 = L(k).z2;
        if z1 <= 0 && z2 <= 0, continue; end
        if (z1 < 0) ~= (z2 < 0)
            pb = pb + abs(z2)/abs(z2-z1) * L(k).len_m;
        else
            pb = pb + L(k).len_m;
        end
    end
    R.path_back_m = pb;
    R.instr_len   = opts.instr;
    R.reach_m     = pb + opts.instr;

    if ~opts.quiet, report_(R); end
end

% =====================================================================
function report_(R)
    fprintf('\n  LEGS (engine truth)  %s\n', R.deck);
    fprintf('    %-3s %-10s %-11s %10s %10s %10s %10s\n', ...
            'i','element','type','x (m)','y (m)','z (m)','foot r mm');
    for k = 1:R.nElt
        fprintf('    %-3d %-10s %-11s %10.4f %10.4f %+10.4f %10.1f\n', k, ...
                R.names{k}, R.type{k}, R.vpt(1,k), R.vpt(2,k), R.vpt(3,k), ...
                R.foot_r(k)*1e3);
    end
    fprintf('    %-24s %10s %10s %10s\n', 'chief leg','length m','z from','z to');
    for k = 1:numel(R.leg)
        fprintf('    %-24s %10.4f %+10.4f %+10.4f\n', ...
                sprintf('%s -> %s', R.leg(k).from, R.leg(k).to), ...
                R.leg(k).len_m, R.leg(k).z1, R.leg(k).z2);
    end
    fprintf(['    M1-M2 spacing (front envelope)  %8.4f m\n' ...
             '    deepest element behind M1       %8.4f m  (%s)\n' ...
             '    OVERHANG                        %+8.4f m  (%.2fx the front span)\n' ...
             '    optics slab behind M1           %+8.4f .. %+.4f m  (depth %.4f)\n' ...
             '    back focal PATH, M1 plane -> interface %8.4f m  (%.2fx)\n' ...
             '    + instrument %.3f m  =>  reach  %8.4f m  (%.2fx)\n' ...
             '    radial extent of any body       %8.4f m\n' ...
             '    rays %d, lost %d, rms WFE %.4f um\n'], ...
            R.span_front_m, R.span_back_m, R.deepest, R.overhang_m, ...
            R.span_back_m/R.span_front_m, R.z_slab(1), R.z_slab(2), ...
            diff(R.z_slab), R.path_back_m, R.path_back_m/R.span_front_m, ...
            R.instr_len, R.reach_m, R.reach_m/R.span_front_m, R.r_env_m, ...
            R.nRays, R.nLost, R.rmsWFE_m*1e6);
end
