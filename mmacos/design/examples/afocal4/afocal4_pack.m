function K = afocal4_pack(P, deck, opts)
%AFOCAL4_PACK  Is this deck BUILDABLE?  Engine-truth packaging gate.
%
%   K = AFOCAL4_PACK(P, DECK) answers the three parts of the S4b
%   buildability constraint (PLAN_AFOCAL4, BUILDABILITY CONSTRAINT) on a
%   COMMITTED prescription, from traced rays rather than from the paraxial
%   layout the design was closed on:
%
%     1  STATIONS.  Every mirror's vertex z, and z(last) - z(M1) against
%        P.pack.m3_behind_min.  Sky is at -z, so BEHIND M1 is +z.
%     2  FOLD DAYLIGHT.  On the last mirror's EXIT leg -- the leg his own
%        recenter fold picks off -- the lateral gap between that bundle and
%        EVERY OTHER bundle crossing the same station.  Not just the feed
%        leg: in a four-mirror train the M2 -> field-mirror leg also runs
%        through the collimator's station, and it is precisely the check
%        nobody ran that let the S3/S4 layouts through (Dave's note on the
%        S3 gap).  A fold fits where gap > its own body margin.
%     3  INSTRUMENT VOLUME.  Where the pupil and the stated instrument
%        envelope END UP once that fold is inserted, and whether any of it
%        re-enters the incoming beam -- the cylinder of radius
%        P.pack.m1_keepout in front of M1 (z < 0).  This is the part the
%        S3 packaging check did not have, and the reason the S4 trade is
%        retracted rather than amended.
%
%   THE FOLD IS NOT INSERTED HERE.  This is the report that says a fold
%   CAN be inserted and where; AFOCAL4_FOLD builds the folded deck and
%   AFOCAL4_S4B demonstrates it.  Keeping the two apart means the gate can
%   run on every trade point for the price of one trace, while the
%   demonstration -- which re-solves nothing but must be rendered and
%   re-scored -- runs once.
%
%   Name-value:
%     'fields'  K x 2 field offsets to trace, rad (default: the box centre
%               only -- the bundle envelope is what matters and the DRAW
%               fan already spans the pupil)
%     'zfold'   candidate fold stations, m (default: 12 across the exit leg)
%     'init'    load the deck (true)
%     'quiet'   (false)
%
%   Returns K with .behind_m1 .z .names .ok_station, the station scan
%   .fold (.z .gap .c_out .hw_out .clear_of), the chosen .fold_pick,
%   .instr (.z_min .z_max .r_min .clears_m1), .ok and .why.
%
%   See also AFOCAL4_BUILD, AFOCAL4_FOLD, AFOCAL4_S4B, FOLD_STATION_REPORT.

    arguments
        P (1,1) struct
        deck (1,:) char
        opts.zfold (1,:) double = []
        opts.init  (1,1) logical = true
        opts.quiet (1,1) logical = false
    end
    pk = P.pack;

    if opts.init || ~macos.has_rx()
        macos.load_rx(deck);
    end
    txt = fileread(deck);
    V   = grab3_(txt,'VptElt');   names = grab_names_(txt);
    nE  = size(V,2);
    nM  = nE - 1;                        % the last element is the interface

    K = struct('deck',deck, 'names',{names}, 'z',V(3,:), ...
               'behind_m1', V(3,nM) - V(3,1), 'ok',false, 'why','');
    K.ok_station = K.behind_m1 >= pk.m3_behind_min;

    % ---- the legs, as traced ------------------------------------------
    % One YZ meridian fan gives every leg's polyline.  Segment k is the
    % bundle between element k and element k+1; segment 0 is the incoming
    % beam, which is what the instrument must stay out of.
    b = macos.draw_rays('YZ', 0, nE);
    seg = cell(1, nE);                   % seg{k} = [z1 y1 z2 y2] for k -> k+1
    for r = 1:b.nray
        for i = 1:b.nper(r)-1
            ea = b.elt(i,r);   eb = b.elt(i+1,r);
            if eb ~= ea + 1, continue; end
            if ea < 1 || ea > nE, continue; end
            seg{ea}(end+1,:) = [b.U(i,r) b.V(i,r) b.U(i+1,r) b.V(i+1,r)]; %#ok<AGROW>
        end
    end
    K.seg = seg;

    % ---- 2. fold daylight on the last mirror's exit leg ----------------
    out = seg{nM};
    if isempty(out)
        K.why = sprintf('no traced rays on the %s exit leg', names{nM});
        if ~opts.quiet, report_(K, pk); end
        return;
    end
    zz = opts.zfold;
    if isempty(zz)
        zlo = min(out(:,[1 3]),[],'all');  zhi = max(out(:,[1 3]),[],'all');
        pad = 0.05*(zhi - zlo);
        zz  = linspace(zlo+pad, zhi-pad, 12);
    end
    others = setdiff(1:nM-1, []);        % every leg upstream of the exit leg
    F = struct('z',{},'c_out',{},'hw_out',{},'gap',{},'clear_of',{});
    for j = 1:numel(zz)
        yo = ycross_(out, zz(j));
        if isempty(yo), continue; end
        co = 0.5*(min(yo)+max(yo));   ho = 0.5*(max(yo)-min(yo));
        g  = Inf;   who = 'nothing else here';
        for k = others
            yi = ycross_(seg{k}, zz(j));
            if isempty(yi), continue; end
            gk = max(min(yi)-max(yo), min(yo)-max(yi));
            if gk < g, g = gk;  who = sprintf('%s->%s', names{k}, names{k+1}); end
        end
        F(end+1) = struct('z',zz(j), 'c_out',co, 'hw_out',ho, 'gap',g, ...
                          'clear_of',who); %#ok<AGROW>
    end
    % ORDER THE STATIONS ALONG THE BEAM, not along z.  After an odd number
    % of mirrors the exit leg runs back toward -z (his three-mirror does),
    % after an even number it runs +z (the four-mirror does), so "first in
    % the list" means opposite things in the two cases -- and picking the
    % wrong end puts the fold on top of the interface plane.
    [~, ord] = sort(abs([F.z] - V(3,nM)));
    F = F(ord);
    K.fold = F;
    if isempty(F)
        K.why = 'no station where the exit leg crosses';
        if ~opts.quiet, report_(K, pk); end
        return;
    end
    % pick the station with the most daylight that still leaves the pupil
    % downstream of the fold -- a fold past the interface plane moves the
    % instrument but leaves the pupil in the beam.
    zi   = V(3,nE);
    keep = arrayfun(@(f) between_(f.z, V(3,nM), zi), F);
    if ~any(keep)
        K.why = sprintf(['no fold station between %s and the interface ' ...
                         'plane: %.0f mm of room'], names{nM}, abs(zi-V(3,nM))*1e3);
        if ~opts.quiet, report_(K, pk); end
        return;
    end
    % EARLIEST station with room, not the roomiest.  Daylight grows
    % monotonically away from the last mirror because the exit leg walks off
    % the feed leg, so "most gap" always lands on top of the interface plane
    % -- a fold there has nothing left to fold.  The earliest compliant
    % station leaves the whole remaining interface distance as lever arm,
    % which is what pushes the instrument clear of the axis.
    Fk = F(keep);
    kbest = find([Fk.gap] > pk.fold_margin, 1, 'first');
    if isempty(kbest), [~, kbest] = max([Fk.gap]); end
    K.fold_pick = Fk(kbest);
    K.ok_fold   = K.fold_pick.gap > pk.fold_margin;

    % ---- 3. where the instrument ends up -------------------------------
    % The fold turns the exit beam into PK.FOLD_TO -- +x by default, because
    % the field bias is in y and the exit chief is already 18 deg off axis
    % THERE, so +x is the direction in which the telescope has nothing.  A
    % flat fold is an isometry: the optics do not change, only where the
    % volume sits.  Three things then have to be true:
    %
    %   (a) the interface pupil and the envelope behind it lie in the x-y
    %       plane at the fold's station, and that station is BEHIND M1 by
    %       more than half the envelope's own girth (nothing pokes forward
    %       into the incoming beam);
    %   (b) the envelope starts far enough off the telescope axis to clear
    %       the bundles that still run near it at that station -- measured
    %       from the traced XZ fan, not assumed;
    %   (c) nothing of it enters the M1 keep-out cylinder in front of M1.
    zf  = K.fold_pick.z;
    dfp = abs(zi - zf);                       % fold -> pupil, along the beam
    hw  = 0.5*pk.instr_dia;
    K.instr = struct('z_fold',zf, 'off_pupil',dfp, ...
                     'z_min',zf - hw, 'z_max',zf + hw, ...
                     'r_min',dfp, 'r_max',dfp + pk.instr_len);

    % (b) engine-truth: how close to the fold axis does anything else come,
    %     inside the slab the envelope occupies?
    bx = macos.draw_rays('XZ', 0, nE);
    xmax = 0;
    for r = 1:bx.nray
        for i = 1:bx.nper(r)-1
            ea = bx.elt(i,r);   eb = bx.elt(i+1,r);
            % real inter-element legs only: the fan's leading segment comes
            % from the SOURCE plane, whose station is a modelling choice and
            % not a body anything can hit.
            if ea < 1 || eb ~= ea + 1, continue; end
            if ea == nM, continue; end        % the exit leg IS the beam folded
            z1 = bx.U(i,r);  z2 = bx.U(i+1,r);
            if max(z1,z2) < K.instr.z_min || min(z1,z2) > K.instr.z_max, continue; end
            % CLIP the leg to the slab before reading it.  A ray is one
            % straight segment from M2 to M3 here -- 1.7 m long, 500 mm wide
            % at one end -- so taking its endpoints charges the envelope
            % with a beam radius from the far side of the telescope.
            [za, zb] = deal(max(min(z1,z2), K.instr.z_min), ...
                            min(max(z1,z2), K.instr.z_max));
            x1 = bx.V(i,r);  x2 = bx.V(i+1,r);
            for zc = [za zb]
                if abs(z2 - z1) < eps, xc = max(abs(x1),abs(x2));
                else, xc = abs(x1 + (zc - z1)/(z2 - z1)*(x2 - x1));
                end
                xmax = max(xmax, xc);
            end
        end
    end
    % The envelope is a CYLINDER of radius instr_dia/2 about the folded
    % chief, not a line along it -- so what has to clear the beams is its
    % WALL, and the margin carries the -hw.  Getting that wrong makes the
    % check disagree with the one AFOCAL4_S4B runs on the folded deck
    % (which measures point-to-axis distances directly), and the folded
    % deck is the one telling the truth.
    K.instr.x_other  = xmax;
    K.instr.clear_m  = dfp - hw - xmax;
    K.instr.clears_beams = K.instr.clear_m > 0;
    % ... and the actionable form of the same statement: the biggest
    % instrument that fits at this standoff.  A binary verdict against an
    % assumed 300 mm envelope hides the fact that the fold's lever arm --
    % and therefore the interface standoff -- is what sets it.
    K.instr.dia_max = 2*max(0, dfp - xmax);
    % (c) and it must all sit behind the primary
    K.instr.clears_m1 = K.instr.z_min >= 0 || ...
                        K.instr.r_min > pk.m1_keepout;

    % ---- 4. incidence angles on the POWERED surfaces -------------------
    % Reported, not gated.  The compliant closures move the field mirror
    % well past the intermediate image, where the chief ray is high and its
    % incidence is not obviously small any more, and a mirror worked at 30
    % degrees is a different part from one worked at 5.  The FLAT fold and
    % the interface plane are excluded on purpose: a fold's incidence is a
    % packaging choice, not an optical constraint, and folding it into the
    % same column is the e2e2 trap.
    K.aoi = aoi_(b, names, nM);
    K.aoi_max_spread = max([0, K.aoi.spread_deg]);

    K.ok = K.ok_station && K.ok_fold && K.instr.clears_m1 && ...
           K.instr.clears_beams;
    if ~K.ok
        w = {};
        if ~K.ok_station, w{end+1} = sprintf('%s only %.0f mm behind M1', ...
                names{nM}, K.behind_m1*1e3); end
        if ~K.ok_fold,    w{end+1} = sprintf('fold daylight %.1f mm', ...
                K.fold_pick.gap*1e3); end
        if ~K.instr.clears_m1, w{end+1} = 'instrument volume in the incoming beam'; end
        if ~K.instr.clears_beams
            w{end+1} = sprintf(['instrument envelope %.0f mm off axis vs ' ...
                                '%.0f mm of beam'], K.instr.r_min*1e3, ...
                               K.instr.x_other*1e3);
        end
        K.why = strjoin(w, '; ');
    end
    if ~opts.quiet, report_(K, pk); end
end

% =====================================================================
function report_(K, pk)
    fprintf('\n  PACKAGING GATE  %s\n', K.deck);
    fprintf('    stations (m, +z = behind M1):');
    for i = 1:numel(K.z), fprintf('  %s %+.3f', K.names{i}, K.z(i)); end
    fprintf('\n    %s behind M1: %+.0f mm  (minimum %.0f)  %s\n', ...
            K.names{max(1,numel(K.z)-1)}, K.behind_m1*1e3, ...
            pk.m3_behind_min*1e3, tick_(K.ok_station));
    if isfield(K,'fold') && ~isempty(K.fold)
        fprintf('    fold stations on the exit leg:\n');
        fprintf('      %8s %10s %9s %9s  %s\n', ...
                'z (m)','ret ctr','halfw','gap (mm)','nearest bundle');
        for j = 1:numel(K.fold)
            fprintf('      %8.3f %10.4f %9.4f %9.1f  %s\n', K.fold(j).z, ...
                    K.fold(j).c_out, K.fold(j).hw_out, K.fold(j).gap*1e3, ...
                    K.fold(j).clear_of);
        end
    end
    if isfield(K,'fold_pick') && ~isempty(K.fold_pick)
        fprintf('    fold at z %+.3f m: daylight %.1f mm vs %.1f mm margin  %s\n', ...
                K.fold_pick.z, K.fold_pick.gap*1e3, pk.fold_margin*1e3, ...
                tick_(K.ok_fold));
    end
    if isfield(K,'instr') && ~isempty(K.instr)
        fprintf(['    instrument volume: pupil %.0f mm off axis, envelope to ' ...
                 '%.0f mm, in the x-y plane at z %+.3f m (slab %+.3f..%+.3f)  %s\n'], ...
                K.instr.r_min*1e3, K.instr.r_max*1e3, K.instr.z_fold, ...
                K.instr.z_min, K.instr.z_max, tick_(K.instr.clears_m1));
        fprintf(['      envelope wall vs the nearest other bundle: %+.1f mm  ' ...
                 '(offset %.0f, radius %.0f, beam %.0f) -- largest instrument ' ...
                 'that fits here: %.0f mm dia  %s\n'], ...
                K.instr.clear_m*1e3, K.instr.r_min*1e3, 500*pk.instr_dia, ...
                K.instr.x_other*1e3, K.instr.dia_max*1e3, ...
                tick_(K.instr.clears_beams));
    end
    if isfield(K,'aoi') && ~isempty(K.aoi)
        fprintf('    incidence on the powered surfaces (reported, not gated):');
        for i = 1:numel(K.aoi)
            fprintf('  %s %.1f+-%.1f deg', K.aoi(i).name, ...
                    K.aoi(i).mid_deg, 0.5*K.aoi(i).spread_deg);
        end
        fprintf('\n');
    end
    if K.ok, fprintf('    => BUILDABLE\n');
    else,    fprintf('    => NOT BUILDABLE: %s\n', K.why);
    end
end

function A = aoi_(b, names, nM)
%AOI_  Incidence over the meridional fan, per powered mirror: the MEDIAN
%   and the full spread.  Not "the chief ray's AOI" -- the fan IS the
%   pupil, so on an f/1.25 primary the spread is the f-number (M1 reads
%   11.8 deg of it) and only the difference BETWEEN mirrors is a statement
%   about the layout.  A mirror turns the beam by 180 - 2*AOI (normal incidence
%   reverses it), so AOI = 90 - acos(d_in . d_out)/2 per ray -- no surface
%   normal needed (the AOI_REPORT identity).  Read from the YZ fan alone,
%   which is exact here because these designs are coaxial and biased in y:
%   the meridian that carries the chief carries the true angles.
%
%   Taken from the POLYLINE, not from the per-leg segment lists: a ray that
%   drops out on one leg would desync two index-matched lists and turn a
%   missing ray into a wrong angle.
    acc = cell(1, nM);
    for r = 1:b.nray
        for i = 2:b.nper(r)-1
            k = b.elt(i,r);
            % M1 included: its in-leg is the incoming beam, which is a real
            % straight ray, and on a biased design its incidence is the
            % field angle -- worth seeing beside the others.
            if k < 1 || k > nM, continue; end
            di = [b.U(i,r)-b.U(i-1,r), b.V(i,r)-b.V(i-1,r)];
            do = [b.U(i+1,r)-b.U(i,r), b.V(i+1,r)-b.V(i,r)];
            if norm(di) < eps || norm(do) < eps, continue; end
            di = di/norm(di);   do = do/norm(do);
            acc{k}(end+1) = 90 - rad2deg(acos(max(-1,min(1, dot(di,do)))))/2;
        end
    end
    A = struct('name',{},'elt',{},'mid_deg',{},'min_deg',{},'max_deg',{}, ...
               'spread_deg',{});
    for k = 1:nM
        a = acc{k};   a = a(isfinite(a));
        if isempty(a), continue; end
        A(end+1) = struct('name',names{k}, 'elt',k, 'mid_deg',median(a), ...
                          'min_deg',min(a), 'max_deg',max(a), ...
                          'spread_deg',max(a)-min(a)); %#ok<AGROW>
    end
end

function s = tick_(b),  if b, s = 'OK'; else, s = '<-- FAILS'; end,  end

function t = between_(x, a, b)
    t = (x - a)*(x - b) < 0;
end

function yy = ycross_(seg, z0)
%   Where a bundle's polylines cross the plane z = z0.  Segment-wise, so a
%   leg that doubles back (it does not here, but a folded deck can) is
%   counted at every crossing rather than once.
    yy = [];
    if isempty(seg), return; end
    for s = 1:size(seg,1)
        z1 = seg(s,1); y1 = seg(s,2); z2 = seg(s,3); y2 = seg(s,4);
        if (z0-z1)*(z0-z2) < 0
            yy(end+1) = y1 + (z0-z1)/(z2-z1)*(y2-y1); %#ok<AGROW>
        end
    end
end

function M = grab3_(txt, key)
    t = regexp(txt, ['(?m)^\s*' key '=\s*([^\n]*)'], 'tokens');
    M = zeros(3, numel(t));
    for i = 1:numel(t), M(:,i) = sscanf(strrep(t{i}{1},'D','E'), '%f', 3); end
end

function n = grab_names_(txt)
    t = regexp(txt, '(?m)^\s*EltName=\s*(\S*)', 'tokens');
    n = cellfun(@(c) c{1}, t, 'UniformOutput', false);
end
