function out = pack_fold(deck_in, folds, deck_out, opts)
%PACK_FOLD  Insert one or more flat folds into a COMMITTED prescription.
%
%   OUT = PACK_FOLD(DECK_IN, FOLDS, DECK_OUT) writes a new prescription in
%   which each entry of FOLDS is a flat Reflector inserted into the train,
%   with everything downstream mapped by that fold plane's reflection
%   isometry.  The OPTICS ARE NOT TOUCHED: conics, radii, spacings and the
%   source are carried across verbatim, so the folded deck is the same
%   design in a different place -- which is exactly what has to be true for
%   a packaging study, and is ASSERTED afterwards rather than assumed (see
%   PACK_NULL).
%
%   Why a deck-level inserter rather than TELESCOPE/ADD_FOLD: the committed
%   afocal4 decks are the published evidence for the trade, and
%   AFOCAL4_BUILD re-closes a design struct from parameters before emitting.
%   Folding the DECK cannot perturb the design it is folding, and it needs
%   no design struct -- so the packaging study operates on precisely the
%   prescription the deck and the record quote.  (AFOCAL4_BUILD also admits
%   only ONE fold, after the last mirror; the depth this study has to
%   remove sits in the M2 -> field-mirror leg, upstream of that.)
%
%   FOLDS is a struct array (or a cell of structs) with fields
%     .name   element name for the flat
%     .after  name OR index of the element the fold follows.  A fold
%             inserted earlier can be named here, which is how a chain of
%             folds inside ONE leg is built.
%     .dist   distance from that element's VERTEX along the leg, m.  The
%             leg direction is vertex-to-vertex, NOT the chief ray: these
%             trains are coaxial, the vertices lie on the axis, and the
%             axis is what a fold is placed against.  (The chief runs 100+
%             mm off the vertex line at the far end of a biased design --
%             see the aperture PACK_FOLD reports, which is measured about
%             the beam, not about the vertex.)
%     .to     1x3 outgoing direction (need not be normalised)
%
%   The reflection is the ADD_FOLD isometry, verbatim:
%       n = unit(d_in - d_out),  M = I - 2*n*n',  psi_fold = -n
%   and for every downstream element  x -> P + M*(x - P),  v -> M*v.
%   The element coordinate frame TElt is REBUILT from the mapped psi with
%   the emitter's own convention (z along psi, y = the global +y component
%   orthogonalised, x = y cross z) rather than transformed: TElt is
%   trace-neutral and derived from psi, and a reflection is
%   orientation-REVERSING, so mapping the frame directly would hand MACOS a
%   left-handed perturbation frame.
%
%   Name-value:
%     'quiet'  (false)
%
%   Returns OUT with .deck (the written path), .M (3x3 per fold), .P (3x1
%   per fold), .names, .vpt/.psi before and after, .zElt.
%
%   See also PACK_LEGS, PACK_ROUTE, PACK_NULL, PACK_CLEAR.

    arguments
        deck_in  (1,:) char
        folds
        deck_out (1,:) char
        opts.quiet (1,1) logical = false
    end
    if iscell(folds), folds = [folds{:}]; end

    D = read_deck_(deck_in);

    for q = 1:numel(folds)
        f = folds(q);
        k = elt_index_(D, f.after);
        if k >= numel(D.blk)
            error('pack_fold:last', 'cannot fold after the last element (%s).', ...
                  D.name{k});
        end
        seg  = D.vpt(:,k+1) - D.vpt(:,k);
        slen = norm(seg);
        if ~(f.dist > 0 && f.dist < slen)
            error('pack_fold:dist', ...
                ['fold ''%s'': dist %.4f m is not inside the %s->%s spacing ' ...
                 '%.4f m.'], f.name, f.dist, D.name{k}, D.name{k+1}, slen);
        end
        din  = seg / slen;
        dout = f.to(:) / norm(f.to);
        if norm(dout - din) < 1e-12
            error('pack_fold:straight', 'fold ''%s'': ''to'' equals the incoming direction.', f.name);
        end
        P = D.vpt(:,k) + f.dist * din;
        n = din - dout;   n = n / norm(n);
        M = eye(3) - 2*(n*n.');

        % ---- map everything downstream -------------------------------
        for j = k+1:numel(D.blk)
            D.vpt(:,j) = P + M*(D.vpt(:,j) - P);
            D.rpt(:,j) = P + M*(D.rpt(:,j) - P);
            D.psi(:,j) = M*D.psi(:,j);
        end

        % ---- the flat itself -----------------------------------------
        fb = fold_block_(f.name, P, -n, slen - f.dist);
        D.blk  = [D.blk(1:k),  {fb},      D.blk(k+1:end)];
        D.name = [D.name(1:k), {f.name},  D.name(k+1:end)];
        D.vpt  = [D.vpt(:,1:k), P,        D.vpt(:,k+1:end)];
        D.rpt  = [D.rpt(:,1:k), P,        D.rpt(:,k+1:end)];
        D.psi  = [D.psi(:,1:k), -n,       D.psi(:,k+1:end)];
        D.zElt = [D.zElt(1:k),  slen-f.dist, D.zElt(k+1:end)];
        D.zElt(k) = f.dist;                    % the shortened leg into the fold
        D.isfold = [D.isfold(1:k), true, D.isfold(k+1:end)];

        folds(q).P = P;  folds(q).M = M;  folds(q).n = n;  folds(q).at = k+1;
    end

    write_deck_(D, deck_out);
    out = struct('deck',deck_out, 'folds',folds, 'names',{D.name}, ...
                 'vpt',D.vpt, 'psi',D.psi, 'zElt',D.zElt, 'isfold',D.isfold);
    if ~opts.quiet
        fprintf('  pack_fold: %d fold(s) -> %s\n', numel(folds), deck_out);
        for j = 1:numel(D.name)
            fprintf('    %-3d %-10s vpt [%+8.4f %+8.4f %+8.4f]  psi [%+7.4f %+7.4f %+7.4f]\n', ...
                    j, D.name{j}, D.vpt(:,j), D.psi(:,j));
        end
    end
end

% =====================================================================
function D = read_deck_(f)
%READ_DECK_  Split a prescription into header / element blocks / trailer and
%   pull the geometry each block carries.  The ELEMENT COUNT COMES FROM THE
%   BLOCKS, not from the declared nElt: several decks in this corpus declare
%   an nElt that disagrees with their Element= block count, and taking the
%   declared value shifts every index (the corpus-indexing lesson).
    txt   = fileread(f);
    lines = strsplit(txt, newline);
    if isempty(lines{end}), lines(end) = []; end

    ib = find(~cellfun(@isempty, regexp(lines, '^\s*iElt=\s*\d+', 'once')));
    if isempty(ib), error('pack_fold:parse', 'no iElt= blocks in %s', f); end
    it = find(~cellfun(@isempty, regexp(lines, '^%\s*Output Coordinate', 'once')), 1);
    if isempty(it), it = numel(lines)+1; end

    D.head = lines(1:ib(1)-1);
    D.tail = lines(it:end);
    ends   = [ib(2:end)-1, it-1];
    n      = numel(ib);
    D.blk  = cell(1,n);
    for k = 1:n, D.blk{k} = lines(ib(k):ends(k)); end

    D.name = cell(1,n);   D.vpt = zeros(3,n);  D.rpt = zeros(3,n);
    D.psi  = zeros(3,n);  D.zElt = zeros(1,n); D.isfold = false(1,n);
    for k = 1:n
        D.name{k} = key_str_(D.blk{k}, 'EltName');
        if isempty(D.name{k}), D.name{k} = sprintf('e%d',k); end
        D.vpt(:,k) = key_vec_(D.blk{k}, 'VptElt', 3);
        r = key_vec_(D.blk{k}, 'RptElt', 3);
        if isempty(r), r = D.vpt(:,k); end
        D.rpt(:,k) = r;
        D.psi(:,k) = key_vec_(D.blk{k}, 'psiElt', 3);
        z = key_vec_(D.blk{k}, 'zElt', 1);
        if isempty(z), z = 0; end
        D.zElt(k) = z;
    end
end

function write_deck_(D, f)
%WRITE_DECK_  Re-emit with the mapped geometry substituted into each block.
%   Every line the fold does not touch is carried across BYTE-FOR-BYTE, so a
%   diff of the folded deck against its parent shows exactly the geometry
%   that moved and nothing else.
    L = D.head;
    for k = 1:numel(D.blk)
        b = D.blk{k};
        b = set_i_(b, 'iElt', k);
        b = set_v_(b, 'VptElt', D.vpt(:,k));
        b = set_v_(b, 'RptElt', D.rpt(:,k));
        b = set_v_(b, 'psiElt', D.psi(:,k));
        b = set_z_(b, 'zElt',   D.zElt(k));
        b = set_telt_(b, D.psi(:,k));
        L = [L, b]; %#ok<AGROW>
    end
    L = [L, D.tail];
    % nElt in the header must follow the block count
    for i = 1:numel(L)
        if ~isempty(regexp(L{i}, '^\s*nElt=', 'once'))
            L{i} = sprintf('             nElt=%4d', numel(D.blk));
        end
    end
    fid = fopen(f, 'w');
    if fid < 0, error('pack_fold:write', 'cannot write %s', f); end
    fprintf(fid, '%s\n', L{:});
    fclose(fid);
end

function b = fold_block_(name, P, psi, zNext)
%FOLD_BLOCK_  A flat Reflector block in the emitter's own shape.  ApType is
%   None on purpose: in this layer a design-phase ap_r is a check_clipping
%   BODY, not a stop (the ApType policy), and an honestly-sized flat emitted
%   as a hard aperture turns a packaging study into a ray-loss study.  The
%   clear aperture the flat NEEDS is measured from the traced beam and
%   reported by PACK_CLEAR.
    v3 = @(u) sprintf('%.16E  %.16E  %.16E', u(1), u(2), u(3));
    b = { ...
      '             iElt=  0'
      ['          EltName=  ' name]
      '          Element=  Reflector'
      '          Surface=  Flat'
      '            KrElt=-1.0000000000000000E+22'
      '            KcElt=0.0000000000000000E+00'
      ['           psiElt=  ' v3(psi)]
      ['           VptElt=  ' v3(P)]
      ['           RptElt=  ' v3(P)]
      '           IndRef=1.0E+00'
      '           Extinc=0.0E+00'
      '             nObs=  0'
      '           ApType=  None'
      '         PropType=  Geometric'
      sprintf('             zElt=%.16E', zNext)
      '          nECoord=  6' }.';
    b = [b, telt_lines_(psi)];
end

function L = telt_lines_(psi)
%TELT_LINES_  The 6 TElt rows for a surface frame built from psi, in the
%   emitter's convention (Telescope/surf_frame_): z along psi, y the global
%   +y direction orthogonalised (+x if psi is nearly y), x = y cross z.
%   REBUILT rather than transformed -- see the header note on handedness.
    R  = surf_frame_(psi);
    v6 = @(u,w) sprintf('%.16E  %.16E  %.16E  %.16E  %.16E  %.16E', ...
                        u(1),u(2),u(3),w(1),w(2),w(3));
    z0 = [0;0;0];
    L = {['             TElt=  ' v6(R(:,1),z0)]
         ['                    ' v6(R(:,2),z0)]
         ['                    ' v6(R(:,3),z0)]
         ['                    ' v6(z0,R(:,1))]
         ['                    ' v6(z0,R(:,2))]
         ['                    ' v6(z0,R(:,3))]}.';
end

function R = surf_frame_(psi)
    z = psi(:)/norm(psi);
    yh = [0;1;0];   if abs(z(2)) > 0.95, yh = [1;0;0]; end
    y  = yh - (yh.'*z)*z;   y = y/norm(y);
    x  = cross(y, z);
    R  = [x, y, z];
end

% ---- block editing --------------------------------------------------
function k = elt_index_(D, who)
    if isnumeric(who), k = who;  return; end
    k = find(strcmp(D.name, who), 1);
    if isempty(k)
        error('pack_fold:after', 'element ''%s'' not found (have: %s)', ...
              who, strjoin(D.name, ' '));
    end
end

function s = key_str_(b, key)
    s = '';
    for i = 1:numel(b)
        t = regexp(b{i}, ['^\s*' key '=\s*(\S*)'], 'tokens', 'once');
        if ~isempty(t), s = strtrim(t{1}); return; end
    end
end

function v = key_vec_(b, key, n)
%KEY_VEC_  Read N reals after KEY=, continuing onto following lines when the
%   first does not carry them all (the parser's own wrapped-value rule).
    v = [];
    for i = 1:numel(b)
        t = regexp(b{i}, ['^\s*' key '=\s*(.*)$'], 'tokens', 'once');
        if isempty(t), continue; end
        s = strrep(t{1}, 'D', 'E');
        v = sscanf(s, '%f');
        j = i;
        while numel(v) < n && j < numel(b)
            j = j + 1;
            if ~isempty(regexp(b{j}, '^\s*\w+=', 'once')), break; end
            v = [v; sscanf(strrep(b{j},'D','E'), '%f')]; %#ok<AGROW>
        end
        v = v(1:min(n,numel(v)));
        return;
    end
end

function b = set_v_(b, key, val)
    v3 = sprintf('%.16E  %.16E  %.16E', val(1), val(2), val(3));
    for i = 1:numel(b)
        if ~isempty(regexp(b{i}, ['^\s*' key '='], 'once'))
            pre  = regexp(b{i}, ['^\s*' key '='], 'match', 'once');
            b{i} = [pre '  ' v3];
            return;
        end
    end
end

function b = set_i_(b, key, val)
    for i = 1:numel(b)
        if ~isempty(regexp(b{i}, ['^\s*' key '='], 'once'))
            pre  = regexp(b{i}, ['^\s*' key '='], 'match', 'once');
            b{i} = sprintf('%s%4d', pre, val);
            return;
        end
    end
end

function b = set_z_(b, key, val)
    for i = 1:numel(b)
        if ~isempty(regexp(b{i}, ['^\s*' key '='], 'once'))
            pre  = regexp(b{i}, ['^\s*' key '='], 'match', 'once');
            b{i} = sprintf('%s%.16E', pre, val);
            return;
        end
    end
end

function b = set_telt_(b, psi)
%SET_TELT_  Replace the TElt row block (the TElt= line plus its unlabelled
%   continuation rows) with one rebuilt from psi.  Leaves a block that has
%   no TElt alone.
    i = find(~cellfun(@isempty, regexp(b, '^\s*TElt=', 'once')), 1);
    if isempty(i), return; end
    j = i;
    while j < numel(b) && isempty(regexp(b{j+1}, '^\s*\w+=', 'once')), j = j + 1; end
    b = [b(1:i-1), telt_lines_(psi), b(j+1:end)];
end
