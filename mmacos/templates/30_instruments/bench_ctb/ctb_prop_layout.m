function info = ctb_prop_layout(opts)
%CTB_PROP_LAYOUT  Generate the CTB diffraction decks from the bare bench deck.
%   info = CTB_PROP_LAYOUT() reads the geometric bench prescription
%   ctb_planar_stageF.in (17 stations, every PropType=Geometric), traces it
%   for the chief-ray geometry, and writes BOTH validated diffraction
%   models beside it:
%
%     ctb_dcr_gen.in      31 elements -- "compact": the three focal-mask
%                         quartets and the far-field terminal, plus the one
%                         DM1->DM2 near-field leg.  Omits the inter-optic
%                         plane-to-plane propagations.
%     ctb_s2s_dcr_gen.in  44 elements -- "full" surface-to-surface: the same
%                         quartets and terminal, plus a near-field pair on
%                         every inter-optic leg.
%
%   The committed hand decks ctb_dcr.in / ctb_s2s_dcr.in stay the reference;
%   the generated decks are written alongside with the _gen suffix and carry
%   the same element count, order and station indices, so a driver's index
%   map works on either.
%
%   PropType is a parse-time enum with no runtime setter, so every
%   propagation choice is baked into the emitted text here.
%
%   ------------------------------------------------------------------
%   THE THREE STRUCTURES
%   ------------------------------------------------------------------
%   1. FOCAL-MASK QUARTET (one per intermediate focus).  Four surfaces,
%      all Element=Return, replacing the bare focus marker:
%
%        <focus>_FPreturn   Flat   Geometric   at the focus   zElt = 1e22
%        <focus>_EPreturn   Conic  NF1         EP sphere      zElt = +R
%        <focus>            Flat   NF2         at the focus   zElt = 1e22
%        <focus>_EPreturn2  Conic  Geometric   SAME sphere    zElt = +R
%
%      Kr = -R on both spheres; both zElt are +R, written from one variable
%      so the digits are identical.  THE SIGN IS LOAD-BEARING.  propsub.F's
%      NF1 branch (PropType 10) computes the Siegman-Sziklas chirp from
%      zStart = zElt(sphere) and zEnd = zElt(iElt+1) -- the element AFTER
%      the mask, i.e. EPreturn2.  Equal zElts give S = 0, so the sandwich is
%      transparent and the round trip is the identity; EPreturn2 at -R gives
%      S ~ 2R, which is the spurious-defocus failure.  The mask element's
%      own zElt is not read by the chirp (the committed hand decks disagree
%      on it: 1e22 compact, +R full, both correct); 1e22 is emitted here.
%
%      R is not a free choice: it is the exit-pupil conjugate of that focus.
%      It is measured by running FEX on a TRUNCATED deck ending in the
%      triple <upstream optics> / FPreturn / EPreturn / focus-as-FocalPlane
%      -- the same geometry the terminal leg uses.  Verified against the
%      committed decks: the measured radii reproduce Dave's to ratio
%      1.000000000 (Focus23 7017.8526, FPM 1000.0841, FieldStop 415.9278)
%      from arbitrary seeds.
%
%   2. NEAR-FIELD PAIR (one per propagated inter-optic leg):
%
%        Prop<k>_start  Reference Flat  NFPlane    zElt = -L
%        Prop<k>_end    Reference Flat  Geometric  zElt =  0
%
%      L is the chief-ray distance between the two stations; the engine
%      propagates the zElt DIFFERENCE, so -L then 0.  Both planes sit on the
%      chief pierce of their station, normal along the chief -- NOT at a
%      fraction along the segment.  The committed compact deck carried
%      L = 399.94 mm on the DM1->DM2 leg against a 499.92 mm chief distance
%      (0.8x it, from a builder that placed the planes at 10% and 90% along
%      the segment, with the end plane landing behind DM1); corrected
%      2026-08-06, and both committed decks now agree with this rule to all
%      digits.
%
%   3. FAR-FIELD TERMINAL (three surfaces, replacing the bare FPA):
%
%        FP_return  Return     Flat   Geometric  at the FPA  zElt = 1e22
%        ExitPupil  Return     Conic  FarField   EP sphere   zElt = +R
%        FPA        FocalPlane Flat   Geometric  at the FPA  zElt = 1e22
%
%      Emitted with a seed sphere, then re-measured by FEX on the assembled
%      deck and rewritten (the two-pass FEX pattern add_pupil uses).
%
%   Every inserted surface has its vertex at the traced CHIEF PIERCE of its
%   station and its axis along the chief -- not at the element vertex, which
%   on these off-axis parabolas is metres away from the beam.
%
%   ------------------------------------------------------------------
%   Name-value
%   ------------------------------------------------------------------
%     'src'       bare geometric deck        (default ctb_planar_stageF.in)
%     'variant'   'compact' | 'full' | 'both'                 (default both)
%     'outdir'    where the decks are written        (default this example)
%     'suffix'    appended to the deck stem                 (default '_gen')
%     'ngridpts'  diffraction ray grid, odd                    (default 255)
%     'model'     macos model size, >= ngridpts                (default 512)
%     'verify'    load each deck, compare the chief ray against the bare
%                 deck at every real optic, and check the PSF centring
%                                                              (default true)
%
%   Returns a struct array, one entry per generated deck:
%     .name    'compact' | 'full'
%     .out     path written
%     .nElt    element count
%     .ix      struct, EltName -> 1-based index
%     .R       struct, focus name (and ExitPupil) -> sphere radius
%     .chk     verification results (empty when 'verify' is false)
%
%   Run:  >> info = ctb_prop_layout;
%   See also: ctb_coro_compare, tCtbProp, Coro_propagation_summary.md.

    arguments
        opts.src      (1,:) char   = ''
        opts.variant  (1,:) char {mustBeMember(opts.variant, ...
                                  {'compact','full','both'})} = 'both'
        opts.outdir   (1,:) char   = ''
        opts.suffix   (1,:) char   = '_gen'
        opts.ngridpts (1,1) double {mustBeInteger,mustBePositive} = 255
        opts.model    (1,1) double {mustBeInteger,mustBePositive} = 512
        opts.verify   (1,1) logical = true
    end

    here = fileparts(mfilename('fullpath'));
    if isempty(opts.src),    opts.src    = fullfile(here,'ctb_planar_stageF.in'); end
    if isempty(opts.outdir), opts.outdir = here;                                  end
    addpath(fullfile(here, '..', '..', '..', 'src'));      % mmacos/src
    assert(opts.model >= opts.ngridpts, ...
        'model_size (%d) must be >= nGridpts (%d)', opts.model, opts.ngridpts);

    % ------------------------------------------------------------------
    % The bare bench, in light order.  KIND drives the whole assembly:
    %   optic  real powered/flat mirror -- copied verbatim
    %   marker passive pupil reference  -- copied verbatim
    %   focus  intermediate focus       -- becomes a quartet
    %   image  detector                 -- becomes the far-field terminal
    % ------------------------------------------------------------------
    ST = { 'OAP1','optic'; 'DM1','optic'; 'DM2','optic'; 'OAP2','optic'
           'Focus23','focus'; 'OAP3','optic'; 'Apodizer','marker'
           'OAP4','optic'; 'FPM','focus'; 'OAP5','optic'
           'Lyot','marker'; 'OAP6','optic'; 'FieldStop','focus'
           'OAP7','optic'; 'Backend','marker'; 'OAP8','optic'
           'FPA','image' };
    name = ST(:,1).';  kind = ST(:,2).';  nST = numel(name);

    % ------------------------------------------------------------------
    % 1) Chief-ray pierce at every station.
    %    Segment directions come from the pierce points, never from the
    %    ray-buffer direction: at a mirror the buffered direction is the
    %    post-reflection one, and which side of a station it belongs to is
    %    exactly the ambiguity this avoids.
    % ------------------------------------------------------------------
    macos.init(opts.model);
    nE = macos.load_rx(opts.src);
    assert(nE == nST, 'expected a %d-station bare deck, got %d', nST, nE);
    cp = zeros(3, nST);
    for k = 1:nST
        s  = macos.trace(k);
        ri = macos.get_ray_info(s.nRays);
        assert(all(ri.ok_trace & ri.ok_pass), ...
            '%s: station %d (%s) vignettes', opts.src, k, name{k});
        cp(:,k) = ri.pos(:,1);
    end
    uin = @(k) unit_(cp(:,k) - cp(:,k-1));     % beam direction arriving at k

    % ------------------------------------------------------------------
    % 2) Leg numbering.  Walk the gaps from DM1 onward; every gap consumes
    %    a leg index EXCEPT the one leaving a focus (that beam is already
    %    accounted for by the focus quartet).  This reproduces the hand
    %    decks' Prop1/2/4/5/7/8/10/11 numbering exactly -- legs 3, 6, 9 are
    %    the quartets and leg 12 is the terminal.
    % ------------------------------------------------------------------
    leg = zeros(1, nST);                        % leg index of gap k -> k+1
    n = 0;
    for k = 2:nST-1
        if strcmp(kind{k}, 'focus'), continue; end
        n = n + 1;  leg(k) = n;
    end

    % ------------------------------------------------------------------
    % 3) Sphere radius per focus, by FEX on a truncated deck.
    % ------------------------------------------------------------------
    blocks = split_elements_(fileread(opts.src));
    hdr = blocks{1};  foot = blocks{end};  ebl = blocks(2:end-1);
    tmpd = tempname;  mkdir(tmpd);
    R = struct();
    for k = find(strcmp(kind,'focus'))
        R.(name{k}) = fex_radius_(hdr, foot, ebl, k, cp(:,k), uin(k), ...
                                  0.5*norm(cp(:,k) - cp(:,k-1)), ...
                                  name{k}, opts.model, tmpd);
        fprintf('[ctb_prop_layout] %-10s exit-pupil radius = %.10f (FEX)\n', ...
                name{k}, R.(name{k}));
    end

    % ------------------------------------------------------------------
    % 4) Emit each variant.
    % ------------------------------------------------------------------
    switch opts.variant
        case 'both',    vs = {'compact','full'};
        otherwise,      vs = {opts.variant};
    end
    info = struct('name',{},'out',{},'nElt',{},'ix',{},'R',{},'chk',{});
    for v = 1:numel(vs)
        info(v) = emit_variant_(vs{v}, hdr, foot, ebl, name, kind, leg, ...
                                cp, R, opts);   %#ok<AGROW>
    end
end

% ======================================================================
function e = emit_variant_(variant, hdr, foot, ebl, name, kind, leg, ...
                           cp, R, opts)
%EMIT_VARIANT_  Assemble, write, FEX the terminal, rewrite, verify.
    nST = numel(name);
    uin = @(k) unit_(cp(:,k) - cp(:,k-1));
    Z_PLANE = 1.0e22;   KR_FLAT = -1.0e22;

    B  = {};                                  % emitted blocks, light order
    for k = 1:nST
        switch kind{k}
            case 'focus'
                r = R.(name{k});  u = uin(k);  F = cp(:,k);
                B{end+1} = render_([name{k} '_FPreturn'], 'Return','Flat', ...
                              'Geometric', F, u, KR_FLAT, Z_PLANE); %#ok<AGROW>
                B{end+1} = render_([name{k} '_EPreturn'], 'Return','Conic', ...
                              'NF1', F - r*u, u, -r, r);            %#ok<AGROW>
                B{end+1} = convert_marker_(ebl{k}, 'Return','Flat','NF2', ...
                              F, u, KR_FLAT, Z_PLANE);              %#ok<AGROW>
                B{end+1} = render_([name{k} '_EPreturn2'], 'Return','Conic', ...
                              'Geometric', F - r*u, u, -r, r);      %#ok<AGROW>

            case 'image'
                % Seed sphere; FEX re-measures it on the assembled deck.
                u = uin(k);  F = cp(:,k);
                rs = 0.5 * norm(cp(:,k) - cp(:,k-1));
                B{end+1} = render_('FP_return', 'Return','Flat','Geometric', ...
                              F, -u, KR_FLAT, Z_PLANE);             %#ok<AGROW>
                B{end+1} = render_('ExitPupil', 'Return','Conic','FarField', ...
                              F - rs*u, u, -rs, rs);                %#ok<AGROW>
                B{end+1} = convert_marker_(ebl{k}, '', 'Flat','Geometric', ...
                              F, u, KR_FLAT, Z_PLANE);              %#ok<AGROW>

            otherwise
                B{end+1} = ebl{k};                                  %#ok<AGROW>
        end

        % --- near-field pair on the gap k -> k+1 ----------------------
        % Skipped on three kinds of gap: the one LEAVING a focus (leg 0 --
        % that beam belongs to the quartet), the one ENTERING a focus (the
        % quartet IS its propagation, legs 3/6/9), and the terminal
        % (leg 12, the far-field triple).
        if leg(k) == 0 || strcmp(kind{k+1},'focus') || k == nST-1
            continue
        end
        u = unit_(cp(:,k+1) - cp(:,k));
        L = norm(cp(:,k+1) - cp(:,k));
        if strcmp(variant,'full')
            B{end+1} = render_(sprintf('Prop%d_start',leg(k)), 'Reference', ...
                          'Flat','NFPlane',   cp(:,k),   u, KR_FLAT, -L); %#ok<AGROW>
            B{end+1} = render_(sprintf('Prop%d_end',leg(k)),   'Reference', ...
                          'Flat','Geometric', cp(:,k+1), u, KR_FLAT, 0);  %#ok<AGROW>
        elseif leg(k) == 1
            % compact: the DM1->DM2 leg is the only propagated one.
            B{end+1} = render_('P1_start', 'Reference','Flat','NFPlane', ...
                          cp(:,k),   u, KR_FLAT, -L);                     %#ok<AGROW>
            B{end+1} = render_('P1_end',   'Reference','Flat','Geometric', ...
                          cp(:,k+1), u, KR_FLAT, 0);                      %#ok<AGROW>
        elseif strcmp(name{k+1}, 'Apodizer')
            % compact: an inert Geometric plane at the leg midpoint, kept
            % so the compact model's station indices match the committed
            % ctb_dcr.in.  It carries no propagation.
            B{end+1} = render_('Apodizer_Pst', 'Reference','Flat','Geometric', ...
                          cp(:,k) + 0.5*L*u, u, KR_FLAT, Z_PLANE);        %#ok<AGROW>
        end
    end

    switch variant
        case 'compact', stem = 'ctb_dcr';
        case 'full',    stem = 'ctb_s2s_dcr';
    end
    out = fullfile(opts.outdir, sprintf('%s%s.in', stem, opts.suffix));
    write_deck_(out, hdr, B, foot, opts.ngridpts);

    % --- two-pass FEX on the terminal exit pupil ----------------------
    macos.init(opts.model);
    nT = macos.load_rx(out);
    s  = macos.fex(1);
    iEP = nT - 1;
    rEP = abs(s.rad);
    B{iEP} = render_('ExitPupil', 'Return','Conic','FarField', ...
                     s.vpt(:), s.psi(:), -rEP, rEP);
    write_deck_(out, hdr, B, foot, opts.ngridpts);
    R.ExitPupil = rEP;
    fprintf('[ctb_prop_layout] %-10s exit-pupil radius = %.10f (FEX)\n', ...
            'ExitPupil', rEP);

    e = struct('name', variant, 'out', out, 'nElt', numel(B), ...
               'ix', index_map_(B), 'R', R, 'chk', []);
    fprintf('[ctb_prop_layout] wrote %s (%d elements, nGridpts=%d)\n', ...
            out, e.nElt, opts.ngridpts);

    if opts.verify
        e.chk = verify_deck_(e, name, kind, cp, opts);
    end
end

% ----------------------------------------------------------------------
function chk = verify_deck_(e, name, kind, cp, opts)
%VERIFY_DECK_  Chief ray at every real optic vs the bare deck, PSF centring.
    macos.init(opts.model);
    nT = macos.load_rx(e.out);
    assert(nT == e.nElt, '%s: loaded %d elements, emitted %d', ...
           e.out, nT, e.nElt);
    dmax = 0;  worst = '';
    for k = 1:numel(name)
        if ~strcmp(kind{k},'optic'), continue; end
        j  = e.ix.(name{k});
        s  = macos.trace(j);
        ri = macos.get_ray_info(s.nRays);
        d  = norm(ri.pos(:,1) - cp(:,k));
        if d > dmax, dmax = d;  worst = name{k}; end
    end
    I = macos.intensity(nT);
    [pk, idx] = max(I(:));  [r,c] = ind2sub(size(I), idx);
    ctr = floor(opts.model/2) + 1;
    chk = struct('chief_max_mm', dmax, 'chief_worst', worst, ...
                 'psf_peak', pk, 'psf_row', r, 'psf_col', c, ...
                 'psf_centre', ctr, 'psf_centred', (r==ctr && c==ctr), ...
                 'dx_fpa_m', macos.dx_at(nT));
    fprintf(['[ctb_prop_layout] %-8s chief-ray match %.3e mm (worst %s); ' ...
             'PSF peak %.4e at [%d,%d] (centre %d)\n'], ...
            e.name, dmax, worst, pk, r, c, ctr);
end

% ----------------------------------------------------------------------
function r = fex_radius_(hdr, foot, ebl, k, F, u, rs, nm, model, tmpd)
%FEX_RADIUS_  Exit-pupil radius conjugate to the focus at station k.
%   Builds <optics 1..k-1> / FPreturn(flat at the focus) / EPreturn(seed
%   sphere) / focus-as-FocalPlane and runs FEX, whose radius is the
%   chief-ray distance from the found pupil to the iElt+1 plane.  The seed
%   only has to load -- FEX overwrites the radius, pose and vertex -- but
%   keep it comparable to the incoming leg so the seed sphere has clean
%   daylight to the real optics: ConSrf picks its intersection root by
%   |L^2 - mpr| proximity, with no flow-of-light sense.
    B = ebl(1:k-1);
    B{end+1} = render_([nm '_FPreturn'], 'Return','Flat','Geometric', ...
                       F, u, -1.0e22, 1.0e22);
    B{end+1} = render_([nm '_EPreturn'], 'Return','Conic','Geometric', ...
                       F - rs*u, u, -rs, rs);
    B{end+1} = render_(nm, 'FocalPlane','Flat','Geometric', ...
                       F, u, -1.0e22, 1.0e22);
    f = fullfile(tmpd, sprintf('fex_%s.in', nm));
    write_deck_(f, hdr, B, foot, []);          % keep the bare ray grid
    macos.init(model);
    macos.load_rx(f);
    s = macos.fex(1);
    r = abs(s.rad);
end

% ----------------------------------------------------------------------
function write_deck_(path, hdr, B, foot, ngridpts)
%WRITE_DECK_  Renumber iElt, patch nElt/nGridpts, write.
%   The iElt substitution MUST be line-anchored: 'iElt' is a substring of
%   'psiElt', so an unanchored pattern eats the leading digit of psiElt.
    n = numel(B);
    for k = 1:n
        B{k} = regexprep(B{k}, '^(\s*)iElt=\s+\d+', ...
                         sprintf('$1iElt=  %d', k), 'lineanchors');
    end
    hdr = regexprep(hdr, '^(\s*)nElt=\s+\d+', sprintf('$1nElt=  %d', n), ...
                    'lineanchors');
    if ~isempty(ngridpts)
        hdr = regexprep(hdr, '^(\s*)nGridpts=\s+\d+', ...
                        sprintf('$1nGridpts=  %d', ngridpts), 'lineanchors');
    end
    fid = fopen(path, 'w');
    assert(fid > 0, 'cannot write %s', path);
    fprintf(fid, '%s', hdr);
    for k = 1:n, fprintf(fid, '%s', B{k}); end
    fprintf(fid, '%s', foot);
    fclose(fid);
end

% ----------------------------------------------------------------------
function s = render_(nm, element, surface, prop, vpt, psi, kr, zElt)
%RENDER_  One inserted surface block.
%   zElt is written with the same format everywhere, so a quartet's two
%   spheres -- fed the same variable -- carry byte-identical digits.
    F = @(v) sprintf('  %.16E  %.16E  %.16E', v(1), v(2), v(3));
    L = {''};
    L{end+1} =         '             iElt=  0';
    L{end+1} = sprintf('          EltName=  %s', nm);
    L{end+1} = sprintf('          Element=  %s', element);
    L{end+1} = sprintf('          Surface=  %s', surface);
    L{end+1} = sprintf('            KrElt=%.16E', kr);
    L{end+1} =         '            KcElt=0.0E+00';
    L{end+1} = sprintf('           psiElt=%s', F(psi));
    L{end+1} = sprintf('           VptElt=%s', F(vpt));
    L{end+1} = sprintf('           RptElt=%s', F(vpt));
    L{end+1} =         '           IndRef=1.0E+00';
    L{end+1} =         '           Extinc=0.0E+00';
    L{end+1} =         '            nCoat=  0';
    L{end+1} =         '             nObs=  0';
    L{end+1} =         '           ApType=  None';
    L{end+1} = sprintf('         PropType=  %s', prop);
    L{end+1} = sprintf('             zElt=%.16E', zElt);
    L{end+1} =         '          nECoord= -6';
    s = [strjoin(L, newline) newline];
end

% ----------------------------------------------------------------------
function blk = convert_marker_(blk, element, surface, prop, vpt, psi, kr, z)
%CONVERT_MARKER_  Retype a bare-deck marker in place, keeping every other
%   field (ApVec, TElt, coating, ...) verbatim.  Pass element='' to leave
%   Element= alone.  All substitutions are line-anchored (see write_deck_).
    v3 = @(v) sprintf('  %.16E  %.16E  %.16E', v(1), v(2), v(3));
    if ~isempty(element)
        blk = sub_(blk, 'Element',  ['  ' element]);
    end
    blk = sub_(blk, 'Surface',  ['  ' surface]);
    blk = sub_(blk, 'PropType', ['  ' prop]);
    blk = sub_(blk, 'KrElt',    sprintf('%.16E', kr));
    blk = sub_(blk, 'zElt',     sprintf('%.16E', z));
    blk = sub_(blk, 'psiElt',   v3(psi));
    blk = sub_(blk, 'VptElt',   v3(vpt));
    blk = sub_(blk, 'RptElt',   v3(vpt));
end

function blk = sub_(blk, key, value)
%SUB_  Replace one key's value in an element block.  The replacement is
%   passed through literally -- do NOT regexptranslate it: that escapes
%   regex metacharacters ('.', '+') which are ordinary text on this side of
%   the substitution and would land in the deck as backslashes.  Only '$'
%   and '\' are special in a replacement, and neither occurs in a numeric
%   or enum value.
    blk = regexprep(blk, ['^(\s*)' key '=.*$'], ['$1' key '=' value], ...
                    'lineanchors', 'dotexceptnewline');
end

% ----------------------------------------------------------------------
function blocks = split_elements_(txt)
%SPLIT_ELEMENTS_  {header; e1..eN; footer}, split on iElt= / nOutCord=.
    lines = regexp(txt, '\r?\n', 'split');
    ei = find(~cellfun('isempty', regexp(lines, '^\s*iElt=',     'once')));
    fi = find(~cellfun('isempty', regexp(lines, '^\s*nOutCord=', 'once')), 1);
    assert(~isempty(ei), 'no iElt= lines found');
    assert(~isempty(fi), 'no nOutCord= footer found');
    join_ = @(a,b) [strjoin(lines(a:b), newline) newline];
    blocks = {join_(1, ei(1)-1)};
    for k = 1:numel(ei)
        if k < numel(ei), b = ei(k+1)-1; else, b = fi-1; end
        blocks{end+1} = join_(ei(k), b);                       %#ok<AGROW>
    end
    blocks{end+1} = join_(fi, numel(lines));
end

% ----------------------------------------------------------------------
function ix = index_map_(B)
    ix = struct();
    for k = 1:numel(B)
        nm = regexp(B{k}, '^\s*EltName=\s*(\S+)', 'tokens', 'once', 'lineanchors');
        if isempty(nm), continue; end
        ix.(matlab.lang.makeValidName(nm{1})) = k;
    end
end

% ----------------------------------------------------------------------
function u = unit_(v)
    u = v(:) / norm(v);
end
