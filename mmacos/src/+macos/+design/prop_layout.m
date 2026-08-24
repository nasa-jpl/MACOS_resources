function info = prop_layout(src, kinds, opts)
%PROP_LAYOUT  Turn a geometric deck into a DIFFRACTION deck.
%
%   info = MACOS.DESIGN.PROP_LAYOUT(SRC, KINDS) reads a bare geometric
%   prescription, traces it for the chief-ray geometry, and writes a deck
%   that carries the complex field: focal-mask QUARTETS at the
%   intermediate foci, a far-field TERMINAL at the detector, and
%   near-field PAIRS on whichever inter-optic legs you ask for.
%
%   KINDS is one label per element of SRC, in light order:
%     'optic'   real powered/flat surface   -- copied verbatim
%     'marker'  passive pupil reference     -- copied verbatim
%     'focus'   intermediate focus          -- becomes a quartet
%     'image'   detector                    -- becomes the far-field triple
%
%   This is `ctb_prop_layout`'s recipe with the station table lifted out
%   as an argument.  That example keeps its own copy: its two committed
%   decks are the reference and `tCtbProp` pins them, so it is not worth
%   re-pointing at a generalization to save duplication.
%
%   ------------------------------------------------------------------
%   THE THREE STRUCTURES (and the sign that is load-bearing)
%   ------------------------------------------------------------------
%   FOCAL-MASK QUARTET, four Element=Return surfaces replacing the bare
%   focus marker:
%
%     <focus>_FPreturn   Flat   Geometric   at the focus   zElt = 1e22
%     <focus>_EPreturn   Conic  NF1         EP sphere      zElt = +R
%     <focus>            Flat   NF2         at the focus -- THE MASK PLANE
%     <focus>_EPreturn2  Conic  Geometric   SAME sphere    zElt = +R
%
%   Kr = -R on both spheres, and BOTH zElt are +R, written from one
%   variable so the digits are identical.  `propsub.F`'s NF1 branch builds
%   the Siegman-Sziklas chirp from `zStart = zElt(sphere)` and
%   `zEnd = zElt(iElt+1)` -- the element AFTER the mask, i.e. EPreturn2.
%   Equal zElts give chirp argument zero, so the sandwich is transparent
%   and the round trip is the identity; EPreturn2 at -R gives an argument
%   of order 2R, a defocus no ray check catches.
%
%   R is MEASURED, not chosen: it is the exit-pupil conjugate of that
%   focus, found by running FEX on a truncated deck ending in the same
%   triple.
%
%   NEAR-FIELD PAIR on a requested leg k -> k+1:
%     Prop<k>_start  Reference Flat NFPlane    at station k    zElt = -L
%     Prop<k>_end    Reference Flat Geometric  at station k+1  zElt =  0
%   L is the CHIEF-RAY distance between the stations and the engine
%   propagates the zElt DIFFERENCE.  Both planes sit on the chief pierce
%   of their own station -- not at a fraction along the segment, which is
%   how the ctb deck once propagated a leg at 0.8x its true length.
%
%   FAR-FIELD TERMINAL, three surfaces replacing the bare detector:
%     FP_return  Return     Flat   Geometric  at the FPA  zElt = 1e22
%     ExitPupil  Return     Conic  FarField   EP sphere   zElt = +R
%     <image>    FocalPlane Flat   Geometric  at the FPA  zElt = 1e22
%   Emitted with a seed sphere, then re-measured by FEX on the ASSEMBLED
%   deck and rewritten -- the two-pass pattern `add_pupil` uses.
%
%   Every inserted surface has its vertex at the traced CHIEF PIERCE of
%   its station and its axis along the chief, never at the element vertex
%   (on an off-axis parabola that is metres from the beam).
%
%   MASKS ARE NEVER DECLARED IN THE DECK.  Propagate to the mask plane,
%   multiply the complex field in MATLAB (`macos.apodize` /
%   `macos.apodize_complex`), continue with 'reset_trace', false.  An
%   obscuration declared on a Reference clips RAYS only and the
%   diffraction wavefront passes through untouched -- a mask that looks
%   right in the deck and suppresses nothing.
%
%   ------------------------------------------------------------------
%   Name-value
%   ------------------------------------------------------------------
%     'out'       output deck path (default: SRC stem + '_prop.in')
%     'nf_legs'   station indices k whose gap k -> k+1 gets a near-field
%                 pair.  Default: the gap ENTERING the first 'marker',
%                 which seeds the field at a PUPIL -- the field has to
%                 exist before any mask plane can multiply it.
%     'ngridpts'  diffraction ray grid, odd ([] = keep SRC's)
%     'model'     macos model size, >= ngridpts (default 512)
%     'stop_name' EltName of the aperture stop.  FEX needs a stop -- the
%                 chief ray, and hence the pupil, is undefined without one
%                 -- and it is resolved BY NAME because the element's index
%                 shifts as quartets are inserted.  Default '' = the deck
%                 declares its own (a Telescope deck carries ApStop= in the
%                 header; a bare Bench deck does not).
%     'verify'    trace the result and check the chief ray against SRC at
%                 every original station (default true)
%
%   Returns .out .nElt .ix (EltName -> index) .R (focus name -> radius)
%   .chk (verification, [] when 'verify' is false).
%
%   See also ctb_prop_layout, macos.design.Telescope/add_pupil,
%   macos.apodize.

    arguments
        src   (1,:) char
        kinds (1,:) cell
        opts.out      (1,:) char   = ''
        opts.nf_legs  (1,:) double = []
        opts.ngridpts double = []
        opts.model    (1,1) double {mustBeInteger,mustBePositive} = 512
        opts.stop_name (1,:) char  = ''
        opts.verify   (1,1) logical = true
    end
    assert(isfile(src), 'prop_layout: %s not found', src);
    valid = {'optic','marker','focus','image'};
    for k = 1:numel(kinds)
        assert(any(strcmp(kinds{k}, valid)), ...
            'prop_layout: kinds{%d} = "%s" is not one of optic|marker|focus|image', ...
            k, kinds{k});
    end
    if isempty(opts.out)
        [d,b] = fileparts(src);  opts.out = fullfile(d, [b '_prop.in']);
    end
    nST = numel(kinds);

    % ---- 1) chief-ray pierce at every station ----------------------------
    % Segment directions come from the PIERCE POINTS, never from the ray
    % buffer's direction: at a mirror the buffered direction is the
    % post-reflection one, and which side of a station it belongs to is
    % exactly the ambiguity this avoids.
    macos.init(opts.model);
    nE = macos.load_rx(src);
    assert(nE == nST, ...
        'prop_layout: %s has %d elements, kinds has %d', src, nE, nST);
    % ONE traced pass, read through the engine's RayPosHist.  Not a
    % per-station macos.trace(k) loop: on a SEGMENTED deck the first N
    % elements are parallel segments of the same mirror, so "trace past
    % element 5" is not a station and the engine rejects it.  ray_hist
    % records every ray's crossing at every element in a single pass, and
    % ray 1 is the chief.
    macos.ray_hist('on');
    tr = macos.trace(nE);
    h  = macos.ray_hist(tr.nRays);
    macos.ray_hist('off');
    cp = squeeze(h.P(:, 1, 2:end));              % 3 x nST, chief pierces
    ok_st = h.ok(1, 2:end);                      % chief reached this station
    uin = @(k) unit_(cp(:,k) - cp(:,k-1));       % direction arriving at k

    need = @(k) assert(ok_st(k), ...
        ['prop_layout: the chief ray does not reach station %d (%s) -- ' ...
         'its pose cannot be measured'], k, kinds{k});

    if isempty(opts.nf_legs)
        im = find(strcmp(kinds,'marker'), 1);
        if ~isempty(im) && im > 1, opts.nf_legs = im - 1; end
    end

    % ---- 2) sphere radius per focus, by FEX on a truncated deck ----------
    blocks = split_elements_(fileread(src));
    hdr = blocks{1};  foot = blocks{end};  ebl = blocks(2:end-1);
    tmpd = tempname;  mkdir(tmpd);
    cu = onCleanup(@() rmdir(tmpd,'s'));
    R = struct();
    fnames = {};
    for k = find(strcmp(kinds,'focus'))
        need(k);  need(k-1);
        nm = eltname_(ebl{k}, sprintf('Focus%d', k));
        fnames{k} = nm; %#ok<AGROW>
        R.(matlab.lang.makeValidName(nm)) = fex_radius_(hdr, foot, ebl, k, ...
            cp(:,k), uin(k), 0.5*norm(cp(:,k)-cp(:,k-1)), nm, opts.model, tmpd, ...
            opts.stop_name);
        fprintf('[prop_layout] %-12s exit-pupil radius = %.10f (FEX)\n', ...
                nm, R.(matlab.lang.makeValidName(nm)));
    end

    % ---- 3) assemble -----------------------------------------------------
    Z_PLANE = 1.0e22;  KR_FLAT = -1.0e22;
    B = {};  rs_seed = NaN;  newidx = zeros(1,nST);
    for k = 1:nST
        switch kinds{k}
            case 'focus'
                nm = fnames{k};  r = R.(matlab.lang.makeValidName(nm));
                u = uin(k);  F = cp(:,k);
                B{end+1} = render_([nm '_FPreturn'], 'Return','Flat', ...
                              'Geometric', F, u, KR_FLAT, Z_PLANE);   %#ok<AGROW>
                B{end+1} = render_([nm '_EPreturn'], 'Return','Conic', ...
                              'NF1', F - r*u, u, -r, r);              %#ok<AGROW>
                B{end+1} = convert_marker_(ebl{k}, 'Return','Flat','NF2', ...
                              F, u, KR_FLAT, Z_PLANE);                %#ok<AGROW>
                newidx(k) = numel(B);         % the MASK plane, 3rd of four
                B{end+1} = render_([nm '_EPreturn2'], 'Return','Conic', ...
                              'Geometric', F - r*u, u, -r, r);        %#ok<AGROW>

            case 'image'
                need(k);  need(k-1);
                u = uin(k);  F = cp(:,k);
                rs_seed = 0.5*norm(cp(:,k) - cp(:,k-1));
                B{end+1} = render_('FP_return', 'Return','Flat','Geometric', ...
                              F, -u, KR_FLAT, Z_PLANE);               %#ok<AGROW>
                B{end+1} = render_('ExitPupil', 'Return','Conic','FarField', ...
                              F - rs_seed*u, u, -rs_seed, rs_seed);   %#ok<AGROW>
                B{end+1} = convert_marker_(ebl{k}, '', 'Flat','Geometric', ...
                              F, u, KR_FLAT, Z_PLANE);                %#ok<AGROW>
                newidx(k) = numel(B);         % the detector, 3rd of three

            otherwise
                B{end+1} = ebl{k};                                    %#ok<AGROW>
                newidx(k) = numel(B);
        end

        if any(opts.nf_legs == k) && k < nST
            need(k);  need(k+1);
            u = unit_(cp(:,k+1) - cp(:,k));
            L = norm(cp(:,k+1) - cp(:,k));
            B{end+1} = render_(sprintf('Prop%d_start',k), 'Reference', ...
                          'Flat','NFPlane',   cp(:,k),   u, KR_FLAT, -L); %#ok<AGROW>
            B{end+1} = render_(sprintf('Prop%d_end',k),   'Reference', ...
                          'Flat','Geometric', cp(:,k+1), u, KR_FLAT, 0);  %#ok<AGROW>
        end
    end
    write_deck_(opts.out, hdr, B, foot, opts.ngridpts);

    % ---- 4) re-measure the terminal sphere on the ASSEMBLED deck ---------
    if ~isnan(rs_seed)
        macos.init(opts.model);
        n = macos.load_rx(opts.out);
        iep = find_named_(B, 'ExitPupil');
        set_stop_(opts.out, opts.stop_name);
        s = fex_(1, opts.out);
        rs = abs(s.rad);
        % Take FEX's POSE, not just its radius.  FEX finds where the exit
        % pupil actually is -- vertex and axis -- and imposing the seed's
        % own pose instead puts the sphere off the true pupil, which tilts
        % the far-field pattern: measured 154 pixels of PSF decentre on a
        % 512 grid before this was corrected.  Only the radius came back
        % right, and a radius alone is not a pupil.
        B{iep} = render_('ExitPupil', 'Return','Conic','FarField', ...
                         s.vpt(:), s.psi(:), -rs, rs);
        write_deck_(opts.out, hdr, B, foot, opts.ngridpts);
        fprintf('[prop_layout] %-12s exit-pupil radius = %.10f (FEX, seed %.10f)\n', ...
                'ExitPupil', rs, rs_seed);
        R.ExitPupil = rs;
    end

    info = struct('out', opts.out, 'nElt', numel(B), 'ix', index_map_(B), ...
                  'R', R, 'station', newidx, 'chk', []);
    if opts.verify
        info.chk = verify_(opts.out, src, newidx, cp, opts.model);
    end
end

% =========================================================================
function chk = verify_(out, src, newidx, cp, model)
%VERIFY_  The diffraction deck must put the chief ray where the geometric
%   one did, at every ORIGINAL station.  An inserted plane that is subtly
%   mis-posed still traces; this is what catches it.  NEWIDX comes from
%   the assembly, not from re-parsing the deck -- reconstructing the map
%   afterwards is exactly the kind of bookkeeping that quietly drifts.
    macos.init(model);
    n = macos.load_rx(out);
    s = macos.trace(n);
    r = macos.get_ray_info(s.nRays);
    chk = struct('nElt',n, 'nRays',s.nRays, ...
                 'nPass', nnz(logical(r.ok_pass) & logical(r.ok_trace)));
    macos.ray_hist('on');  macos.trace(n);
    hh = macos.ray_hist(s.nRays);  macos.ray_hist('off');
    d = nan(1, numel(newidx));
    for k = 1:numel(newidx)
        if ~hh.ok(1, newidx(k)+1), continue; end   % chief not at this station
        d(k) = norm(hh.P(:, 1, newidx(k)+1) - cp(:,k));
    end
    chk.chief_max = max(d(isfinite(d)));
    chk.chief = d;
    % PSF centring: the far-field terminal must put the on-axis PSF on the
    % FFT DC pixel, floor(N/2)+1.  A mis-posed exit-pupil sphere still
    % traces and still makes a PSF -- just not there.
    chk.psf_row = NaN;  chk.psf_col = NaN;  chk.psf_centred = false;
    try
        I = macos.intensity(n);
        [chk.psf_peak, ii] = max(I(:));
        [chk.psf_row, chk.psf_col] = ind2sub(size(I), ii);
        ctr = floor(size(I,1)/2) + 1;
        chk.psf_centre = ctr;
        chk.psf_centred = (chk.psf_row == ctr && chk.psf_col == ctr);
    catch
    end
    fprintf(['[prop_layout] verify: %d elements, %d/%d rays, chief agrees ' ...
             'with %s to %.3g; PSF peak at [%d %d] (centre %d) -> %s\n'], ...
            n, chk.nPass, chk.nRays, src, chk.chief_max, ...
            chk.psf_row, chk.psf_col, chk.psf_centre, ...
            tern_(chk.psf_centred, 'CENTRED', 'OFF-CENTRE'));
end

function i = find_named_(B, nm)
    i = 0;
    for k = 1:numel(B)
        t = regexp(B{k}, '^\s*EltName=\s*(\S+)', 'tokens','once','lineanchors');
        if ~isempty(t) && strcmp(t{1}, nm), i = k;  return; end
    end
    error('prop_layout:noElt','no element named %s in the assembled deck', nm);
end

function nm = eltname_(blk, dflt)
    t = regexp(blk, '^\s*EltName=\s*(\S+)', 'tokens','once','lineanchors');
    if isempty(t), nm = dflt; else, nm = t{1}; end
end

function r = fex_radius_(hdr, foot, ebl, k, F, u, rs, nm, model, tmpd, stopnm)
%FEX_RADIUS_  Exit-pupil radius conjugate to the focus at station k.
%   <optics 1..k-1> / FPreturn(flat at the focus) / EPreturn(seed sphere)
%   / focus-as-FocalPlane, then FEX -- whose radius is the chief-ray
%   distance from the found pupil to the iElt+1 plane.  The seed only has
%   to LOAD (FEX overwrites radius, pose and vertex), but keep it
%   comparable to the incoming leg so the sphere has clean daylight to the
%   real optics: ConSrf picks its intersection root by |L^2 - mpr|
%   proximity, with no flow-of-light sense.
    B = ebl(1:k-1);
    B{end+1} = render_([nm '_FPreturn'], 'Return','Flat','Geometric', ...
                       F, u, -1.0e22, 1.0e22);
    B{end+1} = render_([nm '_EPreturn'], 'Return','Conic','Geometric', ...
                       F - rs*u, u, -rs, rs);
    B{end+1} = render_(nm, 'FocalPlane','Flat','Geometric', ...
                       F, u, -1.0e22, 1.0e22);
    f = fullfile(tmpd, sprintf('fex_%s.in', matlab.lang.makeValidName(nm)));
    write_deck_(f, hdr, B, foot, []);          % keep the bare ray grid
    macos.init(model);
    macos.load_rx(f);
    set_stop_(f, stopnm);
    s = fex_(1, f);
    r = abs(s.rad);
end

function set_stop_(rx, stopnm)
%SET_STOP_  Point the engine's stop at the named element in THIS deck.
%   By name, not index: quartets shift every downstream index, so an
%   index captured on the source deck is wrong on the emitted one.
    if isempty(stopnm), return; end
    nm = regexp(fileread(rx), '^\s*EltName=\s*(\S+)', 'tokens','lineanchors');
    nm = cellfun(@(c) c{1}, nm, 'UniformOutput', false);
    i = find(strcmp(nm, stopnm), 1);
    if isempty(i)
        error('macos:design:prop_layout:stop', ...
              'prop_layout: no element named "%s" to use as the stop', stopnm);
    end
    macos.stop(i, [0 0]);
end

function s = fex_(iElt, rx)
%FEX_  macos.fex with the actionable hint attached.  A missing stop is
%   the usual cause and the raw message does not say which deck.
    try
        s = macos.fex(iElt);
    catch ME
        if strcmp(ME.identifier, 'macos:fex:noStop')
            error('macos:design:prop_layout:nostop', ...
                ['prop_layout: %s declares no aperture stop, so FEX cannot ' ...
                 'find the exit pupil.  Pass ''stop_name'' (the EltName of ' ...
                 'the stop), or emit ApStop= in the deck header.'], rx);
        end
        rethrow(ME);
    end
end

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
    assert(fid > 0, 'prop_layout: cannot write %s', path);
    fprintf(fid, '%s', hdr);
    for k = 1:n, fprintf(fid, '%s', B{k}); end
    fprintf(fid, '%s', foot);
    fclose(fid);
end

function s = render_(nm, element, surface, prop, vpt, psi, kr, zElt)
%RENDER_  One inserted surface block.  zElt is written with the same
%   format everywhere, so a quartet's two spheres -- fed the same variable
%   -- carry byte-identical digits.
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

function blk = convert_marker_(blk, element, surface, prop, vpt, psi, kr, z)
%CONVERT_MARKER_  Retype a bare-deck marker in place, keeping every other
%   field (ApVec, TElt, coating, ...) verbatim.  element='' leaves
%   Element= alone.
    v3 = @(v) sprintf('  %.16E  %.16E  %.16E', v(1), v(2), v(3));
    if ~isempty(element), blk = sub_(blk, 'Element', ['  ' element]); end
    blk = sub_(blk, 'Surface',  ['  ' surface]);
    blk = sub_(blk, 'PropType', ['  ' prop]);
    blk = sub_(blk, 'KrElt',    sprintf('%.16E', kr));
    blk = sub_(blk, 'zElt',     sprintf('%.16E', z));
    blk = sub_(blk, 'psiElt',   v3(psi));
    blk = sub_(blk, 'VptElt',   v3(vpt));
    blk = sub_(blk, 'RptElt',   v3(vpt));
end

function blk = sub_(blk, key, value)
%SUB_  Replace one key's value.  The replacement is passed LITERALLY --
%   do not regexptranslate it: that escapes '.' and '+', which are
%   ordinary text on this side and would land in the deck as backslashes.
    blk = regexprep(blk, ['^(\s*)' key '=.*$'], ['$1' key '=' value], ...
                    'lineanchors', 'dotexceptnewline');
end

function blocks = split_elements_(txt)
%SPLIT_ELEMENTS_  {header; e1..eN; footer}, split on iElt= / nOutCord=.
    lines = regexp(txt, '\r?\n', 'split');
    ei = find(~cellfun('isempty', regexp(lines, '^\s*iElt=',     'once')));
    fi = find(~cellfun('isempty', regexp(lines, '^\s*nOutCord=', 'once')), 1);
    assert(~isempty(ei), 'prop_layout: no iElt= lines found');
    assert(~isempty(fi), 'prop_layout: no nOutCord= footer found');
    join_ = @(a,b) [strjoin(lines(a:b), newline) newline];
    blocks = {join_(1, ei(1)-1)};
    for k = 1:numel(ei)
        if k < numel(ei), b = ei(k+1)-1; else, b = fi-1; end
        blocks{end+1} = join_(ei(k), b);                       %#ok<AGROW>
    end
    blocks{end+1} = join_(fi, numel(lines));
end

function ix = index_map_(B)
    ix = struct();
    for k = 1:numel(B)
        nm = regexp(B{k}, '^\s*EltName=\s*(\S+)', 'tokens','once','lineanchors');
        if isempty(nm), continue; end
        ix.(matlab.lang.makeValidName(nm{1})) = k;
    end
end

function u = unit_(v), u = v(:)/norm(v); end

function s = tern_(c,a,b), if c, s = a; else, s = b; end, end
