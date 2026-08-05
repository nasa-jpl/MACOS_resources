function info = ctb_prop_layout(varargin)
%CTB_PROP_LAYOUT  Build the CTB diffraction-propagation deck from stageF.
%   info = CTB_PROP_LAYOUT() reads the geometric CTB prescription
%   ctb_planar_stageF.in (all PropType=Geometric), traces it to recover the
%   chief-ray geometry at every station, and writes ctb_planar_prop.in --
%   the SAME optics with the diffraction reference / return / sphere
%   surfaces INSERTED so the field can be propagated source->FPA as a
%   chain of near-field / far-field diffraction legs.
%
%   This is a one-time BUILDER (regenerable); the emitted .in is committed.
%   PropType is a parse-time enum with no runtime setter, so it MUST be
%   baked into the deck here.  Sphere POSE/CURVATURE are only load-time
%   seeds -- ctb_propagate.m refines them at runtime with ORS / FEX (the
%   CoroExample.jou pattern), so the seeds only have to make the Rx load
%   and trace.
%
%   The insertion mirrors the canonical encoding of
%   pymacos/tests/Rx/Rx_Coro.in:
%     * a DM near-field plane-to-plane leg  (NFPlane -> Geometric),
%     * a sphere/NF1 -> focus/NF2 -> Return pair bracketing each focus,
%     * NFPlane markers landing the beam on each intermediate pupil,
%     * a terminal FarField sphere before the FPA.
%
%   Name-value:
%     'src'      source geometric deck (default ctb_planar_stageF.in here)
%     'out'      output deck            (default ctb_planar_prop.in here)
%     'ngridpts' diffraction ray grid   (default 255 -- odd, bring-up scale;
%                bump to 511 for a validation-grade run)
%     'model'    macos model size       (default 512)
%     'standoff' sphere daylight off the bounding OAP vertex, BaseUnits mm
%                (default 60): keeps the reference sphere clear of the real
%                optic so ORS picks the right ConSrf root.
%
%   Returns a struct with the emitted path and the element-index map
%   (info.ix) that ctb_prop_legs / ctb_propagate consume.
%
%   See also: ctb_propagate, ctb_prop_legs, CoroExample.jou.
    p = inputParser;
    here = fileparts(mfilename('fullpath'));
    p.addParameter('src', fullfile(here, 'ctb_planar_stageF.in'));
    p.addParameter('out', fullfile(here, 'ctb_planar_prop.in'));
    p.addParameter('ngridpts', 255);
    p.addParameter('model', 512);
    p.addParameter('standoff', 60.0);
    p.parse(varargin{:});
    o = p.Results;

    addpath(fullfile(here, '..', '..', '..', 'src'));   % mmacos/src

    % ------------------------------------------------------------------
    % 1) Trace the geometric deck; capture chief pos + dir at each station.
    % ------------------------------------------------------------------
    macos.init(o.model);
    nE = macos.load_rx(o.src);
    assert(nE == 17, 'expected 17-element stageF deck, got %d', nE);

    % Station names in stageF light order (fixed by the builder).
    name17 = {'OAP1','DM1','DM2','OAP2','Focus23','OAP3','Apodizer', ...
              'OAP4','FPM','OAP5','Lyot','OAP6','FieldStop','OAP7', ...
              'Backend','OAP8','FPA'};
    cp = zeros(3, nE);  cd = zeros(3, nE);
    for k = 1:nE
        s = macos.trace(k);
        ri = macos.get_ray_info(s.nRays);
        assert(all(ri.ok_trace & ri.ok_pass), ...
            'stageF: element %d (%s) vignettes', k, name17{k});
        cp(:, k) = ri.pos(:, 1);
        d = ri.dir(:, 1);  cd(:, k) = d / norm(d);
    end
    iof = containers.Map(name17, num2cell(1:nE));   % name -> stageF index

    % ------------------------------------------------------------------
    % 2) Compute the poses of the inserted diffraction surfaces.
    %    Convention (folded-bench safe -- never a literal [0 0 1]):
    %      plane refs face the beam;  spheres point at their CoC (the
    %      focus), Kr = -|vertex - focus|.  All derived from the traced
    %      chief geometry above.
    % ------------------------------------------------------------------
    ins = struct('after', {}, 'name', {}, 'element', {}, 'surface', {}, ...
                 'prop', {}, 'vpt', {}, 'psi', {}, 'kr', {}, 'zElt', {});

    unit = @(v) v / norm(v);
    % A point a fraction f along the chief segment from element a to b.
    seg = @(a, b, f) cp(:, iof(a)) + f * (cp(:, iof(b)) - cp(:, iof(a)));

    % --- Leg 1: DM p2p (NFPlane, collimated).  Plane refs after DM1 and
    %  before DM2.  zElt RULE for plane-to-plane (Rx_Coro Prop_1_start/end):
    %  the propagation distance is the DIFFERENCE of the two zElts and must
    %  equal the chief-ray L between them -- start zElt = -L, end zElt = 0
    %  (NOT both 1e22, NOT both 0).  Kr stays flat (-1e22).
    d_dm1 = cd(:, iof('DM2'));            % chief leaving DM1 toward DM2
    Vp1s = seg('DM1', 'DM2', 0.10);  Vp1e = seg('DM1', 'DM2', 0.90);
    Lp1  = norm(Vp1e - Vp1s);
    ins(end+1) = mk('DM1', 'P1_start', 'Reference', 'Flat', 'NFPlane', ...
                    Vp1s,  d_dm1, -1.0e22, -Lp1);      % start zElt = -L
    ins(end+1) = mk('P1_start', 'P1_end', 'Reference', 'Flat', 'Geometric', ...
                    Vp1e, -d_dm1, -1.0e22, 0.0);       % end   zElt = 0

    % --- Through-focus spheres bracketing each intermediate focus. -------
    %  Built as a SYMMETRIC MIRROR PAIR about the focus, matching Rx_Coro's
    %  1stPropStart / CorMask / 1stPropEnd (elts 8/9/10) EXACTLY:
    %    Sin  (NF1)          : one radius r UPSTREAM  of the focus,
    %                          psi = incoming beam dir, Kr=-r, zElt=+r.
    %    focus(NF2, edited)  : psi = incoming beam dir, zElt=-r  (step 3).
    %    Sout (Return/Geom)  : one radius r DOWNSTREAM of the focus,
    %                          psi = -incoming (back toward focus), Kr=-r,
    %                          zElt=-r.
    %  The chief legitimately REVERSES across Sout and flips back at the
    %  next Sin -- this is Rx_Coro's negative-length reference-leg behaviour
    %  (its elt10 chief dir is +z while the beam runs -z), NOT a bug.
    %  r is the focus->nearer-bounding-OAP distance minus a standoff, so
    %  BOTH spheres keep clean daylight to the real optics (ConSrf root).
    foci = { 'Focus23','OAP2','OAP3';   % focus, feed-OAP (before), pick-OAP (after)
             'FPM',    'OAP4','OAP5';
             'FieldStop','OAP6','OAP7' };
    focus_r = containers.Map('KeyType','char','ValueType','double');
    for r = 1:size(foci, 1)
        fc = foci{r,1};  feed = foci{r,2};  pick = foci{r,3};
        Fv  = cp(:, iof(fc));
        uin = unit(Fv - cp(:, iof(feed)));           % chief dir INTO the focus
        % radius: fit inside the nearer of the two OAP gaps, minus standoff.
        rad = min(norm(Fv - cp(:, iof(feed))), ...
                  norm(cp(:, iof(pick)) - Fv)) - o.standoff;
        Vin  = Fv - rad * uin;                        % one radius upstream
        Vout = Fv + rad * uin;                        % one radius downstream
        ins(end+1) = mk(feed, [fc '_Sin'], 'Reference', 'Conic', 'NF1', ...
                        Vin,  uin, -rad, rad);         %#ok<AGROW>  z=+r
        ins(end+1) = mk(fc,   [fc '_Sout'], 'Return', 'Conic', 'Geometric', ...
                        Vout, -uin, -rad, -rad);       %#ok<AGROW>  z=-r
        % _Sret: a FLAT Return placed BACK AT THE FOCUS -- Rx_Coro's
        % 'CorMaskReturn' (elt 11).  _Sout reverses the chief (+dir); this
        % flat flips it back to forward so the NEXT OAP sees a correctly-
        % oriented bundle.  Without it the reversed bundle blows up at the
        % downstream powered mirror (marginal rays -> infinity).
        ins(end+1) = mk([fc '_Sout'], [fc '_Sret'], 'Return', 'Flat', ...
                        'Geometric', Fv, uin, -1.0e22, -1.0e22); %#ok<AGROW>
        focus_r(fc) = rad;  %#ok<AGROW>  remember for the NF2 zElt in step 3
    end

    % --- NFPlane markers landing the beam on each intermediate pupil. ----
    %  A plane just before the pupil marker; the marker itself carries the
    %  MATLAB mask and stays Geometric.  Same plane-to-plane zElt RULE: the
    %  START plane gets zElt=-L (L = chief distance to the pupil marker) and
    %  the pupil marker (END, reused stageF block) keeps zElt=0.
    pups = { 'Apodizer','OAP3';   % pupil, feeding OAP (before it)
             'Lyot',    'OAP5' };
    for r = 1:size(pups, 1)
        pu = pups{r,1};  feed = pups{r,2};
        dpu = cd(:, iof(pu));                 % chief through the pupil
        Vst = cp(:, iof(pu)) - 0.5*norm(cp(:,iof(pu)) - cp(:,iof(feed))) * dpu;
        Lpu = norm(cp(:, iof(pu)) - Vst);     % chief L: start plane -> pupil
        ins(end+1) = mk(feed, [pu '_Pst'], 'Reference', 'Flat', 'NFPlane', ...
                        Vst, dpu, -1.0e22, -Lpu);              %#ok<AGROW>
        % the pupil marker (END of this plane-to-plane leg) keeps zElt=0
        % from stageF -- start(-L) minus end(0) = -L = the propagation dist.
    end

    % --- Terminal exit-pupil pair before the FPA.  This mirrors
    %  macos.design.Telescope.add_pupil (Telescope.m:742-758) EXACTLY:
    %  the far-field transform needs BOTH a flat image-reference at the
    %  FPA location AND the exit-pupil sphere -- a lone FarField sphere
    %  gives no valid output plane.  Light order:
    %     OAP8 -> FP_return (flat, Geometric, at FPA) -> ExitPupil
    %          (sphere, FarField) -> FPA.
    %  uIn is the chief OAP8->FPA direction; FP_return faces back up the
    %  beam (-uIn); ExitPupil is seeded a standoff before the FPA and is
    %  RE-PLACED at runtime by FEX (ctb_propagate), which also sets the
    %  EP radius.  zElt (EP<->image hop) is synced to the FEX radius in
    %  the driver; the seed only has to load.
    Fv   = cp(:, iof('FPA'));
    uIn  = unit(Fv - cp(:, iof('OAP8')));
    rSeed = norm(Fv - cp(:, iof('OAP8'))) - o.standoff;   % EP a standoff before FPA
    Vep  = Fv - rSeed * uIn;
    ins(end+1) = mk('OAP8', 'FP_return', 'Return', 'Flat', 'Geometric', ...
                    Fv,  -uIn, -1.0e22, rSeed);            %#ok<AGROW>
    ins(end+1) = mk('FP_return', 'ExitPupil', 'Return', 'Conic', 'FarField', ...
                    Vep, -unit(Fv - Vep), -rSeed, rSeed);  %#ok<AGROW>

    % ------------------------------------------------------------------
    % 3) Splice the inserted blocks into the stageF text, renumber, and
    %    flip the three focal markers to NF2.  Keeps every original block
    %    verbatim (TElt, ApVec, coatings ...), only editing what must change.
    % ------------------------------------------------------------------
    txt = fileread(o.src);
    blocks = split_elements(txt);         % {header; e1; e2; ...; footer}
    hdr = blocks{1};  foot = blocks{end};
    ebl = blocks(2:end-1);                % 17 element blocks, in order
    assert(numel(ebl) == 17, 'parsed %d element blocks, expected 17', numel(ebl));

    % bump nGridpts in the header (ray grid -> diffraction grid).
    hdr = regexprep(hdr, 'nGridpts=\s+\d+', sprintf('nGridpts=  %d', o.ngridpts));

    % flip focal markers to NF2 (they receive the far-field-focused field)
    % and set their zElt to -r (Rx_Coro CorMask convention: the NF2 plane
    % sits one radius from its bracketing spheres).  ALSO re-align each
    % REUSED diffraction marker to the RULE: axis parallel to the chief ray,
    % vertex at the chief incidence point (Dave 2026-08-04).  The stageF
    % markers carry psi=(1,0,0), but the chief arrives tilted by the fold
    % residual (~0.6 deg); a tilted focal-grid axis throws the far-field PSF
    % off in one axis.  cd/cp are the traced chief dir/pos from step 1.
    for fc = {'Focus23','FPM','FieldStop'}
        j = iof(fc{1});
        ebl{j} = set_prop(ebl{j}, 'NF2');
        ebl{j} = set_z(ebl{j}, -focus_r(fc{1}));
        ebl{j} = set_psi(ebl{j}, cd(:, j));       % axis || chief dir
        ebl{j} = set_vpt(ebl{j}, cp(:, j));       % vertex at chief pierce
    end

    % the terminal FocalPlane: sentinel zElt (the far-field ExitPupil->FPA
    % hop needs 1e22, matching Rx_Coro) AND axis || chief / vertex at the
    % chief pierce so the on-axis PSF lands at the grid centre.
    jF = iof('FPA');
    ebl{jF} = set_z(ebl{jF}, 1.0e22);
    ebl{jF} = set_psi(ebl{jF}, cd(:, jF));
    ebl{jF} = set_vpt(ebl{jF}, cp(:, jF));

    % assemble: walk stageF elements, emitting any inserts anchored 'after'
    % each one (P1_start is 'after DM1', its _end 'after P1_start' -- handled
    % by chaining anchors through the running output list).
    outblocks = {};
    emitted = containers.Map('KeyType','char','ValueType','logical');
    function flush_after(anchorName)
        moved = true;
        while moved
            moved = false;
            for q = 1:numel(ins)
                if ~isKey(emitted, ins(q).name) && strcmp(ins(q).after, anchorName)
                    outblocks{end+1} = render_block(ins(q)); %#ok<AGROW>
                    emitted(ins(q).name) = true;
                    anchorName = ins(q).name;   % chain (e.g. P1_start->P1_end)
                    moved = true;
                end
            end
        end
    end
    for k = 1:17
        outblocks{end+1} = ebl{k};                 %#ok<AGROW>
        flush_after(name17{k});
    end
    assert(all(cellfun(@(n) isKey(emitted,n), {ins.name})), ...
        'not all inserts were placed');

    nOut = numel(outblocks);
    % renumber iElt sequentially.  LINE-ANCHOR the match: the token 'iElt'
    % is ALSO a substring of 'psiElt', so an unanchored 'iElt=\s+\d+' also
    % matches 'ps|iElt=  9...' and clobbers the leading digit of psiElt.
    % '^\s*iElt=' only fires on the real iElt line.
    for k = 1:nOut
        outblocks{k} = regexprep(outblocks{k}, '^(\s*)iElt=\s+\d+', ...
                                 sprintf('$1iElt=  %d', k), 'lineanchors');
    end
    hdr = regexprep(hdr, 'nElt=\s+\d+', sprintf('nElt=  %d', nOut));

    fid = fopen(o.out, 'w');
    assert(fid > 0, 'cannot write %s', o.out);
    fprintf(fid, '%s', hdr);
    for k = 1:nOut, fprintf(fid, '%s', outblocks{k}); end
    fprintf(fid, '%s', foot);
    fclose(fid);

    % ------------------------------------------------------------------
    % 4) Element-index map for the augmented deck (name -> new iElt).
    % ------------------------------------------------------------------
    ix = build_index_map(outblocks);
    info = struct('out', o.out, 'nElt', nOut, 'ix', ix, ...
                  'ngridpts', o.ngridpts, 'model', o.model);
    fprintf('[ctb_prop_layout] wrote %s (%d elements, nGridpts=%d)\n', ...
            o.out, nOut, o.ngridpts);
end

% ======================================================================
function e = mk(after, name, element, surface, prop, vpt, psi, kr, zElt)
%MK  Build one inserted-surface descriptor.  zElt defaults to 0 (flat
%   References/foci); the terminal exit-pupil pair passes the EP radius.
    if nargin < 9, zElt = 0.0; end
    e = struct('after', after, 'name', name, 'element', element, ...
               'surface', surface, 'prop', prop, ...
               'vpt', vpt(:).', 'psi', psi(:).', 'kr', kr, 'zElt', zElt);
end

% ----------------------------------------------------------------------
function s = render_block(e)
%RENDER_BLOCK  Emit a minimal Reference/Return element block (Rx_Coro form).
    F = @(v) sprintf('  %.10E  %.10E  %.10E', v(1), v(2), v(3));
    L = {};
    L{end+1} = '';
    L{end+1} = '             iElt=  0';                 % renumbered later
    L{end+1} = sprintf('          EltName=  %s', e.name);
    L{end+1} = sprintf('          Element=  %s', e.element);
    L{end+1} = sprintf('          Surface=  %s', e.surface);
    L{end+1} = sprintf('            KrElt=%.10E', e.kr);
    L{end+1} =         '            KcElt=0.0E+00';
    L{end+1} = sprintf('           psiElt=%s', F(e.psi));
    L{end+1} = sprintf('           VptElt=%s', F(e.vpt));
    L{end+1} = sprintf('           RptElt=%s', F(e.vpt));
    L{end+1} =         '           IndRef=1.0E+00';
    L{end+1} =         '           Extinc=1.0E+22';
    L{end+1} =         '            nCoat=  0';
    L{end+1} =         '             nObs=  0';
    L{end+1} =         '           ApType=  None';
    L{end+1} = sprintf('         PropType=  %s', e.prop);
    L{end+1} = sprintf('             zElt=%.10E', e.zElt);
    L{end+1} =         '          nECoord= -6';
    s = [strjoin(L, newline) newline];
end

% ----------------------------------------------------------------------
function blk = set_prop(blk, prop)
%SET_PROP  Replace the PropType value of an element block (line-anchored).
    blk = regexprep(blk, '^(\s*)PropType=\s+\S+', ...
                    sprintf('$1PropType=  %s', prop), 'lineanchors');
end

% ----------------------------------------------------------------------
function blk = set_z(blk, z)
%SET_Z  Replace the zElt value of an element block (line-anchored).
    blk = regexprep(blk, '^(\s*)zElt=\S+', ...
                    sprintf('$1zElt=%.10E', z), 'lineanchors');
end

% ----------------------------------------------------------------------
function blk = set_psi(blk, v)
%SET_PSI  Replace the psiElt 3-vector of an element block (line-anchored).
    v = v(:);
    blk = regexprep(blk, '^(\s*)psiElt=.*$', ...
        sprintf('$1psiElt=  %.10E  %.10E  %.10E', v(1), v(2), v(3)), ...
        'lineanchors', 'dotexceptnewline');
end

% ----------------------------------------------------------------------
function blk = set_vpt(blk, v)
%SET_VPT  Replace BOTH VptElt and RptElt of an element block with v.
    v = v(:);
    s = sprintf('%.10E  %.10E  %.10E', v(1), v(2), v(3));
    blk = regexprep(blk, '^(\s*)VptElt=.*$', ['$1VptElt=  ' s], ...
                    'lineanchors', 'dotexceptnewline');
    blk = regexprep(blk, '^(\s*)RptElt=.*$', ['$1RptElt=  ' s], ...
                    'lineanchors', 'dotexceptnewline');
end

% ----------------------------------------------------------------------
function blocks = split_elements(txt)
%SPLIT_ELEMENTS  Split a prescription into {header; e1..eN; footer} by the
%   'iElt=' keyword.  Header is everything before the first iElt; footer is
%   everything from the Output-Coordinate section onward (nOutCord).
    lines = regexp(txt, '\r?\n', 'split');
    % locate element starts (lines matching iElt=) and the footer start.
    isElt = ~cellfun('isempty', regexp(lines, '^\s*iElt=', 'once'));
    isFoot = ~cellfun('isempty', regexp(lines, '^\s*nOutCord=', 'once'));
    ei = find(isElt);
    fi = find(isFoot, 1, 'first');
    assert(~isempty(ei), 'no iElt= lines found');
    assert(~isempty(fi), 'no nOutCord= footer found');
    % an element block starts one line above iElt (the blank line) if present.
    starts = ei;
    join = @(a, b) [strjoin(lines(a:b), newline) newline];
    blocks = {};
    blocks{1} = join(1, starts(1) - 1);         % header (up to first iElt-1)
    for k = 1:numel(starts)
        a = starts(k);
        if k < numel(starts), b = starts(k+1) - 1; else, b = fi - 1; end
        blocks{end+1} = join(a, b);              %#ok<AGROW>
    end
    blocks{end+1} = join(fi, numel(lines));      % footer
end

% ----------------------------------------------------------------------
function ix = build_index_map(outblocks)
%BUILD_INDEX_MAP  name -> 1-based iElt from the assembled block list.
    ix = struct();
    for k = 1:numel(outblocks)
        nm = regexp(outblocks{k}, 'EltName=\s*(\S+)', 'tokens', 'once');
        if isempty(nm), continue; end
        key = matlab.lang.makeValidName(nm{1});
        ix.(key) = k;
    end
end
