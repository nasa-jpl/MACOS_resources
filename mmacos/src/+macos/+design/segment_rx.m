function out = segment_rx(parent_in, opts)
%SEGMENT_RX  Segment one element of a MACOS prescription (SegMirMaker splice).
%
%   out = macos.design.segment_rx(parent_in, Name=Value) replaces element
%   `elt` of the parent prescription with SegMirMaker-generated segment
%   blocks and returns the merged, load-ready `.in`.  Sprint 2D S1:
%   orchestrate, do not reimplement — all segment geometry (placement,
%   face triads, FF replication, psi handling) comes from the SegMirMaker
%   run (see macos.design.segmirmaker_run); this function only splices
%   text and renumbers.
%
%   Steps: run SegMirMaker batch (with the emitted iElt numbering started
%   at `elt`) → carry the parent element's Zernike FIGURE onto every
%   segment via the FF channel (SegMirMaker replicates only conic +
%   frames; a Surface=Zernike parent would otherwise lose its solved
%   figure) → replace the parent's source GridType with the segment
%   tiling grid + insert the nSeg/width/gap/SegXgrid/SegCoord block →
%   swap the parent element block for the segment blocks → renumber the
%   downstream blocks' iElt (cosmetic — the engine numbers by read
%   order — but keeps the file SAVE-grade) → update nElt.  ApStop never
%   renumbers (header form = 3-vector position, element form rides its
%   own block), but if the SEGMENTED element carried the element-form
%   stop it is dropped with the block → warning + out.dropped_apstop.
%
%   Options: elt (1) — parent element to segment; out_in ('') — output
%   path (default <name>_seg.in in the SegMirMaker scratch dir, which
%   already holds macos_param.txt + the GridFile= data, so it is a
%   self-contained load directory — the engine resolves GridFile from
%   the cwd); emit_apertures (false) — declare each segment's PHYSICAL
%   boundary as a polygonal aperture in the merged .in
%   (macos.design.seg_apertures: hex corners exact; pie = center hexagon
%   + chorded sectors with inner-sector obscurations), with ap_pad (0 =
%   physical edge; gap/2 = trace-neutral tiling midline) and ap_obs
%   (true) knobs; carry_obs (true) -- carry the segmented element's own
%   obscuration declarations (e.g. the set_hole central perforation)
%   onto the CENTER segment, where they physically live; grid_npts ([])
%   -- rewrite the merged source's nGridpts (segmentation usually wants
%   a denser grid than the parent; e5 corpus runs 128); plus the
%   segmentation options forwarded to segmirmaker_run: rings, grid,
%   gap, dofs, meas_config, seg_size, ap_diam, psi, segxgrid, f, ecc,
%   standoff, binary.
%
%   out fields:
%     .in        merged prescription path
%     .nseg      number of segments
%     .seg_elts  element numbers of the segments (elt .. elt+nseg-1)
%     .n_elt     new total element count
%     .frames    1×nseg struct: name, rpt(3×1), xhat/yhat/zhat(3×1),
%                lmon — each segment's local frame AS EMITTED (pMon =
%                RptElt, {xMon,yMon,zMon} = face triad; provenance
%                derived(segmirmaker))
%     .hx        edge-sensor Hx.m path (ingested in S2)
%     .presc     raw segment-block file path
%     .run       the segmirmaker_run output struct
%
%   See also: macos.design.segmirmaker_run, tSegmentRx.

arguments
    parent_in (1,1) string
    opts.elt (1,1) double {mustBeInteger, mustBePositive} = 1
    opts.out_in (1,1) string = ""
    opts.rings (1,1) double {mustBeInteger, mustBePositive} = 1
    opts.grid (1,1) string = "Hex"
    opts.gap (1,1) double {mustBeNonnegative} = 0
    opts.dofs (1,1) double {mustBeMember(opts.dofs, [3 6])} = 6
    opts.meas_config (1,1) double {mustBeMember(opts.meas_config, [1 2])} = 1
    opts.seg_size double = []
    opts.ap_diam double = []
    opts.psi double = []
    opts.segxgrid double = []
    opts.f double = []
    opts.ecc double = []
    opts.standoff double = []
    opts.binary (1,1) string = ""
    opts.emit_apertures (1,1) logical = false
    opts.ap_pad (1,1) double = 0
    opts.ap_obs (1,1) logical = true
    opts.carry_obs (1,1) logical = true
    opts.grid_npts double = []
end

% --- 0. read + vet the parent BEFORE paying for the SegMirMaker run ------
lines = readlines(parent_in);
tl = strtrim(lines);
starts = find(startsWith(tl, "iElt="));
if opts.elt > numel(starts)
    error('macos:design:segment_rx:elt', ...
        'parent has %d element blocks; cannot segment element %d', ...
        numel(starts), opts.elt);
end
% Unhandled element-index-bearing keywords: refuse rather than silently
% mis-renumber.  (Extend the renumber rules here as these show up.)
for kw = ["OptTgtElt" "RefElt" "tMetElt"]
    if any(startsWith(tl, kw + "="))
        error('macos:design:segment_rx:renumber', ...
            ['parent carries %s= (an element-index keyword segment_rx ' ...
             'does not renumber yet) — splice by hand or extend the rules'], kw);
    end
end

% --- 1. generate the segment blocks -------------------------------------
[~, base, ~] = fileparts(parent_in);
run_out = macos.design.segmirmaker_run(parent_in, ...
    'elt', opts.elt, 'start_elt', opts.elt, 'name', base + "_segblocks", ...
    'rings', opts.rings, 'grid', opts.grid, 'gap', opts.gap, ...
    'dofs', opts.dofs, 'meas_config', opts.meas_config, ...
    'seg_size', opts.seg_size, 'ap_diam', opts.ap_diam, ...
    'psi', opts.psi, 'segxgrid', opts.segxgrid, 'f', opts.f, ...
    'ecc', opts.ecc, 'standoff', opts.standoff, 'binary', opts.binary);

plines = readlines(run_out.presc);
pstarts = find(startsWith(strtrim(plines), "iElt="));
if isempty(pstarts)
    error('macos:design:segment_rx:presc', 'no element blocks in %s', run_out.presc);
end
nseg = numel(pstarts);
phead = plines(1:pstarts(1)-1);              % GridType + nSeg..SegCoord table
gt = regexp(phead(startsWith(strtrim(phead), "GridType=")), ...
            'GridType=\s*(\S+)', 'tokens', 'once');
seg_gridtype = string(gt{1});
seg_header = phead(~startsWith(strtrim(phead), "GridType="));
while ~isempty(seg_header) && strlength(strtrim(seg_header(end))) == 0
    seg_header(end) = [];                    % drop trailing blank lines
end
% SegXgrid: the engine's HSEG anchors the HEX tiling to SegXgrid's
% GLOBAL orientation (PSEG/pie ignores it -- its rotation block is dead
% code), and the frames <-> tiling contract closes only with
% SegXgrid=(-1,0,0) alongside the heritage xGrid=(-1,0,0) below:
% engine-truth verified via ray_hist -- each hex segment's rays then
% land ON its emitted frame.  SegMirMaker's default emission (1,0,0)
% leaves the hex tiling 180 deg off the frames (latent in the e5
% corpus, where no fixture traces hex apertures).
isx = find(startsWith(strtrim(seg_header), "SegXgrid="), 1);
if ~isempty(isx)
    sx = sscanf(char(regexprep(seg_header(isx), '.*SegXgrid=', '')), '%f');
    if numel(sx) == 3 && abs(sx(1) - 1) < 1e-9 && all(abs(sx(2:3)) < 1e-9)
        seg_header(isx) = regexprep(seg_header(isx), ...
            '(SegXgrid=\s*)\S+', "$1-1.0");
    end
end
pblocks = cell(nseg, 1);
for k = 1:nseg
    e = numel(plines); if k < nseg, e = pstarts(k+1) - 1; end
    pblocks{k} = plines(pstarts(k):e);
end

% --- 1b. carry the parent's Zernike FIGURE onto every segment ------------
% SegMirMaker replicates only the parent's conic + frames; a parent with
% Surface=Zernike (e.g. a design-layer primary carrying its solved
% freeform figure) would silently lose that figure -- the segmented
% system then reproduces the BASE conic, not the as-designed surface
% (e2e s3, 2026-07-18: 5.9 nm parent -> 15 um segmented).  The segments
% are FreeForm surfaces with an FF channel reserved for exactly this:
% append the parent figure as FFZern* in the PARENT's Mon frame (pFF/
% xFF/yFF/zFF = parent pMon/xMon/yMon/zMon, lFF = parent lMon), leaving
% the per-segment Mon channel free for segment figure DOFs.
pb0 = blocks_parent_figure_(lines, starts, opts.elt);
if ~isempty(pb0)
    for k = 1:nseg
        b = pblocks{k};
        while ~isempty(b) && strlength(strtrim(b(end))) == 0
            b(end) = [];
        end
        pblocks{k} = [b; pb0];
    end
end

% --- 2. carve up the parent ---------------------------------------------
% Tail = output-coordinate section (plus its banner comments), if present.
noc = find(startsWith(tl, "nOutCord="), 1);
if isempty(noc)
    tail_begin = numel(lines) + 1;
else
    tail_begin = noc;
    while tail_begin > 1 && (strlength(tl(tail_begin-1)) == 0 || ...
                             startsWith(tl(tail_begin-1), "%"))
        tail_begin = tail_begin - 1;
    end
end

pre = lines(1:starts(1)-1);                   % source section + banners
blocks = cell(numel(starts), 1);
for k = 1:numel(starts)
    e = tail_begin - 1; if k < numel(starts), e = starts(k+1) - 1; end
    blocks{k} = lines(starts(k):e);
end
tail = lines(tail_begin:end);

% --- 3. edit the source section -----------------------------------------
% GridType -> the segment tiling grid; insert the nSeg/width/gap/
% SegXgrid/SegCoord block after yGrid (or after GridType if no yGrid).
igt = find(startsWith(strtrim(pre), "GridType="), 1);
if isempty(igt)
    error('macos:design:segment_rx:gridtype', 'parent has no GridType= line');
end
pre(igt) = regexprep(pre(igt), '(GridType=\s*)\S+', "$1" + seg_gridtype);
% Heritage source-grid orientation: the SegMirMaker <-> engine tiling
% contract assumes xGrid=(-1,0,0) (the e5mono/dmt6mono corpus).  A
% design-layer parent emits (+1,0,0), under which the engine's
% ray-to-segment map comes out 180 deg ROTATED from the emitted segment
% frames -- every off-center segment's polygonal aperture then clips
% its own rays (e2e s3, 2026-07-18).  Flip the merged source to the
% heritage orientation (no-op for heritage parents; the engine
% recomputes yGrid/zGrid from it).
ix = find(startsWith(strtrim(pre), "xGrid="), 1);
if ~isempty(ix)
    xg = sscanf(char(regexprep(pre(ix), '.*xGrid=', '')), '%f');
    if numel(xg) == 3 && abs(xg(1) - 1) < 1e-9 && all(abs(xg(2:3)) < 1e-9)
        pre(ix) = regexprep(pre(ix), '(xGrid=\s*)\S+', "$1-1.0");
    end
end
% Optional sampling bump: the parent's nGridpts rides through the
% splice, but segmentation usually wants a denser grid (the e5 corpus
% runs 128 where the design-layer parents emit 41 -- at 41 a 19-segment
% tiling gets ~50 rays/segment and 25 mm gaps are under-resolved).
if ~isempty(opts.grid_npts)
    ig2 = find(startsWith(strtrim(pre), "nGridpts="), 1);
    if ~isempty(ig2)
        pre(ig2) = regexprep(pre(ig2), '(nGridpts=\s*)\d+', ...
                             sprintf('$1%d', round(opts.grid_npts)));
    end
end
iy = find(startsWith(strtrim(pre), "yGrid="), 1);
if isempty(iy), iy = igt; end
pre = [pre(1:iy); ""; seg_header; pre(iy+1:end)];

% nElt: block count is the engine truth (nElt= in old files can be stale).
n_elt_new = numel(starts) + nseg - 1;
ine = find(startsWith(strtrim(pre), "nElt="), 1);
if isempty(ine)
    error('macos:design:segment_rx:nelt', 'parent has no nElt= line');
end
pre(ine) = regexprep(pre(ine), '(nElt=\s*)\d+', sprintf('$1%d', n_elt_new));

% ApStop needs NO renumbering: the header form is a 3-vector POSITION
% (StopPos) and the element form is a 2-vector offset carried inside the
% stop element's OWN block (StopElt = that element, by read order) — both
% survive the splice untouched.  The one hazard: if the SEGMENTED element
% itself carried the element-form ApStop, the stop definition leaves with
% its block (SegMirMaker segments don't carry one).  Warn, don't guess —
% a stop on a single segment would be wrong anyway.
dropped_apstop = any(startsWith(strtrim(blocks{opts.elt}), "ApStop="));
if dropped_apstop
    warning('macos:design:segment_rx:apstop', ...
        ['the segmented element %d carried the ApStop= stop definition, ' ...
         'which is dropped with its block — set a stop explicitly ' ...
         '(header ApStop= x y z position form, or an ApStop= line in ' ...
         'another element''s block)'], opts.elt);
end

% --- 4. assemble ---------------------------------------------------------
merged = pre;
for k = 1:opts.elt-1
    merged = [merged; blocks{k}]; %#ok<AGROW>
end
for k = 1:nseg
    b = pblocks{k};
    if k == nseg && strlength(strtrim(b(end))) > 0
        b(end+1) = "";  %#ok<AGROW> % keep a blank line before the next block
    end
    merged = [merged; b]; %#ok<AGROW>
end
for k = opts.elt+1:numel(starts)
    b = blocks{k};
    j = find(startsWith(strtrim(b), "iElt="), 1);
    b(j) = regexprep(b(j), '(iElt=\s*)\d+', sprintf('$1%d', k + nseg - 1));
    merged = [merged; b]; %#ok<AGROW>
end
merged = [merged; tail];

out_in = opts.out_in;
if strlength(out_in) == 0
    out_in = fullfile(run_out.workdir, base + "_seg.in");
end
writelines(merged, out_in);

% --- 5. per-segment frames (as emitted) ----------------------------------
frames = repmat(struct('name', "", 'rpt', zeros(3,1), 'xhat', zeros(3,1), ...
                       'yhat', zeros(3,1), 'zhat', zeros(3,1), ...
                       'psi', zeros(3,1), 'lmon', NaN), ...
                1, nseg);
    function v = vec3(blk, key)
        t = regexp(blk(startsWith(strtrim(blk), key + "=")), ...
                   key + '=\s*(\S+)\s+(\S+)\s+(\S+)', 'tokens', 'once');
        v = str2double(string(t))';
    end
for k = 1:nseg
    b = pblocks{k};
    nm = regexp(b(startsWith(strtrim(b), "EltName=")), ...
                'EltName=\s*(\S+)', 'tokens', 'once');
    frames(k).name = string(nm{1});
    frames(k).rpt  = vec3(b, "RptElt");
    frames(k).xhat = vec3(b, "xMon");
    frames(k).yhat = vec3(b, "yMon");
    frames(k).zhat = vec3(b, "zMon");
    frames(k).psi  = vec3(b, "psiElt");   % surface normal keyword — the
                                          % ChkDf2 default xObs source
                                          % (NOT zMon: face triads clock)
    lm = regexp(b(startsWith(strtrim(b), "lMon=")), ...
                'lMon=\s*(\S+)', 'tokens', 'once');
    frames(k).lmon = str2double(string(lm{1}));
end

% tile geometry (manual: width = flat-to-flat, gap = inter-segment
% spacing; neighbor center distance = width + gap) from the SegMirMaker
% header, for boundary-true drawing / launcher placement (hex_tile).
gv = @(key) str2double(string(regexp( ...
    seg_header(find(startsWith(strtrim(seg_header), key + "="), 1)), ...
    key + '=\s*(\S+)', 'tokens', 'once')));
out = struct('in', char(out_in), 'nseg', nseg, ...
             'seg_elts', opts.elt:(opts.elt+nseg-1), 'n_elt', n_elt_new, ...
             'frames', frames, 'hx', run_out.hx, 'presc', run_out.presc, ...
             'width', gv("width"), 'gap', gv("gap"), ...
             'grid', char(opts.grid), ...
             'dropped_apstop', dropped_apstop, 'run', run_out);

% --- 6. optional: declare each segment's physical boundary as a
%        polygonal aperture (macos.design.seg_apertures) ------------------
% Appended at the END of each segment block: SegMirMaker .presc blocks
% carry no ApType/nObs lines (ChkDf2 defaults them), so there is nothing
% to replace — and the parser is last-wins anyway.
if opts.emit_apertures
    ap = macos.design.seg_apertures(out, 'pad', opts.ap_pad, ...
                                    'obs', opts.ap_obs);
    tlm = strtrim(merged);
    mstarts = find(startsWith(tlm, "iElt="));
    for s = nseg:-1:1                        % bottom-up, indices stay valid
        k = out.seg_elts(s);
        b1 = numel(merged); if k < numel(mstarts), b1 = mstarts(k+1) - 1; end
        ie = b1;
        while ie > mstarts(k) && strlength(strtrim(merged(ie))) == 0
            ie = ie - 1;
        end
        merged = [merged(1:ie); ap.blocks{s}; merged(ie+1:end)];
    end
    writelines(merged, out_in);
    out.apertures = ap;
end

% --- 7. carry the segmented element's obscurations onto the CENTER
%        segment (Seg1) -- e.g. a perforated primary's central hole
%        (Telescope set_hole emission).  The parent block leaves the
%        file with its obscurations; physically they live in the center
%        segment (Dave 2026-07-18).  The group is appended at the END
%        of Seg1's block (the parser is last-wins, and obscuration
%        declarations are contiguous: nObs= N followed by N ObsType=/
%        ObsVec= [+ vertex-row] sets), so it also wins over any nObs= 0
%        a seg_apertures emission wrote earlier in the block.
%        Limitation: not merged with per-segment obscurations
%        seg_apertures itself declares (pie wedges) -- those live on
%        OTHER segments today, so the groups never collide.
out.carried_obs = 0;
if opts.carry_obs
    pb = blocks{opts.elt};  tpb = strtrim(pb);
    iN = find(startsWith(tpb, "nObs="), 1);
    nOb = 0;
    if ~isempty(iN)
        tk = regexp(tpb(iN), 'nObs=\s*(\d+)', 'tokens', 'once');
        if ~isempty(tk), nOb = str2double(tk{1}); end
    end
    if nOb > 0
        j = iN + 1;
        while j <= numel(pb) && (startsWith(tpb(j), "ObsType=") || ...
              startsWith(tpb(j), "ObsVec=") || ...
              startsWith(tpb(j), "PolyObsVec=") || ...
              startsWith(tpb(j), "Poly3DObsVec=") || ...
              ~isempty(regexp(tpb(j), '^[-+0-9.eEdD ]+$', 'once')))
            j = j + 1;
        end
        grp = pb(iN:j-1);
        merged = readlines(out_in);
        tlm = strtrim(merged);
        mstarts = find(startsWith(tlm, "iElt="));
        k = out.seg_elts(1);
        b1 = numel(merged); if k < numel(mstarts), b1 = mstarts(k+1) - 1; end
        ie = b1;
        while ie > mstarts(k) && strlength(strtrim(merged(ie))) == 0
            ie = ie - 1;
        end
        merged = [merged(1:ie); grp; merged(ie+1:end)];
        writelines(merged, out_in);
        out.carried_obs = nOb;
    end
end
end

% =========================================================================
function grp = blocks_parent_figure_(lines, starts, elt)
%BLOCKS_PARENT_FIGURE_  The parent element's Zernike figure, re-keyed for
%   the segment FF channel (step 1b).  Returns [] when the parent block
%   carries no ZernCoef= (bare-conic parents: no-op, e5 fixtures
%   unchanged).  The figure is carried VERBATIM -- same type, modes,
%   coefs, normalization radius (lMon -> lFF) and frame (pMon/xMon/yMon/
%   zMon -> pFF/xFF/yFF/zFF) -- so every segment evaluates conic +
%   parent figure exactly as the monolithic parent did.
b1 = starts(elt);
b2 = numel(lines); if elt < numel(starts), b2 = starts(elt+1) - 1; end
pb = lines(b1:b2);  tp = strtrim(pb);

    function [v, rows] = take_(key)
    % the Key= line plus its bare-number continuation rows
        i = find(startsWith(tp, key + "="), 1);
        v = string.empty;  rows = string.empty;
        if isempty(i), return; end
        v = pb(i);
        j = i + 1;
        while j <= numel(pb) && strlength(tp(j)) > 0 && ...
              ~contains(pb(j), "=") && ...
              ~isempty(regexp(tp(j), '^[-+0-9.eEdD ]+$', 'once'))
            rows(end+1,1) = pb(j);  j = j + 1; %#ok<AGROW>
        end
    end

[cl, crows] = take_("ZernCoef");
if isempty(cl), grp = string.empty; return; end
[tl, ~]     = take_("ZernType");
[nl, ~]     = take_("nZernCoef");
[ml, mrows] = take_("ZernModes");
[ll, ~]     = take_("lMon");
rekey = @(s, from, to) regexprep(s, "(^\s*)" + from + "=", "$1" + to + "=");
grp = string.empty;
if ~isempty(tl), grp(end+1,1) = rekey(tl, "ZernType", "FFZernType"); end
if ~isempty(nl), grp(end+1,1) = rekey(nl, "nZernCoef", "nFFZernCoef"); end
if ~isempty(ml), grp = [grp; rekey(ml, "ZernModes", "FFZernModes"); mrows]; end
grp = [grp; rekey(cl, "ZernCoef", "FFZernCoef"); crows];
if ~isempty(ll), grp(end+1,1) = rekey(ll, "lMon", "lFF"); end
for fk = ["pMon" "xMon" "yMon" "zMon"; "pFF" "xFF" "yFF" "zFF"]
    [fl, ~] = take_(fk(1));
    if ~isempty(fl), grp(end+1,1) = rekey(fl, fk(1), fk(2)); end %#ok<AGROW>
end
end
