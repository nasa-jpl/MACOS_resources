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
%   at `elt`) → replace the parent's source GridType with the segment
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
%   the cwd); plus the segmentation options forwarded to
%   segmirmaker_run: rings, grid, gap, dofs, meas_config, seg_size,
%   ap_diam, psi, segxgrid, f, ecc, standoff, binary.
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
pblocks = cell(nseg, 1);
for k = 1:nseg
    e = numel(plines); if k < nseg, e = pstarts(k+1) - 1; end
    pblocks{k} = plines(pstarts(k):e);
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
                       'yhat', zeros(3,1), 'zhat', zeros(3,1), 'lmon', NaN), ...
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
end
