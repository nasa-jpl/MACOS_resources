function out = grid_augment_rx(rx_in, rx_out, opts)
%GRID_AUGMENT_RX  Add per-segment grid channels in the CLOCKED Mon frames.
%
%   out = macos.design.grid_augment_rx(RX_IN, RX_OUT) rewrites every
%   Element= Segment block of RX_IN with a flat grid-data channel whose
%   coordinate frame is that segment's OWN clocked Mon frame
%   (pData..zData = pMon..zMon) -- the frame macos.segment_grid_basis
%   assumes when it builds per-segment influence bases, and therefore
%   the frame dw_dgrid pokes need to LOCALIZE on the segment.
%
%   Any grid lines already present in a Segment block are REPLACED, not
%   appended.  This matters: SegMirMaker replicates the parent
%   element's data into every segment block, so SMM-derived fixtures
%   (the e5 corpus: e5pie.in, e5hex2.in, e5_seg_met.in, ...) carry a
%   PARENT-centered grid channel (pData = parent vertex, full-aperture
%   span) in each segment.  Poking those grids with a segment-frame
%   basis paints the mode about the APERTURE center instead of the
%   segment -- the "central dot" failure: responses pile up on the
%   center segment and the Jacobian rank-collapses (e5pie: rank 15 of
%   42, cond 1e7, vs 42 / 1.26 with correct frames).  Because the
%   prescription parser is last-key-wins, appending correct lines after
%   stale ones does NOT fix it -- the stale block must go.  (Diagnosed
%   2026-07-19; the e2e s4 augmentation built segment frames from
%   scratch, which is why it was healthy.)
%
%   OPTIONS:
%     'ng'         grid size nGridMat (default 256; model_size >= ng)
%     'span_frac'  grid span as a fraction of the parent Aperture
%                  (default 1.0: GridSrfdx = Aperture/(ng-1), same for
%                  all segments).  This is the dxGrid convention (Dave
%                  2026-07-19): the grid matrix scaling follows the
%                  BEAM -- at the pupil the ray grid spans the
%                  Aperture (dxGrid*(nGridpts-1) = Aperture), and the
%                  e5mono heritage GridSrfdx = 31.25 = 8000/256 is
%                  exactly this scaling.  Do NOT size the span from
%                  lMon: for pie WEDGES lMon is the hex-heritage
%                  'length' (one value for every segment), NOT a
%                  circumscribing radius about pMon -- a 2.2*lMon span
%                  clips the wedge outer corners, the influence maps
%                  stop filling the segment, and the dwdgrid channel
%                  goes non-physical (Dave caught this on s4)
%     'gdx'        explicit GridSrfdx (overrides span_frac; scalar or
%                  per-segment vector), prescription BaseUnits
%     'gridfile'   GridFile name (default 'flat.txt'); written beside
%                  RX_OUT as an ng x ng zero grid if missing (the
%                  engine resolves GridFile= relative to the cwd at
%                  load time -- keep RX_OUT's directory current)
%     'write_grid' write the flat grid file if absent (default true)
%
%   out fields: rx_out, nseg, ng, gdx (per segment), lmon (per
%   segment), replaced (per segment: true if stale grid lines were
%   removed), gridfile (path).
%
%   See also: macos.segment_grid_basis, macos.dw_dgrid_multi,
%             macos.design.segment_rx, macos.write_grid_file.

arguments
    rx_in  (1,1) string
    rx_out (1,1) string
    opts.ng (1,1) double {mustBeInteger, mustBePositive} = 256
    opts.span_frac (1,1) double {mustBePositive} = 1.0
    opts.gdx double = []
    opts.gridfile (1,1) string = ""   % default "flat<ng>.txt": a name
                                      % keyed to ng, so a stale same-name
                                      % grid file of a DIFFERENT size in
                                      % the cwd can't be picked up
    opts.write_grid (1,1) logical = true
end
assert(isfile(rx_in), 'grid_augment_rx: %s not found', rx_in);
if strlength(opts.gridfile) == 0
    opts.gridfile = sprintf("flat%d.txt", opts.ng);
end
if isempty(opts.gdx)
    ap = regexp(fileread(char(rx_in)), 'Aperture=\s*([\d.eEdD+-]+)', ...
                'tokens', 'once');
    assert(~isempty(ap), ['grid_augment_rx: no Aperture= in the Rx ' ...
        'header -- pass ''gdx'' explicitly']);
    opts.gdx = opts.span_frac * str2double(strrep(upper(ap{1}), 'D', 'E')) ...
               / (opts.ng - 1);
end

GRID_KEYS = ["nGridMat" "GridFile" "GridSrfdx" "GridType" ...
             "pData" "xData" "yData" "zData" "lData"];

L = splitlines(string(fileread(rx_in)));
outL   = strings(0, 1);
inseg  = false;
monf   = containers.Map;
segn   = 0;
lmon   = [];  gdx_out = [];  replaced = logical([]);
pending = strings(0, 1);          % grid block queued for insert at zMon
stale   = false;

flush_ = @(x) x;  %#ok<NASGU>  (clarity only)
for i = 1:numel(L)
    ln = L(i);
    tl = strtrim(ln);
    isNewElt = startsWith(tl, 'Element=');
    if isNewElt
        inseg = contains(tl, 'Segment');
        if inseg
            segn = segn + 1;
            monf = containers.Map;
            replaced(segn) = false; %#ok<AGROW>
        end
    end
    if inseg && ~isNewElt
        % drop any pre-existing grid-channel line in a Segment block
        key = extractBefore(tl, '=');
        if ~ismissing(key) && any(strtrim(key) == GRID_KEYS)
            replaced(segn) = true;
            stale = true; %#ok<NASGU>
            continue;                      % REPLACED, not kept
        end
        for k = ["lMon" "pMon" "xMon" "yMon" "zMon"]
            if startsWith(tl, k + "=")
                monf(char(k)) = regexprep(char(tl), '^\s*\w+=', '');
            end
        end
    end
    outL(end+1) = ln; %#ok<AGROW>
    if inseg && startsWith(tl, "zMon=") && monf.Count >= 4
        assert(isKey(monf, 'pMon') && isKey(monf, 'xMon') && ...
               isKey(monf, 'yMon') && isKey(monf, 'zMon'), ...
            'grid_augment_rx: segment %d lacks a full clocked Mon frame', segn);
        lm = NaN;
        if isKey(monf, 'lMon'), lm = sscanf(monf('lMon'), '%g'); end
        assert(isfinite(lm) && lm > 0, ...
            'grid_augment_rx: segment %d has no usable lMon (need span)', segn);
        lmon(segn) = lm; %#ok<AGROW>
        if isscalar(opts.gdx), g = opts.gdx; else, g = opts.gdx(segn); end
        gdx_out(segn) = g; %#ok<AGROW>
        pending = [ ...
            sprintf("         nGridMat=  %d", opts.ng)
            "         GridFile=  " + opts.gridfile
            sprintf("        GridSrfdx=%.6E", g)
            "            pData=" + monf('pMon')
            "            xData=" + monf('xMon')
            "            yData=" + monf('yMon')
            "            zData=" + monf('zMon')];
        outL = [outL(:); pending(:)]; %#ok<AGROW>
        pending = strings(0, 1);
    end
end
assert(segn > 0, 'grid_augment_rx: no Element= Segment blocks in %s', rx_in);

fid = fopen(rx_out, 'w');
fprintf(fid, '%s\n', outL);
fclose(fid);

gf = fullfile(fileparts(char(rx_out)), char(opts.gridfile));
if opts.write_grid && ~isfile(gf)
    macos.write_grid_file(gf, zeros(opts.ng));
end

out = struct('rx_out', string(rx_out), 'nseg', segn, 'ng', opts.ng, ...
    'gdx', gdx_out, 'lmon', lmon, 'replaced', replaced, ...
    'gridfile', string(gf));
end
