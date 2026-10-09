function out = promote_segments_freeform(rx_in, rx_out, opts)
%PROMOTE_SEGMENTS_FREEFORM  Turn conic optics into inert FreeForm carriers.
%
%   out = macos.design.promote_segments_freeform(RX_IN, RX_OUT) rewrites
%   every  Element= Segment  block of RX_IN into a  Surface= FreeForm
%   (SrfType 14) block that carries a MonZernike figure CHANNEL -- so the
%   dw_dz_zernike (MonZern) and, after macos.design.grid_augment_rx, the
%   dw_dgrid rungs have something to harvest -- while adding ZERO
%   wavefront: the conic base (Kr/Kc) is kept verbatim and every figure
%   coefficient is zero.  Everything else (which blocks, their clocked
%   frames, their lMon) is DERIVED FROM THE Rx; the only required input is
%   the prescription itself.
%
%   Reusable beyond segments (Dave 2026-08-21): pass 'elts' to promote a
%   named set instead of every Segment -- e.g. a monolithic SM/TM
%   Reflector, so the figure rungs harvest it too.  A block with no
%   clocked Mon frame (a bare on-axis segment, or a Reflector) gets one
%   synthesized from its psiElt (surface normal) + RptElt/VptElt (vertex);
%   a block that already carries pMon/xMon/yMon/zMon keeps it verbatim.
%
%   This is the fixture-side half of PLAN_CONFIGURATIONS.md departure #6:
%   a deck of conic optics cannot feed the figure rungs of the
%   configuration-axis family, and promoting it must be shown optically
%   INERT (a change to a committed deck).  The engine facts it relies on
%   (verified against the Fortran 2026-08-21):
%
%     * A FreeForm's Mon channel is live only when lMon > 0
%       (SetFreeFormFlags: ifMon = (lMon>0), elt_mod.F).  lMon > 0 is the
%       sleeper requirement, not the type code -- a promoted block with no
%       lMon harvests silently NOTHING.
%     * MonZernType= BornWolf sets MonZernTypeL = 2, which dispatches the
%       MonZernCoef -> MonCoef conversion (ZerntoMon2).  A promoted-from-Rx
%       FreeForm does get ChkDf2's ANSI default even without it, but ANSI
%       and BornWolf are DIFFERENT mode-index bases, so a poke of "mode 4"
%       is a different shape; we set it explicitly (and it matches the
%       reference figured deck, e5hex1.in).
%     * All-zero MonZernCoef is inert: MonomialEval returns zero sag and
%       zero surface-normal perturbation when every coefficient is zero,
%       and ZerntoMon is a pure linear map (zero in -> zero out).  A poked
%       coefficient produces a real, localized response.
%     * The Mon frame pMon/xMon/yMon/zMon must be the optic's CLOCKED
%       triad or a per-optic poke de-localizes (the e5 "central dot").
%
%   Stale grid lines are DROPPED.  A conic optic may carry an inert
%   nGridMat / GridFile= none / GridSrfdx (the zoom fixture's segments
%   do): inert under Conic, but under FreeForm ifGridTerm=(nGridMat>0)
%   goes LIVE and the engine would try to consume GridFile= none.  Two
%   ways to give the promoted deck a real grid:
%     * leave 'grid' false (default) and grid-augment the SEGMENTS at
%       harvest time with macos.design.grid_augment_rx (the segment path);
%     * pass 'grid' true to write a flat grid channel into the promoted
%       blocks here -- needed for NON-segment optics (SM/TM), which
%       grid_augment_rx (segment-only) will not touch.
%   A ZernCoef= block (a SrfType-8 Zernike artifact, unused by FreeForm)
%   is dropped too, to match e5hex1.
%
%   OPTIONS
%     'elts'       element ids to promote ([] = every Element= Segment,
%                  the default and back-compatible behaviour).  Ids not
%                  found as element blocks are an error.
%     'zern_type'  MonZernType string (default 'BornWolf').
%     'n_mon'      nMonZernCoef to declare (default 1, matching e5hex1;
%                  the poke veneer addresses higher modes regardless).
%     'lmon'       lMon for a frame-LESS block.  Scalar, or a
%                  containers.Map(iElt->lMon) for a per-optic value (SM/TM
%                  want their own footprint radius, not the segment's).
%                  [] = median of the blocks that already declare an lMon.
%                  A block that already carries lMon keeps it.  The vertex
%                  is NOT the optic centre in general and lMon is a real
%                  footprint radius, so derive both by TRACING all fields x
%                  all configurations (see design/src, e.g. the
%                  light-bearing-optics helper) and pass the results here.
%     'pmon'       containers.Map(iElt->[3x1]) frame-centre override for a
%                  frame-LESS block -- e.g. the traced footprint CENTROID,
%                  which is where the figure should centre (the vertex may
%                  be far from the beam).  [] = vertex (RptElt/VptElt).
%     'grid'       also write a flat grid channel (nGridMat/GridFile/
%                  GridSrfdx + pData..zData = the Mon frame) into every
%                  promoted block (default false).  For non-segment optics
%                  grid_augment_rx cannot reach, this is how the grid rung
%                  gets a surface to poke.
%     'ng'         grid size for 'grid' (default 64).
%     'gridfile'   grid file name for 'grid' (default "flat<ng>.txt");
%                  written beside RX_OUT as an ng x ng zero grid if absent.
%     'gdx'        GridSrfdx for 'grid' (scalar, or Map iElt->gdx).  [] =
%                  2*lMon/(ng-1) per block (span covers the footprint).
%
%   out fields: rx_out, elts (promoted ids), nseg (count), n_synth_frame,
%   lmon (per promoted elt), dropped_grid / dropped_zern (per block),
%   gridfile (when 'grid').
%
%   See also: macos.design.grid_augment_rx, macos.find_freeform_elts,
%             macos.zernike_grid_basis, run_sensitivities.  Reference
%             figured deck: templates/50_sensitivities/run_dwdx_multi/e5hex1.in.

arguments
    rx_in  (1,1) string
    rx_out (1,1) string
    opts.elts double = []
    opts.zern_type (1,:) char = 'BornWolf'
    opts.n_mon (1,1) double {mustBeInteger, mustBePositive} = 1
    opts.lmon = []                        % scalar | containers.Map | []
    opts.pmon = []                        % containers.Map(iElt->[3x1]) frame
                                          % centre override (a traced
                                          % footprint centroid); frame-less
                                          % blocks only, [] = vertex
    opts.grid (1,1) logical = false
    opts.ng (1,1) double {mustBeInteger, mustBePositive} = 64
    opts.gridfile (1,1) string = ""
    opts.gdx = []                         % scalar | containers.Map | []
end
assert(isfile(rx_in), 'promote_segments_freeform: %s not found', rx_in);
% guard the in-place footgun: reading and writing the same file mid-stream
% yields a corrupt deck.  Chain promotions through DISTINCT paths.
assert(~strcmp(fullpath_(char(rx_in)), fullpath_(char(rx_out))), ...
    ['promote_segments_freeform: rx_in and rx_out are the same file ' ...
     '(%s).  Write to a distinct path (chain promotions file-to-file).'], ...
    char(rx_in));
want = unique(double(opts.elts(:)).');    % explicit id set, or empty = all Segments
if opts.grid && strlength(opts.gridfile) == 0
    opts.gridfile = sprintf("flat%d.txt", opts.ng);
end

% grid-channel keys that must NOT survive verbatim onto a FreeForm (a
% stale grid would go live); we re-emit them ourselves when 'grid' is set.
% A ZernCoef block is dropped separately (it is multi-line).
GRID_KEYS = ["nGridMat" "GridFile" "GridSrfdx" "GridType" ...
             "pData" "xData" "yData" "zData" "lData"];
FRAME_KEYS = ["pMon" "xMon" "yMon" "zMon"];

L = splitlines(string(fileread(rx_in)));

% ---- pass 1: survey every element block --------------------------------
% Record, per element block: its iElt, whether it is a Segment, whether it
% carries a full Mon frame, its lMon (if any), and psi/rpt/vpt (for frame
% synthesis).  This lets 'elts' name any block and lets a frame-less block
% (bare centre segment, or a Reflector) get a synthesized triad.
blk = struct('iElt',{}, 'is_seg',{}, 'has_frame',{}, 'lmon',{}, ...
             'psi',{}, 'rpt',{}, 'vpt',{}, ...
             'pMon',{}, 'xMon',{}, 'yMon',{}, 'zMon',{});
lmon_seen = [];  b = 0;  cur = -1;
for i = 1:numel(L)
    tl = strtrim(L(i));
    m = regexp(char(tl), '^iElt=\s*(-?\d+)', 'tokens', 'once');
    if ~isempty(m), cur = str2double(m{1}); end
    if startsWith(tl, 'Element=')
        b = b + 1;
        blk(b) = struct('iElt',cur, 'is_seg',contains(tl,'Segment'), ...
            'has_frame',0, 'lmon',NaN, 'psi',[0;0;-1], 'rpt',[0;0;0], ...
            'vpt',[0;0;0], 'pMon',[], 'xMon',[], 'yMon',[], 'zMon',[]);
        continue
    end
    if b == 0, continue; end
    for k = FRAME_KEYS
        if startsWith(tl, k + "="), blk(b).has_frame = blk(b).has_frame + 1; end
    end
    if startsWith(tl, "pMon="), blk(b).pMon = vec3_(tl); end
    if startsWith(tl, "xMon="), blk(b).xMon = vec3_(tl); end
    if startsWith(tl, "yMon="), blk(b).yMon = vec3_(tl); end
    if startsWith(tl, "zMon="), blk(b).zMon = vec3_(tl); end
    if startsWith(tl, "lMon=")
        v = vec3_(tl);  blk(b).lmon = v(1);  lmon_seen(end+1) = v(1); %#ok<AGROW>
    end
    if startsWith(tl, "psiElt="), blk(b).psi = vec3_(tl); end
    if startsWith(tl, "RptElt="), blk(b).rpt = vec3_(tl); end
    if startsWith(tl, "VptElt="), blk(b).vpt = vec3_(tl); end
end
assert(~isempty(blk), 'promote_segments_freeform: no Element= blocks in %s', rx_in);

% which blocks to promote
allids = [blk.iElt];
if isempty(want)
    promote_ids = allids([blk.is_seg]);
else
    missing = setdiff(want, allids);
    assert(isempty(missing), ['promote_segments_freeform: elts ' ...
        '%s are not element blocks in the Rx'], mat2str(missing));
    promote_ids = want;
end
assert(~isempty(promote_ids), ...
    'promote_segments_freeform: nothing to promote (no Segment blocks / empty elts)');

% default lMon for a frame-less block with none declared: the median of
% the blocks that DO declare one (the segment scale).
good = lmon_seen(isfinite(lmon_seen) & lmon_seen > 0);
lmon_default = NaN;  if ~isempty(good), lmon_default = median(good); end

% ---- pass 2: rewrite ---------------------------------------------------
outL = strings(0, 1);
b = 0;  cur = -1;  promoting = false;  drop_cont = false;
lmon_used = containers.Map('KeyType','double','ValueType','double');
n_synth = 0;  dropped_grid = 0;  dropped_zern = 0;  gdx_used = [];
for i = 1:numel(L)
    ln = L(i);  tl = strtrim(ln);
    isNewElt = startsWith(tl, 'Element=');
    isKeyLine = ~isempty(regexp(char(tl), '^[A-Za-z]\w*\s*=', 'once'));
    m = regexp(char(tl), '^iElt=\s*(-?\d+)', 'tokens', 'once');
    if ~isempty(m), cur = str2double(m{1}); end

    % drop a multi-line ZernCoef block (only inside a promoted block)
    if drop_cont
        if isKeyLine || isNewElt
            drop_cont = false;
        else
            dropped_zern = dropped_zern + 1;  continue
        end
    end

    if isNewElt
        b = b + 1;
        promoting = any(blk(b).iElt == promote_ids);
        outL(end+1) = ln; %#ok<AGROW>
        continue
    end

    if promoting
        keytok = regexp(char(tl), '^([A-Za-z]\w*)\s*=', 'tokens', 'once');
        thiskey = ""; if ~isempty(keytok), thiskey = string(keytok{1}); end

        if any(thiskey == GRID_KEYS)          % drop stale grid lines
            dropped_grid = dropped_grid + 1;  continue
        end
        if thiskey == "ZernCoef"              % drop the SrfType-8 artifact
            dropped_zern = dropped_zern + 1;  drop_cont = true;  continue
        end
        if thiskey == "Surface"
            this = blk(b);
            lm = resolve_scalar_(opts.lmon, this.iElt);      % explicit override
            if isnan(lm), lm = this.lmon; end                % block's own lMon
            if isnan(lm), lm = lmon_default; end             % segment median
            assert(isfinite(lm) && lm > 0, ['promote_segments_freeform: ' ...
                'no lMon for elt %d (pass ''lmon'')'], this.iElt);
            lmon_used(this.iElt) = lm;

            % the block's Mon frame -- its own if it carries one, else a
            % triad synthesized from psiElt/centre.  ONE resolver, so the
            % grid frame (pData..zData) below is provably the SAME frame.
            this.pmon = resolve_vec_(opts.pmon, this.iElt);   % traced centre override
            [fp, fx, fy, fz, synth] = mon_frame_(this);

            outL(end+1) = "          Surface=  FreeForm"; %#ok<AGROW>
            outL = [outL(:); mon_lines_(opts)]; %#ok<AGROW>
            if synth
                % emit the synthesized Mon frame + lMon (a pre-framed block
                % keeps its own frame, emitted verbatim below)
                outL = [outL(:);
                    fmtv_("             pMon=", fp)
                    fmtv_("             xMon=", fx)
                    fmtv_("             yMon=", fy)
                    fmtv_("             zMon=", fz)
                    sprintf("             lMon=  %.10E", lm)]; %#ok<AGROW>
                n_synth = n_synth + 1;
            end
            if opts.grid
                g = resolve_scalar_(opts.gdx, this.iElt);
                if isnan(g), g = 2*lm/(opts.ng-1); end        % span ~ 2*lMon
                gdx_used(end+1) = g; %#ok<AGROW>
                outL = [outL(:);
                    sprintf("         nGridMat=  %d", opts.ng)
                    "         GridFile=  " + opts.gridfile
                    sprintf("        GridSrfdx=%.6E", g)
                    fmtv_("            pData=", fp)
                    fmtv_("            xData=", fx)
                    fmtv_("            yData=", fy)
                    fmtv_("            zData=", fz)]; %#ok<AGROW>
            end
            continue
        end
    end
    outL(end+1) = ln; %#ok<AGROW>
end

fid = fopen(rx_out, 'w');
assert(fid > 0, 'promote_segments_freeform: cannot write %s', rx_out);
fprintf(fid, '%s\n', outL);
fclose(fid);

if opts.grid
    gf = fullfile(fileparts(char(rx_out)), char(opts.gridfile));
    if ~isfile(gf), macos.write_grid_file(gf, zeros(opts.ng)); end
end

out = struct('rx_out', string(rx_out), 'elts', promote_ids, ...
    'nseg', numel(promote_ids), 'n_synth_frame', n_synth, ...
    'lmon', lmon_used, 'dropped_grid', dropped_grid, ...
    'dropped_zern', dropped_zern);
if opts.grid, out.gridfile = string(opts.gridfile); out.gdx = gdx_used; end
end


% =====================================================================
function s = mon_lines_(opts)
%MON_LINES_  The MonZernike channel lines injected into every promoted block.
s = [ ...
    "      MonZernType=  " + string(opts.zern_type)
    sprintf("     nMonZernCoef=  %d", opts.n_mon)
    "      MonZernCoef=  " + strjoin(repmat("0.0E+00", 1, opts.n_mon), "  ")];
end


% ---------------------------------------------------------------------
function [p, x, y, z, synth] = mon_frame_(blk)
%MON_FRAME_  The block's clocked Mon frame: its own, or a synthesized one.
%   synth=false when the block already carries a full pMon/xMon/yMon/zMon
%   (returned verbatim -- the Mon and grid frames then match the deck).
%   synth=true otherwise: zMon = surface normal (psiElt); pMon = the
%   optic centre (blk.pmon override if given -- e.g. a traced footprint
%   centroid; else RptElt, fallback VptElt); x/y an orthonormal in-plane
%   pair.  Inert-safe: amplitudes are zero, so the clocking never touches
%   the light; a valid orthonormal triad + lMon>0 is all the engine needs.
if blk.has_frame >= 4 && ~isempty(blk.pMon)
    p = blk.pMon(:); x = blk.xMon(:); y = blk.yMon(:); z = blk.zMon(:);
    synth = false;  return
end
synth = true;
z = blk.psi(:);  z = z / norm(z);
if isfield(blk,'pmon') && ~isempty(blk.pmon)
    p = blk.pmon(:);                      % traced footprint centre (override)
else
    p = blk.rpt(:);  if all(p == 0), p = blk.vpt(:); end
end
ref = [1;0;0];  if abs(z(1)) > 0.9, ref = [0;1;0]; end
x = ref - (ref.' * z) * z;  x = x / norm(x);
y = cross(z, x);                          % right-handed: x cross y = z
end


% ---------------------------------------------------------------------
function v = resolve_scalar_(spec, iElt)
%RESOLVE_SCALAR_  spec may be [] | scalar | containers.Map(iElt->val).
%   Returns NaN when nothing applies to this element.
if isempty(spec), v = NaN; return; end
if isa(spec, 'containers.Map')
    if isKey(spec, iElt), v = spec(iElt); else, v = NaN; end
    return
end
v = double(spec);          % scalar applies to every promoted block
end


% ---------------------------------------------------------------------
function v = resolve_vec_(spec, iElt)
%RESOLVE_VEC_  spec may be [] | containers.Map(iElt->[3x1]).  [] if none.
v = [];
if isa(spec, 'containers.Map') && isKey(spec, iElt)
    v = double(spec(iElt));  v = v(:);
end
end


% ---------------------------------------------------------------------
function p = fullpath_(p)
%FULLPATH_  Absolutise a path for the in-place-write guard (no I/O).
if ~startsWith(p, filesep), p = fullfile(pwd, p); end
end


% ---------------------------------------------------------------------
function s = fmtv_(tag, v)
s = string(tag) + sprintf("  %.10E", v(1)) + sprintf("  %.10E", v(2)) + ...
    sprintf("  %.10E", v(3));
end


% ---------------------------------------------------------------------
function v = vec3_(line)
%VEC3_  First 1-3 numbers after Key= , Fortran D exponents included.
%   Returns a 3x1 (missing entries 0); a scalar caller reads v(1).
rhs = regexprep(char(line), '^\s*\w+\s*=', '');
rhs = regexprep(rhs, '%.*$', '');                 % strip inline comment
rhs = regexprep(rhs, '([dD])([+-]?\d)', 'e$2');   % D exponent -> e
n = sscanf(rhs, '%g', 3);
v = zeros(3, 1);  v(1:numel(n)) = n;
end
