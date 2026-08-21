function out = promote_segments_freeform(rx_in, rx_out, opts)
%PROMOTE_SEGMENTS_FREEFORM  Turn Conic segments into inert FreeForm carriers.
%
%   out = macos.design.promote_segments_freeform(RX_IN, RX_OUT) rewrites
%   every  Element= Segment  block of RX_IN whose  Surface=  is Conic into
%   a  Surface= FreeForm  (SrfType 14) block that carries a MonZernike
%   figure CHANNEL -- so the dw_dz_zernike (MonZern) and, after
%   macos.design.grid_augment_rx, the dw_dgrid rungs have something to
%   harvest -- while adding ZERO wavefront: the conic base (Kr/Kc) is
%   kept verbatim and every figure coefficient is zero.
%
%   This is the fixture-side half of PLAN_CONFIGURATIONS.md departure #6:
%   a deck of Conic segments cannot feed the figure rungs of the
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
%     * The Mon frame pMon/xMon/yMon/zMon must be the segment's CLOCKED
%       triad or a per-segment poke de-localizes (the e5 "central dot").
%       Segments that already carry a frame keep it verbatim; a segment
%       with none (a bare on-axis centre segment) gets one synthesized
%       from its psiElt (surface normal) and RptElt (centre).
%
%   Stale grid lines are DROPPED.  A Conic segment may carry an inert
%   nGridMat / GridFile= none / GridSrfdx (the zoom fixture does): inert
%   under Conic, but under FreeForm ifGridTerm=(nGridMat>0) goes LIVE and
%   the engine would try to consume GridFile= none.  The dwdgrid rung adds
%   a real grid channel with macos.design.grid_augment_rx; the promoted
%   deck itself stays pure-Mon.  A ZernCoef= block (a SrfType-8 Zernike
%   artifact, unused by FreeForm) is dropped too, to match e5hex1.
%
%   OPTIONS
%     'zern_type'  MonZernType string (default 'BornWolf').
%     'n_mon'      nMonZernCoef to declare (default 1, matching e5hex1;
%                  the poke veneer addresses higher modes regardless).
%     'lmon'       lMon to synthesize for a frame-less segment ([] =
%                  median of the segments that already declare one).
%
%   out fields: rx_out, nseg (segments promoted), n_synth_frame (how many
%   had a frame synthesized), lmon_synth (the value used), dropped_grid /
%   dropped_zern (per-segment line counts removed).
%
%   See also: macos.design.grid_augment_rx, macos.find_freeform_elts,
%             run_sensitivities.  Reference figured deck:
%             templates/50_sensitivities/run_dwdx_multi/e5hex1.in.

arguments
    rx_in  (1,1) string
    rx_out (1,1) string
    opts.zern_type (1,:) char = 'BornWolf'
    opts.n_mon (1,1) double {mustBeInteger, mustBePositive} = 1
    opts.lmon double = []
end
assert(isfile(rx_in), 'promote_segments_freeform: %s not found', rx_in);

% grid-channel keys that must NOT survive onto a FreeForm (they would go
% live); a ZernCoef block is dropped separately (it is multi-line).
GRID_KEYS = ["nGridMat" "GridFile" "GridSrfdx" "GridType" ...
             "pData" "xData" "yData" "zData" "lData"];
FRAME_KEYS = ["pMon" "xMon" "yMon" "zMon"];

L = splitlines(string(fileread(rx_in)));

% ---- pass 1: per-segment survey (frame present? geometry for synth?) ----
seg = struct('has_frame', {}, 'has_lmon', {}, 'psi', {}, 'rpt', {}, 'vpt', {});
lmon_seen = [];
inseg = false;  s = 0;
for i = 1:numel(L)
    tl = strtrim(L(i));
    if startsWith(tl, 'Element=')
        inseg = contains(tl, 'Segment');
        if inseg
            s = s + 1;
            seg(s) = struct('has_frame', 0, 'has_lmon', false, ...
                'psi', [0;0;-1], 'rpt', [0;0;0], 'vpt', [0;0;0]);
        end
        continue
    end
    if ~inseg, continue; end
    for k = FRAME_KEYS
        if startsWith(tl, k + "="), seg(s).has_frame = seg(s).has_frame + 1; end
    end
    if startsWith(tl, "lMon=")
        seg(s).has_lmon = true;
        v = vec3_(tl);  lmon_seen(end+1) = v(1); %#ok<AGROW>  (lMon is scalar)
    end
    if startsWith(tl, "psiElt="), seg(s).psi = vec3_(tl); end
    if startsWith(tl, "RptElt="), seg(s).rpt = vec3_(tl); end
    if startsWith(tl, "VptElt="), seg(s).vpt = vec3_(tl); end
end
nseg = s;
assert(nseg > 0, 'promote_segments_freeform: no Element= Segment blocks in %s', rx_in);

lmon_synth = opts.lmon;
if isempty(lmon_synth)
    good = lmon_seen(isfinite(lmon_seen) & lmon_seen > 0);
    assert(~isempty(good), ['promote_segments_freeform: no segment ' ...
        'declares an lMon and none was passed -- cannot size a frame']);
    lmon_synth = median(good);
end

% ---- pass 2: rewrite ---------------------------------------------------
outL = strings(0, 1);
inseg = false;  s = 0;  drop_cont = false;
dropped_grid = zeros(1, nseg);  dropped_zern = zeros(1, nseg);
n_synth = 0;
for i = 1:numel(L)
    ln = L(i);
    tl = strtrim(ln);
    isNewElt = startsWith(tl, 'Element=');
    isKeyLine = ~isempty(regexp(char(tl), '^[A-Za-z]\w*\s*=', 'once'));

    % a multi-line ZernCoef block: drop the ZernCoef= line and its
    % continuation rows (lines with no Keyword=) until the next key
    if drop_cont
        if isKeyLine || isNewElt
            drop_cont = false;       % fall through and handle this line
        else
            dropped_zern(s) = dropped_zern(s) + 1;
            continue                 % a ZernCoef continuation row
        end
    end

    if isNewElt
        inseg = contains(tl, 'Segment');
        if inseg, s = s + 1; end
        outL(end+1) = ln; %#ok<AGROW>
        continue
    end

    if inseg
        keytok = regexp(char(tl), '^([A-Za-z]\w*)\s*=', 'tokens', 'once');
        thiskey = ""; if ~isempty(keytok), thiskey = string(keytok{1}); end

        % drop stale grid channel lines (single-line keys)
        if any(thiskey == GRID_KEYS)
            dropped_grid(s) = dropped_grid(s) + 1;
            continue
        end
        % drop the ZernCoef block (SrfType-8 artifact; unused by FreeForm)
        if thiskey == "ZernCoef"
            dropped_zern(s) = dropped_zern(s) + 1;
            drop_cont = true;        % and its continuation rows
            continue
        end
        % promote the Surface line and inject the Mon channel after it
        if thiskey == "Surface"
            outL(end+1) = "          Surface=  FreeForm"; %#ok<AGROW>
            outL = [outL(:); mon_lines_(opts, lmon_synth)]; %#ok<AGROW>
            % synthesize a clocked Mon frame + lMon for a frame-less
            % segment (a bare on-axis centre); segments that already
            % carry one keep it verbatim below
            if seg(s).has_frame < 4
                outL = [outL(:); frame_lines_(seg(s), lmon_synth)]; %#ok<AGROW>
                n_synth = n_synth + 1;
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

out = struct('rx_out', string(rx_out), 'nseg', nseg, ...
    'n_synth_frame', n_synth, 'lmon_synth', lmon_synth, ...
    'dropped_grid', dropped_grid, 'dropped_zern', dropped_zern);
end


% =====================================================================
function s = mon_lines_(opts, lmon)
%MON_LINES_  The MonZernike channel lines injected into every segment.
s = [ ...
    "      MonZernType=  " + string(opts.zern_type)
    sprintf("     nMonZernCoef=  %d", opts.n_mon)
    "      MonZernCoef=  " + strjoin(repmat("0.0E+00", 1, opts.n_mon), "  ")];
end


% ---------------------------------------------------------------------
function s = frame_lines_(seg, lmon)
%FRAME_LINES_  Synthesize a clocked Mon frame for a frame-less segment.
%   zMon = surface normal (psiElt); pMon = the segment centre (RptElt,
%   falling back to VptElt); x/y an orthonormal in-plane pair.  Inert-safe:
%   the amplitudes are zero, so the exact clocking never touches the light;
%   a valid orthonormal triad is all the engine needs.
z = seg.psi(:);  z = z / norm(z);
p = seg.rpt(:);  if all(p == 0), p = seg.vpt(:); end
ref = [1;0;0];  if abs(z(1)) > 0.9, ref = [0;1;0]; end
x = ref - (ref.' * z) * z;  x = x / norm(x);
y = cross(z, x);                       % right-handed: x cross y = z
s = [ ...
    fmtv_("             pMon=", p)
    fmtv_("             xMon=", x)
    fmtv_("             yMon=", y)
    fmtv_("             zMon=", z)
    sprintf("             lMon=  %.10E", lmon)];
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
