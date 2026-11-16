function R = seg_route(rx, varargin)
%SEG_ROUTE  Ray->segment routing verdict for one segmented prescription.
%
%   R = SEG_ROUTE(RX) measures whether the engine routes rays to the
%   segment ELEMENT that geometrically owns them, using physical arbiters
%   only.  Run ONE DECK PER MATLAB PROCESS (a second load_rx of a 256-grid
%   segmented deck can kill the process).
%
%   INSTRUMENTS
%
%     A  APERTURE RAY COUNT  (gold, where available).  Only for decks whose
%        segments carry per-segment `ApType= Polygonal`.  A permuted deck
%        clips each segment's rays against the aperture polygon of the
%        segment 180 deg across the ring, so most of the pupil is lost.
%        Verdict = committed-header pupil count vs a SegXgrid-negated copy.
%        No frame, no angle, no array map.
%
%     B  DRAW-3D GLOBAL ATTRIBUTION  (primary, available everywhere).
%        macos.draw_rays3d returns each crossing's TRUE GLOBAL (x,y,z)
%        (engine Draw3DVec = RayPos verbatim) alongside its element label.
%        For each segment element, the mean crossing position is compared
%        to that element's own RptElt and to its 180-partner's: nearest
%        wins.  A distance comparison in the deck's own global frame --
%        no projection, no array map, no angle convention.
%        Cross-validated against A on every deck that carries both.
%
%     C  DRAW-2D FRAME DEMONSTRATION  (diagnostic; the round-1 trap).
%        macos.draw_rays projects onto the SOURCE/RAY-GRID basis
%        (xDraw=xGrid, yDraw=yGrid for plane 'XY' -- macos_cmd_loop.inc
%        DRAW).  The whole e5mono family has xGrid = (-1,0,0), so b.U is
%        MINUS global X and reading b.U as a global X inverts the verdict
%        on the two segments that straddle the X axis.  Reported both ways
%        so the slip is visible rather than inferred.
%
%     D  TILT-PIVOT PISTON  (opt-in, 'pivot',true).  Rotate segment k about
%        local y with the pivot at (a) its own RptElt and (b) the
%        180-partner's, and read the mean OPD over the region defined by a
%        separate clean PISTON poke.  KNOWN CONFOUND: on a fast curved
%        parent, moving the pivot adds a rigid common-mode piston and the
%        residual is dominated by surface curvature, so neither reading
%        goes to zero -- and on aperture-carrying decks set_elt_rpt also
%        MOVES the aperture polygon.  Kept for the record, not for
%        verdicts.  See SEG_AUDIT_STATUS.md.
%
%     T3 INERTNESS PROBE.  Peak |dOPD| for a local/global rotation and a
%        local/global translation of one ring segment, so a deck whose
%        segment rotations are bit-inert is recorded rather than mistaken
%        for a routing result.
%
%   Name/value options:
%     'model_size'  []       macos.init size (default 512; 1024 if nGridpts>128)
%     'opd_elt'     []       OPD plane (default: first non-Segment element)
%     'piston'      2.5e-4   piston probe, deck BaseUnits
%     'tilt'        5e-7     tilt probe, radians
%     'pivot'       false    also run instrument D
%     'nProbe'      3        ring segments instrument D tests
%     'verbose'     true
%
%   Example:
%     addpath ~/dev/MACOS_res_dev/mmacos/src
%     addpath ~/dev/MACOS_res_dev/segmirmaker
%     R = seg_route('~/dev/macos/ZGD_test_files/e5pie.in');
%
%   See also SEG_AUDIT, SEG_READ_RX.

p = inputParser;
p.addParameter('model_size', []);
p.addParameter('opd_elt', []);
p.addParameter('piston', 2.5e-4);
p.addParameter('tilt', 5e-7);
p.addParameter('pivot', false);
p.addParameter('nProbe', 3);
p.addParameter('verbose', true);
p.parse(varargin{:});
o = p.Results;

rx = char(rx);
if startsWith(rx, '~'), rx = fullfile(getenv('HOME'), rx(3:end)); end
R = struct('rx', rx, 'err', '', 'verdict', 'UNDETERMINED', 'by', '');

D = seg_read_rx(rx);
segs = find(strcmpi(D.eltType, 'Segment'));
R.iSeg = segs(:).'; R.nSeg = numel(segs);
if isempty(segs), R.err = 'no Element=Segment elements'; return, end

R.gridType = getdef(D.hdr, 'GridType', '');
R.nGridpts = getdef(D.hdr, 'nGridpts', NaN);
R.SegXgrid = getdef(D.hdr, 'SegXgrid', [1;0;0]);
xg = getdef(D.hdr, 'xGrid', [1;0;0]); xg = xg(1:3)/norm(xg(1:3));
yg = getdef(D.hdr, 'yGrid', [0;1;0]); yg = yg(1:3)/norm(yg(1:3));
R.xGrid = xg; R.yGrid = yg;
su = R.SegXgrid(1:3)/norm(R.SegXgrid(1:3));
R.segX2ang = atan2d(dot(yg,su), dot(xg,su));   % sourcsub.F:201 basis

R.Rpt = D.vecs.RptElt(:,segs);
R.Vpt = D.vecs.VptElt(:,segs);
R.psi = D.vecs.psiElt(:,segs(1)); R.psi = R.psi/norm(R.psi);

% ---- 180-deg partner map, in the PARENT's own geometry -----------------
% Rotate (Rpt - Vpt) by 180 deg about the parent axis psi:  u -> 2(u.n)n - u.
% Then take the deck segment whose RptElt is nearest that point.  Pure deck
% geometry: no header basis, no array map, no angle convention.
V = R.Vpt(:,1); n = R.psi;
R.partner = zeros(1, R.nSeg); R.partnerErr = zeros(1, R.nSeg);
R.RptRad  = zeros(1, R.nSeg);
for k = 1:R.nSeg
    u = R.Rpt(:,k) - V;
    R.RptRad(k) = norm(u - dot(u,n)*n);
    tgt = V + 2*dot(u,n)*n - u;
    [R.partnerErr(k), R.partner(k)] = min(vecnorm(R.Rpt - tgt, 2, 1));
end
ring = R.RptRad > 1e-6*max([1, R.RptRad]);
R.ringSeg = find(ring);
R.selfConj = (R.partner == 1:R.nSeg);

R.hasPolyAp = any(strcmpi(D.apType(segs), 'Polygonal'));
if isempty(o.model_size)
    o.model_size = 512; if R.nGridpts > 128, o.model_size = 1024; end
end
R.model_size = o.model_size;

if o.verbose
    fprintf('\n%s\n%s\n', repmat('=',1,88), rx);
    fprintf('  GridType=%s  nSeg=%d  nGridpts=%g  SegX2=%+.1f deg  xGrid=[%s]\n', ...
        R.gridType, R.nSeg, R.nGridpts, R.segX2ang, num2str(xg.', '%+.3g '));
    fprintf('  180-partner map: %s   (worst nearest-match err %.3g)\n', ...
        mat2str(R.partner), max([0 R.partnerErr(ring)]));
    fprintf('  polygon apertures on segments: %s ; model_size %d\n', ...
        tf(R.hasPolyAp), o.model_size);
end

macos.init(o.model_size);
here = pwd; cleanup = onCleanup(@() cd(here));
cd(fileparts(rx));

% ---- baseline ----------------------------------------------------------
macos.load_rx(rx);
nElt = macos.num_elt();
ie = o.opd_elt;
if isempty(ie)
    ie = max(segs) + 1;
    while ie <= nElt && strcmpi(D.eltType{ie}, 'Segment'), ie = ie + 1; end
    if ie > nElt, ie = nElt; end
end
R.opd_elt = ie;
s0 = macos.trace(ie); W0 = macos.opd();
R.nRays0 = s0.nRays; R.nPupil0 = nnz(W0 ~= 0);
if o.verbose
    fprintf('  baseline: OPD at elt %d, nRays=%d, pupil px=%d\n', ie, R.nRays0, R.nPupil0);
end

% ======================================================================
% Instrument B -- draw-3d global attribution  (primary)
% ======================================================================
R.B = struct('ran', false, 'seg', [], 'nCross', [], 'dOwn', [], ...
             'dPart', [], 'ctr', [], 'verdict', '');
try
    P = []; E = [];
    for pl = {'YZ','XZ'}                     % two orthogonal meridian fans
        b = macos.draw_rays3d(pl{1}, 1, max(segs));
        nd = size(b.P, 2);
        P = [P, reshape(b.P, 3, [])]; %#ok<AGROW>
        E = [E; reshape(b.elt(1:nd,:), [], 1)]; %#ok<AGROW>
    end
    for k = 1:R.nSeg
        e = segs(k); q = R.partner(k);
        sel = (E == e);
        if nnz(sel) < 3, continue, end
        c = mean(P(:,sel), 2);
        R.B.seg(end+1)    = e;
        R.B.nCross(end+1) = nnz(sel);
        R.B.ctr(:,end+1)  = c;
        R.B.dOwn(end+1)   = norm(c - R.Rpt(:,k));
        R.B.dPart(end+1)  = norm(c - R.Rpt(:,q));
    end
    % A segment votes only if (a) it is a ring segment, (b) it is not its
    % own 180-partner, and (c) the tiling REALLY has a 180-partner for it --
    % i.e. the nearest deck RptElt to the rotated position is actually close.
    % Guard (c) matters: a Flower/petal layout has no 180 symmetry, the
    % partner map degenerates, and a vote from it would be meaningless.
    tol = 0.05 * max(R.RptRad);
    vote = R.B.dPart > 0 & (R.B.dOwn ./ max(R.B.dPart, realmin) < 1e6);
    for i = 1:numel(R.B.seg)
        k = find(segs == R.B.seg(i));
        vote(i) = vote(i) && ~R.selfConj(k) && ring(k) && R.partnerErr(k) < tol;
    end
    R.B.votes = vote;
    if any(vote)
        R.B.ran = true;
        good = R.B.dOwn(vote) < R.B.dPart(vote);
        if all(good),      R.B.verdict = 'CORRECT';
        elseif ~any(good), R.B.verdict = 'PERMUTED';
        else,              R.B.verdict = 'MIXED';
        end
    end
catch ME
    R.B.err = ME.message;
end
if o.verbose && ~isempty(R.B.seg) && isfield(R.B, 'votes')
    fprintf('  [B] draw-3d global attribution (mean crossing position vs deck RptElt)\n');
    fprintf('      %-5s %-7s %8s %12s %12s %-6s\n', ...
        'elt','partnr','crossings','|d to own|','|d to 180|','votes');
    for i = 1:numel(R.B.seg)
        k = find(segs == R.B.seg(i));
        fprintf('      %-5d %-7d %8d %12.4g %12.4g %-6s\n', R.B.seg(i), ...
            segs(R.partner(k)), R.B.nCross(i), R.B.dOwn(i), R.B.dPart(i), ...
            tf(R.B.votes(i)));
    end
    if R.B.ran
        fprintf('      -> %s\n', R.B.verdict);
    else
        fprintf(['      -> NO VOTE: no segment has a usable 180-partner (worst\n' ...
                 '         nearest-match %.4g vs a %.4g tolerance).  A layout with no\n' ...
                 '         180 symmetry -- e.g. Flower -- cannot be judged this way.\n'], ...
            max([0 R.partnerErr(R.ringSeg)]), 0.05*max(R.RptRad));
    end
end

% ======================================================================
% Instrument A -- aperture ray count vs a SegXgrid-negated copy
% ======================================================================
R.A = struct('ran', false, 'n0', NaN, 'n1', NaN, 'ratio', NaN, 'verdict', '');
if R.hasPolyAp
    flip = flip_segxgrid(rx);
    guard = onCleanup(@() delete_if(flip));
    try
        macos.load_rx(flip);
        s1 = macos.trace(ie); W1 = macos.opd();
        R.A.ran = true;
        R.A.n0 = R.nPupil0;   R.A.n1 = nnz(W1 ~= 0);
        R.A.nRays0 = R.nRays0; R.A.nRays1 = s1.nRays;
        R.A.ratio = R.A.n0 / max(R.A.n1, 1);
        if     R.A.n0 > 2*R.A.n1, R.A.verdict = 'CORRECT';
        elseif R.A.n1 > 2*R.A.n0, R.A.verdict = 'PERMUTED';
        else,                     R.A.verdict = 'INCONCLUSIVE';
        end
    catch ME
        R.A.err = ME.message;
    end
    clear guard
    if o.verbose
        fprintf(['  [A] aperture pupil count: committed %d / SegXgrid-flipped %d' ...
                 '  (ratio %.1f)  -> %s\n'], R.A.n0, R.A.n1, R.A.ratio, R.A.verdict);
    end
end

% ======================================================================
% Instrument C -- draw-2d projection basis, reported BOTH ways
% ======================================================================
R.C = struct('ran', false);
try
    macos.load_rx(rx); macos.trace(ie);
    b = macos.draw_rays('XY', 1, max(segs));
    R.C.eltSeen = []; R.C.uv = []; R.C.gridPred = []; R.C.globPred = [];
    for k = 1:R.nSeg
        e = segs(k);
        sel = (b.elt == e);
        if nnz(sel) < 3, continue, end
        R.C.eltSeen(end+1)    = e;
        R.C.uv(:,end+1)       = [mean(b.U(sel)); mean(b.V(sel))];
        R.C.gridPred(:,end+1) = [dot(xg, R.Rpt(:,k)); dot(yg, R.Rpt(:,k))];
        R.C.globPred(:,end+1) = R.Rpt(1:2,k);
    end
    if ~isempty(R.C.eltSeen)
        R.C.ran     = true;
        R.C.resGrid = max(vecnorm(R.C.uv - R.C.gridPred, 2, 1));
        R.C.resGlob = max(vecnorm(R.C.uv - R.C.globPred, 2, 1));
        R.C.scale   = max(vecnorm(R.C.gridPred, 2, 1));
    end
catch ME
    R.C.err = ME.message;
end
if o.verbose && R.C.ran
    fprintf(['  [C] draw-2d ''XY'' elts %s: max residual vs deck RptElt read in the\n' ...
             '      DRAW basis (xGrid,yGrid) = %.4g ; read as GLOBAL x,y = %.4g' ...
             '  (radius %.4g)\n'], mat2str(R.C.eltSeen), R.C.resGrid, ...
        R.C.resGlob, R.C.scale);
end

% ======================================================================
% T3 -- rotation / translation inertness on one ring segment
% ======================================================================
R.T3 = struct('ran', false);
if ~isempty(R.ringSeg)
    e = segs(R.ringSeg(1));
    try
        macos.load_rx(rx); cbm = mmacos('base_unit_to_metres');
        R.T3.trLocal  = peak_dopd(rx, ie, W0, e, 'translation', [0;0;o.piston*cbm], 'local');
        R.T3.trGlobal = peak_dopd(rx, ie, W0, e, 'translation', [0;0;o.piston*cbm], 'global');
        R.T3.rotLocal = peak_dopd(rx, ie, W0, e, 'rotation',    [0;o.tilt;0],       'local');
        R.T3.rotGlobal= peak_dopd(rx, ie, W0, e, 'rotation',    [0;o.tilt;0],       'global');
        R.T3.elt = e; R.T3.ran = true;
        R.T3.rotInert = (R.T3.rotLocal == 0) && (R.T3.rotGlobal == 0);
        R.T3.trInert  = (R.T3.trLocal  == 0) && (R.T3.trGlobal  == 0);
    catch ME
        R.T3.err = ME.message;
    end
end
if o.verbose && R.T3.ran
    fprintf(['  [T3] elt %d peak |dOPD|: trans local %.3e / global %.3e ;' ...
             ' rot local %.3e / global %.3e%s\n'], R.T3.elt, R.T3.trLocal, ...
        R.T3.trGlobal, R.T3.rotLocal, R.T3.rotGlobal, ...
        tern(R.T3.rotInert, '   <== ROTATIONS BIT-INERT', ''));
end

% ======================================================================
% Instrument D -- tilt-pivot piston (opt-in diagnostic, see header)
% ======================================================================
R.D = struct('ran', false, 'seg', [], 'part', [], 'pOwn', [], 'pPart', [], ...
             'dPart', [], 'nMask', [], 'tStar', []);
if o.pivot && ~isempty(R.ringSeg) && ~(R.T3.ran && R.T3.rotInert)
    macos.load_rx(rx); cbm = mmacos('base_unit_to_metres');
    pick = R.ringSeg(round(linspace(1, numel(R.ringSeg), ...
                     min(o.nProbe, numel(R.ringSeg)))));
    for k = unique(pick, 'stable')
        e = segs(k); q = R.partner(k);
        if q == k, continue, end
        macos.load_rx(rx);
        macos.perturb(e, 'translation', [0;0;o.piston*cbm], 'frame', 'local');
        macos.trace(ie); dW = macos.opd() - W0;
        m = two_level_mask(dW, W0 ~= 0);
        if nnz(m) == 0, continue, end
        macos.load_rx(rx);
        macos.perturb(e, 'rotation', [0;o.tilt;0], 'frame', 'local');
        macos.trace(ie); dTa = macos.opd() - W0;
        macos.load_rx(rx);
        macos.set_elt_rpt(e, R.Rpt(:,q));
        macos.perturb(e, 'rotation', [0;o.tilt;0], 'frame', 'local');
        macos.trace(ie); dTb = macos.opd() - W0;
        a = mean(dTa(m)); c = mean(dTb(m));
        R.D.seg(end+1)   = e;     R.D.part(end+1)  = segs(q);
        R.D.nMask(end+1) = nnz(m);
        R.D.pOwn(end+1)  = a;     R.D.pPart(end+1) = c;
        R.D.dPart(end+1) = norm(R.Rpt(:,q) - R.Rpt(:,k));
        % piston is linear in pivot position; t*=0 => responding region at
        % the element's own RptElt, t*=1 => at the 180-partner's
        R.D.tStar(end+1) = a / (a - c);
    end
    R.D.ran = ~isempty(R.D.seg);
end
if o.verbose && R.D.ran
    fprintf('  [D] tilt-pivot piston, tilt %.1e rad (DIAGNOSTIC -- see header)\n', o.tilt);
    fprintf('      %-5s %-7s %8s %12s %12s %8s %10s\n', ...
        'elt','partnr','maskpx','piston@own','piston@180','t*','|Rk-Rq|');
    for i = 1:numel(R.D.seg)
        fprintf('      %-5d %-7d %8d %12.4e %12.4e %8.3f %10.4g\n', ...
            R.D.seg(i), R.D.part(i), R.D.nMask(i), R.D.pOwn(i), ...
            R.D.pPart(i), R.D.tStar(i), R.D.dPart(i));
    end
end

% ---- roll the verdict up ----------------------------------------------
% A is gold where it exists; B is the everywhere instrument.  Any A/B
% disagreement is reported loudly rather than silently resolved.
if R.A.ran && ~strcmp(R.A.verdict, 'INCONCLUSIVE')
    R.verdict = R.A.verdict; R.by = 'A aperture';
    if R.B.ran && ~strcmp(R.B.verdict, R.A.verdict)
        R.verdict = sprintf('%s  ** A/B DISAGREE (B=%s) **', R.A.verdict, R.B.verdict);
        R.by = 'A (gold), B dissents';
    else
        R.by = 'A aperture + B draw3d agree';
    end
elseif R.B.ran
    R.verdict = R.B.verdict; R.by = 'B draw3d';
end
if o.verbose
    fprintf('  ==> ROUTING: %s   [%s]\n', R.verdict, R.by);
end
end

% ======================================================================
function v = peak_dopd(rx, ie, W0, e, kind, val, frame)
macos.load_rx(rx);
macos.perturb(e, kind, val, 'frame', frame);
macos.trace(ie);
W = macos.opd();
v = max(abs(W(:) - W0(:)));
end

function f = flip_segxgrid(rx)
%FLIP_SEGXGRID  Copy RX beside itself with the HEADER SegXgrid negated.
%   Written into the deck's own directory so relative GridFile= paths keep
%   resolving.  Caller deletes it.
txt = strsplit(fileread(rx), newline);
iFirstElt = find(~cellfun(@isempty, regexp(txt, '^\s*iElt\s*=', 'once')), 1);
if isempty(iFirstElt), iFirstElt = numel(txt); end
done = false;
for n = 1:iFirstElt-1
    tok = regexp(txt{n}, '^(\s*SegXgrid\s*=\s*)(.*)$', 'tokens', 'once');
    if isempty(tok), continue, end
    v = sscanf(strrep(strrep(tok{2},'D','E'),'d','e'), '%f');
    txt{n} = sprintf('%s%.10E  %.10E  %.10E', tok{1}, -v(1), -v(2), -v(3));
    done = true; break
end
if ~done, error('seg_route:noSegXgrid', 'no header SegXgrid= in %s', rx); end
[d, nm] = fileparts(rx);
f = fullfile(d, [nm '__segflip_tmp.in']);
if isempty(strtrim(txt{end})), txt(end) = []; end
fid = fopen(f, 'w');
fprintf(fid, '%s\n', txt{:});
fclose(fid);
end

function delete_if(f)
if exist(f, 'file'), delete(f); end
end

function m = two_level_mask(dW, pup)
% A piston poke of one segment lands as two levels after the engine's mean
% removal: the poked region at -2d(1-f), everything else at +2df.  Split at
% the midpoint and keep the MINORITY cluster.  Valid ONLY on a piston
% response -- a ramp response splits its own wedge.
v = dW(pup);
mid = (max(v) + min(v))/2;
lo = pup & (dW < mid); hi = pup & (dW > mid);
if nnz(lo) <= nnz(hi), m = lo; else, m = hi; end
end

function v = getdef(s, f, d)
v = d;
if ~isfield(s, f) || isempty(s.(f)), return, end
x = s.(f);
if ischar(x), v = x; return, end
if any(isnan(x(:))), return, end
v = x;
end
function s = tf(b), if b, s = 'yes'; else, s = 'no'; end, end
function s = tern(c, a, b), if c, s = a; else, s = b; end, end
