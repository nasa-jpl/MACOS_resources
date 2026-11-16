function R = seg_audit(decks, varargin)
%SEG_AUDIT  Systematic audit of segmented-mirror prescriptions.
%
%   R = SEG_AUDIT(DECKS)  runs the segmentation audit over a cell array of
%   MACOS .in prescription paths and prints one OK/FLAG table per deck plus
%   a corpus roll-up.  R is a struct array (one entry per deck) with the
%   raw measurements, so a caller can re-tabulate without re-running.
%
%   Checks (numbering follows BRIEF_seg_audit.md):
%     1  Rpt/Vpt placement    per-segment RptElt inside its own footprint;
%                             shared parent vertex (0,0,0) on a NON-central
%                             segment is a FLAG.  VptElt is EXPECTED to be
%                             the shared parent vertex (segments are pieces
%                             of one parent conic) -- reported, not flagged.
%     2  Mon-frame clocking   empirical: local tilt about yMon -> the OPD
%                             ramp direction inside the segment's own mask
%                             gives xMon as the pupil sees it.  Compared
%                             against the deck's xMon clock.  The pupil may
%                             image M1 rotated, so the CONSISTENCY of the
%                             per-segment offset is the test, not its value.
%     3  zMon sense           +zMon piston must SHORTEN the path:
%                             wedge dOPD = -2*d*(1-f), f = mask area frac.
%     4  Frame agreement      Mon vs FF vs Data frames per segment, and
%                             whether the FF/Data frame is shared by all
%                             segments (SegMirMaker replicates the PARENT
%                             frame verbatim -- shared is CORRECT there).
%     5  Localization         piston response confined to one segment;
%                             mask area fraction ~ 1/nSeg.
%     6  Rigid local tilt     RptElt really is the rotation pivot: A/B the
%                             same local tilt against one with RptElt forced
%                             to the parent vertex.  (A "ramp crosses zero at
%                             the centroid" test does NOT work on a fast
%                             curved parent -- rotating a Segment re-points
%                             the parent conic, so neither pivot gives a
%                             pure ramp.)
%     7  Housekeeping         nGridpts parity; GLOBAL-frame rigid pokes
%                             inert on Element=Segment (known open engine
%                             question -- recorded, not chased).
%
%   THIS FUNCTION DOES NOT DECIDE RAY->SEGMENT ROUTING.  Its check-2
%   position/clock offsets live in OPD-ARRAY index space, whose (i,j) ->
%   (X,Y) map is deck-dependent, so only their SPREADS are meaningful
%   without a per-deck pin.  For the routing verdict use SEG_ROUTE
%   (instrument A = per-segment aperture ray count; instrument B =
%   draw-3d global attribution).  Reading an unpinned offset -- or any
%   clock measured in the header SegXgrid basis, which rotates with the
%   patch -- as a verdict is what inverted the round-1 audit tables.
%   See SEG_AUDIT_STATUS.md, "Instrument doctrine".
%
%   Name/value options:
%     'model_size'  512      macos.init size
%     'opd_elt'     []       element to take OPD at (default: auto = the
%                            element before the FocalPlane)
%     'piston'      2.5e-4   piston probe, deck BaseUnits
%     'tilt'        2e-7     tilt probe, radians
%     'static_only' false    skip the engine probes (checks 2,3,5,6)
%     'verbose'     true     per-deck detail (the roll-up prints regardless)
%
%   The [R1] line predicts what the proposed PSEG fix (make it use the
%   xt/yt it already computes, as HSEG does) would do to each deck: BREAKS
%   / fixes / none.  Only GridType=Pie decks with a non-identity SegX2 can
%   move -- i.e. only decks whose SegXgrid is not parallel to xGrid.
%
%   Example:
%     addpath ~/dev/MACOS_res_dev/mmacos/src
%     R = seg_audit({'~/dev/macos/ZGD_test_files/e5pie.in', ...
%                    '~/dev/macos/ZGD_test_files/e2e_pie.in'});
%
%   Corpus-wide R1 warning list (static, no engine, fast):
%     f = split(string(evalc(['!find ~/dev -name ''*.in'' -not -path ''*/build*'' ' ...
%           '| xargs grep -lE "GridType= *(Pie|pie)"'])));
%     R = seg_audit(cellstr(f(strlength(f)>0)), 'static_only', true, 'verbose', false);
%     % then read the roll-up's R1 IMPACT column
%
%   See BRIEF_seg_audit.md and SEG_AUDIT_STATUS.md for the findings this
%   script was written to pin.

p = inputParser;
p.addParameter('model_size', 512);
p.addParameter('opd_elt', []);
p.addParameter('piston', 2.5e-4);
p.addParameter('tilt', 2e-7);
p.addParameter('static_only', false);
p.addParameter('verbose', true);
p.parse(varargin{:});
o = p.Results;

if ischar(decks) || isstring(decks), decks = {char(decks)}; end
R = struct([]);

if ~o.static_only
    macos.init(o.model_size);
end

for id = 1:numel(decks)
    rx = char(decks{id});
    if o.verbose
        fprintf('\n%s\n%s\n%s\n', repmat('=',1,100), rx, repmat('=',1,100));
    end
    r = audit_one(rx, o);
    % decks that bail early carry fewer fields -- union them so the struct
    % array can grow (a corpus sweep will always hit a few odd decks)
    if isempty(R)
        R = r;
    else
        for f = setdiff(fieldnames(R), fieldnames(r)).', r.(f{1}) = []; end
        for f = setdiff(fieldnames(r), fieldnames(R)).', [R.(f{1})] = deal([]); end
        R(end+1) = orderfields(r, R); %#ok<AGROW>
    end
end

% roll-up always prints for a multi-deck run; 'verbose' controls only the
% per-deck detail, so a corpus sweep can be quiet but still tabulated.
if numel(R) > 1, rollup(R); end
end

% ======================================================================
function r = audit_one(rx, o)
r = struct('rx', rx, 'ok', false, 'err', '', 'flags', {{}});

D = seg_read_rx(rx);   % shared reader; also used by seg_route
r.hdr = D.hdr;
segs = find(strcmpi(D.eltType, 'Segment'));
r.iSeg = segs(:).';
r.nSeg = numel(segs);
if isempty(segs)
    r.err = 'no Element=Segment elements';
    fprintf('  %s\n', r.err); return
end

% ---- two in-plane bases, kept STRICTLY apart -------------------------
% (a) DECK basis -- built from the parent axis and global x-hat alone.  It
%     does not move when the header moves, so every geometric quantity
%     below is reported against a fixed reference.  USE THIS BY DEFAULT.
% (b) HEADER basis -- the SegMirMaker in-plane basis implied by the header
%     SegXgrid.  It ROTATES WITH THE PATCH, so a clock measured in it is
%     blind to exactly the thing a routing audit is looking for.  It is
%     legitimate for ONE question only: "does the header name the basis the
%     elements were built in" (the ring1Clk / term1 test).  Anything else
%     measured in it is a trap -- this is what invalidated the round-1
%     absolute verdicts.  See SEG_AUDIT_STATUS.md "Instrument doctrine".
psi = D.vecs.psiElt(:, segs(1)); psi = psi/norm(psi);
seed = [1;0;0]; if abs(dot(seed,psi)) > 0.9, seed = [0;1;0]; end
xd = seed - dot(seed,psi)*psi; xd = xd/norm(xd);
yd = cross(psi, xd);
r.xd = xd; r.yd = yd; r.psi = psi;
clk = @(v) atan2d(dot(v,yd), dot(v,xd));           % DECK basis

sxg = getfield_default(D.hdr, 'SegXgrid', [1;0;0]);
ys  = cross(psi, sxg); ys = ys/norm(ys);
xs  = cross(ys, psi);  xs = xs/norm(xs);
r.xs = xs; r.ys = ys;
clkHdr = @(v) atan2d(dot(v,ys), dot(v,xs));        % HEADER basis (term1 only)

% ---------------- static checks (1, 4, 7) -----------------------------
nSeg = r.nSeg;
r.Rpt = D.vecs.RptElt(:, segs);
r.Vpt = D.vecs.VptElt(:, segs);
r.pMon = D.vecs.pMon(:, segs);
r.xMon = D.vecs.xMon(:, segs);
r.yMon = D.vecs.yMon(:, segs);
r.zMon = D.vecs.zMon(:, segs);

r.RptClk    = arrayfun(@(k) clk(r.Rpt(:,k)), 1:nSeg);      % deck basis
r.MonClk    = arrayfun(@(k) clk(r.xMon(:,k)), 1:nSeg);     % deck basis
r.RptClkHdr = arrayfun(@(k) clkHdr(r.Rpt(:,k)), 1:nSeg);   % header basis
r.RptRad = arrayfun(@(k) hypot(dot(r.Rpt(:,k),xd), dot(r.Rpt(:,k),yd)), 1:nSeg);
r.dClk   = wrap180(r.MonClk - r.RptClk);
r.pMonEqRpt = max(vecnorm(r.pMon - r.Rpt, 2, 1));
r.zMonDotPsi = arrayfun(@(k) dot(r.zMon(:,k), D.vecs.psiElt(:,segs(k))), 1:nSeg);

% Mon triad health
r.ortho = zeros(1,nSeg); r.hand = zeros(1,nSeg);
for k = 1:nSeg
    X = r.xMon(:,k); Y = r.yMon(:,k); Z = r.zMon(:,k);
    r.ortho(k) = max(abs([dot(X,Y) dot(X,Z) dot(Y,Z) norm(X)-1 norm(Y)-1 norm(Z)-1]));
    r.hand(k)  = dot(cross(X,Y), Z);
end

% check 1: RptElt inside its own footprint.  The pitch is `width`
% (SegMirMaker: dx=width/2, dy=3*length/4 -> |ring-1 offset| = width;
% engine sourcsub HSEG shrinks each hexagon inward by gap/2, so the gap
% is NOT part of the pitch).  Expected radii: ring m corners m*width,
% edge-centres sqrt(3)*width for ring 2.
w = getfield_default(D.hdr, 'width', NaN);
r.width = w;
r.gap   = getfield_default(D.hdr, 'gap', NaN);
r.pitchRatio = r.RptRad / w;                     % 0, 1, 2, sqrt(3) ...
r.sharedVertexRpt = (r.RptRad < 1e-6*max(1,w));  % at the parent vertex
% central segment legitimately sits at the vertex; any OTHER one is a FLAG
r.flagRpt = find(r.sharedVertexRpt);
r.flagRpt(r.flagRpt == 1) = [];   % seg 1 is the ring-0 centre
r.VptAllShared = max(vecnorm(r.Vpt - r.Vpt(:,1), 2, 1)) < 1e-9*max(1,w);
r.VptIsOrigin  = norm(r.Vpt(:,1)) < 1e-9*max(1,w);

% check 4: frame agreement
r.hasFF = isfield(D.vecs,'xFF') && any(any(D.vecs.xFF(:,segs) ~= 0));
r.hasDat = isfield(D.vecs,'xData') && any(any(D.vecs.xData(:,segs) ~= 0));
r.zFFdotzMon = nan(1,nSeg); r.FFshared = NaN; r.DatEqFF = NaN;
if r.hasFF
    xF = D.vecs.xFF(:,segs); zF = D.vecs.zFF(:,segs); pF = D.vecs.pFF(:,segs);
    r.zFFdotzMon = arrayfun(@(k) dot(zF(:,k), r.zMon(:,k)), 1:nSeg);
    r.FFshared = max([vecnorm(xF-xF(:,1),2,1) vecnorm(zF-zF(:,1),2,1) ...
                      vecnorm(pF-pF(:,1),2,1)]) < 1e-9*max(1,w);
    if r.hasDat
        r.DatEqFF = max([vecnorm(D.vecs.xData(:,segs)-xF,2,1) ...
                         vecnorm(D.vecs.zData(:,segs)-zF,2,1) ...
                         vecnorm(D.vecs.pData(:,segs)-pF,2,1)]) < 1e-9*max(1,w);
    end
end

% check 1b: does the header SegXgrid name the basis the ELEMENTS were
% built in?  SegMirMaker always starts ring 1 at pin = dx*xs + dy*ys with
% dx = width/2, dy = 3*length/4 -- i.e. element 2 sits at clock +60 deg in
% the basis actually used.  For a back-facing parent (zs(3)<0) SMM rotates
% that basis 180 deg (commit 11481cd) and, since a later fix, emits the
% ROTATED basis as SegXgrid.  A deck generated between those two commits
% has a flipped tiling with an unflipped header -> element 2 reads -120.
% That matters because the engine assigns rays to segment ELEMENTS BY
% ORDINAL (tracesub.F:3455 RayToSegMap == EltToSegMap) using a ray->segment
% map built from the HEADER's SegXgrid + SegCoord.
% SMM's walk always starts ring 1 at pin = dx*xs + dy*ys, i.e. element 2
% at +60 deg in the basis actually used -- or at -120 deg if the header
% was written in the un-rotated basis.  Anything else is NOT an SMM walk
% (the pre-SMM SegDemo/FFSegDemo family starts at -60), and the
% header-vs-element comparison below does not apply to it.
r.ring1Clk = NaN; r.segXgridConsistent = NaN; r.smmWalk = false;
if nSeg > 1 && r.RptRad(2) > 1e-6*max(1,w)
    r.ring1Clk = r.RptClkHdr(2);   % HEADER basis: this IS a header question
    if abs(wrap180(r.ring1Clk - 60)) < 5
        r.segXgridConsistent = true;  r.smmWalk = true;
    elseif abs(wrap180(r.ring1Clk + 120)) < 5
        r.segXgridConsistent = false; r.smmWalk = true;
    end
end

% frame convention: Pie tilings should clock xMon along each segment's own
% radial bisector (dClk ~ 0, commit e701f87); Hex keeps the heritage
% convention (dClk ~ -30, xMon along a hex flat normal).
ring = r.RptRad > 1e-6*max(1,w);
r.dClkMedian = median(r.dClk(ring));
r.gridType = getfield_default(D.hdr, 'GridType', '');
r.isPie = ~isempty(r.gridType) && any(upper(r.gridType(1)) == 'P');
r.framesArePieBisector = abs(wrap180(r.dClkMedian)) < 5;
% ---- predicted ray->element routing (CORRECTED, round 2) --------------
% SegMirMaker PLACES the elements using the in-plane basis implied by the
% header SegXgrid.  The engine BUILDS the ray->segment map by rotating the
% grid coordinates by SegX2 = (xGrid.SegXgrid, yGrid.SegXgrid)
% (sourcsub.F:201) -- the SAME quantity.  So the two agree BY CONSTRUCTION,
% and routing is correct, provided
%   (a) the predicate actually applies that rotation, and
%   (b) the header names the basis the elements were really built in
%       (term1: ring-1 at +60 deg, not -120 deg).
% Pre-R1, PSEG computed the rotation into xt/yt and never used it, so (a)
% failed for GridType=Pie whenever SegX2 was not the identity.  R1 fixed
% that, leaving term1 as the ONLY live term:
%   current engine (post-R1):  permuted <=> term1
%   pre-R1 (historical):       permuted <=> term1 XOR (Pie AND SegX2 ~= 0)
%
% Round 1 had term2's SENSE inverted -- it counted the engine APPLYING the
% rotation as the thing that permutes.  Together with the draw-basis frame
% slip that produced a self-consistent but wrong table.  This prediction is
% now confirmed against SEG_ROUTE's physical instruments on 35 classified
% decks (26 term1=no -> CORRECT, 2 term1=yes -> PERMUTED, 7 non-SMM-walk
% decks measured directly).  See SEG_AUDIT_STATUS.md.
xg = getfield_default(D.hdr, 'xGrid', [1;0;0]);
yg = getfield_default(D.hdr, 'yGrid', [0;1;0]);
su = sxg(:)/norm(sxg);
r.segX2    = [dot(xg(:)/norm(xg), su); dot(yg(:)/norm(yg), su)];
r.segX2ang = atan2d(r.segX2(2), r.segX2(1));
r.xGridDotSegXgrid = r.segX2(1);

% did the PRE-R1 engine drop the rotation for this deck?
r.pseg_dropped = r.isPie && abs(wrap180(r.segX2ang)) > 1e-6;
if r.smmWalk
    r.term1      = ~r.segXgridConsistent;
    r.permNow    = wrap180(180*r.term1);                          % post-R1
    r.permPreR1  = wrap180(180*r.term1 + 180*r.pseg_dropped);     % historical
    r.routingPredOK      = abs(r.permNow)   < 5;
    r.routingPredOK_preR1= abs(r.permPreR1) < 5;
else
    % not a SegMirMaker walk (pre-SMM SegDemo family, legacy old_Rx): the
    % header-vs-element test does not apply.  MEASURE it with SEG_ROUTE.
    r.term1 = NaN; r.permNow = NaN; r.permPreR1 = NaN;
    r.routingPredOK = NaN; r.routingPredOK_preR1 = NaN;
end

% ---- what R1 did to this deck (HISTORICAL -- R1 landed in macos 1ce1f5c)
r.r1Rotation      = wrap180(180*r.pseg_dropped);
r.r1MovesGeometry = abs(wrap180(r.segX2ang)) > 1e-6 && r.isPie && ...
                    abs(r.segX2ang - 60*round(r.segX2ang/60)) > 1e-6;
if ~r.pseg_dropped
    if ~isnan(r.routingPredOK) && ~r.routingPredOK
        r.r1Impact = 'no change (PERMUTED before and after)';
    else
        r.r1Impact = 'no change';
    end
elseif isnan(r.routingPredOK)
    r.r1Impact = 'rotated the map (baseline unknown -- measure)';
elseif r.routingPredOK && ~r.routingPredOK_preR1
    r.r1Impact = 'FIXED by R1';
elseif ~r.routingPredOK && r.routingPredOK_preR1
    r.r1Impact = 'BROKEN by R1 (needs a header re-pin)';
else
    r.r1Impact = 'no change (PERMUTED before and after)';
end

% check 7: grid parity
r.nGridpts = getfield_default(D.hdr, 'nGridpts', NaN);
r.evenGrid = mod(r.nGridpts, 2) == 0;

if o.verbose, print_static(r); end
if o.static_only, r.ok = true; return, end

% ---------------- engine probes (2, 3, 5, 6, 7-global) ----------------
try
    r = probe(r, rx, D, o);
    r.ok = true;
catch ME
    r.err = ME.message;
    fprintf('  PROBE FAILED: %s\n', ME.message);
end
if o.verbose && r.ok, print_probe(r); end
end

% ======================================================================
function r = probe(r, rx, D, o)
% GridFile= paths in the deck are relative -- load from the deck's dir
here = pwd; cd(fileparts(rx)); cleanup = onCleanup(@() cd(here));
macos.load_rx(rx);
nElt = macos.num_elt();
ie = o.opd_elt;
if isempty(ie)
    % Default: the FIRST NON-SEGMENT element.  Measuring here makes the
    % placement check ABSOLUTE -- the beam has not reached a focus, so the
    % footprint is an erect, demagnified image of the segmented pupil and a
    % segment's mask centroid sits at that segment's own RptElt direction.
    % At a downstream exit pupil the same masks appear rotated/inverted by
    % whatever imaging sits in between, which only supports the weaker
    % "is the per-segment offset CONSISTENT" test.  (The engine refuses
    % OPD at an Element=Segment, so nSeg+1 is the earliest legal plane.)
    ie = max(r.iSeg) + 1;
    while ie <= nElt && strcmpi(D.eltType{ie}, 'Segment'), ie = ie + 1; end
    if ie > nElt, ie = nElt; end
end
r.opd_elt = ie;

macos.trace(ie); W0 = macos.opd();
r.pupilMask = W0 ~= 0;
r.nPupil = nnz(r.pupilMask);
segs = r.iSeg; nSeg = r.nSeg;

% deck units -> the perturb veneer takes SI metres for translations
cbm = mmacos('base_unit_to_metres');
dp  = o.piston;                               % deck units
dth = o.tilt;                                 % radians

r.maskFrac = zeros(1,nSeg); r.wedgeMean = zeros(1,nSeg);
r.predMean = zeros(1,nSeg); r.maskClk = zeros(1,nSeg);
r.maskCtr  = zeros(2,nSeg); r.rampClk = zeros(1,nSeg);
r.rampZeroOff = zeros(1,nSeg); r.globalInert = false(1,nSeg);
r.globalPk = zeros(1,nSeg); r.localPk = zeros(1,nSeg);
r.maskR = zeros(1,nSeg); r.tiltMaskMean = zeros(1,nSeg);
r.pivotDiff = nan(1,nSeg); r.globalMean = zeros(1,nSeg);
masks = cell(1,nSeg);

pupCtr = mask_centroid(r.pupilMask);

for k = 1:nSeg
    e = segs(k);

    % ---- piston probe along +zMon (LOCAL frame) ----
    macos.load_rx(rx);
    macos.perturb(e, 'translation', [0;0;dp*cbm], 'frame', 'local');
    macos.trace(ie); dW = macos.opd() - W0;
    m = wedge_mask(dW, r.pupilMask);
    masks{k} = m;
    r.maskFrac(k)  = nnz(m) / r.nPupil;
    r.wedgeMean(k) = mean(dW(m));
    r.predMean(k)  = -2*dp*(1 - r.maskFrac(k));
    c = mask_centroid(m);
    r.maskCtr(:,k) = c;
    r.maskClk(k)   = atan2d(c(2)-pupCtr(2), c(1)-pupCtr(1));   % i=X, j=Y
    r.maskR(k)     = norm(c - pupCtr);                         % px
    r.localPk(k)   = max(abs(dW(:)));

    % ---- global-frame piston, same magnitude (check 7) ----
    macos.load_rx(rx);
    macos.perturb(e, 'translation', [0;0;dp*cbm], 'frame', 'global');
    macos.trace(ie); dWg = macos.opd() - W0;
    r.globalPk(k)    = max(abs(dWg(:)));
    r.globalMean(k)  = mean(dWg(m));
    r.globalInert(k) = r.globalPk(k) == 0;

    % ---- tilt about yMon (LOCAL) -> ramp inside the segment mask ----
    macos.load_rx(rx);
    macos.perturb(e, 'rotation', [0;dth;0], 'frame', 'local');
    macos.trace(ie); dT = macos.opd() - W0;
    [g, z0] = ramp_fit(dT, m);
    r.rampClk(k)     = atan2d(g(2), g(1));
    r.rampZeroOff(k) = z0;                     % |zero line - mask centroid|, px
    r.tiltMaskMean(k) = mean(dT(m));

    % ---- check 6: is RptElt actually the rotation pivot? ----
    % A/B against the SAME rotation with RptElt forced to the parent
    % vertex.  Calibration-free: a nonzero difference proves the engine
    % pivots where the deck says, and its size shows how much the shipped
    % RptElt buys over the shared-vertex alternative.  (A naive
    % "ramp crosses zero at the centroid" test does NOT discriminate on a
    % fast curved parent: rotating a Segment re-points the parent conic,
    % so neither pivot gives a pure ramp.)
    if r.RptRad(k) > 1e-9
        macos.load_rx(rx);
        macos.set_elt_rpt(e, [0;0;0]);
        macos.perturb(e, 'rotation', [0;dth;0], 'frame', 'local');
        macos.trace(ie); dTv = macos.opd() - W0;
        r.pivotDiff(k) = max(abs(dT(:) - dTv(:))) / max(abs(dT(:)));
    else
        r.pivotDiff(k) = NaN;    % ring-0 segment already sits at the vertex
    end
end
r.masks = masks;

% overlap between masks = localization failure
r.maskOverlap = 0;
for a = 1:nSeg
    for b = a+1:nSeg
        r.maskOverlap = max(r.maskOverlap, nnz(masks{a} & masks{b}));
    end
end
r.maskUnionFrac = nnz(cat(3, masks{:}) ~= 0) / r.nPupil;
u = false(size(r.pupilMask));
for k = 1:nSeg, u = u | masks{k}; end
r.maskUnionFrac = nnz(u) / r.nPupil;

% ---- CONSISTENCY of the pupil<->deck rotation (check 2) ----------------
% NOT A ROUTING VERDICT.  maskClk lives in OPD-ARRAY index space and the
% array's (i,j) -> (X,Y) map is deck- and lattice-dependent (the notes
% 63-pt deck reads (i=-X, j=+Y); test_in at 128 reads (+X,-Y)).  Without a
% per-deck map pin an absolute offset here is meaningless, and reading one
% as a verdict is what inverted the round-1 tables.  What IS meaningful
% without a pin is the SPREAD: a single rigid offset with small spread
% means every segment responds in a consistent place and every Mon frame
% is clocked consistently.  The absolute value is reported as UNPINNED.
% For the routing verdict, call SEG_ROUTE (instruments A/B).
ring = r.RptRad > 1e-6*max(1,r.width);
dA = wrap180(r.maskClk - r.RptClk);
dB = wrap180(-r.maskClk - r.RptClk);
if spread180(dA(ring)) <= spread180(dB(ring))
    r.arrayParity = +1; d1 = dA;
else
    r.arrayParity = -1; d1 = dB;
end
d2 = wrap180(r.arrayParity*r.rampClk - r.MonClk);   % empirical vs deck Mon-x
r.posOffset  = d1;  r.posOffsetSpread  = spread180(d1(ring));
r.monOffset  = d2;  r.monOffsetSpread  = spread180(d2(ring));
r.posOffsetMean = atan2d(mean(sind(d1(ring))), mean(cosd(d1(ring))));
r.monOffsetMean = atan2d(mean(sind(d2(ring))), mean(cosd(d2(ring))));
end

% ======================================================================
function m = wedge_mask(dW, pup)
% A wedge piston lands as two levels after mean removal: the wedge at
% -2d(1-f) and everything else at +2df.  Split at the midpoint and keep
% the MINORITY cluster (the wedge).
v = dW(pup);
mid = (max(v) + min(v))/2;
lo = pup & (dW < mid);
hi = pup & (dW > mid);
if nnz(lo) <= nnz(hi), m = lo; else, m = hi; end
end

function c = mask_centroid(m)
[i, j] = find(m);
c = [mean(i); mean(j)];
end

function [g, z0] = ramp_fit(dT, m)
% least-squares plane over the mask: dT ~ a*i + b*j + c.  g = (a,b) is the
% ramp gradient in (X,Y) index space; z0 is the distance (px) from the mask
% centroid to the ramp's zero line -- 0 means the pivot is at the centroid.
[i, j] = find(m);
v = dT(m);
A = [i(:) j(:) ones(numel(i),1)];
s = A \ v(:);
g = s(1:2);
c = [mean(i) mean(j)];
if norm(g) > 0
    z0 = abs(s(1)*c(1) + s(2)*c(2) + s(3)) / norm(g);
else
    z0 = NaN;
end
end

function d = wrap180(d), d = mod(d + 180, 360) - 180; end

function s = spread180(d)
% circular peak-to-peak spread of a set of angles, in degrees
if isempty(d), s = NaN; return, end
c = atan2d(mean(sind(d)), mean(cosd(d)));
s = max(abs(wrap180(d - c)));
end

function v = getfield_default(s, f, d)
if isfield(s, f) && ~isempty(s.(f)), v = s.(f); else, v = d; end
end

% ======================================================================
function print_static(r)
fprintf('  nSeg=%d  nGridpts=%g (%s)  width=%.6g  gap=%.6g  SegXgrid clock basis xs=[%s]\n', ...
    r.nSeg, r.nGridpts, tern(r.evenGrid,'EVEN - half-pixel class','odd'), ...
    r.width, r.gap, num2str(r.xs(:).', '%.4g '));
fprintf('  VptElt: all segments share one vertex = %s (at origin = %s)  [expected for a common parent conic]\n', ...
    tf(r.VptAllShared), tf(r.VptIsOrigin));
fprintf(['  clocks below are in the DECK basis (parent axis + global x-hat), NOT the\n' ...
         '  header SegXgrid basis; RptClkH is the header-basis value, used ONLY for the\n' ...
         '  [1b] "does the header name the element basis" test.\n']);
fprintf('  %-4s %-8s %10s %8s %8s %8s %8s %8s %7s %7s %8s\n', ...
    'iE','name','|Rpt|','r/width','RptClk','RptClkH','MonClk','dClk','zM.psi','hand','zFF.zM');
for k = 1:r.nSeg
    fprintf('  %-4d %-8s %10.4f %8.4f %8.2f %8.2f %8.2f %8.3f %7.4f %7.2f %8.3f\n', ...
        r.iSeg(k), '', r.RptRad(k), r.pitchRatio(k), r.RptClk(k), r.RptClkHdr(k), ...
        r.MonClk(k), r.dClk(k), r.zMonDotPsi(k), r.hand(k), r.zFFdotzMon(k));
end
fprintf('  [1] non-central segments at the parent vertex: %s\n', ...
    tern(isempty(r.flagRpt), 'none  -> OK', sprintf('%d  -> FLAG', numel(r.flagRpt))));
fprintf('  [4] pMon==RptElt to %.2g ; FF frame shared by all segments = %s ; Data frame == FF = %s\n', ...
    r.pMonEqRpt, tf(r.FFshared), tf(r.DatEqFF));
fprintf('  [3s] zMon.psiElt min = %.4f  (+1 = zMon toward the source, the SegMirMaker convention)\n', ...
    min(r.zMonDotPsi));
fprintf('  Mon-triad orthonormality worst = %.2g ; handedness min = %.3f\n', ...
    max(r.ortho), min(r.hand));
% Predicted ray->element routing under the CURRENT engine.  sourcsub.F
% builds SegX2 = (xGrid.SegXgrid, yGrid.SegXgrid) and hands it to the
% grid-type predicate; HSEG (Hex) applies that rotation, PSEG (Pie)
% computes it into xt/yt and then never uses them -- see
% SEG_AUDIT_STATUS.md.  On this deck family (xGrid = -xhat, SegXgrid =
% +xhat) SegX2 is a 180 deg rotation, so the two grid types disagree:
% Post-R1 BOTH predicates apply it, so the rotation is no longer a source
% of divergence: routing is correct iff the header names the basis the
% elements were built in (ring-1 at +60 deg).  The verdict itself comes
% from SEG_ROUTE's physical instruments -- this is the static prediction.
if abs(r.segX2ang) < 1e-6
    note = 'identity -- Pie and Hex agree here';
elseif r.isPie
    note = 'Pie applies it post-R1 (PSEG DROPPED it before 1ce1f5c)';
else
    note = 'GridType Hex applies it (HSEG, always did)';
end
fprintf('  [1b] SegX2 (SegXgrid in the ray-grid basis, sourcsub.F:201) = %+.2f deg -- %s\n', ...
    r.segX2ang, note);
if ~r.smmWalk
    fprintf(['       ring-1 segment sits at %+.2f deg, not the SegMirMaker walk (+60 or -120),\n' ...
             '       so the header-vs-element test does not apply.  MEASURE this deck with\n' ...
             '       SEG_ROUTE -- do not predict it.\n'], r.ring1Clk);
else
    fprintf('       ring-1 segment at %+.2f deg => header %s the element basis\n', ...
        r.ring1Clk, tern(r.segXgridConsistent, 'AGREES with', 'is 180 deg FROM'));
    if r.routingPredOK
        fprintf('       -> ray->element routing predicted CORRECT\n');
    else
        fprintf(['       -> ray->element routing predicted **%+.0f deg PERMUTED**.\n' ...
                 '          Nominal traces are unaffected (all segments share one parent surface);\n' ...
                 '          per-segment pokes / sensitivities / MET land on the wrong segment.\n'], ...
            r.permNow);
    end
end
print_r1(r);
end

function print_r1(r)
%PRINT_R1  What R1 (make PSEG honour SegXgrid, as HSEG does) did to this
%   deck.  HISTORICAL: R1 landed in macos 1ce1f5c.  The [1b] line above is
%   the CURRENT-engine prediction; this is the before/after.
switch r.r1Impact
case 'no change'
    if abs(r.segX2ang) < 1e-6
        fprintf('  [R1] SegX2 is the identity -> R1 could not change this deck.\n');
    elseif r.isPie
        fprintf('  [R1] no change.\n');
    else
        fprintf('  [R1] GridType=%s: R1 touched PSEG only -> this deck could not move.\n', r.gridType);
    end
case 'no change (PERMUTED before and after)'
    fprintf(['  [R1] R1 could not reach this deck, and it is %+.0f deg PERMUTED.\n' ...
             '       The header names a basis 180 deg from the one its elements were built\n' ...
             '       in (stale SegMirMaker generation).  A header flip CANNOT fix it --\n' ...
             '       regenerate, or rotate the element block order by half a ring.\n'], r.permNow);
case 'rotated the map (baseline unknown -- measure)'
    fprintf(['  [R1] **WARNING** R1 rotated this deck''s ray->segment map by %+.0f deg.  Its walk\n' ...
             '       is not the SegMirMaker one, so the verdict must be MEASURED (seg_route).\n'], ...
        r.r1Rotation);
case 'FIXED by R1'
    fprintf('  [R1] R1 FIXED this deck: %+.0f deg permuted before -> correct now.\n', r.permPreR1);
case 'BROKEN by R1 (needs a header re-pin)'
    fprintf(['  [R1] **R1 BROKE THIS DECK**: correct before -> %+.0f deg PERMUTED now.\n' ...
             '       It needs its header SegXgrid negated (this is what macos ecfda5c did\n' ...
             '       for test_in/e5pie + view_rx_demo/e5pie).\n'], r.permNow);
end
if abs(r.r1Rotation) > 1e-6
    if r.r1MovesGeometry
        fprintf(['       R1 rotation %+.1f deg is NOT a multiple of 60, so it MOVED the ray grid:\n' ...
                 '       ray counts, gaps, RMS OPD, spot and PSF change too.\n'], r.r1Rotation);
    else
        fprintf(['       R1 rotation %+.1f deg is a lattice symmetry -> segment INDICES permute,\n' ...
                 '       every nominal number (rays, RMS OPD, spot, PSF) stays bit-identical.\n'], ...
            r.r1Rotation);
    end
end
if r.isPie && ~r.framesArePieBisector
    fprintf(['  [2s] GridType=%s but the Mon frames are HEX-HERITAGE (dClk median %+.2f deg, not 0):\n' ...
             '       this deck predates the pie-bisector frame convention (e701f87).\n'], ...
        r.gridType, r.dClkMedian);
else
    fprintf('  [2s] frame convention: %s (dClk median %+.2f deg), GridType=%s\n', ...
        tern(r.framesArePieBisector,'pie-bisector','hex-heritage'), r.dClkMedian, r.gridType);
end
end

function print_probe(r)
fprintf('  --- engine probes (OPD at element %d, %d pupil rays) ---\n', r.opd_elt, r.nPupil);
fprintf('  %-4s %8s %8s %12s %12s %9s %9s %9s %9s %7s %7s %7s %10s\n', ...
    'iE','maskfrac','1/nSeg','wedge dOPD','-2d(1-f)','maskClk','posOff','rampClk','monOff', ...
    'rampZ','pivDiff','globMean','globalPk');
for k = 1:r.nSeg
    fprintf('  %-4d %8.4f %8.4f %12.4e %12.4e %9.2f %9.2f %9.2f %9.2f %7.2f %7.3f %10.3e %10.3e\n', ...
        r.iSeg(k), r.maskFrac(k), 1/r.nSeg, r.wedgeMean(k), r.predMean(k), ...
        r.maskClk(k), r.posOffset(k), r.rampClk(k), r.monOffset(k), r.rampZeroOff(k), ...
        r.pivotDiff(k), r.globalMean(k), r.globalPk(k));
end
ring = r.RptRad > 1e-9;
fprintf(['  [6] RptElt honoured as the rotation pivot: relative change when RptElt is\n' ...
        '      forced to the parent vertex = %s  (0 would mean RptElt is IGNORED)\n'], ...
    num2str(r.pivotDiff(ring), ' %.3f'));
fprintf('  [5] mask overlap (max pairwise px) = %d ; union covers %.3f of pupil\n', ...
    r.maskOverlap, r.maskUnionFrac);
fprintf('  [3] wedge dOPD vs -2d(1-f): worst rel err = %.2e\n', ...
    max(abs((r.wedgeMean - r.predMean) ./ r.predMean)));
fprintf(['  [1e/2] OPD-array parity %+d ; segment POSITION offset %+.2f deg (spread %.3f) ;\n' ...
         '        Mon-frame CLOCK offset %+.2f deg (spread %.3f)\n' ...
         '        -- the SPREADS are the finding here: small = every segment responds in a\n' ...
         '           consistent place and every Mon frame is clocked consistently.  The two\n' ...
         '           MEAN offsets are UNPINNED (OPD-array (i,j)->(X,Y) is deck-dependent);\n' ...
         '           they are NOT a routing verdict -- run SEG_ROUTE for that.\n'], ...
    r.arrayParity, r.posOffsetMean, r.posOffsetSpread, r.monOffsetMean, r.monOffsetSpread);
fprintf('  [7] global-frame piston inert on: %s\n', mat2str(r.iSeg(r.globalInert)));
end

function rollup(R)
fprintf('\n%s\n CORPUS ROLL-UP\n%s\n', repmat('=',1,100), repmat('=',1,100));
fprintf('%-44s %4s %5s %5s %7s %7s %8s %8s %7s  %s\n', ...
    'deck','nSeg','ngpt','Grid','SegX2','dClk(1)','posSprd','monSprd','routing','R1 IMPACT');
for k = 1:numel(R)
    r = R(k);
    [pth, nm, ex] = fileparts(r.rx);
    [~, par] = fileparts(pth);
    nm = [par '/' nm];
    if isempty(r.iSeg) || ~isfield(r,'segX2ang') || isempty(r.segX2ang), continue, end
    ring = r.RptRad > 1e-9;
    if isfield(r,'maskFrac') && ~isempty(r.maskFrac)
        signerr = max(abs((r.wedgeMean - r.predMean)./r.predMean));
        mf = max(r.maskFrac); ps = r.posOffsetSpread; ms = r.monOffsetSpread;
    else
        signerr = NaN; mf = NaN; ps = NaN; ms = NaN;
    end
    dclk = median(r.dClk(ring));
    if isnan(r.routingPredOK),   rt = 'n/a';
    elseif r.routingPredOK,      rt = 'ok';
    else,                        rt = sprintf('%+.0f!', r.permNow);
    end
    fprintf('%-44s %4d %5g %5s %7.1f %7.2f %8.3f %8.3f %7s  %s%s\n', ...
        [nm ex], r.nSeg, r.nGridpts, r.gridType(1:min(4,end)), r.segX2ang, ...
        dclk, ps, ms, rt, r.r1Impact, ...
        tern(r.r1MovesGeometry, ' (MOVES GEOMETRY)', ''));
end
fprintf(['\nrouting = STATIC PREDICTION for the CURRENT engine (post-R1): correct iff the header\n' ...
         'SegXgrid names the basis the elements were built in (ring-1 at +60 deg, not -120).\n' ...
         '  ok | +/-N! = predicted permutation | n/a = not a SegMirMaker walk -> MEASURE it.\n' ...
         'R1 IMPACT is HISTORICAL (R1 landed in macos 1ce1f5c): what the PSEG fix did to this\n' ...
         'deck.  Only GridType=Pie decks with a non-identity SegX2 could move.\n' ...
         '\nThese are PREDICTIONS.  For the measured verdict run SEG_ROUTE (instrument A =\n' ...
         'per-segment aperture ray count, instrument B = draw-3d global attribution).\n']);
end

function s = tf(b)
if isnan(b), s = 'n/a'; elseif b, s = 'yes'; else, s = 'NO'; end
end
function s = tern(c, a, b), if c, s = a; else, s = b; end, end
