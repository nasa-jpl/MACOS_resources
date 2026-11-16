function ap = seg_apertures(seg, opts)
%SEG_APERTURES  Per-segment polygonal aperture blocks from the tiling truth.
%   ap = macos.design.seg_apertures(SEG, Name=Value) generates, for every
%   segment of SEG (= macos.design.segment_rx output), the prescription
%   lines that declare its PHYSICAL boundary as a polygonal aperture:
%   ApType=Polygonal + an explicit xObs (the engine ChkDf2 default
%   permutation (psi3,psi1,psi2)) + a 3-D global-vertex PolyApVec, plus
%   an optional convex PolyObsVec obscuration where the physical shape
%   is non-convex (engine convention: non-convex = convex aperture minus
%   convex obscurations).
%
%   Physical model (manual 4.x segment coordinates): width = flat-to-flat
%   (hex) / radial band (pie), gap = the inter-segment spacing, taken at
%   INTERNAL shared edges only (g/2 each side); a tiling rim edge carries
%   no gap.  Shapes:
%
%     Hex        the exact hex-tile corners (macos.design.hex_tile),
%                apothem width/2 + pad.
%     Pie        center segment = a HEXAGON (the (X,L,R) hex-coordinate
%                tiling's central cell -- verified against the traced ray
%                footprint, NOT a disc), apothem (width-gap)/2 + pad,
%                flats facing the ring-1 wedge centers; ring-1 wedges =
%                the TRUE physical wedge as one CONVEX polygon (outer
%                circumscribed arc + two side edges PARALLEL to the
%                sector boundary rays at offset g/2, + the straight
%                CHORD facing the center hexagon's flat) -- gaps are
%                uniform-width slots, they do NOT converge at the
%                tiling center, and no obscuration is needed (Dave
%                2026-07-18; geometry: macos.design.pie_wedge_geom).
%                Deeper rings (inner edge = an arc, non-convex) emit a
%                convex sector + an inner-sector PolyObsVec when
%                obs=true.
%
%   Options:
%     pad     (0)     outward clearance, same units: 0 = the physical
%                     edge (rays the source tiling puts in the gaps get
%                     clipped -- physically honest); pad=gap/2 puts the
%                     aperture at the tiling midline (trace-neutral).
%     obs     (true)  emit the inner-sector obscuration on DEEPER-ring
%                     pie wedges (their arc inner edge is non-convex;
%                     ring-1 wedges are convex chorded polygons and
%                     never need one).
%     nchord  (12)    chords per wedge outer/inner arc (total vertices
%                     per polygon must stay <= engine mPolySide=128).
%
%   Returns:
%     ap.blocks{s}  string column of prescription lines for segment s
%     ap.poly{s}    3 x nv aperture polygon vertices (global, open)
%     ap.obs{s}     3 x nv obscuration polygon vertices ([] if none)
%     ap.kind       'hex' | 'pie'
%
%   See also: macos.design.segment_rx (emit_apertures=true wires these
%   into the merged .in), macos.design.seg_boundary ('rxpoly' reads them
%   back).
arguments
    seg (1,1) struct
    opts.pad (1,1) double = 0
    opts.obs (1,1) logical = true
    opts.nchord (1,1) double {mustBeInteger, mustBePositive} = 12
end
kind = 'hex';
if isfield(seg, 'grid'), kind = lower(char(seg.grid)); end
fr = seg.frames;
n  = numel(fr);
g  = 0;
if isfield(seg, 'gap') && isfinite(seg.gap), g = seg.gap; end
w  = seg.width;
pad = opts.pad;

poly = cell(1, n); obs = cell(1, n);
switch kind
    case 'hex'
        % hex tiles: neighbor distance = width + gap, so the physical
        % edge IS the apothem width/2 -- hex_tile gives exact corners
        T = macos.design.hex_tile(seg, pad);
        for s = 1:n, poly{s} = T.corners{s}; obs{s} = []; end

    case 'pie'
        u = fr(1).xhat;  vN = fr(1).zhat;  v = cross(vN, u);
        c0 = mean([fr.rpt], 2);
        C2 = [u.'; v.'] * ([fr.rpt] - c0);
        R  = macos.design.pie_rings(C2, w);   % width-scaled tolerance
        rc = R.rc;  isctr = R.isctr;
        lift = @(P2) c0 + u*P2(1,:) + v*P2(2,:);
        for s = 1:n
            if isctr(s)
                % central cell of the hex-coordinate tiling: a hexagon
                % with flats facing the ring-1 wedge centers
                az = atan2(C2(2, R.iring == 1), C2(1, R.iring == 1));
                flat_ang = angle(mean(exp(1i*6*az))) / 6;
                a0h = (w - g)/2 + pad;
                phic = flat_ang + pi/6 + (0:5)*pi/3;
                poly{s} = lift(C2(:,s) + (a0h/cos(pi/6))*[cos(phic); sin(phic)]);
                obs{s} = [];
            else
                dth = 2*pi / R.nmem(R.iring(s));
                a0 = atan2(C2(2,s), C2(1,s));
                has_outer  = R.iring(s) < numel(R.rings);
                inner_ring = R.iring(s) == 1;
                W = macos.design.pie_wedge_geom(a0, dth, rc(s), w, g, ...
                        pad, has_outer, inner_ring && any(isctr));
                thc = linspace(W.th1, W.th2, opts.nchord+1);
                % circumscribed outer arc: the polygon never cuts
                % inside the rim
                Pout = (W.ro/cos((thc(2)-thc(1))/2)) * [cos(thc); sin(thc)];
                if inner_ring && any(isctr)
                    % ring-1 wedge = the physical shape, one convex
                    % polygon (chord + parallel sides + arc): uniform-
                    % width gaps, no apex at the center, no obscuration
                    poly{s} = lift([W.A, Pout, W.B]);
                    obs{s} = [];
                elseif opts.obs
                    % deeper ring: convex sector from the side-line
                    % apex + inner-sector arc obscuration
                    poly{s} = lift([W.X, Pout]);
                    thi = linspace(W.ti1, W.ti2, opts.nchord+1);
                    obs{s} = lift([W.X, W.ri*[cos(thi); sin(thi)]]);
                else
                    poly{s} = lift([W.X, Pout]);
                    obs{s} = [];
                end
            end
        end

    otherwise
        error('macos:design:seg_apertures:grid', ...
              'no aperture model for GridType %s (hex | pie)', kind);
end

% prescription text: explicit xObs = the ChkDf2 default permutation of
% psiElt (= the face zMon triad axis), so PolyApVec projects at parse
% time regardless of keyword order
fmt3 = @(P) arrayfun(@(q) sprintf("  %.10E  %.10E  %.10E", ...
            P(1,q), P(2,q), P(3,q)), 1:size(P,2))';
blocks = cell(1, n);
for s = 1:n
    if isfield(fr, 'psi') && norm(fr(s).psi) > 0
        ps = fr(s).psi;                    % psiElt as emitted
    else
        ps = fr(s).zhat;                   % geometry-only fallback
    end
    xo = ps([3 1 2]);
    b = [ ...
        sprintf("           ApType=  Polygonal"); ...
        sprintf("             xObs=  %.10E  %.10E  %.10E", xo); ...
        sprintf("        PolyApVec=  %d", size(poly{s}, 2)); ...
        fmt3(poly{s})];
    if ~isempty(obs{s})
        b = [b; ...
            sprintf("             nObs=  1"); ...
            sprintf("          ObsType=  Polygon"); ...
            sprintf("       PolyObsVec=  %d", size(obs{s}, 2)); ...
            fmt3(obs{s})];
    end
    blocks{s} = b;
end
ap = struct('blocks', {blocks}, 'poly', {poly}, 'obs', {obs}, 'kind', kind);
end
