%ASSESS_E5PIE_POLYAP  Poly apertures on pie segments: the assessment.
%
% Question (Dave 2026-07-16): should each segment carry a polygonal
% aperture in the Rx, with that polygon as the basis for launcher
% placement (the general case, covering imported segmented Rx too)?
%
% This prototype, on the SegMirMaker e5 PIE segmentation:
%   1. builds the pie system through the normal flow (segment_rx),
%      traces the BASELINE (segment-ness = source tiling only,
%      ApType=None on every segment);
%   2. renders the pie MET scene (first real-fixture exercise of
%      seg_boundary's pie wedges + edge launcher placement);
%   3. EMITS a poly-aperture variant: each ring wedge gets
%      ApType=Polygonal + xObs + PolyApVec (a convex chorded SECTOR out
%      to the wedge outer arc) + a convex PolyObsVec inner-sector
%      obscuration (annular sectors are non-convex; the engine builds
%      non-convex shapes as convex aperture minus convex obscurations,
%      Lou item 13).  The center segment gets a circumscribed regular
%      NCTR-gon (Dave 2026-07-16: Polygonal for Elt 1 too, no circular
%      special case -- SetCvxPolyApVtx generates its ApVec).
%   4. traces the variant and compares: nominal parity (source tiling
%      already excludes gap rays, so the apertures should be nearly
%      invisible), then a 100 mm segment decenter, where the aperture
%      SHOULD bite (rays walking off the physical segment get clipped
%      -- the physical-honesty gain);
%   5. reads the polygons back out of the variant Rx and compares them
%      to seg_boundary's tiling reconstruction (the 'rxpoly' reader
%      feasibility check).
%
% Findings land in findings.txt + PNGs beside this script.

NCHORD = 12;                 % chords per arc (<= mPolySide=128 total)
NCTR   = 24;                 % vertices of the center-segment polygon
WITH_OBS = false;            % include the inner-sector obscuration
                             % (discriminator: outer-ring clipping with
                             % obs OFF isolates the aperture polygon)
POKE   = 0.100;              % decenter for the clipping demo, metres
PAD    = 25;                 % aperture clearance pad, mm: the mask sits
                             % just OUTSIDE the tiling edge so it is
                             % trace-neutral at nominal and bites only
                             % on perturbation walk-off

here = fileparts(mfilename('fullpath'));
res_root = fullfile(getenv('HOME'), 'dev', 'MACOS_resources');
tin = fullfile(res_root, 'segmirmaker', 'test_in');
log_ = fopen(fullfile(here, 'findings.txt'), 'w');
out = @(varargin) fprintf(1, varargin{:}) + fprintf(log_, varargin{:});

%% ---- 1. pie system through the normal flow ----------------------------
seg = macos.design.segment_rx(fullfile(tin, 'e5mono.in'), 'elt', 1, ...
    'rings', 1, 'grid', 'Pie', 'gap', 50, 'dofs', 6, 'meas_config', 1);
out('pie system: %d segments, width %.1f, gap %g\n', seg.nseg, seg.width, seg.gap);

old = cd(seg.run.workdir); restore = onCleanup(@() cd(old));
macos.init(512);
macos.load_rx(seg.in);
t0 = macos.trace();
pass_ = @(t) nnz(subsref(macos.get_ray_info(t.nRays), ...
    substruct('.', 'ok_pass')));
p0 = pass_(t0);
out('baseline:      %6d src rays, %6d pass, rmsWFE %.6g\n', ...
    t0.nRays, p0, t0.rmsWFE);

%% ---- 2. pie MET scene (seg_boundary pie wedges, real fixture) ---------
am = macos.design.add_met(seg.in, seg, 'hub', seg.nseg+1, 'r_fid', 590, ...
    'nf', 6, 'extra_sources', seg.n_elt-2, 'r_extra', 100);
macos.load_rx(am.in); macos.trace();
fv = macos.design.met_view(seg, am, 'visible', false, 'edge_off', 5, ...
    'title', 'e5 PIE segmentation: boundary-true wedges + edge launchers', ...
    'save', fullfile(here, 'e5pie_met_layout.png'));
close(fv);
out('pie MET scene rendered (e5pie_met_layout.png)\n');

%% ---- 3. emit the poly-aperture variant --------------------------------
B = macos.design.seg_boundary(seg);              % tiling-truth wedges
lines = readlines(seg.in);
tl = strtrim(lines);
starts = find(startsWith(tl, "iElt="));
fprintf('emit: %d lines, %d element blocks, seg_elts [%s]\n', ...
    numel(lines), numel(starts), join(string(seg.seg_elts), ' '));
C2 = [B.u.'; B.v.'] * ([seg.frames.rpt] - B.c0);
rc = vecnorm(C2);
isctr = rc < 1e-6 * max(rc);
w = seg.width; g = seg.gap;
for s = seg.nseg:-1:1                            % bottom-up, indices stay valid
    k = seg.seg_elts(s);
    fprintf('  emit s=%d k=%d: %d blocks, %d lines\n', s, k, ...
        numel(starts), numel(lines));
    b0 = starts(k);
    b1 = numel(lines); if k < numel(starts), b1 = starts(k+1) - 1; end
    % aperture frame: engine ChkDf2 default xObs = (psi3, psi1, psi2)
    ps = sscanf(regexprep(lines(find(startsWith(tl(b0:b1), "psiElt="),1)+b0-1), ...
                          '.*psiElt=', ''), '%f');
    xo = ps([3 1 2]);
    fmt3 = @(P) arrayfun(@(q) sprintf("  %.10E  %.10E  %.10E", ...
                P(1,q), P(2,q), P(3,q)), 1:size(P,2))';
    if isctr(s)
        % center segment: circumscribed regular NCTR-gon at the disc
        % radius (Polygonal like the wedges -- no circular special case)
        r0c = min(w/2, min(rc(~isctr)) - w/2 - g) + PAD;
        thc = 2*pi*(0:NCTR-1)/NCTR;
        Pc = B.c0 + (r0c/cos(pi/NCTR)) * (B.u*cos(thc) + B.v*sin(thc));
        newap = [ ...
            sprintf("           ApType=  Polygonal"); ...
            sprintf("             xObs=  %.10E  %.10E  %.10E", xo); ...
            sprintf("        PolyApVec=  %d", size(Pc,2)); ...
            fmt3(Pc)];
    else
        a0 = atan2(C2(2,s), C2(1,s));
        nring = nnz(~isctr);
        ha = pi/nring - (g/2 - PAD)/rc(s);       % wedge half-span + pad
        ri = rc(s) - w/2 - PAD;  ro = rc(s) + w/2 + PAD;
        thc = linspace(a0-ha, a0+ha, NCHORD+1);
        % aperture = convex sector to the outer arc (circumscribed
        % chords so the polygon never clips inside the wedge)
        Pout = B.c0 + (ro/cos((thc(2)-thc(1))/2)) * ...
                      (B.u*cos(thc) + B.v*sin(thc));
        Vap = [B.c0, Pout];                      % apex + arc
        % obscuration = convex inner sector (inscribed chords)
        Pin = B.c0 + ri*(B.u*cos(thc) + B.v*sin(thc));
        Vob = [B.c0, Pin];
        newap = [ ...
            sprintf("           ApType=  Polygonal"); ...
            sprintf("             xObs=  %.10E  %.10E  %.10E", xo); ...
            sprintf("        PolyApVec=  %d", size(Vap,2)); ...
            fmt3(Vap)];
        if WITH_OBS
            newap = [newap; ...
                sprintf("             nObs=  1"); ...
                sprintf("          ObsType=  Polygon"); ...
                sprintf("       PolyObsVec=  %d", size(Vob,2)); ...
                fmt3(Vob)]; %#ok<AGROW>
        end
    end
    % append at the end of the segment's block: SegMirMaker .presc
    % blocks carry NO ApType/nObs lines (ChkDf2 defaults them), so
    % there is nothing to replace -- and the parser is last-wins anyway
    ie = b1;
    while ie > b0 && strlength(strtrim(lines(ie))) == 0, ie = ie - 1; end
    lines = [lines(1:ie); newap; lines(ie+1:end)];
    tl = strtrim(lines); starts = find(startsWith(tl, "iElt="));
end
poly_in = fullfile(seg.run.workdir, 'e5pie_polyap.in');
writelines(lines, poly_in);
copyfile(poly_in, fullfile(here, 'e5pie_polyap.in'));
out('poly-aperture variant emitted (e5pie_polyap.in)\n');

%% ---- 4. trace parity + the clipping demo ------------------------------
macos.load_rx(poly_in);
t1 = macos.trace();
p1 = pass_(t1);
out('poly variant:  %6d src rays, %6d pass, rmsWFE %.6g\n', ...
    t1.nRays, p1, t1.rmsWFE);
rs = macos.get_ray_status(t1.nRays);
fe = rs.fail_elt(rs.status ~= 0);
ue = unique(fe(fe > 0));
for e2 = ue.'
    out('  lost rays first failed at elt %d: %d\n', e2, nnz(fe == e2));
end
out('nominal parity: dPass = %d, d(rmsWFE) = %.3g (%.4f%%)\n', ...
    p1 - p0, t1.rmsWFE - t0.rmsWFE, ...
    100*abs(t1.rmsWFE - t0.rmsWFE)/t0.rmsWFE);

% WHERE do the clipped rays live?  Pupil-grid OPD masks tell directly.
w1map = macos.opd(); m1 = isfinite(w1map) & (w1map ~= 0);
macos.load_rx(seg.in); macos.trace();
w0map = macos.opd(); m0 = isfinite(w0map) & (w0map ~= 0);
dm = m0 & ~m1;                                   % clipped by the apertures
N = size(w0map, 1);
[ii, jj] = find(dm);
uu = 2*(jj - (N+1)/2)/N;  vv = 2*(ii - (N+1)/2)/N;   % pupil coords +-1
rr = hypot(uu, vv) * 4000;                       % mm at the 8 m pupil
aa = mod(rad2deg(atan2(vv, uu)), 60);            % angle within a wedge
out('clipped pixels: %d;  radius [%.0f .. %.0f] mm;  median %.0f\n', ...
    nnz(dm), min(rr), max(rr), median(rr));
out('  angle-in-wedge quartiles: %.1f / %.1f / %.1f deg\n', ...
    prctile(aa, 25), prctile(aa, 50), prctile(aa, 75));
fdm = figure('Visible','off');
imagesc(dm); axis image; title('rays clipped by poly apertures (pupil)');
print(fdm, fullfile(here, 'clipped_mask.png'), '-dpng', '-r120'); close(fdm);
macos.load_rx(poly_in); macos.trace();           % restore variant state

% decenter segment 2 by POKE: with a physical aperture, edge rays must
% clip; without one, they silently stay
for variant = 1:2
    if variant == 1, macos.load_rx(seg.in);  nm = 'no aperture  '; pn = p0;
    else,            macos.load_rx(poly_in); nm = 'poly aperture'; pn = p1;
    end
    macos.perturb(seg.seg_elts(2), 'translation', [POKE; 0; 0], ...
                  'frame', 'local');
    macos.modify();
    tp = macos.trace();
    out('poke %.0f mm, %s: %6d pass (clipped %d vs its nominal)\n', ...
        POKE*1e3, nm, pass_(tp), pn - pass_(tp));
end

%% ---- 5. rxpoly reader feasibility -------------------------------------
% read the polygons back from the emitted Rx and compare to the tiling
% reconstruction (what a seg_boundary('rxpoly') source would do)
lines2 = readlines(poly_in); tl2 = strtrim(lines2);
st2 = find(startsWith(tl2, "iElt="));
maxdev = 0; nread = 0;
nring = nnz(~isctr);
for s = 1:seg.nseg
    k = seg.seg_elts(s);
    b0 = st2(k); b1 = numel(lines2); if k < numel(st2), b1 = st2(k+1)-1; end
    ip = find(startsWith(tl2(b0:b1), "PolyApVec="), 1) + b0 - 1;
    nv = sscanf(regexprep(lines2(ip), '.*PolyApVec=', ''), '%d');
    V = zeros(3, nv);
    for q = 1:nv, V(:,q) = sscanf(lines2(ip+q), '%f'); end
    nread = nread + 1;
    % arc vertices must sit at the circumscribed radius used at
    % emission -- the round-trip fidelity of an 'rxpoly' reader.
    % Wedges: vertex 1 is the sector apex, 2:end the outer arc.
    % Center: all NCTR vertices are on the circumscribed circle.
    if isctr(s)
        Vp = [B.u.'; B.v.'] * (V - B.c0);
        r0c = min(w/2, min(rc(~isctr)) - w/2 - g) + PAD;
        r_circ = r0c / cos(pi/NCTR);
    else
        Vp = [B.u.'; B.v.'] * (V(:,2:end) - B.c0);
        ha = pi/nring - (g/2 - PAD)/rc(s);
        r_circ = (rc(s) + w/2 + PAD) / cos(ha/NCHORD);
    end
    maxdev = max(maxdev, max(abs(vecnorm(Vp) - r_circ)));
end
out('rxpoly read-back: %d segment polygons (incl. center), arc radius consistent to %.3g mm\n', ...
    nread, maxdev);
fclose(log_);
fprintf('findings.txt + figures written to %s\n', here);
