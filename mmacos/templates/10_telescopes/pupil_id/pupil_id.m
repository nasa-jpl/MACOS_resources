% pupil_id.m  (mmacos/templates/10_telescopes/ -- a pupil-ID showcase driver)
% =====================================================================
%  BEYOND FEX: THE EXIT PUPIL AS A SURFACE
% =====================================================================
%  tma_onaxis's own note is that "the exit pupil after M3 is ASSESSED
%  (FEX), NOT constrained."  FEX reduces the exit pupil to a SINGLE
%  chief-ray conjugate sphere: one field point, forced Kc=0, one radius.
%  It gives the XP location and the far-field propagation distance and
%  NOTHING about pupil-imaging quality -- no pupil spherical aberration,
%  no pupil astigmatism, no pupil WALK across field.
%
%  This is the next chapter.  It takes tma_onaxis as a representative
%  telescope (M1 IS the stop, so the entrance pupil is the M1 beam rim),
%  runs the grid-of-cone-sources method (design/src/pupil_map) to find the
%  exit pupil as a fitted SURFACE, cross-checks it against the engine's
%  two-ray XPS (macos.pupil_quality), reports how sharply the EP images to
%  the XP (macos.pupil_zone_map), and tracks the pupil WALK vs field.
%
%  IT COMPOSES EXISTING, TEST-GATED TOOLS -- no new engine or veneer code.
%
%  THE TWO XP RADII (do not conflate).  FEX's Kr is the XP->detector
%  FAR-FIELD radius -- the PROPAGATION sphere the diffraction code needs;
%  that is what stays in the Rx.  The cone-crossing sag fit gives the
%  CURVATURE of the pupil-imaging convergence surface -- a pupil-QUALITY
%  diagnostic, a different quantity, NOT written to the Rx.  We keep FEX's
%  propagation radius (with the bundle vertex) and report the convergence
%  curvature + departure-from-sphere as the beyond-FEX pupil quality.
%
%  THREE EP CONVENTIONS, one deck (pupil_map 'anchor'):
%    'rim'     the EP as the beam RIM (classic telescope; M1 rim here) -- PRIMARY.
%    'stop'    the EP as a POINT through ApStop (when another element -- M2,
%              a DM, a pupil stop -- defines it); also what the engine XPS
%              computes, so it is the cross-check anchor.
%    'surface' on the curved entrance ELEMENT surface (pupil imagers that
%              must resolve the EP surface: a segmented primary, DM1 in a
%              coronagraph).
%
%  Run:  >> run('.../templates/10_telescopes/pupil_id/pupil_id.m')
%        (headless: matlab -batch "run('.../pupil_id.m'); exit(0)")
% =====================================================================

here = fileparts(mfilename('fullpath'));  if isempty(here), here = pwd; end
root = fileparts(fileparts(fileparts(here)));      % pupil_id -> 10_telescopes -> templates -> mmacos
run(fullfile(root,'mmacos_setup.m'));
addpath(fullfile(root,'design','src'));             % pupil_map
deck = fullfile(fileparts(here),'tma_onaxis','tma_onaxis.in');
assert(isfile(deck), 'need the tma_onaxis deck: %s', deck);

% ====================  USER CHOICES  =================================
FOV_ARCMIN   = 2;                 % half-box for the cone field grid
NGRID        = 3;                 % NGRID x NGRID field cones (>=3 required)
NODES        = 21;                % node lattice on the entrance surface
WALK_ARCMIN  = linspace(-2,2,5);  % field points for the pupil-walk sweep
MODEL_SIZE   = 256;               % one process, one size (do not transition)
% ====================================================================

macos.init(MODEL_SIZE);
R = struct();  R.deck = deck;  R.fov_arcmin = FOV_ARCMIN;  R.nodes = NODES;

% --------------------------------------------------------------------
% [1] BASELINE -- the single-chief-ray XP sphere (FEX; the thing we beat)
% --------------------------------------------------------------------
macos.load_rx(deck);  nE = macos.num_elt();
assert(nE == 6, 'expected the 6-element tma_onaxis deck; got nElt=%d', nE);
EP_ELT = 1;  XP_ELT = nE - 1;                        % M1 stop ; ExitPupil Return
macos.stop(EP_ELT);  macos.trace(nE);
f0 = macos.fex(1);                                   % chief-ray XP sphere
R.fex = struct('vpt',f0.vpt(:).', 'rad',f0.rad, 'xp_z',f0.vpt(3));
fprintf('[1] FEX baseline (one chief ray): XP z=%.5f m  propagation rad=%.6f m\n', ...
    f0.vpt(3), f0.rad);

% --------------------------------------------------------------------
% [2] EP declaration -- M1 rim, realized by anchor='rim' (no element)
% --------------------------------------------------------------------
R_M1 = 0.5;                                          % D/2 for tma_onaxis (D=1 m)
R.R_M1 = R_M1;
fprintf('[2] EP = M1 rim (M1 is the stop, ApStop=0 0 0). R_M1=%.3f m. ', R_M1);
fprintf('Declared via anchor=''rim'' -- no physical element inserted.\n');

% --------------------------------------------------------------------
% [3] CONE-CONVERGENCE XP SURFACE (beyond-FEX) -- three EP anchors
% --------------------------------------------------------------------
F = macos.design.field_grid(FOV_ARCMIN, NGRID, 'units','arcmin');
fprintf('[3] cone field grid: %d cones (%dx%d, +/-%g arcmin)\n', ...
    size(F,1), NGRID, NGRID, FOV_ARCMIN);
o_rim = pupil_map(deck, F, 'anchor','rim',     'obj_elt',EP_ELT, 'img_elt',XP_ELT, 'nodes',NODES, 'init',false);
o_srf = pupil_map(deck, F, 'anchor','surface', 'obj_elt',EP_ELT, 'img_elt',XP_ELT, 'nodes',NODES, 'init',false);
R.rim = ladder_(o_rim);  R.surface = ladder_(o_srf);
fprintf(['    RIM anchor: blur.rms=%.3e m  defocus=%.3e  astig=%.3e  spherical=%.3e\n' ...
         '                sag_rms=%.3e m  anchor.blur_ratio=%.3f (%d/%d nodes)\n'], ...
    o_rim.blur.rms, o_rim.surface.defocus, o_rim.surface.astig(1), o_rim.surface.spherical, ...
    o_rim.surface.sag_rms, o_rim.anchor.blur_ratio, nnz(o_rim.good), numel(o_rim.good));

% --------------------------------------------------------------------
% [4] XP sphere -> Rx (FEX propagation radius + bundle vertex) and the
%     pupil-QUALITY departure from a perfect sphere (the beyond-FEX bit)
% --------------------------------------------------------------------
% Exit-frame projection of the cone-crossing cloud.
X  = o_rim.X(:, o_rim.good);
X0 = o_rim.exit_frame.origin(:);  nn = o_rim.exit_frame.n(:);
e1 = o_rim.exit_frame.e1(:);      e2 = o_rim.exit_frame.e2(:);
d  = X - X0;
uu = (e1.'*d).';  vv = (e2.'*d).';  ww = (nn.'*d).';  rho = hypot(uu,vv);
% Sag fit w = c0 + b1*u + b2*v + a*rho^2 : a best-fit sphere can be POSITIONED
% and TILTED, so the fit absorbs piston (c0) + tilt (b1,b2) + curvature (a).
% The tilt terms are FRAME orientation (the exit axis is tipped ~1' by the
% field bias), not pupil aberration -- removing them leaves the true
% departure-from-sphere.  R_conv = convergence-surface curvature (a QUALITY
% metric, NOT the propagation radius); dep = the residual pupil aberration.
sol = [ones(numel(ww),1) uu vv rho.^2] \ ww;
c0 = sol(1);  a = sol(4);  R_conv = 1/(2*a);
dep = ww - [ones(numel(ww),1) uu vv rho.^2]*sol;  dep_rms = sqrt(mean(dep.^2));
vtx = X0 + c0*nn;                                    % bundle XP vertex
R.quality = struct('conv_radius',R_conv, 'dep_rms',dep_rms, ...
    'vtx',vtx(:).', 'uv',[uu vv], 'dep',dep(:));
% Write the PROPAGATION sphere (FEX radius) at the bundle vertex.
macos.load_rx(deck);  macos.stop(EP_ELT);
psi = nn;  if psi(3) > 0, psi = -psi; end            % normal toward the image
macos.set_xp(vtx, psi, f0.rad);
x1 = macos.get_xp();
R.written = struct('vpt',x1.vpt(:).', 'psi',x1.psi(:).', 'rad',x1.rad);
fprintf(['[4] XP sphere -> Rx: vertex=[% .5f % .5f %.5f] m, propagation rad=%.6f m (FEX)\n' ...
         '    pupil-quality: convergence curvature R=%.5f m, departure-from-sphere RMS=%.3e m\n' ...
         '    (vertex bundle-vs-FEX: %.2e m ; set_xp round-trip dvpt=%.1e drad=%.1e)\n'], ...
    x1.vpt, x1.rad, R_conv, dep_rms, norm(vtx-f0.vpt(:)), norm(x1.vpt-vtx), abs(x1.rad-f0.rad));

% --------------------------------------------------------------------
% [5] ENGINE TWO-RAY CROSS-CHECK (XPS) -- pupil_map 'stop' == pupil_quality
% --------------------------------------------------------------------
macos.load_rx(deck);  macos.stop(EP_ELT);
pq = macos.pupil_quality(XP_ELT, 'quiet',true);      % engine XPS, tangential differential
o_stop = pupil_map(deck, [0 0; 0 2e-5; 0 -2e-5], 'anchor','stop', ...
    'img_elt',XP_ELT, 'nodes',25, 'min_fields',2, 'init',false);
Rq = max(hypot(pq.uv(:,1), pq.uv(:,2)));
s  = (Rq / o_stop.surface.norm_radius)^2;            % renormalize to pq's disk
rel = @(x,y) 100*abs(x-y)/max(abs(y),eps);
R.xps = struct('pq_zern',pq.zern(:).', 'map_zern',(s*o_stop.surface.zern(:)).', ...
    'names',{pq.names}, 's',s, ...
    'defocus_relpct',rel(s*o_stop.surface.defocus, pq.defocus), ...
    'astig_relpct',  rel(s*o_stop.surface.astig(1), pq.astig(1)));
fprintf(['[5] XPS cross-check (pupil_map anchor=''stop'' vs engine pupil_quality):\n' ...
         '    defocus %.3e vs %.3e (%.2f%%)  astig %.3e vs %.3e (%.2f%%)  [rescale s=%.4f]\n'], ...
    s*o_stop.surface.defocus, pq.defocus, R.xps.defocus_relpct, ...
    s*o_stop.surface.astig(1), pq.astig(1), R.xps.astig_relpct, s);

% --------------------------------------------------------------------
% [6] PUPIL IMAGING SHARPNESS -- how sharply the EP images to the XP
% --------------------------------------------------------------------
macos.load_rx(deck);  macos.stop(EP_ELT);  macos.trace(XP_ELT);
z = macos.pupil_zone_map(EP_ELT, XP_ELT, 'shape','annular', 'ngrid',5, 'quiet',true);
R.zones = struct('nzone',z.nzone, 'med_spot',z.med_spot, 'max_spot',z.max_spot, ...
    'global_spot',z.global_spot, 'spots',z.spots(:).', 'zctr',z.zctr);
fprintf('[6] EP->XP imaging: %d zones, median spot=%.3e m, worst=%.3e m\n', ...
    z.nzone, z.med_spot, z.max_spot);

% --------------------------------------------------------------------
% [7] PUPIL WALK vs FIELD -- the across-field piece FEX cannot give
% --------------------------------------------------------------------
macos.load_rx(deck);  cur = macos.get_src_fov();  d0 = cur.src_dir(:)/norm(cur.src_dir);
walk = struct('dth_arcmin',WALK_ARCMIN, 'xp_z',zeros(size(WALK_ARCMIN)), 'rad',zeros(size(WALK_ARCMIN)));
for k = 1:numel(WALK_ARCMIN)
    th = deg2rad(WALK_ARCMIN(k)/60);                 % tangential tip about x
    Rx = [1 0 0; 0 cos(th) -sin(th); 0 sin(th) cos(th)];
    macos.load_rx(deck);  macos.set_src_fov('src_dir', Rx*d0);
    macos.stop(EP_ELT);  macos.trace(nE);
    fw = macos.fex(1);
    walk.xp_z(k) = fw.vpt(3);  walk.rad(k) = fw.rad;
end
R.walk = walk;
fprintf('[7] pupil walk over %+.0f..%+.0f arcmin: XP z %.6f..%.6f m, rad %.6f..%.6f m\n', ...
    WALK_ARCMIN(1), WALK_ARCMIN(end), min(walk.xp_z), max(walk.xp_z), min(walk.rad), max(walk.rad));

% --------------------------------------------------------------------
% [8] figures
% --------------------------------------------------------------------
fig_cloud_(R, o_rim, f0, pq, here);
fig_zern_(R, here);
fig_walk_(R, here);

% --------------------------------------------------------------------
% [9] save + [10] summary
% --------------------------------------------------------------------
save(fullfile(here,'pupil_id_results.mat'), 'R', '-v7.3');
fprintf('\n=====================================================================\n');
fprintf(' PUPIL ID -- tma_onaxis, beyond FEX\n');
fprintf('   FEX (1 chief ray):  XP z=%.5f m   propagation rad=%.6f m\n', R.fex.xp_z, R.fex.rad);
fprintf('   cone-convergence :  blur=%.2e m   departure-from-sphere=%.2e m\n', R.rim.blur_rms, R.quality.dep_rms);
fprintf('   pupil aberration :  defocus=%.2e  astig=%.2e  spherical=%.2e (rim anchor)\n', ...
    R.rim.defocus, R.rim.astig, R.rim.spherical);
fprintf('   XPS cross-check  :  defocus %.2f%% / astig %.2f%% vs engine pupil_quality\n', ...
    R.xps.defocus_relpct, R.xps.astig_relpct);
fprintf('   EP->XP imaging   :  median zone spot=%.2e m (%d zones)\n', R.zones.med_spot, R.zones.nzone);
fprintf('   written to Rx    :  XP vertex + FEX propagation radius (set_xp)\n');
fprintf('=====================================================================\n');

% ====================================================================
%  local helpers (compose-only; no library code lives here)
% ====================================================================
function L = ladder_(o)
%LADDER_  Pull the four-part pupil_map ladder into a flat struct for the table.
    L = struct('blur_rms',o.blur.rms, 'blur_max',o.blur.max, ...
        'defocus',o.surface.defocus, 'astig',o.surface.astig(1), ...
        'coma',o.surface.coma(1), 'spherical',o.surface.spherical, ...
        'sag_rms',o.surface.sag_rms, 'sag_pv',o.surface.sag_pv, ...
        'mag',o.map.mag, 'anamorph',o.map.anamorph, ...
        'wander_rms',o.wander.rms, 'blur_ratio',o.anchor.blur_ratio, ...
        'ngood',nnz(o.good), 'ntot',numel(o.good));
end

function fig_cloud_(R, o_rim, f0, pq, here)
%FIG_CLOUD_  XP crossing cloud + best-fit sphere vertex + departure-from-sphere map.
    fig = figure('Visible','off','Color','w','Position',[60 60 1280 560]);
    tl = tiledlayout(fig,1,2,'TileSpacing','compact','Padding','compact');
    title(tl, sprintf('tma_onaxis exit pupil -- cone-convergence surface (%d cones, %d nodes)', ...
        size(o_rim.fields,1), R.nodes), 'FontWeight','bold','Interpreter','none');

    X = o_rim.X(:, o_rim.good);
    ax = nexttile(tl); hold(ax,'on'); grid(ax,'on'); box(ax,'on');
    scatter3(ax, X(1,:)*1e3, X(2,:)*1e3, X(3,:)*1e3, 14, X(3,:)*1e3, 'filled');
    plot3(ax, f0.vpt(1)*1e3, f0.vpt(2)*1e3, f0.vpt(3)*1e3, 'kp', 'MarkerSize',14, 'MarkerFaceColor','y');
    plot3(ax, pq.vertex(1)*1e3, pq.vertex(2)*1e3, pq.vertex(3)*1e3, 'r+', 'MarkerSize',12, 'LineWidth',1.5);
    xlabel(ax,'x (mm)'); ylabel(ax,'y (mm)'); zlabel(ax,'z (mm)'); view(ax,35,18);
    legend(ax,{'crossing cloud','FEX vertex','XPS vertex'},'Location','northeast','FontSize',9);
    title(ax,'XP crossing cloud vs FEX / XPS vertex','Interpreter','none');

    ax = nexttile(tl); hold(ax,'on'); grid(ax,'on'); box(ax,'on'); axis(ax,'equal');
    scatter(ax, R.quality.uv(:,1)*1e3, R.quality.uv(:,2)*1e3, 24, R.quality.dep*1e9, 'filled');
    cb = colorbar(ax); cb.Label.String = 'departure from sphere (nm)';
    xlabel(ax,'u (mm)'); ylabel(ax,'v (mm)');
    title(ax, sprintf('departure-from-sphere  RMS=%.1f nm', R.quality.dep_rms*1e9));

    p = fullfile(here,'pupil_id_cloud.png');  exportgraphics(fig, p, 'Resolution',150);  close(fig);
    fprintf('    wrote %s\n', p);
end

function fig_zern_(R, here)
%FIG_ZERN_  Pupil-aberration Zernikes: pupil_map anchor='stop' vs engine XPS.
%   Only the ABERRATION terms are compared -- piston/tilt are removable
%   REFERENCE-FRAME terms (pupil_map fits in a centred frame and nulls them;
%   pupil_quality leaves them in its raw M1 coords), so they are not expected
%   to agree and would dwarf the physics.  Drop them.
    keep = find(~ismember(R.xps.names, {'piston','tilt_x','tilt_y'}));
    fig = figure('Visible','off','Color','w','Position',[60 60 1000 520]);
    ax = axes(fig); hold(ax,'on'); grid(ax,'on'); box(ax,'on');
    Y = [R.xps.map_zern(keep).', R.xps.pq_zern(keep).'];
    b = bar(ax, Y);  b(1).FaceColor=[0.20 0.50 0.80];  b(2).FaceColor=[0.85 0.45 0.15];
    set(ax,'XTick',1:numel(keep),'XTickLabel',R.xps.names(keep),'XTickLabelRotation',20);
    ylabel(ax,'Zernike coefficient (m)');  legend(ax,{'pupil\_map (anchor=stop)','engine XPS (pupil\_quality)'},'Location','best');
    title(ax, sprintf(['pupil aberration -- cone-convergence vs engine two-ray XPS ' ...
        '(defocus %.2f%%, astig %.2f%%; rescale s=%.3f; frame terms removed)'], ...
        R.xps.defocus_relpct, R.xps.astig_relpct, R.xps.s), 'Interpreter','none');
    p = fullfile(here,'pupil_id_zernikes.png');  exportgraphics(fig, p, 'Resolution',150);  close(fig);
    fprintf('    wrote %s\n', p);
end

function fig_walk_(R, here)
%FIG_WALK_  Pupil walk vs field: XP z and reference radius vs field angle.
    w = R.walk;
    fig = figure('Visible','off','Color','w','Position',[60 60 1000 460]);
    tl = tiledlayout(fig,1,2,'TileSpacing','compact','Padding','compact');
    title(tl,'exit-pupil walk vs field (FEX at each field point)','FontWeight','bold');
    ax = nexttile(tl); plot(ax, w.dth_arcmin, w.xp_z*1e3, 'o-','LineWidth',1.6,'MarkerFaceColor','w');
    grid(ax,'on'); box(ax,'on'); xlabel(ax,'field (arcmin)'); ylabel(ax,'XP axial position z (mm)');
    title(ax,'XP longitudinal walk');
    ax = nexttile(tl); plot(ax, w.dth_arcmin, w.rad*1e3, 's-','LineWidth',1.6,'Color',[0.85 0.33 0.10],'MarkerFaceColor','w');
    grid(ax,'on'); box(ax,'on'); xlabel(ax,'field (arcmin)'); ylabel(ax,'XP reference radius (mm)');
    title(ax,'XP radius walk');
    p = fullfile(here,'pupil_id_walk.png');  exportgraphics(fig, p, 'Resolution',150);  close(fig);
    fprintf('    wrote %s\n', p);
end
