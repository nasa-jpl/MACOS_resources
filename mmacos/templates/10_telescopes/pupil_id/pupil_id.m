% pupil_id.m  (mmacos/templates/10_telescopes/ -- a pupil-ID showcase driver)
% =====================================================================
%  BEYOND FEX: THE EXIT PUPIL AS A SURFACE  (general-purpose)
% =====================================================================
%  FEX reduces the exit pupil to a SINGLE chief-ray conjugate sphere: one
%  field point, forced Kc=0, one radius.  It gives the XP location and the
%  far-field propagation distance and NOTHING about pupil-imaging quality --
%  no pupil spherical aberration, no pupil astigmatism, no pupil WALK vs field.
%
%  This is the next chapter, and it is a GENERAL-PURPOSE driver: give it a
%  telescope Rx, it finds the exit pupil as a cone-convergence SURFACE, writes
%  a REVISED Rx (the XP reference sphere re-placed at the cone-bundle vertex,
%  an improvement on the one-chief-ray FEX sphere), cross-checks against the
%  engine's two-ray XPS, reports how sharply the EP images to the XP, and
%  tracks the pupil walk vs field.
%
%  TWO LAYERS:
%    - design/src/pupil_find.m  -- the CORE finder: Rx in, cone convergence,
%      set_xp into the internal Rx IN PLACE (exactly as FEX modifies engine
%      state), metrics out; no I/O, no figures.  Sensitivity drivers
%      (run_dwd*.m) call this directly.
%    - pupil_id.m (this file)   -- the WRAPPER: calls pupil_find, then does
%      the report, the XPS cross-check, the zone + walk metrics, the figures,
%      and writes the revised Rx to disk (macos.save_rx).
%
%  IT COMPOSES EXISTING, TEST-GATED TOOLS -- no new engine or veneer code.
%
%  Run (template default -- tma_onaxis):
%    >> pupil_id                          % or run('.../pupil_id.m')
%  Run on YOUR telescope:
%    >> out = pupil_id('/path/to/my_tel.in');
%    >> out = pupil_id(rx, 'ep_elt',E, 'xp_elt',X, 'fov_arcmin',3, ...
%                          'out_rx','my_tel_xp.in');
%  Headless:  matlab -batch "pupil_id('/path/to/my_tel.in'); exit(0)"
% =====================================================================
function out = pupil_id(rx, opts)
    arguments
        rx (1,:) char = ''                      % Rx path ('' -> tma_onaxis default)
        opts.ep_elt      (1,1) double = 1        % entrance-pupil (stop) element
        opts.xp_elt      (1,1) double = 0        % exit-pupil Return/Reference element; 0 -> nElt-1
        opts.fov_arcmin  (1,1) double = 2        % half-box for the cone field grid
        opts.ngrid       (1,1) double = 3        % ngrid x ngrid field cones (>=3)
        opts.nodes       (1,1) double = 21       % node lattice on the entrance surface
        opts.walk_arcmin (1,:) double = linspace(-2,2,5)  % pupil-walk field sweep
        opts.model_size  (1,1) double = 256      % one process, one size (do not transition)
        opts.anchor      (1,:) char {mustBeMember(opts.anchor,{'rim','surface','stop'})} = 'rim'
        opts.write_rx    (1,1) logical = true    % write the revised Rx to disk
        opts.out_rx      (1,:) char = ''         % revised-Rx path ('' -> <rx>_xp.in beside input)
        opts.outdir      (1,:) char = ''         % figure/results dir ('' -> input Rx's dir)
        opts.figures     (1,1) logical = true
    end
    here = fileparts(mfilename('fullpath'));  if isempty(here), here = pwd; end
    root = fileparts(fileparts(fileparts(here)));   % pupil_id -> 10_telescopes -> templates -> mmacos
    run(fullfile(root,'mmacos_setup.m'));
    addpath(fullfile(root,'design','src'));          % pupil_find, pupil_map

    if isempty(rx), rx = fullfile(fileparts(here),'tma_onaxis','tma_onaxis.in'); end   % template default
    assert(isfile(rx), 'Rx not found: %s', rx);
    if isempty(opts.outdir), opts.outdir = fileparts(rx);  if isempty(opts.outdir), opts.outdir = here; end, end
    if isempty(opts.out_rx), [d,n] = fileparts(rx);  opts.out_rx = fullfile(d,[n '_xp.in']); end

    macos.init(opts.model_size);
    F = macos.design.field_grid(opts.fov_arcmin, opts.ngrid, 'units','arcmin');

    % --------------------------------------------------------------------
    % [1-4] CORE: find the XP surface + place the bundle sphere in the Rx
    % --------------------------------------------------------------------
    pf = pupil_find(rx, F, 'ep_elt',opts.ep_elt, 'xp_elt',opts.xp_elt, ...
        'anchor',opts.anchor, 'nodes',opts.nodes, 'init',false, 'place',true);
    EP = pf.ep_elt;  XP = pf.xp_elt;  nE = pf.nElt;
    R = struct('rx',rx, 'ep_elt',EP, 'xp_elt',XP, 'nElt',nE, 'xp_type',pf.xp_type, ...
        'anchor',opts.anchor, 'fov_arcmin',opts.fov_arcmin, 'nodes',opts.nodes, ...
        'fex',pf.fex, ...
        'written',struct('vpt',pf.vtx, 'psi',pf.psi, 'rad',pf.rad), ...
        'quality',struct('conv_radius',pf.conv_radius, 'dep_rms',pf.dep_rms, ...
                         'uv',pf.uv, 'dep',pf.dep), ...
        'rim',ladder_(pf.o_rim));
    fprintf('[0] Rx=%s  nElt=%d  EP(stop)=%d  XP(%s)=%d\n', rx, nE, EP, pf.xp_type, XP);
    fprintf('[1] FEX baseline (one chief ray): XP z=%.5f m  propagation rad=%.6f m\n', pf.fex.vpt(3), pf.fex.rad);
    fprintf('[2] EP at element %d via anchor=''%s'' (%s)\n', EP, opts.anchor, anchor_note_(opts.anchor));
    fprintf('[3] cone field grid: %d cones (%dx%d, +/-%g arcmin); blur.rms=%.3e m  blur_ratio=%.3f\n', ...
        size(F,1), opts.ngrid, opts.ngrid, opts.fov_arcmin, pf.blur.rms, pf.o_rim.anchor.blur_ratio);
    fprintf(['[4] XP sphere -> Rx: vertex=[% .5f % .5f %.5f] m, propagation rad=%.6f m (FEX)\n' ...
             '    pupil-quality: convergence curvature R=%.5f m, departure-from-sphere RMS=%.3e m\n' ...
             '    (vertex bundle-vs-FEX: %.2e m)\n'], ...
        pf.vtx, pf.rad, pf.conv_radius, pf.dep_rms, norm(pf.vtx(:)-pf.fex.vpt(:)));

    % --------------------------------------------------------------------
    % [5] ENGINE TWO-RAY CROSS-CHECK (XPS) -- pupil_map 'stop' == pupil_quality
    % --------------------------------------------------------------------
    macos.load_rx(rx);  macos.stop(EP);
    pq = macos.pupil_quality(XP, 'quiet',true);          % engine XPS, tangential differential
    o_stop = pupil_map(rx, [0 0; 0 2e-5; 0 -2e-5], 'anchor','stop', ...
        'img_elt',XP, 'nodes',25, 'min_fields',2, 'init',false);
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
    %     (optional: a heavily-vignetted or tilted-aperture deck can leave a
    %     zone with no live rays; report and continue rather than abort.)
    % --------------------------------------------------------------------
    R.zones = struct([]);
    try
        macos.load_rx(rx);  macos.stop(EP);  macos.trace(XP);
        z = macos.pupil_zone_map(EP, XP, 'shape','annular', 'ngrid',5, 'quiet',true);
        R.zones = struct('nzone',z.nzone, 'med_spot',z.med_spot, 'max_spot',z.max_spot, ...
            'global_spot',z.global_spot, 'spots',z.spots(:).', 'zctr',z.zctr);
        fprintf('[6] EP->XP imaging: %d zones, median spot=%.3e m, worst=%.3e m\n', z.nzone, z.med_spot, z.max_spot);
    catch me
        fprintf('[6] EP->XP imaging: SKIPPED (%s) -- zone map needs a fully-lit pupil.\n', me.message);
    end

    % --------------------------------------------------------------------
    % [7] PUPIL WALK vs FIELD -- the across-field piece FEX cannot give
    % --------------------------------------------------------------------
    macos.load_rx(rx);  cur = macos.get_src_fov();  d0 = cur.src_dir(:)/norm(cur.src_dir);
    W = opts.walk_arcmin;
    walk = struct('dth_arcmin',W, 'xp_z',zeros(size(W)), 'rad',zeros(size(W)));
    for k = 1:numel(W)
        th = deg2rad(W(k)/60);                           % tangential tip about x
        Rx = [1 0 0; 0 cos(th) -sin(th); 0 sin(th) cos(th)];
        macos.load_rx(rx);  macos.set_src_fov('src_dir', Rx*d0);
        macos.stop(EP);  macos.trace(nE);
        fw = macos.fex(1);  walk.xp_z(k) = fw.vpt(3);  walk.rad(k) = fw.rad;
    end
    R.walk = walk;
    fprintf('[7] pupil walk over %+.0f..%+.0f arcmin: XP z %.6f..%.6f m, rad %.6f..%.6f m\n', ...
        W(1), W(end), min(walk.xp_z), max(walk.xp_z), min(walk.rad), max(walk.rad));

    % --------------------------------------------------------------------
    % [8] REVISED Rx -> disk (re-place the bundle sphere, then save)
    % --------------------------------------------------------------------
    if opts.write_rx
        macos.load_rx(rx);  macos.stop(EP);
        macos.set_xp(pf.vtx(:), pf.psi(:), pf.rad);
        macos.save_rx(opts.out_rx);
        R.out_rx = opts.out_rx;
        fprintf('[8] revised Rx written: %s (XP vertex + FEX propagation radius)\n', opts.out_rx);
    else
        R.out_rx = '';
    end

    % --------------------------------------------------------------------
    % [9] figures  [10] save results  [11] summary
    % --------------------------------------------------------------------
    if opts.figures
        fig_cloud_(R, pf.o_rim, pf.fex, pq, opts.outdir);
        fig_zern_(R, opts.outdir);
        fig_walk_(R, opts.outdir);
    end
    [~,tag] = fileparts(rx);
    save(fullfile(opts.outdir, sprintf('pupil_id_%s.mat',tag)), 'R', '-v7.3');

    fprintf('\n=====================================================================\n');
    fprintf(' PUPIL ID -- %s, beyond FEX\n', tag);
    fprintf('   FEX (1 chief ray):  XP z=%.5f m   propagation rad=%.6f m\n', R.fex.vpt(3), R.fex.rad);
    fprintf('   cone-convergence :  blur=%.2e m   departure-from-sphere=%.2e m\n', R.rim.blur_rms, R.quality.dep_rms);
    fprintf('   pupil aberration :  defocus=%.2e  astig=%.2e  spherical=%.2e (%s anchor)\n', ...
        R.rim.defocus, R.rim.astig, R.rim.spherical, R.anchor);
    fprintf('   XPS cross-check  :  defocus %.2f%% / astig %.2f%% vs engine pupil_quality\n', ...
        R.xps.defocus_relpct, R.xps.astig_relpct);
    if ~isempty(R.zones)
        fprintf('   EP->XP imaging   :  median zone spot=%.2e m (%d zones)\n', R.zones.med_spot, R.zones.nzone);
    else
        fprintf('   EP->XP imaging   :  (skipped -- pupil not fully lit on this deck)\n');
    end
    if opts.write_rx, fprintf('   revised Rx       :  %s\n', R.out_rx); end
    fprintf('=====================================================================\n');
    out = R;
end

% ====================================================================
%  local helpers (report/figure only; the finder lives in design/src)
% ====================================================================
function s = anchor_note_(a)
    switch a
        case 'rim',     s = 'EP as the beam rim -- classic telescope';
        case 'stop',    s = 'EP as a point through ApStop -- also the engine-XPS anchor';
        case 'surface', s = 'EP on the entrance-element surface -- pupil imagers / DM1';
    end
end

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

function fig_cloud_(R, o_rim, fex, pq, outdir)
%FIG_CLOUD_  XP crossing cloud + best-fit sphere vertex + departure map.
    fig = figure('Visible','off','Color','w','Position',[60 60 1280 560]);
    tl = tiledlayout(fig,1,2,'TileSpacing','compact','Padding','compact');
    title(tl, sprintf('exit pupil -- cone-convergence surface (%d cones, %d nodes)', ...
        size(o_rim.fields,1), R.nodes), 'FontWeight','bold','Interpreter','none');

    X = o_rim.X(:, o_rim.good);
    ax = nexttile(tl); hold(ax,'on'); grid(ax,'on'); box(ax,'on');
    scatter3(ax, X(1,:)*1e3, X(2,:)*1e3, X(3,:)*1e3, 14, X(3,:)*1e3, 'filled');
    plot3(ax, fex.vpt(1)*1e3, fex.vpt(2)*1e3, fex.vpt(3)*1e3, 'kp', 'MarkerSize',14, 'MarkerFaceColor','y');
    plot3(ax, pq.vertex(1)*1e3, pq.vertex(2)*1e3, pq.vertex(3)*1e3, 'r+', 'MarkerSize',12, 'LineWidth',1.5);
    xlabel(ax,'x (mm)'); ylabel(ax,'y (mm)'); zlabel(ax,'z (mm)'); view(ax,35,18);
    legend(ax,{'crossing cloud','FEX vertex','XPS vertex'},'Location','northeast','FontSize',9);
    title(ax,'XP crossing cloud vs FEX / XPS vertex','Interpreter','none');

    ax = nexttile(tl); hold(ax,'on'); grid(ax,'on'); box(ax,'on'); axis(ax,'equal');
    scatter(ax, R.quality.uv(:,1)*1e3, R.quality.uv(:,2)*1e3, 24, R.quality.dep*1e9, 'filled');
    cb = colorbar(ax); cb.Label.String = 'departure from sphere (nm)';
    xlabel(ax,'u (mm)'); ylabel(ax,'v (mm)');
    title(ax, sprintf('departure-from-sphere  RMS=%.1f nm', R.quality.dep_rms*1e9));

    p = fullfile(outdir,'pupil_id_cloud.png');  exportgraphics(fig, p, 'Resolution',150);  close(fig);
    fprintf('    wrote %s\n', p);
end

function fig_zern_(R, outdir)
%FIG_ZERN_  Pupil-aberration Zernikes: pupil_map anchor='stop' vs engine XPS.
%   Piston/tilt are removable REFERENCE-FRAME terms (pupil_map nulls them in a
%   centred frame; pupil_quality leaves them in raw M1 coords), so they are
%   not expected to agree and would dwarf the physics -- drop them.
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
    p = fullfile(outdir,'pupil_id_zernikes.png');  exportgraphics(fig, p, 'Resolution',150);  close(fig);
    fprintf('    wrote %s\n', p);
end

function fig_walk_(R, outdir)
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
    p = fullfile(outdir,'pupil_id_walk.png');  exportgraphics(fig, p, 'Resolution',150);  close(fig);
    fprintf('    wrote %s\n', p);
end
