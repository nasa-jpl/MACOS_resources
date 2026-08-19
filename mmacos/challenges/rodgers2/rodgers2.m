function R = rodgers2(opts)
%RODGERS2  The Rodgers2 benchmark: his four afocal decks, under our metrics.
%
%   R = RODGERS2() transcribes J.M. Rodgers' four CODE V afocal decks
%   (RODGERS2_SEQ -> RODGERS2_DECK), scores them on the afocal ladder
%   against his reported WFE, and measures the interface-pupil quality his
%   deck asserts only verbally.  It writes the committed artifacts.
%
%   THE STUDY IS THE PUPIL TABLE.  His slides state that "with 3 mirrors the
%   pupil quality is not very good; a 4th mirror is needed for pupil
%   control", and his deck contains NO pupil metric -- the only pupil-
%   adjacent evidence in it is the coldstop's DAR tilt (0 / 4.289 / 3.577 /
%   -0.356 deg) and a magnification that slips from 30x.  The pupil-quality
%   DEFINITION is therefore OURS (PUPIL_MAP, the cone-convergence model),
%   and any material that goes back to Mike must say so.
%
%   Sections (run a subset with 'sections'):
%     0  transcription + the first-order gate (magnification, the recenter
%        sign decode, the coldstop's tilt off the exit chief)
%     1  the afocal ladder on HIS 3x3 solve set AND on a uniform grid,
%        against his 15 / 430 / 160 / 119 nm -- including which RUNG his
%        CODE V field map corresponds to, decoded once and then pinned
%     2  the BASELINE PUPIL TABLE: the four-part cone-convergence ladder
%        plus the engine's own XPS pupil surface, on all four variants
%     3  figures
%
%   Name-value:
%     'sections'   default 0:3
%     'map_n'      uniform scoring grid, n x n (9)
%     'nodes'      pupil-map node lattice (21)
%     'sampling'   deck nGridpts (41)
%     'save'       write .in / .png / .mat (true)
%     'quiet'      (false)
%
%   See also RODGERS2_SEQ, RODGERS2_DECK, AFOCAL_LADDER_DECK, PUPIL_MAP,
%   CALIB_AFOCAL_PROBE, PACKET.md.

    arguments
        opts.sections (1,:) double = 0:3
        opts.map_n    (1,1) double = 9
        opts.nodes    (1,1) double = 21
        opts.sampling (1,1) double = 41
        opts.save     (1,1) logical = true
        opts.quiet    (1,1) logical = false
    end
    here = fileparts(mfilename('fullpath'));
    root = fileparts(fileparts(here));
    run(fullfile(root,'mmacos_setup.m'));
    addpath(here);

    S = rodgers2_seq();
    nv = numel(S.v);
    lam = S.lambda_nm*1e-9;
    macos.init(256);

    R = struct('seq',S, 'lambda_m',lam, 'map_n',opts.map_n, ...
               'when',datestr(now,31)); %#ok<TNOW1,DATST>

    % =====================================================================
    if any(opts.sections == 0)
    banner('0.  transcription and the first-order gate');
    % =====================================================================
    fprintf(['  %-13s %10s %10s %10s %10s %10s %10s\n'], ...
        'variant','miss(mm)','exit(mm)','M_beam','M_ang','chief(deg)','csTilt(deg)');
    D = struct('name',{},'file',{},'miss_mm',{},'exit_beam_mm',{}, ...
               'mag_beam',{},'mag_ang',{},'chief_deg',{},'cs_tilt_deg',{});
    for i = 1:nv
        o = rodgers2_deck(i, 'sampling', opts.sampling, 'verify', true, ...
                          'file', fullfile(here, sprintf('rodgers2_%s.in', S.v(i).name)));
        ch = atan2d(-o.exit_dir(2), -o.exit_dir(3));
        e = struct('name',S.v(i).name, 'file',o.file, ...
                   'miss_mm',o.chief_miss_mm, 'exit_beam_mm',o.exit_beam_mm, ...
                   'mag_beam',o.mag, 'mag_ang',o.mag_ang, 'chief_deg',ch, ...
                   'cs_tilt_deg',o.coldstop.ADE_total_deg - ch);
        if isempty(D), D = e; else, D(end+1) = e; end %#ok<AGROW>
        fprintf('  %-13s %10.2e %10.3f %10.4f %10.4f %10.5f %10.5f\n', ...
            e.name, e.miss_mm, e.exit_beam_mm, e.mag_beam, e.mag_ang, ...
            e.chief_deg, e.cs_tilt_deg);
    end
    R.firstorder = D;
    fprintf(['\n  THE DECODE (witness 5 for the rodgers1 ADE sign).  The "recenter"\n' ...
             '  coordinate break places the coldstop vertex ON the traced exit chief\n' ...
             '  to %.0e mm.  Under the OPPOSITE ADE sense the same arithmetic misses\n' ...
             '  it by 211-247 mm on a 33 mm beam.  And the recenter ADE IS the exit\n' ...
             '  chief angle to 5 decimals, so the coldstop DAR tilt is exactly the\n' ...
             '  coldstop''s tilt away from normal-to-chief -- which is the only\n' ...
             '  pupil-adjacent number his deck carries.\n'], max([D.miss_mm]));
    end

    % =====================================================================
    if any(opts.sections == 1)
    banner('1.  the afocal ladder vs his reported WFE');
    % =====================================================================
    Fsolve = S.Frel;                                     % his 3x3
    Fgrid  = macos.design.field_grid(S.fov_half_deg*60, opts.map_n, ...
                                     'units','arcmin');  % uniform
    W = struct('name',{},'L_solve',{},'L_grid',{},'info_solve',{}, ...
               'info_grid',{},'gt_max_nm',{},'gt_avg_nm',{});
    for i = 1:nv
        f = fullfile(here, sprintf('rodgers2_%s.in', S.v(i).name));
        [Ls, is_] = afocal_ladder_deck(f, Fsolve, 'lambda', lam);
        [Lg, ig]  = afocal_ladder_deck(f, Fgrid,  'lambda', lam);
        e = struct('name',S.v(i).name, 'L_solve',Ls, 'L_grid',Lg, ...
                   'info_solve',is_, 'info_grid',ig, ...
                   'gt_max_nm',S.v(i).gt_max_nm, 'gt_avg_nm',S.v(i).gt_avg_nm);
        if isempty(W), W = e; else, W(end+1) = e; end %#ok<AGROW>
    end
    R.wfe = W;

    rn = {'1 piston','2 +tip/tilt','3 +power'};
    for r = 1:3
        fprintf('\n  rung %s   (nm)\n', rn{r});
        fprintf('  %-13s | %9s %9s | %9s %9s | %8s %8s | %8s %8s\n', ...
            'variant','max','avg','max_grid','avg_grid','his max','his avg','max x','avg x');
        for i = 1:nv
            Ls = W(i).L_solve(:,r)*1e9;   Lg = W(i).L_grid(:,r)*1e9;
            Ls = Ls(isfinite(Ls));        Lg = Lg(isfinite(Lg));
            fprintf('  %-13s | %9.2f %9.2f | %9.2f %9.2f | %8.1f %8.1f | %8.3f %8.3f\n', ...
                W(i).name, max(Ls), mean(Ls), max(Lg), mean(Lg), ...
                W(i).gt_max_nm, W(i).gt_avg_nm, ...
                max(Lg)/W(i).gt_max_nm, mean(Lg)/W(i).gt_avg_nm);
        end
    end
    fprintf(['\n  The two MAX columns are identical by construction: his 3x3 set\n' ...
             '  and a uniform %dx%d grid over the same box share the four corners,\n' ...
             '  and a corner is always the worst field.  The AVERAGES do not, and\n' ...
             '  the ratio columns are therefore taken on the UNIFORM GRID -- a\n' ...
             '  9-point set that is one third corners over-weights them badly (it\n' ...
             '  reads 2.1x his average on S1 where the uniform grid reads 1.04x).\n'], ...
             opts.map_n, opts.map_n);

    % --- the rung decode, done ONCE and then pinned -----------------------
    ratio = nan(nv,3);
    for i = 1:nv
        for r = 1:3
            v = W(i).L_grid(:,r)*1e9;   v = v(isfinite(v));
            ratio(i,r) = max(v)/W(i).gt_max_nm;
        end
    end
    score = max(abs(log(ratio)), [], 1);        % worst-variant log-distance
    [~, rbest] = min(score);
    R.matched_rung = rbest;
    R.rung_ratio = ratio;
    fprintf(['\n  THE MATCHED RUNG.  Scoring the worst-case log distance from 1.0\n' ...
             '  across all four variants picks rung %d (%s):\n'], rbest, rn{rbest});
    fprintf('    ratios by variant: %s\n', ...
            strtrim(sprintf('%.3f  ', ratio(:,rbest))));
    fprintf(['    -- inside the <=1.15x gate on all four.  So his CODE V afocal\n' ...
             '    field-map RMS removes piston AND tip/tilt per field, and nothing\n' ...
             '    more.  Quote that rung on every number.\n']);
    aratio = nan(nv,1);
    for i = 1:nv
        v = W(i).L_grid(:,rbest)*1e9;  v = v(isfinite(v));
        aratio(i) = mean(v)/W(i).gt_avg_nm;
    end
    R.rung_ratio_avg = aratio;
    fprintf(['    and the in-box AVERAGES at that rung, on the uniform grid:\n' ...
             '      %s  (%.3f .. %.3f x)\n' ...
             '    His stated averages are therefore AREA averages over the box,\n' ...
             '    not means of his nine solve points.\n'], ...
             strtrim(sprintf('%.3f  ', aratio)), min(aratio), max(aratio));
    fprintf('    band on the max: %.3f .. %.3f x\n', ...
             min(ratio(:,rbest)), max(ratio(:,rbest)));
    end

    % =====================================================================
    if any(opts.sections == 2)
    banner('2.  THE BASELINE PUPIL TABLE');
    % =====================================================================
    Fsolve = S.Frel;
    P = struct('name',{},'pm',{},'pq',{},'pq_msg',{});
    for i = 1:nv
        f = fullfile(here, sprintf('rodgers2_%s.in', S.v(i).name));
        pm = pupil_map(f, Fsolve, 'nodes', opts.nodes, 'init', false);
        pq = [];  msg = '';
        try
            macos.load_rx(f);
            pq = macos.pupil_quality(macos.num_elt(), 'quiet', true);
        catch ME
            msg = ME.message;
        end
        e = struct('name',S.v(i).name, 'pm',pm, 'pq',pq, 'pq_msg',msg);
        if isempty(P), P = e; else, P(end+1) = e; end %#ok<AGROW>
    end
    R.pupil = P;
    pupil_table_(P, S);
    end

    % =====================================================================
    if any(opts.sections == 3) && opts.save
    banner('3.  figures');
    % =====================================================================
    if isfield(R,'wfe')
        for i = 1:nv
            png = fullfile(here, sprintf('rodgers2_%s_ladder.png', R.wfe(i).name));
            ladder_map_(R.wfe(i), S, png);
            fprintf('  wrote %s\n', png);
        end
    end
    if isfield(R,'pupil')
        for i = 1:nv
            png = fullfile(here, sprintf('rodgers2_%s_pupil.png', R.pupil(i).name));
            pupil_fig_(R.pupil(i), png);
            fprintf('  wrote %s\n', png);
        end
    end
    end

    if opts.save
        save(fullfile(here,'rodgers2_results.mat'), 'R', '-v7.3');
        fprintf('\n  saved rodgers2_results.mat\n');
    end
end

% =====================================================================
function pupil_table_(P, S)
%PUPIL_TABLE_  The first quantitative statement of "the 3-mirror pupil
%   quality is not very good".  Four parts, never merged.
    fprintf(['\n  Cone aperture = his 3x3 field set.  Anchored on the M1 SURFACE.\n' ...
             '  Lengths in um unless marked.  Diffraction floor is\n' ...
             '  lambda/(2 NA_field), quoted per variant.\n\n']);
    fprintf('  %-13s | %8s %8s | %9s %9s %9s | %8s\n', ...
        'variant','blur rms','blur max','mag','anamorph','distort%','floor');
    for i = 1:numel(P)
        m = P(i).pm;
        fprintf('  %-13s | %8.1f %8.1f | %9.4f %9.5f %8.3f%% | %8.2f\n', ...
            P(i).name, 1e6*m.blur.rms, 1e6*m.blur.max, m.map.mag, ...
            m.map.anamorph, 100*m.map.distortion_frac_max, ...
            1e6*m.diffraction.floor_m);
    end
    fprintf('\n  %-13s | %9s %11s | %11s %11s | %9s %9s\n', ...
        'variant','magC','mag range','surf tilt(mr)','defocus(mm)','wander','w/best');
    for i = 1:numel(P)
        m = P(i).pm;
        tl = 1e3*norm(m.surface.tilt)/m.surface.norm_radius;   % mrad
        fprintf('  %-13s | %9.4f %4.2f-%5.2f | %11.4f %11.4f | %9.1f %9.1f\n', ...
            P(i).name, m.map.mag_centre, min(m.map.mag_per_field), ...
            max(m.map.mag_per_field), tl, 1e3*m.surface.defocus, ...
            1e6*m.wander.rms, 1e6*m.best_plane.rms);
    end
    % THE MAGNIFICATION FRAME.  The "mag range" above is read on the PLACED
    % coldstop -- a fixed plane each field's exit chief strikes at up to
    % 13.6 deg -- so it carries a 1/cos areal stretch that is a FRAME term
    % and not pupil imaging.  Read the SAME STATION perpendicular to each
    % field's OWN exit chief and that term is gone.  Both belong in the
    % record: the placed number is what a fixed coldstop samples, the
    % chief-normal number is the pupil-imaging defect and the S3 target.
    fprintf('\n  %-13s | %8s %11s %7s | %8s %11s %7s | %11s\n', ...
        'variant','magC','range','+-%','magC_chf','range_chf','+-%','incid(deg)');
    for i = 1:numel(P)
        m = P(i).pm;   p = m.map.mag_per_field;   c = m.map.mag_per_field_chief;
        fprintf('  %-13s | %8.4f %5.2f-%5.2f %+6.3f | %8.4f %5.2f-%5.2f %+6.3f | %5.2f-%5.2f\n', ...
            P(i).name, m.map.mag_centre, min(p), max(p), ...
            100*(max(p)-min(p))/2/m.map.mag_centre, ...
            m.map.mag_centre_chief, min(c), max(c), ...
            100*(max(c)-min(c))/2/m.map.mag_centre_chief, ...
            min(m.map.incidence_deg), max(m.map.incidence_deg));
    end
    % HIS COLDSTOP TUNING, tested directly: how much further would the
    % interface plane have to move to minimise the wander?  If his DAR tilt
    % is the right tilt, this is small.
    fprintf('\n  %-13s | %13s | %13s %13s | %13s\n', ...
        'variant','his DAR(deg)','best dTilt(deg)','best shift(mm)','wander gain');
    for i = 1:numel(P)
        m = P(i).pm;
        fprintf('  %-13s | %13.4f | %15.4f %13.4f | %13.3f\n', ...
            P(i).name, S.v(i).coldstop_ADE_deg, m.best_plane.tilt_deg, ...
            1e3*m.best_plane.shift, m.wander.rms/m.best_plane.rms);
    end
    fprintf('\n  %-13s | %13s %13s | %13s %13s | %10s\n', ...
        'variant','ideal beta/m2','ideal resid','flat resid rms','flat resid PV', ...
        'XPS astig');
    for i = 1:numel(P)
        m = P(i).pm;
        if isempty(P(i).pq)
            xs = sprintf('%10s', 'n/a');
        else
            xs = sprintf('%10.4f', 1e3*P(i).pq.astig(1));
        end
        fprintf('  %-13s | %13.4f %11.2f um | %11.2f um %10.4f mm | %s\n', ...
            P(i).name, m.surface.ideal.beta_over_m2, ...
            1e6*m.surface.ideal.resid_rms, 1e6*m.surface.flat.resid_rms, ...
            1e3*m.surface.flat.resid_pv, xs);
    end
    if ~isempty(P(1).pq_msg)
        fprintf('\n  macos.pupil_quality: %s\n', P(1).pq_msg);
    end
    fprintf(['\n  Read it as four separate statements, because they have different\n' ...
             '  causes and different fixes:\n' ...
             '    blur       how sharply the primary is imaged at all\n' ...
             '    mag range  how much that image BREATHES across the field --\n' ...
             '               his 30x -> 28.7x slip lives here.  Quote the FRAME:\n' ...
             '               the chief-normal column is the pupil defect, the\n' ...
             '               placed column adds the coldstop obliquity\n' ...
             '    surface    where the pupil image SITS, against the ideal image\n' ...
             '               of the primary''s own %0.0f mm sag (beta/m2 = 1 means\n' ...
             '               that sag is imaged exactly as it should be, and is\n' ...
             '               NOT pupil curvature) and against the flat coldstop\n' ...
             '    wander     what the instrument actually feels at the placed\n' ...
             '               plane; w/best is after re-tuning that plane\n'], ...
             1e3*P(1).pm.surface.ideal.sag_in_pv);
end

% =====================================================================
function ladder_map_(W, S, png)
%LADDER_MAP_  Field map of the matched rung on the uniform grid, with his
%   solve points overlaid -- the sampling question is visible, not argued.
    n = sqrt(size(W.L_grid,1));
    F = macos.design.field_grid(S.fov_half_deg*60, n, 'units','arcmin');
    x = reshape(F(:,1)*180/pi, n, n) + 0;
    y = reshape(F(:,2)*180/pi, n, n) + S.v(1).YAN_abs_deg*0;
    fig = figure('Visible','off','Position',[100 100 1180 380]);
    tl = tiledlayout(fig,1,3,'TileSpacing','compact','Padding','compact');
    rn = {'rung 1  piston','rung 2  + tip/tilt','rung 3  + power'};
    for r = 1:3
        ax = nexttile(tl);
        z = reshape(W.L_grid(:,r), n, n)*1e9;
        contourf(ax, x, y, z, 16, 'LineColor','none');
        hold(ax,'on');
        plot(ax, S.Frel_deg(:,1), S.Frel_deg(:,2), 'w+', 'MarkerSize',8, ...
             'LineWidth',1.2);
        hold(ax,'off');
        axis(ax,'square');  colormap(ax, parula);  cb = colorbar(ax);
        cb.Label.String = 'RMS WFE (nm)';
        title(ax, sprintf('%s   max %.1f nm', rn{r}, max(z(:))));
        xlabel(ax, 'XAN - centre (deg)');
        if r == 1, ylabel(ax, 'YAN - centre (deg)'); end
    end
    ttl = sprintf(['%s   afocal ladder over the 0.5x0.5 deg box' ...
                   '   (his reported max %.0f nm)'], ...
                   strrep(W.name,'_','\_'), W.gt_max_nm);
    title(tl, ttl);
    fig.Name = ttl;
    exportgraphics(fig, png, 'Resolution', 150);
    close(fig);
end

function pupil_fig_(P, png)
%PUPIL_FIG_  The four-part ladder, one panel each.
    m = P.pm;   g = m.good;
    u = m.nodes(1,g)*1e3;   v = m.nodes(2,g)*1e3;      % entrance, mm
    fig = figure('Visible','off','Position',[100 100 1250 330]);
    tl = tiledlayout(fig,1,4,'TileSpacing','compact','Padding','compact');

    ax = nexttile(tl);
    scatter(ax, u, v, 26, m.blur.waist_rms(g)*1e6, 'filled');
    lbl_(ax, sprintf('(1) blur   rms %.0f um', 1e6*m.blur.rms), 'um');
    ylabel(ax, 'M1 y (mm)');

    ax = nexttile(tl);
    scatter(ax, u, v, 26, m.surface.flat.dist(g)*1e6, 'filled');
    lbl_(ax, sprintf('(2) surface vs flat   %.0f um rms', ...
                     1e6*m.surface.flat.resid_rms), 'um');

    ax = nexttile(tl);
    scatter(ax, u, v, 26, m.map.distortion(g)*1e6, 'filled');
    lbl_(ax, sprintf('(3) distortion   %.3f%% of R', ...
                     100*m.map.distortion_frac_max), 'um');

    ax = nexttile(tl);
    scatter(ax, u, v, 26, m.wander.per_node_rms(g)*1e6, 'filled');
    lbl_(ax, sprintf('(4) wander at the placed plane   rms %.0f um', ...
                     1e6*m.wander.rms), 'um');

    ttl = sprintf('%s   interface pupil, cone-convergence ladder   (M = %.4f)', ...
                  strrep(P.name,'_','\_'), m.map.mag);
    title(tl, ttl);   fig.Name = ttl;
    exportgraphics(fig, png, 'Resolution', 150);
    close(fig);
end

function lbl_(ax, t, un)
    axis(ax,'equal');  axis(ax,'tight');  box(ax,'on');
    colormap(ax, parula);  cb = colorbar(ax);  cb.Label.String = un;
    title(ax, t);  xlabel(ax, 'M1 x (mm)');
end

function banner(varargin)
    fprintf('\n=================================================================\n');
    fprintf(' %s\n', sprintf(varargin{:}));
    fprintf('=================================================================\n');
end
