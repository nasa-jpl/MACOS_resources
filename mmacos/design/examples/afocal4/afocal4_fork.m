function R = afocal4_fork(opts)
%AFOCAL4_FORK  The 343 mm fork, re-scored under the RIM pupil convention.
%
%   R = AFOCAL4_FORK() answers the S4c brief's third task.  At 343 mm -- the
%   only interface standoff where the S4b package closes around a real
%   instrument -- there are two buildable designs and the choice between them
%   costs a factor of 40 in wavefront error.  That fork was scored with the
%   cones anchored on the PRIMARY'S SURFACE and the blur and wander averaged
%   over the WHOLE pupil.  Dave's 2026-08-04 ruling says a coldstop metric
%   should be neither: the pupil object is the RIM of the primary, and what
%   the coldstop cares about is the EDGE of the pupil image.  So the question
%   is whether the physically-correct metric SOFTENS the fork.
%
%   Sections:
%     0  THE PREMISE, MEASURED.  The ruling rests on rim- and pole-conjugate
%        planes being resolvably different: the pupil-imaging depth of focus
%        lambda/(2*NA_field^2) has to be SMALLER than the primary's own sag
%        imaged at m^2.  Both numbers come out of PUPIL_MAP, so the premise
%        is checked rather than repeated.  The rim/stop identity is checked
%        here too: on Rodgers' decks the declared stop plane IS the rim
%        plane, and that is a measurement (~1e-13 mm), not a coincidence
%        anyone has to take on trust.
%     1  THE LADDER IN BOTH CONVENTIONS, deck by deck: surface-anchored
%        full-aperture (the S4/S4b numbers, unchanged) beside rim-anchored,
%        with the rim ZONE reported separately.
%     2  THE FORK.  Both branches at 343 mm, in both conventions, and the
%        exchange rate each one implies.
%     3  THE RIM-WEIGHTED RE-SOLVE (with 'resolve',true).  If the rim metric
%        softens the fork, the powered branch is re-solved against it -- same
%        log-domain merit, rim-zone blur and wander replacing the full-
%        aperture ones -- and the new exchange rate is reported.
%     4  RODGERS' OWN LADDER in both conventions, for the PACKET.
%
%   Name-value:
%     'nodes'   pupil_map lattice (P.solve.nodes_score)
%     'zone'    rim-zone width as a fraction of the pupil radius (0.10)
%     'resolve' 'off' (default) | 'run' (solve it -- an hour) | 'load'
%               (score the committed rim-solved deck)
%     'context' run section 4 (true)
%     'save'    write afocal4_fork.mat + the re-solved deck (true)
%
%   See also PUPIL_MAP, AFOCAL4_SCORE, AFOCAL4_BASIN2, AFOCAL4_S4B.

    arguments
        opts.nodes   (1,1) double  = 0
        opts.zone    (1,1) double  = 0.10
        opts.resolve (1,:) char {mustBeMember(opts.resolve, ...
                     {'off','run','load'})} = 'off'
        opts.context (1,1) logical = true
        opts.save    (1,1) logical = true
    end
    here = fileparts(mfilename('fullpath'));
    P = afocal4_params();
    if opts.nodes <= 0, opts.nodes = P.solve.nodes_score; end
    macos.init(P.model_size);
    R = struct('P',P, 'when',datestr(now,31), 'zone',opts.zone); %#ok<TNOW1,DATST>
    matf = fullfile(here,'afocal4_fork.mat');

    % the fork, and its controls.  The flat-M4 branch IS his three-mirror, so
    % his parent is carried beside it: if the two do not read alike, the
    % claim "declining pupil control returns you to the telescope Rodgers
    % already has" is the thing that is wrong.
    D = {
      'basin 1, flat M4 (343)',   'afocal4_b_trade_343mm.in'
      'basin 2, powered M4 (343)','afocal4_b2_trade_343mm.in'
      'basin 2, LONG-SOLVED',     'afocal4_b2long_343mm.in'
      'his 3-mirror parent',      'afocal4_parent3.in'
      };
    keep = cellfun(@(f) isfile(fullfile(here,f)), D(:,2));
    if ~all(keep)
        for i = find(~keep(:).')
            fprintf('  (%s missing -- skipped)\n', D{i,2});
        end
    end
    D = D(keep,:);

    banner('0.  THE PREMISE, MEASURED');
    fprintf(['  The rim-vs-pole choice is only real if the pupil imager can\n' ...
             '  RESOLVE the difference: the depth of focus of the pupil image,\n' ...
             '  lambda/(2*NA_field^2), against the primary''s own sag imaged at\n' ...
             '  the longitudinal magnification m^2.\n\n']);
    fprintf('  %-28s %10s %10s %12s %12s %10s\n', 'deck', 'NA_field', ...
            'DoF um', 'sag P-V mm', 'imaged um', 'resolved?');
    Pre = struct('name',{},'deck',{},'dof_um',{},'imaged_um',{},'na',{});
    for i = 1:size(D,1)
        f = fullfile(here, D{i,2});
        o = pupil_map(f, P.Fsolve, 'nodes',opts.nodes, 'init',false);
        dof = o.lambda/(2*max(o.diffraction.NA_field,realmin)^2);
        img = o.surface.ideal.sag_imaged_pv;
        fprintf('  %-28s %10.4g %10.2f %12.4f %12.2f %10s\n', D{i,1}, ...
                o.diffraction.NA_field, dof*1e6, o.surface.ideal.sag_in_pv*1e3, ...
                img*1e6, yn_(dof < img));
        Pre(end+1) = struct('name',D{i,1}, 'deck',f, 'dof_um',dof*1e6, ...
                            'imaged_um',img*1e6, ...
                            'na',o.diffraction.NA_field); %#ok<AGROW>
    end
    R.premise = Pre;

    % ---- the rim/stop identity, on the deck that declares a stop --------
    ad = fullfile(fileparts(fileparts(here)),'rodgers2','rodgers2_S1_onaxis.in');
    if isfile(ad)
        orm = pupil_map(ad, P.Fsolve, 'anchor','rim', 'nodes',opts.nodes, ...
                        'init',false);
        fprintf(['\n  RIM = HIS DECLARED STOP, measured on rodgers2_S1: rim sag ' ...
                 '%.6f mm at\n  r = %.4f mm (%s edge); his stop sits %.6f mm ' ...
                 'ahead of the pole, i.e.\n  %.3e mm from the rim plane.  ' ...
                 'Analytic sag of R = -2500 mm at r = 500 mm: %.6f mm.\n'], ...
                orm.rim.sag*1e3, orm.rim.radius*1e3, orm.rim.edge, ...
                orm.rim.stop_offset*1e3, orm.rim.stop_minus_rim*1e3, ...
                1e3*orm.rim.radius^2/(2*2.5));
        R.identity = orm.rim;
    end

    % =====================================================================
    banner('1.  THE LADDER IN BOTH CONVENTIONS');
    % =====================================================================
    T = struct('name',{},'deck',{},'S',{},'surf',{},'rim',{});
    for i = 1:size(D,1)
        f = fullfile(here, D{i,2});
        S  = afocal4_score(P, f, 'nodes',opts.nodes, 'grid',P.grid_n);
        os = pupil_map(f, P.Fsolve, 'nodes',opts.nodes, 'init',false, ...
                       'rim_zone',opts.zone);
        or = pupil_map(f, P.Fsolve, 'nodes',opts.nodes, 'init',false, ...
                       'anchor','rim', 'rim_zone',opts.zone);
        T(end+1) = struct('name',D{i,1}, 'deck',f, 'S',S, ...
                          'surf',os, 'rim',or); %#ok<AGROW>
        print_pair_(D{i,1}, S, os, or, opts.zone);
    end
    R.table = T;
    if opts.save, save(matf,'R','-v7.3'); end

    % =====================================================================
    banner('2.  THE FORK');
    % =====================================================================
    k1 = find(strcmp({T.name}, 'basin 1, flat M4 (343)'), 1);
    k2 = find(strcmp({T.name}, 'basin 2, LONG-SOLVED'), 1);
    if isempty(k2), k2 = find(strcmp({T.name},'basin 2, powered M4 (343)'),1); end
    if ~isempty(k1) && ~isempty(k2)
        R.fork = fork_table_(T(k1), T(k2), opts.zone);
    end
    if opts.save, save(matf,'R','-v7.3'); end

    % =====================================================================
    if ~strcmp(opts.resolve,'off')
    banner('3.  THE RIM-WEIGHTED RE-SOLVE of the powered branch');
    % =====================================================================
    fprintf(['  Same merit, same DOFs, same sampling; the blur and wander\n' ...
             '  terms now score the outer %.0f%% of the pupil, anchored on the\n' ...
             '  rim.  What changes is what the solve is being asked for.\n\n'], ...
            100*opts.zone);
    Q = P;
    Q.pupil = struct('anchor','rim', 'zone','rim', 'rim_zone',opts.zone);
    Q.solve.fd_step = 3e-4;   Q.solve.fd_type = 'forward';
    Q.solve.tol_fun = 1e-8;   Q.solve.tol_x = 1e-9;   Q.solve.tol_opt = 1e-8;
    Q.solve.max_fev = 500;
    [D0, src] = seed_design_(here);
    dk = fullfile(here,'afocal4_b2rim_343mm.in');
    if strcmp(opts.resolve,'load') && ~isfile(dk)
        fprintf('  no %s to load -- section skipped\n', dk);
    elseif isempty(D0) && ~strcmp(opts.resolve,'load')
        fprintf('  no design struct for the powered branch -- section skipped\n');
    else
        if strcmp(opts.resolve,'run')
            fprintf('  seeded from %s\n', src);
            if ~opts.save, dk = [tempname '.in']; end
            s = afocal4_solve(Q, D0, 'dofs',{'conic','standoff','front'}, ...
                              'deck',dk, 'label','rim-weighted 343 mm', ...
                              'max_iter',5000);
            R.resolve = s;
        else
            fprintf('  loading the committed rim-solved deck %s\n', dk);
            R.resolve = struct('deck',dk, 'loaded',true);
        end
        % and score the RESULT in BOTH conventions, because a design solved
        % against one metric has to be quotable in the other
        Sr_rim  = afocal4_score(P, dk, 'nodes',opts.nodes, 'grid',P.grid_n, ...
                                'anchor','rim', 'zone','rim');
        Sr_surf = afocal4_score(P, dk, 'nodes',opts.nodes, 'grid',P.grid_n);
        R.resolve_scores = struct('rim',Sr_rim, 'surf',Sr_surf);
        fprintf(['\n  rim-solved design: WFE %.1f nm | rim-zone blur %.1f um, ' ...
                 'wander %.1f um\n                     full-aperture blur ' ...
                 '%.1f um, wander %.1f um, breathing %.4f%%\n'], ...
                Sr_surf.wfe_max_nm, Sr_rim.blur_um, Sr_rim.wander_um, ...
                Sr_surf.blur_um, Sr_surf.wander_um, Sr_surf.breathe_pct);
        if ~isempty(k1)
            fprintf(['  exchange rate against the flat-M4 branch (%.1f nm): ' ...
                     '%.1fx in wavefront\n'], T(k1).S.wfe_max_nm, ...
                    Sr_surf.wfe_max_nm/T(k1).S.wfe_max_nm);
        end
    end
    if opts.save, save(matf,'R','-v7.3'); end
    end

    % =====================================================================
    if opts.context
    banner('4.  RODGERS'' OWN LADDER, both conventions (for the PACKET)');
    % =====================================================================
    rd = fullfile(fileparts(fileparts(here)),'rodgers2');
    C = struct('name',{},'surf',{},'rim',{});
    for c = {{'S1 on-axis','rodgers2_S1_onaxis.in'}, ...
             {'S2 offset','rodgers2_S2_offset.in'}, ...
             {'S3 newconics','rodgers2_S3_newconics.in'}, ...
             {'S4 tilt/dec','rodgers2_S4_tiltdec.in'}}
        f = fullfile(rd, c{1}{2});
        if ~isfile(f), continue; end
        os = pupil_map(f, P.Fsolve, 'nodes',opts.nodes, 'init',false, ...
                       'rim_zone',opts.zone);
        or = pupil_map(f, P.Fsolve, 'nodes',opts.nodes, 'init',false, ...
                       'anchor','rim', 'rim_zone',opts.zone);
        C(end+1) = struct('name',c{1}{1}, 'surf',os, 'rim',or); %#ok<AGROW>
    end
    fprintf(['  %-14s %9s %9s %9s %7s | %9s %9s %9s | %9s %7s %7s\n'], ...
            'variant', 'blur surf','blur rim','rim zone','edge x', ...
            'wand surf','wand rim','zone wand','M rim','DoF um','imgd um');
    for i = 1:numel(C)
        dof = C(i).surf.lambda/(2*max(C(i).surf.diffraction.NA_field,realmin)^2);
        fprintf(['  %-14s %9.1f %9.1f %9.1f %7.3f | %9.1f %9.1f %9.1f | ' ...
                 '%9.4f %7.2f %7.2f\n'], ...
            C(i).name, C(i).surf.blur.rms*1e6, C(i).rim.blur.rms*1e6, ...
            C(i).rim.rim_zone.blur_rms*1e6, ...
            C(i).rim.rim_zone.blur_rms/C(i).rim.blur.rms, ...
            C(i).surf.best_plane.rms*1e6, ...
            C(i).rim.best_plane.rms*1e6, C(i).rim.rim_zone.wander_best_rms*1e6, ...
            C(i).rim.map.mag_centre_chief, dof*1e6, ...
            C(i).surf.surface.ideal.sag_imaged_pv*1e6);
    end
    fprintf(['  (DoF is lambda/(2*NA_field^2); the lambda/NA^2 form Dave ' ...
             'quoted is twice it.)\n']);
    R.context = C;
    end

    if opts.save, save(matf,'R','-v7.3');  fprintf('\n  saved %s\n', matf); end
end

% =====================================================================
function print_pair_(name, S, os, or, zf)
%PRINT_PAIR_  One deck, both conventions, side by side.  The full-aperture
%   columns under the two anchors answer "does the object plane matter?";
%   the rim-zone column answers "does looking at the EDGE matter?"; and they
%   are different questions, so they are printed as different columns.
    fprintf('\n  %s\n    WFE rung 2 max %.1f nm (grid max %.1f nm)\n', ...
            name, S.wfe_max_nm, val_(S,'wfe_grid_max_nm'));
    fprintf('    %-26s %12s %12s %12s\n', '', 'SURFACE anch', 'RIM anch', ...
            sprintf('RIM zone %.0f%%', 100*zf));
    row = @(n, a, b, c) fprintf('    %-26s %12.4f %12.4f %12.4f\n', n, a, b, c);
    row('blur rms (um)',      os.blur.rms*1e6,        or.blur.rms*1e6, ...
                              or.rim_zone.blur_rms*1e6);
    row('blur max (um)',      os.blur.max*1e6,        or.blur.max*1e6, ...
                              or.rim_zone.blur_max*1e6);
    row('wander, refit (um)', os.best_plane.rms*1e6,  or.best_plane.rms*1e6, ...
                              or.rim_zone.wander_best_rms*1e6);
    row('wander, placed (um)',os.wander.rms*1e6,      or.wander.rms*1e6, ...
                              or.rim_zone.wander_rms*1e6);
    row('breathing (%)',      breathe_(os),           breathe_(or), NaN);
    row('M centre, chief',    os.map.mag_centre_chief,or.map.mag_centre_chief, NaN);
    row('surface vs ideal (mm)', os.surface.ideal.resid_max*1e3, ...
                              or.surface.ideal.resid_max*1e3, NaN);
    row('distortion (% of R)',100*os.map.distortion_frac_max, ...
                              100*or.map.distortion_frac_max, NaN);
    row('anchoring resid (um)', os.anchor.resid_max*1e6, ...
                              or.anchor.resid_max*1e6, NaN);
end

function F = fork_table_(a, b, zf)
%FORK_TABLE_  The one row-pair the study turns on, in both conventions, and
%   the exchange rate each convention implies.  A ratio is quoted for every
%   column so that "softens" is a number rather than an impression.
    fprintf(['  At 343 mm, both buildable: a fourth mirror gone to a FLAT ' ...
             '(his three-mirror)\n  against a real fourth mirror.  The ' ...
             'wavefront ratio is the fork.\n\n']);
    fprintf('  %-24s %12s %12s %10s\n', 'metric', 'flat M4', 'powered M4', ...
            'ratio');
    r = @(n,x,y) fprintf('  %-24s %12.4f %12.4f %10.2fx\n', n, x, y, y/x);
    fprintf('  %-24s %12.1f %12.1f %10.2fx\n', 'WFE rung 2 (nm)', ...
            a.S.wfe_max_nm, b.S.wfe_max_nm, b.S.wfe_max_nm/a.S.wfe_max_nm);
    fprintf('  -- SURFACE anchor, full aperture (the S4b convention) --\n');
    r('blur rms (um)',        a.surf.blur.rms*1e6,       b.surf.blur.rms*1e6);
    r('wander refit (um)',    a.surf.best_plane.rms*1e6, b.surf.best_plane.rms*1e6);
    r('breathing (%)',        breathe_(a.surf),          breathe_(b.surf));
    fprintf('  -- RIM anchor, full aperture --\n');
    r('blur rms (um)',        a.rim.blur.rms*1e6,        b.rim.blur.rms*1e6);
    r('wander refit (um)',    a.rim.best_plane.rms*1e6,  b.rim.best_plane.rms*1e6);
    r('breathing (%)',        breathe_(a.rim),           breathe_(b.rim));
    fprintf('  -- RIM anchor, rim zone (outer %.0f%%) --\n', 100*zf);
    r('blur rms (um)',        a.rim.rim_zone.blur_rms*1e6, ...
                              b.rim.rim_zone.blur_rms*1e6);
    r('wander refit (um)',    a.rim.rim_zone.wander_best_rms*1e6, ...
                              b.rim.rim_zone.wander_best_rms*1e6);
    F = struct('flat',a, 'powered',b, ...
        'wfe_ratio', b.S.wfe_max_nm/a.S.wfe_max_nm, ...
        'blur_ratio_surface', b.surf.blur.rms/a.surf.blur.rms, ...
        'blur_ratio_rim',     b.rim.blur.rms/a.rim.blur.rms, ...
        'blur_ratio_rimzone', b.rim.rim_zone.blur_rms/a.rim.rim_zone.blur_rms, ...
        'breathe_ratio_surface', breathe_(b.surf)/breathe_(a.surf), ...
        'breathe_ratio_rim',     breathe_(b.rim)/breathe_(a.rim), ...
        'edge_penalty_flat',    a.rim.rim_zone.blur_rms/a.rim.blur.rms, ...
        'edge_penalty_powered', b.rim.rim_zone.blur_rms/b.rim.blur.rms);
    % THE QUESTION IS ASKED IN TWO PARTS AND THEY HAVE DIFFERENT ANSWERS, so
    % the verdict is printed as numbers rather than as a word.  (1) Does the
    % flat branch's EDGE look better than its average?  (2) Does the powered
    % branch's requirement relax?
    fprintf(['\n  ANCHOR vs ZONE.  Moving the object plane from the pole to ' ...
             'the rim changes the\n  blur by %.3f%% (flat) and %.3f%% ' ...
             '(powered) -- nothing.  Looking at the EDGE\n  instead of the ' ...
             'average changes it by %.0f%% and %.0f%%.  Everything the rim ' ...
             'convention\n  does to these numbers, it does through the ZONE.' ...
             '\n'], ...
            100*(a.rim.blur.rms/a.surf.blur.rms - 1), ...
            100*(b.rim.blur.rms/b.surf.blur.rms - 1), ...
            100*(F.edge_penalty_flat - 1), 100*(F.edge_penalty_powered - 1));
    fprintf(['\n  So: the fourth mirror buys %.1fx of aperture-average pupil ' ...
             'blur and %.1fx at the\n  rim, for %.1fx of wavefront either ' ...
             'way.  The flat branch''s edge is %.0f%% WORSE\n  than its own ' ...
             'average, not better, and still %.0fx its target.  The rim ' ...
             'metric takes\n  %.0f%% off what the fourth mirror buys and ' ...
             'adds nothing to doing without it.\n'], ...
            1/F.blur_ratio_surface, 1/F.blur_ratio_rimzone, F.wfe_ratio, ...
            100*(F.edge_penalty_flat - 1), a.rim.rim_zone.blur_rms*1e6/47, ...
            100*(1 - F.blur_ratio_surface/F.blur_ratio_rimzone));
end

function b = breathe_(o)
    c = o.map.mag_per_field_chief;
    b = 100*(max(c)-min(c))/2/o.map.mag_centre_chief;
end

function [D, src] = seed_design_(here)
%SEED_DESIGN_  The powered branch's design struct, preferring the long solve.
%   Which one it came from is RETURNED and printed, because a re-solve is
%   only comparable to the design it started from.
    D = [];   src = '';
    for f = {'afocal4_basin2_343mm.mat','afocal4_basin2.mat','afocal4_s4b.mat'}
        p = fullfile(here, f{1});
        if ~isfile(p), continue; end
        q = load(p);
        if isfield(q,'R') && isfield(q.R,'pt') && ~isempty(q.R.pt)
            k = find(abs([q.R.pt.iface] - 0.343) < 1e-9, 1);
            if ~isempty(k)
                D = q.R.pt(k).D;  src = [f{1} ' (long-solved)'];  return;
            end
        end
        if isfield(q,'R') && isfield(q.R,'trade2') && ~isempty(q.R.trade2)
            k = find(abs([q.R.trade2.iface] - 0.343) < 1e-9, 1);
            if ~isempty(k)
                D = q.R.trade2(k).D;  src = [f{1} ' (S4b sweep)'];  return;
            end
        end
    end
end

function v = val_(S, f)
    if isfield(S,f), v = S.(f); else, v = NaN; end
end
function s = yn_(b),  if b, s = 'YES'; else, s = 'no'; end,  end
function v = ternary_(c,a,b),  if c, v = a; else, v = b; end,  end

function banner(s)
    fprintf('\n%s\n%s\n%s\n', repmat('=',1,74), s, repmat('=',1,74));
end
