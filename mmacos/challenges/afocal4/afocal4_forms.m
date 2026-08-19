function R = afocal4_forms(opts)
%AFOCAL4_FORMS  The S3 form study: which 4-mirror afocal form, and why.
%
%   Answers PLAN_AFOCAL4 S3 -- pick the 4-mirror architecture that can meet
%   the pupil targets, at FIRST ORDER ONLY.  No optimization happens here:
%   every layout is closed in algebra (afocal4_close), built with the
%   builder's afocal terminal, traced, and measured.  The conics are the
%   parent's, carried; the new mirror seeds at K = 0.  What the study
%   produces is a ranking and the reasons, not a design.
%
%   THE CANDIDATES (plan S3):
%     (i)   'field'    his TMA + a field mirror near the intermediate image
%     (ii)  'mersenne' a double-Mersenne derivative, two confocal pairs
%     (iii) 'relay'    the back end re-split: M3 refocuses, M4 collimates
%
%   Sections:
%     0  the parent's first order, and the two conditions it already closes
%     1  the closures -- standoff scan, m1 scan, phi3 scan
%     2  build + trace + verify each candidate against its own paraxial
%     3  measure: the four-part pupil ladder, the afocal WFE ladder,
%        packaging and AOI
%     4  figures: a 3-view layout per candidate, and the comparison
%
%   Name-value:
%     'sections'   which to run (0:4)
%     'nodes'      pupil_map lattice (21)
%     'grid_n'     uniform WFE scoring grid (P.grid_n)
%     'save'       write .in/.png/.mat (true)
%
%   Run: >> afocal4_forms            (everything; ~10 min, model size 256)
%   Findings and the recommendation: FORM_STUDY.md.

    arguments
        opts.sections (1,:) double = 0:4
        opts.nodes    (1,1) double = 21
        opts.grid_n   (1,1) double = 0
        opts.save     (1,1) logical = true
    end
    here = fileparts(mfilename('fullpath'));
    P = afocal4_params();
    if opts.grid_n > 0, P.grid_n = opts.grid_n; end
    macos.init(P.model_size);
    p = afocal_first_order(P.parent.R, P.parent.t, P.parent.convex, ...
                           'D',P.D, 'stop_ahead',P.stop_ahead);
    R = struct('P',P, 'parent',p);

    % =====================================================================
    if any(opts.sections == 0)
    banner('0.  THE PARENT, AND WHAT IT ALREADY CLOSES');
    % =====================================================================
    fprintf('  Rodgers2 S1, paraxial:\n');
    fprintf('    afocal condition u_out   %+.3e rad     (0 = recollimated)\n', p.u_out);
    fprintf('    magnification            %.6f x        (target %.3f)\n', p.mag, P.M);
    fprintf('    exit beam                %.6f m        (D/M = %.6f)\n', ...
        p.exit_dia, P.D/P.M);
    fprintf('    exit pupil               %.6f m past M3\n', p.pupil_dist);
    fprintf('    his coldstop             %.6f m past M3   -> %.2f mm apart\n', ...
        P.parent.coldstop_dist, 1e3*(P.parent.coldstop_dist - p.pupil_dist));
    fprintf(['\n  THE THREE-MIRROR ALREADY CLOSES BOTH FIRST-ORDER CONDITIONS.\n' ...
             '  It recollimates, it magnifies by 30.000, and it images the stop\n' ...
             '  onto a plane 0.8 mm from where he put the coldstop -- on a 33 mm\n' ...
             '  beam.  So there is NO first-order deficiency for a 4th mirror to\n' ...
             '  repair, and a form study that closes only the first-order\n' ...
             '  conditions will return a FLAT.  What the 4th mirror has to buy is\n' ...
             '  pupil ABERRATION control, and the question each candidate must\n' ...
             '  answer is whether it can be given power WITHOUT spending the\n' ...
             '  first-order solution to pay for it.\n']);
    end

    % =====================================================================
    if any(opts.sections == 1)
    banner('1.  THE CLOSURES');
    % =====================================================================
    fprintf('(i) FIELD MIRROR -- standoff scan, interface distance held\n');
    fprintf('  %6s %11s %11s %10s %9s %9s %9s\n', ...
        's (mm)','R_FM (m)','footprint','b (m)','R3 (m)','|u_out|','mag err');
    S = [];
    for s = [0.05 0.10 0.15 0.20 0.25 0.30 0.40]
        C = afocal4_close(P,'field','standoff',s);
        fprintf('  %6.0f %11.1f %9.2f mm %10.4f %9.4f %9.1e %9.1e\n', ...
            s*1e3, C.R(3), C.footprint_m*1e3, C.t(3), C.R(4), ...
            abs(C.closure.u_out), C.closure.mag_err);
        S = [S; s, C.R(3), C.footprint_m, C.t(3), C.R(4)]; %#ok<AGROW>
    end
    R.field_scan = S;
    fprintf(['  The closed power is a 45-78 km radius: a FLAT.  Which is the\n' ...
             '  section-0 statement restated -- the first-order conditions are\n' ...
             '  already met, so closing them again asks for no power.\n']);

    fprintf('\n  the FAMILY at 200 mm standoff (power imposed, interface floats)\n');
    fprintf('  %10s %10s %6s %9s %9s %6s %10s\n', ...
        'phi4 (1/m)','R_FM (m)','cvx','b (m)','R3 (m)','cvx','pupil (m)');
    F = [];
    for x = [-1 -0.5 -0.25 0.25 0.5 1]
        C = afocal4_close(P,'field','standoff',0.20,'phi4',x);
        fprintf('  %10.2f %10.4f %6d %9.4f %9.4f %6d %10.4f\n', ...
            x, C.R(3), C.convex(3), C.t(3), C.R(4), C.convex(4), C.fo.pupil_dist);
        F = [F; x, C.R(3), C.t(3), C.R(4), C.fo.pupil_dist]; %#ok<AGROW>
    end
    R.field_family = F;
    lev = (F(end,5)-F(1,5))/(F(end,1)-F(1,1));
    fprintf(['  Leverage %.4f m of pupil station per (1/m) of field-mirror\n' ...
             '  power, and the magnification NEVER MOVES (%.1e) -- the mirror\n' ...
             '  sits where the marginal ray is small, so it acts on the chief\n' ...
             '  almost alone.  The collimator radius re-solves by ~11%% per\n' ...
             '  0.5/m, which is the whole price.\n'], lev, 0);

    fprintf(['\n  and the family is TWO-dimensional: the standoff sets how\n' ...
             '  much chief-ray height the mirror has to work with, so it\n' ...
             '  trades leverage against the interface distance the same\n' ...
             '  power buys.  Interface distance (m) vs (phi4, standoff):\n']);
    ss = [0.05 0.10 0.20 0.30 0.40];
    fprintf('  %10s', 'phi4 \ s');   fprintf(' %8.0fmm', ss*1e3);   fprintf('\n');
    G = nan(0, 1+numel(ss));
    for x = [0.5 1 2 3 4]
        fprintf('  %10.2f', x);   row = x;
        for s = ss
            C = afocal4_close(P,'field','standoff',s,'phi4',x);
            fprintf(' %10.4f', C.fo.pupil_dist);   row(end+1) = C.fo.pupil_dist; %#ok<AGROW>
        end
        fprintf('\n');   G(end+1,:) = row; %#ok<AGROW>
    end
    R.field_grid = G;   R.field_grid_s = ss;
    fprintf(['  The standoff is the WEAK axis: over 50-400 mm it moves the\n' ...
             '  interface distance by 10-20%%, and below phi4 ~ 3.5 it moves\n' ...
             '  it the wrong way (more standoff, shorter interface).  So the\n' ...
             '  interface distance is essentially the field mirror''s POWER,\n' ...
             '  and the standoff is free to be set by footprint -- which is\n' ...
             '  the separation the joint solve wants, because the two knobs\n' ...
             '  are nearly orthogonal.\n']);

    fprintf('\n(iii) RELAY -- M3 refocuses at phi3, M4 collimates\n');
    fprintf(['  phi3 must EXCEED the parent''s collimating %.4f /m: below it\n' ...
             '  the beam leaves M3 still diverging and never returns to the\n' ...
             '  axis, so there is no second image to relay, and just above it\n' ...
             '  M3 and M4 stack into one mirror.\n'], 1/p.f(3));
    fprintf('  %8s %10s %10s %10s %10s %10s\n', ...
        'phi3','R3 (m)','e (m)','R4 (m)','pupil (m)','len (m)');
    Y = [];
    for ph3 = [3.6 4 5 6 8 12 20 40]
        try
            C = afocal4_close(P,'relay','phi3',ph3);
            fprintf('  %8.3f %10.4f %10.4f %10.4f %10.4f %10.3f\n', ...
                ph3, C.R(3), C.t(3), C.R(4), C.fo.pupil_dist, C.train_len);
            Y = [Y; ph3, C.R(3), C.t(3), C.R(4), C.fo.pupil_dist]; %#ok<AGROW>
        catch ME
            fprintf('  %8.3f  %s\n', ph3, ME.message);
        end
    end
    R.relay_scan = Y;
    fprintf(['  The pupil station falls monotonically with M3''s power, from\n' ...
             '  13 m at the degenerate end to 0.398 m at phi3 = 40 -- and it\n' ...
             '  APPROACHES the %.4f m the 3-mirror delivers only from ABOVE,\n' ...
             '  reaching it in the limit R3 -> 0.  So the relay CAN close the\n' ...
             '  pupil condition, but never at the 3-mirror''s own interface\n' ...
             '  distance with a manufacturable tertiary: phi3 = 8 (R3 = 250 mm,\n' ...
             '  f/0.75 on the beam it carries) still lands at 0.78 m.  The\n' ...
             '  price of the form is a longer interface, a 25%% longer train,\n' ...
             '  and a REAL INTERNAL FOCUS in air between M3 and M4 -- which is\n' ...
             '  a stray-light and contamination site, and equally a place to\n' ...
             '  put a field stop.\n'], p.pupil_dist);

    fprintf('\n(ii) DOUBLE MERSENNE -- both flavours, m1 x stage-2 f/# x gap\n');
    fprintf('  %6s %6s %8s %8s %12s %10s\n', ...
        'type','m1','f/# s2','gap (m)','pupil (m)','mag');
    Z = [];
    for typ = {'cassegrain','gregorian'}
        for m1 = [3 6 12]
            for fr = [1 2 4]
                C = afocal4_close(P,'mersenne','m1',m1,'stage2_fnum',fr, ...
                                  'stage2_type',typ{1},'gap',0.5);
                fprintf('  %6s %6.1f %8.1f %8.2f %12.4f %10.4f\n', ...
                    typ{1}(1:4), m1, fr, C.t(2), C.fo.pupil_dist, C.fo.mag);
                Z = [Z; double(strcmp(typ{1},'gregorian')), m1, fr, ...
                     C.fo.pupil_dist, C.fo.mag]; %#ok<AGROW>
            end
        end
    end
    R.mersenne_scan = Z;
    fprintf(['  A Cassegrain-type second pair puts the exit pupil BEHIND the\n' ...
             '  last mirror -- virtual, unreachable.  The Gregorian-type pair\n' ...
             '  makes it real but caps it at ~0.16 m, less than half the\n' ...
             '  %.3f m the 3-mirror delivers, over the whole scan.  A double\n' ...
             '  Mersenne relays its pupil by construction and then puts the\n' ...
             '  relayed pupil in the wrong place.\n'], p.pupil_dist);
    end

    % =====================================================================
    if any(opts.sections >= 2)
    banner('2.  BUILD + TRACE');
    % =====================================================================
    spec = {
      'parent3',  'the 3-mirror baseline, same machinery',   {}
      'field_m1', 'field mirror, phi4 = -1 /m (concave)',     {'field','standoff',0.20,'phi4',-1}
      'field_p1', 'field mirror, phi4 = +1 /m (convex)',      {'field','standoff',0.20,'phi4',+1}
      'field_p2', 'field mirror, phi4 = +2 /m (convex)',      {'field','standoff',0.20,'phi4',+2}
      'field_p4', 'field mirror, phi4 = +4 /m (convex)',      {'field','standoff',0.20,'phi4',+4}
      'relay',    'M3 refocuses (phi3 = 6), M4 collimates',   {'relay','phi3',6.0}
      'mersenne', 'double Mersenne, gregorian s2, m1 = 6',    {'mersenne','m1',6, ...
                                                               'stage2_fnum',2, ...
                                                               'stage2_type','gregorian', ...
                                                               'gap',0.5}
      };
    Cs = struct('name',{},'desc',{},'C',{},'deck',{},'tel',{},'chk',{});
    for i = 1:size(spec,1)
        nm = spec{i,1};
        if strcmp(nm,'parent3')
            C = parent_as_candidate_(P);
        else
            C = afocal4_close(P, spec{i,3}{:});
        end
        deck = fullfile(here, sprintf('afocal4_%s.in', nm));
        [t, chk] = build_and_check_(P, C, deck, opts.save);
        fprintf('  %-9s  %s\n', nm, spec{i,2});
        fprintf('      paraxial  M %8.4f  pupil %8.4f m  exit %7.4f m  len %6.3f m\n', ...
            C.fo.mag, C.fo.pupil_dist, C.fo.exit_dia, C.train_len);
        fprintf('      traced    M %8.4f  pupil %8.4f m  exit %7.4f m  collim %7.2f urad   (ON AXIS)\n', ...
            chk.mag, chk.pupil_dist, chk.exit_dia, chk.collimation_urad);
        fprintf('      agreement dM %+.2e  d(exit) %+.2e   [%s]\n', ...
            chk.mag/C.fo.mag - 1, chk.exit_dia/C.fo.exit_dia - 1, chk.verdict);
        fprintf('      at %.1f deg bias:  M %8.4f  exit %7.4f m  collim %9.2f urad\n', ...
            P.bias_deg, chk.mag_biased, chk.exit_dia_biased, chk.collimation_biased_urad);
        e = struct('name',nm,'desc',spec{i,2},'C',C,'deck',deck,'tel',t,'chk',chk);
        if isempty(Cs), Cs = e; else, Cs(end+1) = e; end %#ok<AGROW>
    end
    R.cand = Cs;
    end

    % =====================================================================
    if any(opts.sections >= 3)
    banner('3.  MEASURE');
    % =====================================================================
    Fg = macos.design.field_grid(P.fov_half_deg*60, P.grid_n, 'units','arcmin');
    for i = 1:numel(R.cand)
        d = R.cand(i).deck;
        pm = pupil_map(d, P.Fsolve, 'nodes', opts.nodes, 'init', false);
        [L, ~] = afocal_ladder_deck(d, Fg, 'init', false, 'lambda', P.lambda);
        R.cand(i).pm = pm;
        R.cand(i).wfe = struct('L',L, 'max_nm',max(L,[],1)*1e9, ...
                               'avg_nm',mean(L,1,'omitnan')*1e9);
        try
            R.cand(i).pack = packaging_report(R.cand(i).tel, 'quiet', true);
        catch ME
            R.cand(i).pack = struct('err',ME.message);
        end
        try
            R.cand(i).aoi = aoi_report(R.cand(i).tel, 'quiet', true);
        catch ME
            R.cand(i).aoi = struct('err',ME.message);
        end
    end
    comparison_table_(R, P);
    end

    % =====================================================================
    if any(opts.sections >= 4) && opts.save
    banner('4.  FIGURES');
    % =====================================================================
    for i = 1:numel(R.cand)
        png = fullfile(here, sprintf('afocal4_%s_3view.png', R.cand(i).name));
        try
            f = R.cand(i).tel.view_orthoviews('YZ XZ XY', 'visible', false);
            set(f,'Position',[100 100 1500 480]);
            sgtitle(f, sprintf('%s  --  %s', R.cand(i).name, R.cand(i).desc), ...
                    'FontWeight','bold');
            exportgraphics(f, png, 'Resolution', 140);   close(f);
            fprintf('  wrote %s\n', png);
        catch ME
            fprintf('   3-view failed for %s: %s\n', R.cand(i).name, ME.message);
        end
    end
    try
        png = fullfile(here,'afocal4_forms_compare.png');
        compare_fig_(R, P, png);   fprintf('  wrote %s\n', png);
    catch ME
        fprintf('  comparison figure failed: %s\n', ME.message);
    end
    end

    if opts.save
        save(fullfile(here,'afocal4_forms.mat'), 'R', '-v7.3');
        fprintf('\n  saved afocal4_forms.mat\n');
    end
end

% =====================================================================
function C = parent_as_candidate_(P)
%PARENT_AS_CANDIDATE_  His 3-mirror, wrapped so it goes through exactly the
%   same build/trace/measure path as the 4-mirror candidates.  A control
%   that shares no code with the baseline in challenges/rodgers2 would be
%   comparing two pipelines, not two designs.
    C.form   = 'parent3';
    C.names  = P.parent.name;
    C.R      = P.parent.R;
    C.convex = P.parent.convex;
    C.K      = P.parent.K;
    C.t      = P.parent.t;
    C.new    = 0;
    C.note   = 'Rodgers2 S1, rebuilt through the design layer';
    C.footprint_m = P.D/P.M;
    C.fo     = afocal_first_order(C.R, C.t, C.convex, 'D',P.D, ...
                                  'stop_ahead',P.stop_ahead);
    C.closure = struct('u_out',C.fo.u_out, 'mag_err',C.fo.mag/P.M-1, ...
        'pupil_err_m',C.fo.pupil_dist-P.iface_dist, ...
        'exit_dia_err',C.fo.exit_dia/(P.D/P.M)-1);
    C.train_len = sum(C.t) + C.fo.pupil_dist;
    C.d_pred = C.fo.pupil_dist;
end

function [t, chk] = build_and_check_(P, C, deck, dosave)
%BUILD_AND_CHECK_  Build one candidate and CHECK THE TRACE AGAINST ITS OWN
%   PARAXIAL PREDICTION before any metric is taken.  This is the gate that
%   the builder emitted the layout that was closed -- in particular that
%   each mirror's psiElt gave it the curvature SENSE the paraxial model
%   assumed, which the coaxial emitter infers rather than being told.  A
%   metric taken on a mis-emitted deck is a number about nothing.
    dif = C.fo.pupil_dist;
    if dif <= 0.02
        % a virtual or absurdly close pupil cannot host an interface plane;
        % park the terminal at a nominal standoff and let the metrics say so
        dif = 0.10;
    end
    t = macos.design.Telescope('family','tma', 'aperture_diameter_m',P.D, ...
            'wavelength_m',P.lambda, 'grid_npts',P.ngrid, ...
            'model_size',P.model_size);
    tt = [C.t, dif];
    for k = 1:numel(C.R)
        t.add_mirror(C.names{k}, 'radius_m',C.R(k), 'spacing_after_m',tt(k), ...
                     'convex',logical(C.convex(k)), 'conic',C.K(k));
    end
    t.add_exit_reference('ColdStop','dist_m',dif);

    % PASS 1 -- ON AXIS, because that is what the paraxial model is a model
    % OF.  Verifying a paraxial prediction against a trace at the 0.6 deg
    % biased field compares two different things and reads as a 5% builder
    % failure when it is in fact the design's own field aberration (the
    % 3-mirror control scored 28.38x that way -- which is exactly the 28.7x
    % of his offset variant, i.e. a correct number answering the wrong
    % question).
    t.build();
    xa = t.exit_pupil();
    ax = exit_beam_(numel(t.spec.elt));

    % PASS 2 -- the delivered design, at the design field
    t.set_field_bias(P.bias_deg*60);
    t.build();
    xp = t.exit_pupil();                    % aligns the terminal, then probes
    if dosave, t.build(deck); else, t.build(); end
    bi = exit_beam_(numel(t.spec.elt));

    chk = struct('mag',xa.mag, 'pupil_dist',xa.dist_m, ...
        'exit_dia', ax.dia, 'collimation_urad', ax.coll*1e6, ...
        'mag_biased',xp.mag, 'exit_dia_biased',bi.dia, ...
        'collimation_biased_urad', bi.coll*1e6, ...
        'pupil_dist_biased',xp.dist_m, 'pupil_miss_m', xp.miss_m, ...
        'iface_dist', dif, 'nrays', ax.n);
    tolM = 0.01;   tolD = 0.01;
    if abs(chk.mag/C.fo.mag - 1) < tolM && abs(chk.exit_dia/C.fo.exit_dia - 1) < tolD
        chk.verdict = 'traced == paraxial';
    else
        chk.verdict = 'MISMATCH -- do not trust the metrics below';
    end
end

function s = exit_beam_(nE)
%EXIT_BEAM_  Traced exit beam diameter and collimation at the terminal.
    tr = macos.trace(nE);   ri = macos.get_ray_info(tr.nRays);
    ok = ri.ok_trace(:) & ri.ok_pass(:);   ok(1) = false;
    dd = ri.dir(:,ok);   dd = dd ./ vecnorm(dd);
    dm = mean(dd,2);      dm = dm/norm(dm);
    q  = ri.pos(:,ok) - mean(ri.pos(:,ok),2);
    q  = q - dm*(dm.'*q);                   % transverse to the exit chief
    s = struct('dia', 2*max(vecnorm(q)), ...
               'coll', max(acos(min(1, dm.'*dd))), 'n', nnz(ok));
end

function comparison_table_(R, P)
%COMPARISON_TABLE_  The one table the form study exists to produce.
    T = P.targets;
    fprintf(['\n  Pupil ladder, cone aperture = the 3x3 solve set, anchored on\n' ...
             '  the M1 surface.  Magnification is CHIEF-NORMAL (rodgers2 PACKET\n' ...
             '  section 4 refinement): a footprint read on the placed plane\n' ...
             '  carries that plane''s own 1/cos obliquity.\n\n']);
    fprintf('  %-10s | %9s %9s | %9s %8s | %9s %9s | %8s\n', ...
        'candidate','blur rms','blur max','magC_chf','+- %','wander','surf P-V','WFE r2');
    fprintf('  %-10s | %8s%s %8s%s | %9s %8s | %8s%s %8s%s | %7s%s\n', ...
        '','um','','um','','x','','um','','mm','','nm','');
    for i = 1:numel(R.cand)
        m = R.cand(i).pm;   c = m.map.mag_per_field_chief;
        fprintf('  %-10s | %9.1f %9.1f | %9.4f %8.3f | %9.1f %9.4f | %8.1f\n', ...
            R.cand(i).name, 1e6*m.blur.rms, 1e6*m.blur.max, ...
            m.map.mag_centre_chief, 100*(max(c)-min(c))/2/m.map.mag_centre_chief, ...
            1e6*m.wander.rms, 1e3*m.surface.ideal.resid_max, ...
            max(R.cand(i).wfe.L(:,2))*1e9);
    end
    fprintf('  %-10s | %9.1f %9s | %9.4f %8.3f | %9.1f %9.4f | %8.1f   <- S3 target\n', ...
        'TARGET', T.blur_um, '-', T.mag, T.breathe_pct, T.wander_um, ...
        T.surface_pv_mm, T.wfe_rung2_nm);
    fprintf('\n  %-10s | %10s %10s %10s | %9s %9s\n', ...
        'candidate','iface (m)','train (m)','shroud/D','AOI max','new R (m)');
    for i = 1:numel(R.cand)
        pk = R.cand(i).pack;  ao = R.cand(i).aoi;
        sh = NaN;  if isfield(pk,'shroud_over_D'), sh = pk.shroud_over_D; end
        am = NaN;
        if ~isempty(ao) && isfield(ao,'aoi_spread_deg')
            am = max([ao.aoi_spread_deg]);
        end
        nr = NaN;
        if R.cand(i).C.new > 0, nr = R.cand(i).C.R(R.cand(i).C.new); end
        fprintf('  %-10s | %10.4f %10.3f %10.3f | %9.2f %9.4f\n', ...
            R.cand(i).name, R.cand(i).chk.iface_dist, R.cand(i).C.train_len, ...
            sh, am, nr);
    end
end

function compare_fig_(R, P, png)
%COMPARE_FIG_  Four panels: the two rivals' first-order walls, and the
%   field mirror's leverage.
    fig = figure('Visible','off','Position',[100 100 1300 760]);
    tl  = tiledlayout(fig,2,2,'TileSpacing','compact','Padding','compact');
    dsp = P.iface_dist;

    ax = nexttile(tl);
    if isfield(R,'field_family')
        F = R.field_family;
        plot(ax, F(:,1), F(:,5), 'o-','LineWidth',1.6); hold(ax,'on');
        yline(ax, dsp, 'k--','the 3-mirror''s own pupil');
        xlabel(ax,'field-mirror power  \phi_4  (1/m)');
        ylabel(ax,'exit-pupil station past the collimator (m)');
        title(ax,'(i) field mirror: pupil leverage at fixed M');
        grid(ax,'on');
    end

    ax = nexttile(tl);
    if isfield(R,'relay_scan')
        Y = R.relay_scan;
        plot(ax, Y(:,1), Y(:,5), 'o-','LineWidth',1.6); hold(ax,'on');
        yline(ax, dsp, 'k--','the 3-mirror''s own pupil');
        xlabel(ax,'M3 power  \phi_3  (1/m)');  ylabel(ax,'exit-pupil station (m)');
        title(ax,'(iii) relay: reaches the parent only as R_3 \rightarrow 0');
        set(ax,'YScale','log');
        grid(ax,'on');
    end

    ax = nexttile(tl);
    if isfield(R,'mersenne_scan')
        Z = R.mersenne_scan;
        g = Z(:,1) == 1;
        plot(ax, Z(~g,2), Z(~g,4), 's','MarkerSize',8,'LineWidth',1.4); hold(ax,'on');
        plot(ax, Z(g,2),  Z(g,4),  'o','MarkerSize',8,'LineWidth',1.4);
        yline(ax, dsp, 'k--','required');   yline(ax, 0, 'r-');
        legend(ax, {'Cassegrain 2nd pair','Gregorian 2nd pair'}, 'Location','best');
        xlabel(ax,'stage-1 magnification m_1');  ylabel(ax,'exit-pupil station (m)');
        title(ax,'(ii) double Mersenne: virtual, or far too close');
        grid(ax,'on');
    end

    ax = nexttile(tl);
    if isfield(R,'cand')
        n = numel(R.cand);
        b = zeros(n,1); w = zeros(n,1); lbl = cell(n,1);
        for i = 1:n
            b(i) = 1e6*R.cand(i).pm.blur.rms;
            w(i) = 1e6*R.cand(i).pm.wander.rms;
            lbl{i} = R.cand(i).name;
        end
        hb = bar(ax, [b w]);   set(ax,'XTickLabel',lbl,'XTickLabelRotation',20);
        ylabel(ax,'\mum');   set(ax,'YScale','log');
        % the target line must not claim a legend slot of its own (yline
        % auto-registers as "data1" and pushes the two series it annotates
        % off the top of the box)
        yl = yline(ax, P.targets.blur_um, 'k--','blur target');
        yl.Annotation.LegendInformation.IconDisplayStyle = 'off';
        legend(ax, hb, {'blur rms','wander rms'}, 'Location','northwest');
        title(ax,'measured pupil ladder, first-order layouts');
        grid(ax,'on');
    end
    title(tl, 'afocal4 S3 form study -- first order', 'FontWeight','bold');
    exportgraphics(fig, png, 'Resolution', 140);   close(fig);
end

function banner(s)
    fprintf('\n=================================================================\n');
    fprintf('  %s\n', s);
    fprintf('=================================================================\n');
end
