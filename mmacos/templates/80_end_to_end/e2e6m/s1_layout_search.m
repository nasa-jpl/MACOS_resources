function OUT = s1_layout_search(opts)
%S1_LAYOUT_SEARCH  Pick the 6 m unobscured 0th-order layout for e2e6m.
%
%   The e2e6m front end is a sphere+Zernike (freeform) unobscured
%   telescope -- the `freeform_unobscured` template's topology: three
%   base SPHERES placed for packaging, fold TILTS doing the
%   unobscuration, all correction living in Zernike departures that do
%   not move the chief ray.  The template's committed instance is
%   D = 8 m; this search picks the 0th-order layout for OUR instrument
%   against the campaign's three hard numbers:
%
%     f/# at the FP     in [12, 20]   (Dave 2026-08-24, the OTA band)
%     shroud diameter   <= 8 m        (deployed, diameter-only:
%                                      packaging_report's radial extent
%                                      about the incoming-beam axis)
%     clearance         UNOBSCURED    (check_clipping: every body clear
%                                      of every foreign beam)
%
%   and reports the coronagraph AOI preference (< 15 deg spread per
%   mirror) as a soft score, since it trades directly against
%   compactness.
%
%   Why this designer and not the brief's offset_imager: the field-offset
%   form has an APERTURE CEILING.  Enumerating every first-order root of
%   that three-mirror chain at EPD 6 m (bisection, not the template's
%   Newton, so no root is missed) over L1 6-18 m, |R1|/L1 1.9-3.2,
%   L3/L1 0.2-2.0 and F/12-20 under EFL-exact + Petzval = 0: the compact
%   real-focus roots all have |R2| = 2.6-7 m, while unobscuration needs
%   >= ~8 deg of offset, which walks the beam 1.7+ m at M2 -- off the
%   edge of the sphere, so the offset box does not trace at all before
%   decenters open at S4.  (And the rodgers3 form itself has the
%   marginal ray at 2.837 entrance RADII on M3, i.e. a 17 m M3 at any
%   form-true 6 m scaling.)  Ratified by Dave 2026-08-24.
%
%   APERTURES ARE OFF throughout (Dave's 2026-07-06 doctrine, and the
%   open realize_apertures frame defect: footprint centres measured in
%   global XY but emitted as local ApVec, so a saved tilted-fold .in
%   loses every ray on reload).  Apertures enter the arc later, via the
%   S2 segmentation machinery for the PM and aperture_full_field for the
%   rest of the train.
%
%   OUT = S1_LAYOUT_SEARCH() runs the default two-stage search and
%   writes s1_layout_search.txt + .mat beside this file.
%
%   See also FREEFORM_UNOBSCURED, PACKAGING_REPORT, AOI_REPORT,
%   macos.design.Telescope/check_clipping.

    arguments
        opts.D_m        (1,1) double = 6.0
        opts.lambda_m   (1,1) double = 500e-9
        opts.fno_band   (1,2) double = [12 20]
        opts.shroud_D_m (1,1) double = 8.0
        opts.model      (1,1) double = 256
        opts.gridn      (1,1) double = 21
        opts.R1_m       (1,:) double = [38.4 44.0]        % f/3.2, f/3.67
        opts.T1_m       (1,:) double = [13 15 17]
        opts.T2_m       (1,:) double = [13 16 19 22]
        opts.R2_m       (1,:) double = [5.6 6.4 7.2]
        opts.R3_m       (1,:) double = [1.9 2.25 2.6]
        opts.tilt_deg   (:,3) double = [-7.2 8.46 12.0]
        opts.efl_target (1,1) double = NaN   % NaN = R2 is a grid axis;
                                             % else R2 is SOLVED (secant
                                             % on the live EFL) so the
                                             % f/# gate is met by
                                             % construction and the grid
                                             % spends its budget on the
                                             % packaging knobs
        opts.stage2     (1,1) logical = true
        opts.outdir     (1,:) char   = ''
        opts.verbose    (1,1) logical = true
    end
    here = fileparts(mfilename('fullpath'));
    setup_(here);   % NOT run() in this scope: this function has nested
                    % functions, so its workspace is STATIC and run()
                    % cannot introduce mmacos_setup's variables into it
    if isempty(opts.outdir), opts.outdir = here; end

    L = {};
    say = @(varargin) push_(varargin{:});
    function push_(varargin)
        s = sprintf(varargin{:});
        L{end+1} = s; %#ok<AGROW>
        if opts.verbose, fprintf('%s\n', s); end
    end

    say('==== e2e6m S1 layout search ====');
    say('D = %.2f m | lambda = %g nm | f/# band [%g %g] | shroud <= %.1f m', ...
        opts.D_m, opts.lambda_m*1e9, opts.fno_band, opts.shroud_D_m);
    say('apertures OFF (design doctrine + realize_apertures frame defect)');

    % ---- stage 1: geometry grid at the heritage tilts ---------------------
    [A,B,C,E,F] = ndgrid(opts.R1_m, opts.T1_m, opts.T2_m, opts.R2_m, opts.R3_m);
    cand = [A(:) B(:) C(:) E(:) F(:)];
    say('\nstage 1: %d geometries x %d tilt triples', ...
        size(cand,1), size(opts.tilt_deg,1));
    R1 = [];
    for q = 1:size(opts.tilt_deg,1)
        Rq = score_grid_(cand, opts.tilt_deg(q,:), opts, say);
        if isempty(R1), R1 = Rq; else, R1 = [R1, Rq]; end %#ok<AGROW>
    end

    % ---- stage 2: tilt refinement on the stage-1 winners -------------------
    R2s = R1([]);
    if opts.stage2
        keep = rank_(R1, opts);
        nk = min(4, numel(keep));
        say('\nstage 2: tilt refinement on the %d best stage-1 geometries', nk);
        for i = 1:nk
            g = R1(keep(i));
            base = [g.tilt1 g.tilt2 g.tilt3];
            T = tilt_grid_(base);
            cg = repmat([g.R1 g.T1 g.T2 g.R2 g.R3], size(T,1), 1);
            for j = 1:size(T,1)
                r = score_one_(cg(j,:), T(j,:), opts);
                if isempty(R2s), R2s = r; else, R2s(end+1) = r; end %#ok<AGROW>
                if r.ok_all, say('%s', row_(r)); end
            end
        end
    end

    ALL = R1;
    if ~isempty(R2s), ALL = [R1, R2s]; end
    ord = rank_(ALL, opts);
    say('\n==== ranked survivors (meet f/# band + shroud + unobscured) ====');
    say('%s', hdr_());
    for i = 1:min(12, numel(ord))
        say('%s', row_(ALL(ord(i))));
    end
    if isempty(ord)
        say('NO candidate meets all three gates -- widen the grid.');
    else
        w = ALL(ord(1));
        say('\nWINNER: R = [%.4f %.4f %.4f] m, T = [%.4f %.4f] m, tilt = [%.3f %.3f %.3f] deg', ...
            w.R1, w.R2, w.R3, w.T1, w.T2, w.tilt1, w.tilt2, w.tilt3);
        say('  EFL %.3f m -> f/%.2f | shroud %.3f m (%.3f D) | train %.2f m | AOI spread max %.1f deg', ...
            w.EFL, w.fno, w.shroud_m, w.shroud_over_D, w.train_m, w.aoi_max);
    end

    txt = strjoin(L, newline);
    fid = fopen(fullfile(opts.outdir,'s1_layout_search.txt'),'w');
    fprintf(fid,'%s\n',txt); fclose(fid);
    OUT = struct('all',ALL, 'rank',ord, 'opts',opts, 'text',txt, ...
                 'when',datestr(now,31)); %#ok<TNOW1,DATST>
    save(fullfile(opts.outdir,'s1_layout_search.mat'),'OUT');
    fprintf('\nwrote s1_layout_search.{txt,mat}\n');
end

% =========================================================================
function R2 = solve_R2_(g, tilt, opts)
%SOLVE_R2_  Secant-solve the M2 radius so the LIVE EFL hits the target.
%   R2 is the strongest EFL lever in this topology and the weakest
%   packaging lever, so it is the right variable to spend on the f/#
%   gate -- the grid then buys packaging with the tilts and spacings.
%   Returns the seed unchanged if the secant will not close (the caller
%   scores the resulting f/# and the gate rejects it honestly).
    f = @(R2) efl_of_(g, R2, tilt, opts) - opts.efl_target;
    a = g(4);  b = g(4)*1.05;
    fa = f(a);  fb = f(b);
    for it = 1:25
        if ~isfinite(fa) || ~isfinite(fb) || abs(fb-fa) < 1e-12, break; end
        c = b - fb*(b-a)/(fb-fa);
        if ~isfinite(c) || c <= 0.5 || c > 100, break; end
        a = b;  fa = fb;  b = c;  fb = f(b);
        if abs(fb) < 1e-3*opts.efl_target, break; end
    end
    R2 = b;
    if ~isfinite(R2) || R2 <= 0.5 || R2 > 100, R2 = g(4); end
end

function E = efl_of_(g, R2, tilt, opts)
    E = NaN;
    try
        t = macos.design.Telescope('family','TMA', ...
            'aperture_diameter_m',opts.D_m, 'model_size',opts.model, ...
            'wavelength_m',opts.lambda_m, 'grid_npts',11);
        t.set_base_sphere(true);
        t.add_mirror('M1','radius_m',g(1),'spacing_after_m',g(2),'tilt_deg',tilt(1));
        t.add_mirror('M2','radius_m',R2,'spacing_after_m',g(3), ...
                     'tilt_deg',tilt(2),'convex',true);
        t.add_mirror('M3','radius_m',g(5),'spacing_after','derive','tilt_deg',tilt(3));
        t.add_focal_plane('FP');
        t.build();
        E = efl_(t, numel(t.spec.elt));
    catch
    end
end

function setup_(here)
    run(fullfile(here,'..','..','..','mmacos_setup.m'));
end

function R = score_grid_(cand, tilt, opts, say)
    R = [];
    say('%s', hdr_());
    for i = 1:size(cand,1)
        r = score_one_(cand(i,:), tilt, opts);
        if isempty(R), R = r; else, R(end+1) = r; end %#ok<AGROW>
        say('%s', row_(r));
    end
    say('  (%d of %d meet all three gates)', nnz([R.ok_all]), numel(R));
end

function r = score_one_(g, tilt, opts)
%SCORE_ONE_  Build one 0th-order layout and measure it.  Never throws:
%   a geometry that will not build is a scored failure, not a crash.
    r = struct('R1',g(1),'T1',g(2),'T2',g(3),'R2',g(4),'R3',g(5), ...
               'tilt1',tilt(1),'tilt2',tilt(2),'tilt3',tilt(3), ...
               'EFL',NaN,'fno',NaN,'shroud_m',NaN,'shroud_over_D',NaN, ...
               'train_m',NaN,'aoi_max',NaN,'nray_frac',NaN,'wfe0_waves',NaN, ...
               'r_elt',nan(1,4),'y_m2',NaN,'y_m3',NaN, ...
               'nobs',NaN,'obs_by','','margin_min',NaN, ...
               'clear_ok',false,'ok_fno',false,'ok_shroud',false,'ok_all',false, ...
               'note','');
    try
        if isfinite(opts.efl_target)
            g(4) = solve_R2_(g, tilt, opts);
            r.R2 = g(4);
        end
        t = macos.design.Telescope('family','TMA', ...
            'aperture_diameter_m',opts.D_m, 'model_size',opts.model, ...
            'wavelength_m',opts.lambda_m, 'grid_npts',opts.gridn);
        t.set_base_sphere(true);
        t.add_mirror('M1','radius_m',g(1),'spacing_after_m',g(2),'tilt_deg',tilt(1));
        t.add_mirror('M2','radius_m',g(4),'spacing_after_m',g(3), ...
                     'tilt_deg',tilt(2),'convex',true);
        t.add_mirror('M3','radius_m',g(5),'spacing_after','derive','tilt_deg',tilt(3));
        t.add_focal_plane('FP');
        t.build();
        nE = numel(t.spec.elt);
        s  = macos.trace(nE);
        ri = macos.get_ray_info(s.nRays);
        r.nray_frac = nnz(logical(ri.ok_pass) & logical(ri.ok_trace))/s.nRays;
        if r.nray_frac < 0.99, r.note = 'rays lost'; return; end
        W = macos.opd();  v = W(isfinite(W) & W~=0 & abs(W)<1e30);
        r.wfe0_waves = std(v)/opts.lambda_m;

        r.EFL = efl_(t, nE);
        r.fno = r.EFL/opts.D_m;

        pk = packaging_report(t,'quiet',true);
        r.shroud_m = 2*pk.shroud_radius_m;
        r.shroud_over_D = pk.shroud_over_D;
        r.train_m = pk.length_m;
        re = max([pk.elts.r_body], [pk.elts.r_beam]);
        r.r_elt(1:min(4,numel(re))) = re(1:min(4,numel(re)));
        r.y_m2 = t.spec.elt(2).Vpt(2);
        r.y_m3 = t.spec.elt(3).Vpt(2);

        rep = t.check_clipping('noload',true,'quiet',true);
        r.clear_ok = all([rep.ok]);
        r.nobs = sum([rep.obstructs]);
        r.margin_min = min([rep.margin]);
        bad = {rep([rep.obstructs] > 0).name};
        r.obs_by = strjoin(bad, '+');

        ao = aoi_report(t,'quiet',true);
        r.aoi_max = max([ao.aoi_spread_deg]);

        r.ok_fno    = r.fno >= opts.fno_band(1) && r.fno <= opts.fno_band(2);
        r.ok_shroud = r.shroud_m <= opts.shroud_D_m;
        r.ok_all    = r.ok_fno && r.ok_shroud && r.clear_ok;
    catch ME
        r.note = regexprep(ME.message,'\s+',' ');
        if numel(r.note) > 30, r.note = r.note(1:30); end
    end
end

function E = efl_(t, fk)
%EFL_  Live EFL: chief-ray image displacement per field angle (the
%   design_report measure -- the Seidel seed is unreliable on a folded
%   convex-secondary reimager).
    del = 0.1*pi/180/60;
    t.trace_at_field([0  del]);  s = macos.trace(fk);
    b1 = macos.get_ray_info(s.nRays);
    t.trace_at_field([0 -del]);        macos.trace(fk);
    b2 = macos.get_ray_info(s.nRays);
    t.trace_at_field([]);
    E = norm(b1.pos(:,1) - b2.pos(:,1))/(2*del);
end

function T = tilt_grid_(base)
%TILT_GRID_  Local refinement about a stage-1 tilt triple.  The fold
%   tilts are what buy the unobscuration, so they are swept last and
%   locally: a big tilt change re-poses the whole chain.
    d1 = base(1) + [-1.5 -0.75 0 0.75 1.5];
    d2 = base(2) + [-1.5 -0.75 0 0.75 1.5];
    d3 = base(3) + [-2 -1 0 1 2];
    [A,B,C] = ndgrid(d1,d2,d3);
    T = [A(:) B(:) C(:)];
end

function idx = rank_(R, opts)
%RANK_  Survivors first, tightest shroud first, AOI spread as tiebreak.
    ok = [R.ok_all];
    idx = find(ok);
    if isempty(idx), return; end
    key = [R(idx).shroud_over_D].' + 0.001*[R(idx).aoi_max].';
    [~,o] = sort(key);
    idx = idx(o);
end

function h = hdr_()
    h = sprintf(['  %8s %6s %6s %6s %6s | %7s %7s %6s | %7s %6s %6s %6s %5s' ...
                 ' | %6s %6s %6s %6s %6s %6s'], ...
                'R1','T1','T2','R2','R3','tilt1','tilt2','tilt3', ...
                'EFL','f/#','shroud','train','AOI', ...
                'rM1','rM2','rM3','rFP','yM2','yM3');
end

function s = row_(r)
    s = sprintf(['  %8.3f %6.2f %6.2f %6.3f %6.3f | %7.2f %7.2f %6.2f | ' ...
                 '%7.2f %6.2f %6.3f %6.2f %5.1f | %6.2f %6.2f %6.2f %6.2f ' ...
                 '%6.2f %6.2f%s'], ...
                r.R1, r.T1, r.T2, r.R2, r.R3, r.tilt1, r.tilt2, r.tilt3, ...
                r.EFL, r.fno, r.shroud_m, r.train_m, r.aoi_max, ...
                r.r_elt(1), r.r_elt(2), r.r_elt(3), r.r_elt(4), ...
                r.y_m2, r.y_m3, flags_(r));
end

function s = flags_(r)
%FLAGS_  Which gate a row failed, so a barren grid says WHY.
    if ~isempty(r.note), s = ['  x ' r.note];  return; end
    s = '';
    if ~r.ok_fno,    s = [s ' fno'];    end
    if ~r.ok_shroud, s = [s ' shroud']; end
    if ~r.clear_ok
        if r.nobs > 0
            s = [s sprintf(' OBSC(%s)', r.obs_by)];
        else
            s = [s sprintf(' AP(margin %+.2f)', r.margin_min)];
        end
    end
    if isempty(s),   s = '  <== MEETS'; else, s = ['  x' s]; end
end

function s = tern_(c,a,b), if c, s = a; else, s = b; end, end
