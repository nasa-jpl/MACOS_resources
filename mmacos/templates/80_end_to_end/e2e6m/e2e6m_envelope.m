function OUT = e2e6m_envelope(opts)
%E2E6M_ENVELOPE  Shroud-aware envelope screen for the 6 m offset imager.
%
%   The S1 envelope of an offset_imager instance is a DESIGNER INPUT
%   (z_m1, spacings, seed_R1, offset), and at EPD = 6 m it is what
%   decides whether the observatory fits the campaign's 8 m shroud gate.
%   The template README's feasibility screen (tan(offset) x |M1->stop|
%   >= 1.5 x EPD) is a CONSERVATIVE advisory compiled from a failure
%   14x below it; the physical requirement is only that M2's glass clear
%   the incoming beam.  So: measure, do not argue.
%
%   OUT = E2E6M_ENVELOPE() sweeps candidate envelopes and reports, per
%   candidate, the first-order solve (EFL exact, Petzval = 0), the
%   TRACED footprint geometry of the seed design at the offset box, and
%   two packaging numbers:
%
%     shroud_D_m   diameter of the minimum enclosing circle of every
%                  footprint point and beam-leg sample, projected on the
%                  plane normal to the SHROUD AXIS (= +z, the entry
%                  boresight -- stated, not assumed)
%     shroud_L_m   z extent of the same point set
%
%   plus oi_clear's signed clearance floor for the survivors.  The seed
%   is SPHERES with no tilts/decenters: S4/S5 will move the geometry, so
%   this is a first cut that scopes the envelope, not the S3 shroud gate.
%
%   Name-value:
%     'Fno','offset_deg','leg_m','R1_frac','L3_frac'  sweep axes (vectors)
%     'EPD_m','lambda_m','box_deg'                 fixed instrument
%     'clear_top'   how many survivors get the oi_clear pass (default 8)
%     'shroud_D_max'  gate used to rank survivors (default 8 m)
%
%   'R1_frac' is |R1| / |M1->stop|: the primary radius in units of the
%   first leg (f1 = R1_frac*L1/2, so R1_frac = 2 puts M2 exactly at the
%   primary focus).  rodgers3 sits at 12.2 -- a nearly powerless primary
%   with an M2 almost as large as M1, the form that does not scale to
%   6 m.  'L3_frac' is |M2->M3| / |M1->stop|.
%
%   See also OI_SEED, OI_CLOSE, OI_CLEAR, OFFSET_IMAGER_PARAMS.

    arguments
        opts.Fno            (1,:) double = [12 15 20]
        opts.offset_deg     (1,:) double = [12 16 22.5 30]
        opts.leg_m          (1,:) double = [8 10 12 14 16]
        opts.R1_frac        (1,:) double = [2.2 2.3 2.4]
        opts.L3_frac        (1,:) double = [0.7 0.9 1.1]
        opts.stop_frac      (1,:) double = 0.05
        opts.EPD_m          (1,1) double = 6.0
        opts.lambda_m       (1,1) double = 500e-9
        opts.box_deg        (1,2) double = [2/60 2/60]
        opts.clear_top      (1,1) double = 8
        opts.shroud_D_max   (1,1) double = 8.0
        opts.model          (1,1) double = 256
        opts.sampling       (1,1) double = 21
        opts.outdir         (1,:) char   = ''
    end

    here = fileparts(mfilename('fullpath'));
    run(fullfile(here,'..','..','..','mmacos_setup.m'));
    oidir = fullfile(here,'..','..','10_telescopes','offset_imager');
    addpath(oidir);
    if isempty(opts.outdir), opts.outdir = here; end
    macos.init(opts.model);

    % ---- build the candidate list ----------------------------------------
    [F,O,L,Fi,L3,SF] = ndgrid(opts.Fno, opts.offset_deg, opts.leg_m, ...
                              opts.R1_frac, opts.L3_frac, opts.stop_frac);
    n = numel(F);
    fprintf('\nE2E6M ENVELOPE SCREEN -- %d candidates, EPD %.2f m, lambda %g nm\n', ...
            n, opts.EPD_m, opts.lambda_m*1e9);
    fprintf('shroud axis = +z (entry boresight); gate D <= %.1f m\n\n', ...
            opts.shroud_D_max);

    C = [];
    fprintf('%5s %5s %6s %6s %5s %5s | %7s %8s %8s %8s %8s %8s\n', ...
            'idx','F#','off','L1','R1f','L3f','F/1','sep','r_fp','shrD','shrL','note');
    for i = 1:n
        c = struct('Fno',F(i),'offset_deg',O(i),'leg_m',L(i), ...
                   'R1_frac',Fi(i),'L3_frac',L3(i),'stop_frac',SF(i));
        f1 = Fi(i)*L(i)/2;
        c.f1_m = f1;  c.F1 = f1/opts.EPD_m;
        P = params_(c, opts);
        c.EFL_m = P.EFL_m;
        c.sep_m = 2*tand(abs(P.offset_deg))*c.leg_m;   % M2-beam vs entry-beam
                                                       % centre separation
        try
            X = oi_seed(P);
            X.eliminate = 'R2R3';
            [X, G, fo] = oi_close(X, P, 'offset_deg', P.offset_deg);
            c.R = X.R;
            g = geom_(X, G, P);  g.BFD = fo.BFD_m;
            c.shroud_D_m = g.D;  c.shroud_L_m = g.L;
            c.r_m2_m = g.r(2);   c.r_m3_m = g.r(3);  c.r_fp_m = g.r(4);
            c.BFD_m = g.BFD;
            c.traced = true;  c.note = '';
            c.X = X;  c.G = G;  c.P = P;
        catch ME
            c.R = [NaN NaN NaN];
            c.shroud_D_m = NaN;  c.shroud_L_m = NaN;
            c.r_m2_m = NaN;      c.r_m3_m = NaN;  c.r_fp_m = NaN;
            c.BFD_m = NaN;
            c.traced = false;    c.note = short_(ME.message);
            c.X = [];  c.G = [];  c.P = P;
        end
        if isempty(C), C = c; else, C(end+1) = c; end %#ok<AGROW>
        fprintf('%5d %5g %6.1f %6.1f %5.2f %5.2f | %7.2f %8.3f %8.4f %8.3f %8.2f %8s\n', ...
                i, c.Fno, c.offset_deg, c.leg_m, c.R1_frac, c.L3_frac, c.F1, ...
                c.sep_m, c.r_fp_m, c.shroud_D_m, c.shroud_L_m, c.note);
    end

    % ---- rank the survivors and price their clearance ---------------------
    ok = [C.traced] & ([C.shroud_D_m] <= opts.shroud_D_max);
    fprintf('\n%d / %d candidates trace AND fit D <= %.1f m\n', ...
            nnz(ok), n, opts.shroud_D_max);
    idx = find(ok);
    if ~isempty(idx)
        [~, ord] = sort([C(idx).shroud_D_m]);
        idx = idx(ord);
        nk = min(opts.clear_top, numel(idx));
        fprintf('\nclearance pass on the %d tightest-packing survivors:\n', nk);
        fprintf('%5s %5s %6s %6s %5s %5s | %8s %10s\n', ...
                'idx','F#','off','L1','R1f','L3f','shrD','clear_mm');
        for k = 1:nk
            i = idx(k);
            try
                dmin = oi_clear(C(i).X, C(i).G, C(i).P, C(i).offset_deg);
            catch
                dmin = NaN;
            end
            C(i).clear_mm = dmin*1e3;
            fprintf('%5d %5g %6.1f %6.1f %5.2f %5.2f | %8.3f %10.1f\n', ...
                    i, C(i).Fno, C(i).offset_deg, C(i).leg_m, C(i).R1_frac, ...
                    C(i).L3_frac, C(i).shroud_D_m, C(i).clear_mm);
        end
    end

    OUT = struct('C',C, 'opts',opts, 'when',datestr(now,31)); %#ok<TNOW1,DATST>
    save(fullfile(opts.outdir,'e2e6m_envelope.mat'), 'OUT');
    fprintf('\nsaved %s\n', fullfile(opts.outdir,'e2e6m_envelope.mat'));
end

% =========================================================================
function P = params_(c, opts)
%PARAMS_  Candidate -> offset_imager parameter set.
%   z_m1 is placed so the FOLD sits near z = 0: M1 at +|leg|/2, so the
%   train straddles the origin and the z extent is the honest length.
    leg = abs(c.leg_m);
    P = offset_imager_params(struct( ...
        'name','e2e6m-screen', 'tag','e2e6m_screen', ...
        'EPD_m',opts.EPD_m, 'Fno',c.Fno, 'lambda_m',opts.lambda_m, ...
        'box_deg',opts.box_deg, 'offset_deg',c.offset_deg, ...
        'z_m1_m', leg/2, ...
        'spacings_m', [-c.stop_frac*leg, -(1-c.stop_frac)*leg, c.L3_frac*leg], ...
        'seed_R1_m', -c.R1_frac*leg, ...
        'clear_m',[0.150 0.100], 'exit_dir',[], ...
        'model',opts.model, 'sampling',opts.sampling, ...
        'solve_sampling',opts.sampling, 'outdir',opts.outdir));
end

% =========================================================================
function g = geom_(X, G, P)
%GEOM_  Traced footprints of the seed at the offset box -> shroud metrics.
%   Three fields (box centre + the two YAN extremes), every element, plus
%   the leg samples between them; the entry corridor is carried one leg
%   length upstream of M1 so the incoming beam counts against the shroud.
    tmp = [tempname '.in'];
    cu  = onCleanup(@() del_(tmp));
    D = X;  D.EPD_m = P.EPD_m;  D.WL_m = P.lambda_m;
    D.sampling = P.sampling;  D.name = P.name;
    txt = oi_deck(D);
    by = P.box_deg(2)/2;
    yans = P.offset_deg + [0, -by, +by];
    ie = [1 3 4 5];
    Pall = [];  r = nan(1,4);
    for q = 1:3
        cdir = [0; tand(yans(q)); 1];  cdir = cdir/norm(cdir);
        emit_(txt, tmp, seedpos_(G, cdir), cdir);
        macos.load_rx(tmp);
        if ~macos.has_rx(), error('geom_:load','candidate would not load'); end
        macos.stop(2, [0 0]);
        macos.ray_hist('on');
        tr = macos.trace(macos.num_elt());
        h = macos.ray_hist(tr.nRays);
        macos.ray_hist('off');
        S = cell(1,4);
        for k = 1:4
            m = h.ok(:, ie(k)+1);  m(1) = false;
            if nnz(m) < 5
                error('geom_:lost','field %d lost the beam at station %d', q, ie(k));
            end
            S{k} = h.P(:, m, ie(k)+1);
            r(k) = max(r(k), max(vecnorm(S{k} - mean(S{k},2), 2, 1)));
            Pall = [Pall S{k}]; %#ok<AGROW>
        end
        % entry corridor: one leg length upstream of M1, along -cdir
        span = abs(P.spacings_m(1));
        Pall = [Pall (S{1} - span*cdir)]; %#ok<AGROW>
        % leg samples (10 per leg) so a long diagonal leg counts
        for k = 1:3
            nA = min(size(S{k},2), size(S{k+1},2));
            A = S{k}(:,1:nA);  B = S{k+1}(:,1:nA);
            for s = linspace(0.1,0.9,9)
                Pall = [Pall (A + s*(B-A))]; %#ok<AGROW>
            end
        end
    end
    [~, Dcirc] = minenc_(Pall(1:2,:).');
    g = struct('D', 2*Dcirc, 'L', max(Pall(3,:)) - min(Pall(3,:)), 'r', r);
end

function [c, R] = minenc_(Pxy)
%MINENC_  Minimum enclosing circle of a 2-D point set (Welzl, iterative
%   shrink form -- the point count here is thousands, so the O(n) move-
%   to-front randomized version is overkill; a convex-hull + 1/2/3-point
%   check is exact and fast enough).
    if size(Pxy,1) > 3
        try
            k = convhull(Pxy(:,1), Pxy(:,2));
            Pxy = Pxy(unique(k), :);
        catch
        end
    end
    n = size(Pxy,1);
    c = mean(Pxy,1);  R = max(vecnorm(Pxy - c, 2, 2));
    % 2-point diameters
    for i = 1:n
        for j = i+1:n
            cc = (Pxy(i,:) + Pxy(j,:))/2;
            RR = norm(Pxy(i,:) - cc);
            if RR < R && all(vecnorm(Pxy - cc, 2, 2) <= RR + 1e-9)
                c = cc;  R = RR;
            end
        end
    end
    % 3-point circumcircles
    for i = 1:n
        for j = i+1:n
            for k = j+1:n
                [cc, RR] = circ3_(Pxy(i,:), Pxy(j,:), Pxy(k,:));
                if ~isempty(cc) && RR < R && ...
                        all(vecnorm(Pxy - cc, 2, 2) <= RR + 1e-9)
                    c = cc;  R = RR;
                end
            end
        end
    end
end

function [c, R] = circ3_(a, b, d)
    A = 2*[b(1)-a(1), b(2)-a(2); d(1)-a(1), d(2)-a(2)];
    if abs(det(A)) < 1e-12, c = [];  R = inf;  return; end
    rhs = [sum(b.^2) - sum(a.^2); sum(d.^2) - sum(a.^2)];
    c = (A\rhs).';
    R = norm(a - c);
end

function p = seedpos_(G, cdir)
    cdR = [cdir(1); cdir(2); -cdir(3)];
    tq  = (G.z_m1 - G.stopC(3))/cdir(3);
    q   = G.stopC - tq*cdR;
    p   = q - (oi_standoff(G.EPD_m)/cdir(3))*cdir;
end

function emit_(txt0, tmp, p0, cdir)
    v3 = @(v) sprintf('%.16E  %.16E  %.16E', v(1), v(2), v(3));
    s = regexprep(txt0, '(ChfRayDir=\s*)[^\n]*', ['$1' v3(cdir)]);
    s = regexprep(s,    '(ChfRayPos=\s*)[^\n]*', ['$1' v3(p0)]);
    fid = fopen(tmp,'w');  fprintf(fid,'%s',s);  fclose(fid);
end

function del_(p), if exist(p,'file'), delete(p); end, end

function s = short_(m)
    s = regexprep(m, '\s+', ' ');
    if numel(s) > 8, s = s(1:8); end
end
