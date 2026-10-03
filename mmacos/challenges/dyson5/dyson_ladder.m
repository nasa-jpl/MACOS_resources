function L = dyson_ladder(P, tag, opts)
%DYSON_LADDER  The Dyson departure ladder, rung by rung, under the scorer.
%   L = dyson_ladder(P, tag) solves successive rungs of the Dyson form on the
%   exact chain (spectrometer_geom with straight-ruled grooves, which the
%   engine reproduces ray for ray -- tSpectrometerRx), scoring each iterate
%   with spectrometer_score_chain in PIXEL units, with the smile and
%   keystone operands in the merit FROM THE FIRST PASS (Mouroulis & Green
%   Sec. 5.3's corollary), then emits each rung's deck and scores it in the
%   ENGINE (spectrometer_score).  The ladder is the product; every number
%   is a re-runnable record.
%
%   Rungs (each warm-starts from the previous solution):
%     R0  the concentric seed (s1/s2): block r, R_g at the Dyson condition,
%         face offset, slit offset as in P.
%     R1  concentric knobs: R_g/R_dyson factor and face offset at the
%         seed's block radius ('free_r' adds the radius, which then walks
%         to its bound -- size buys blur by s0's law, not a departure); the
%         groove period and the FPA focus are re-solved inside the chain at
%         every iterate, so the band always spans the FPA.
%     R2  + the block's convex face: conic constant and h^4, h^6 asphere
%         coefficients (Carbon-I's even asphere; engine AsphCoef convention).
%     R3  + the block's centre off the grating's (dz along the axis, dy along
%         the dispersion): the de-concentric departure.
%     R4  + a meniscus corrector in the air gap (the paper's compact variant):
%         vertex z, thickness, two face curvatures; seeded as a concentric
%         null shell.
%
%   Merit (lsqnonlin residual vector, all in pixels over the scoring grid):
%     w_dist * [smile_ij ; keystone_ij]   with smile_ij = v_c(x_i,l_j) - v_c(x_mid,l_j),
%                                         keystone_ij = u_c(x_i,l_j) - u_c(x_i,l_mid)
%     w_blur * [SU_ij ; SV_ij]            the rms spot per point
%   plus a one-sided wall on the slit-to-FPA clearance (>= P.clear_m).
%
%   Returns L.rung(k): .name .vars .x .P (the parameter set), .chain (the
%   chain score), .engine (the engine score), .file (the deck), .merit.
    arguments
        P struct
        tag (1,:) char
        opts.rungs (1,:) double = 0:5
        opts.nx (1,1) double = 5
        opts.nlam (1,1) double = 5
        opts.nring (1,1) double = 4
        opts.w_dist (1,1) double = 10
        opts.w_blur (1,1) double = 1
        opts.clear_m (1,1) double = 3e-3
        opts.max_iter (1,1) double = 60
        opts.free_r (1,1) logical = false
        opts.quiet (1,1) logical = false
        opts.seed = []                         % a parameter set to start from (R5 starts from R4's)
        opts.deck (1,:) char = ''              % deck file name override (one rung)
        opts.bounds = {}                       % {name, lb, ub; ...} overrides of a variable's bounds (the envelope widens the meniscus curvatures)
    end
    base = struct('Fno', P.Fno, 'pixel_m', P.pixel_m, 'npix', P.npix, 'band_m', P.band_m, ...
                  'lambda_ref_m', P.lambda_ref_m, 'order', P.order, 'y_slit', P.y_slit_m, ...
                  'block_r', P.block_r_m, 'glass', P.glass, 'face_offset', P.face_offset_m, ...
                  'Rg_factor', P.Rg_factor, 'grating_model', 'planes', 'slit_px', P.slit_px, ...
                  'block_Kc', 0, 'block_asph', [0 0], 'block_dz', 0, 'block_dy', 0, ...
                  'men_z', 0, 'men_t', 0.010, 'men_ca', 0, 'men_cb', 0);
    for f5 = {'fold_h', 'plate', 'slit_gap', 'fpa_gap'}          % R5's fold prism, when the runner carries it
        if isfield(P, f5{1}), base.(f5{1}) = P.(f5{1}); end
    end
    base.slit_dz = 0;                          % the slit/FPA plane's axial position off the grating's centre (an R5 variable)
    % variable sets per rung: name, lower, upper, scale (the optimizer works in
    % scaled units so every variable is O(1))
    % the block radius is HELD at the seed's unless opts.free_r: freed, the
    % optimizer walks it to its bound and buys blur with size (the h^4/r^3
    % law of s0) -- the ladder's question is what each DEPARTURE buys at a
    % fixed scale
    R1 = {'Rg_factor', 0.90, 1.10, 1;  'face_offset', 1e-4, 5e-3, 1e-3};
    if opts.free_r, R1 = [R1; {'block_r', 0.15, 0.35, 0.1}]; end
    R2 = [R1; {'block_Kc', -2, 2, 0.5;  'asph4', -200, 200, 10;  'asph6', -2e5, 2e5, 1e4}];
    % R3 breaks concentricity: the block's centre leaves the grating's (axial
    % dz, dispersion-direction dy), with the asphere kept open -- the paper's
    % compact variant "operates closer to the concentric-aplanatic condition"
    % with a separate mirror; here the equivalent single-block freedom
    R3 = [R2; {'block_dz', -0.05, 0.05, 1e-2;  'block_dy', -0.03, 0.03, 1e-2}];
    % R4, the paper's COMPACT variant (Fig. 22): a meniscus corrector in the
    % air gap (vertex z_a, thickness t_m, face curvatures c_a, c_b) on top of
    % R3; seeded as a concentric shell (c = 1/z, a null), so R4 starts at R3's
    % merit and departs from there
    % R4 is solved in two steps: (a) the meniscus ALONE (vertex, thickness,
    % two curvatures) from R3's solution, seeded as a shell concentric with
    % the BLOCK's centre (the rays leave the block nearly radially about
    % it, so that shell is the near-null; a shell about the grating's centre
    % is not, once R3 has moved the block); (b) everything together.
    % curvatures of either sign (a plate, a meniscus either way round), the
    % plate down to 2 mm: the first R4 solve sat on the old lower bounds
    % (c = 0.5 /m, t = 4 mm), which is not a solution
    % The R4 landscape is MULTIMODAL (measured, 2026-10-01, three solves from
    % the same R3 seed): with these bounds the solve ends ON them (c = 0.5 /m,
    % t = 4 mm, vertex 0.24 m -- a thin weak plate right after the block) at
    % CRF 1.33 px / EE 0.76; releasing the curvature sign and the plate
    % thickness lands in worse basins (vertex 0.44 m: CRF 1.49 / EE 0.66;
    % vertex held 0.235-0.30 m: CRF 1.57 / EE 0.63).  The record keeps the
    % bounded solve, states that it sits on its bounds, and leaves a global
    % search over the meniscus for beat 4.
    % the meniscus vertex's lower bound follows the block (20 mm beyond its
    % radius): 0.24 m at the record's 220 mm, so the record is unchanged; the
    % closure-envelope sweep (addendum 11) runs other radii through here
    Rm = {'men_z', base.block_r + 0.02, 0.60, 0.1;  'men_t', 0.004, 0.040, 0.01;  'men_ca', 0.5, 8, 1;  'men_cb', 0.5, 8, 1};
    R4a = Rm;  R4 = [R3; Rm];
    % R5, the fold prism: R4's variables with the face offset at the fold's
    % scale (the plate's thickness on the slit side, the prism's depth on the
    % image side; bounds leave the exit face >= 7.5 mm beyond the fold plane)
    R5 = R4;  i5 = find(strcmp(R5(:,1), 'face_offset'));
    if isfield(base, 'fold_h') && base.fold_h > 0
        sg = 0.5e-3;  if isfield(base, 'slit_gap'), sg = base.slit_gap; end
        fg = sg;      if isfield(base, 'fpa_gap'),  fg = base.fpa_gap;  end
        % exit distance = (face - slit_gap) - fold_h + (slit_gap - fpa_gap)/n  >= 7.5 mm
        R5(i5, 2:4) = {base.fold_h + 7.5e-3 + sg + (fg - sg)/1.45, 40e-3, 1e-2};
        % with air on both sides of the block the conjugate plane is no longer
        % the grating's centre plane: the slit/FPA plane's axial position is
        % the knob that recovers it (+-5 mm)
        R5 = [R5; {'slit_dz', -5e-3, 5e-3, 1e-3}];
    end
    rungs = {struct('name', 'R0 concentric seed', 'vars', {{}}), ...
             struct('name', 'R1 concentric knobs (R_g factor, face offset, block r)', 'vars', {R1}), ...
             struct('name', 'R2 + conic + h^4,h^6 asphere on the block face', 'vars', {R2}), ...
             struct('name', 'R3 + block centre off the grating centre (dz, dy)', 'vars', {R3}), ...
             struct('name', 'R4a meniscus alone (vertex, thickness, 2 curvatures)', 'vars', {R4a}), ...
             struct('name', 'R4 + meniscus corrector, all variables (compact variant)', 'vars', {R4}), ...
             struct('name', 'R5 + fold prism (plate on the slit side), all variables', 'vars', {R5})};
    % meniscus seed: a concentric shell 30 mm beyond the block face, 10 mm thick
    base.men_z = 0;  base.men_t = 0.010;  base.men_ca = 0;  base.men_cb = 0;
    Pcur = base;
    if ~isempty(opts.seed)                     % warm start: the seed's DESIGN KNOBS over the base (never its spec fields --
        knobs = {'Rg_factor', 'face_offset', 'block_Kc', 'block_asph', 'block_dz', 'block_dy', 'men_z', 'men_t', 'men_ca', 'men_cb', 'slit_dz'};
        for f = knobs, if isfield(opts.seed, f{1}), Pcur.(f{1}) = opts.seed.(f{1}); end, end   % the envelope changes F-number, slit, pixel, glass in the base)
    end
    for q = 1:size(opts.bounds, 1)             % bound overrides, applied to every rung's variable table
        for rk = 1:numel(rungs)
            V = rungs{rk}.vars;  if isempty(V), continue; end          % R0 has no variables
            i = find(strcmp(V(:,1), opts.bounds{q,1}));
            if ~isempty(i), V(i, 2:3) = opts.bounds(q, 2:3);  rungs{rk}.vars = V; end
        end
    end
    L.rung = struct('name', {}, 'vars', {}, 'x', {}, 'P', {}, 'chain', {}, 'engine', {}, 'file', {}, 'merit', {}, 'on_bounds', {});
    for k = opts.rungs
        rg = rungs{k+1};  V = rg.vars;
        if k >= 4 && Pcur.men_z == 0           % entering R4: the near-null shell about the block's centre
            Pcur.men_z = Pcur.block_dz + Pcur.block_r + 0.030;  Pcur.men_t = 0.010;
            Pcur.men_ca = 1/(Pcur.men_z - Pcur.block_dz);  Pcur.men_cb = 1/(Pcur.men_z + Pcur.men_t - Pcur.block_dz);
        end
        if ~isempty(V)
            x0 = cellfun(@(n) get_(Pcur, n), V(:,1)) ./ cell2mat(V(:,4));
            lb = cell2mat(V(:,2)) ./ cell2mat(V(:,4));  ub = cell2mat(V(:,3)) ./ cell2mat(V(:,4));
            f = @(x) resid_(set_(Pcur, V, x), opts);
            o = optimoptions('lsqnonlin', 'Display', 'iter', 'MaxIterations', opts.max_iter, ...
                             'FunctionTolerance', 1e-10, 'StepTolerance', 1e-8, 'FiniteDifferenceStepSize', 1e-4);
            if opts.quiet, o.Display = 'off'; end
            [x, rn] = lsqnonlin(f, x0(:), lb(:), ub(:), o);
            Pcur = set_(Pcur, V, x);
        else
            x = [];  rn = sum(resid_(Pcur, opts).^2);
        end
        G = spectrometer_geom('dyson', Pcur);
        Rc = spectrometer_score_chain(G, Pcur, 'nx', P.score_nx, 'nlam', P.score_nlam, 'nring', 6);
        if ~isempty(opts.deck), file = opts.deck;
        elseif k <= 5, file = sprintf('%s_s3_r%d.in', tag, k);
        else, file = sprintf('%s_s5_r%d.in', tag, k);
        end
        M = spectrometer_rx(G, file, 'ngridpts', P.ngridpts, 'name', sprintf('%s_r%d', P.tag, k), ...
                            'apertures', true, 'margin', ap_margin_(P));
        macos.load_rx(file);
        Re = spectrometer_score(G, M, Pcur, 'nx', P.score_nx, 'nlam', P.score_nlam, 'quiet', true);
        onb = {};
        if ~isempty(V)
            lbv = cell2mat(V(:,2)) ./ cell2mat(V(:,4));  ubv = cell2mat(V(:,3)) ./ cell2mat(V(:,4));
            hit = abs(x(:) - lbv) < 1e-6*max(1, abs(lbv)) | abs(x(:) - ubv) < 1e-6*max(1, abs(ubv));
            onb = V(hit, 1)';
        end
        L.rung(end+1) = struct('name', rg.name, 'vars', {V}, 'x', x, 'P', Pcur, 'chain', Rc, ...
                               'engine', Re, 'file', file, 'merit', rn, 'on_bounds', {onb});
        if ~opts.quiet
            fprintf('%s: merit %.4g | engine smile %.4f keystone %.4f CRF %.3f SRF %.3f EE %.3f | clear %.2f mm | R_g %.1f mm r %.1f mm face %.3f mm Kc %.3f A %s\n', ...
                rg.name, rn, Re.smile_max, Re.keystone_max, Re.crf_max, Re.srf_max, Re.ee_min, G.fpa.clear_to_slit*1e3, ...
                G.Rg*1e3, G.r*1e3, Pcur.face_offset*1e3, Pcur.block_Kc, mat2str(Pcur.block_asph, 4));
        end
    end
end

function m = ap_margin_(P)
    if isfield(P, 'ap_margin_m'), m = P.ap_margin_m; else, m = 5e-3; end
end

function r = resid_(Pc, opts)
    try
        G = spectrometer_geom('dyson', Pc);
    catch
        r = 1e3*ones(2*opts.nx*opts.nlam*2 + 1, 1);  return
    end
    R = spectrometer_score_chain(G, Pc, 'nx', opts.nx, 'nlam', opts.nlam, 'nring', opts.nring);
    im = ceil(opts.nx/2);  jm = ceil(opts.nlam/2);
    smile = R.V - R.V(im, :);  keystone = R.U - R.U(:, jm);
    bad = isnan(R.SU) | isnan(R.SV);
    smile(bad) = 10;  keystone(bad) = 10;  SU = R.SU;  SV = R.SV;  SU(bad) = 10;  SV(bad) = 10;
    if opts.clear_m > 0, wall = max(0, opts.clear_m - G.fpa.clear_to_slit)/opts.clear_m*100; else, wall = 0; end   % R5: the fold separates slit and FPA; the clearance gate judges
    r = [opts.w_dist*smile(:); opts.w_dist*keystone(:); opts.w_blur*SU(:); opts.w_blur*SV(:); wall];
end

function v = get_(Pc, name)
    switch name
        case 'asph4', v = Pc.block_asph(1);
        case 'asph6', v = Pc.block_asph(2);
        otherwise,    v = Pc.(name);
    end
end

function Pc = set_(Pc, V, x)
    for i = 1:size(V, 1)
        val = x(i)*V{i,4};
        switch V{i,1}
            case 'asph4', Pc.block_asph(1) = val;
            case 'asph6', Pc.block_asph(2) = val;
            otherwise,    Pc.(V{i,1}) = val;
        end
    end
end
