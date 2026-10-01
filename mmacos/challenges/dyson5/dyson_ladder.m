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
        opts.rungs (1,:) double = 0:2
        opts.nx (1,1) double = 5
        opts.nlam (1,1) double = 5
        opts.nring (1,1) double = 4
        opts.w_dist (1,1) double = 10
        opts.w_blur (1,1) double = 1
        opts.clear_m (1,1) double = 3e-3
        opts.max_iter (1,1) double = 60
        opts.free_r (1,1) logical = false
        opts.quiet (1,1) logical = false
    end
    base = struct('Fno', P.Fno, 'pixel_m', P.pixel_m, 'npix', P.npix, 'band_m', P.band_m, ...
                  'lambda_ref_m', P.lambda_ref_m, 'order', P.order, 'y_slit', P.y_slit_m, ...
                  'block_r', P.block_r_m, 'glass', P.glass, 'face_offset', P.face_offset_m, ...
                  'Rg_factor', P.Rg_factor, 'grating_model', 'planes', 'slit_px', P.slit_px, ...
                  'block_Kc', 0, 'block_asph', [0 0]);
    % variable sets per rung: name, lower, upper, scale (the optimizer works in
    % scaled units so every variable is O(1))
    % the block radius is HELD at the seed's unless opts.free_r: freed, the
    % optimizer walks it to its bound and buys blur with size (the h^4/r^3
    % law of s0) -- the ladder's question is what each DEPARTURE buys at a
    % fixed scale
    R1 = {'Rg_factor', 0.90, 1.10, 1;  'face_offset', 1e-4, 5e-3, 1e-3};
    if opts.free_r, R1 = [R1; {'block_r', 0.15, 0.35, 0.1}]; end
    R2 = [R1; {'block_Kc', -2, 2, 0.5;  'asph4', -200, 200, 10;  'asph6', -2e5, 2e5, 1e4}];
    rungs = {struct('name', 'R0 concentric seed', 'vars', {{}}), ...
             struct('name', 'R1 concentric knobs (R_g factor, face offset, block r)', 'vars', {R1}), ...
             struct('name', 'R2 + conic + h^4,h^6 asphere on the block face', 'vars', {R2})};
    Pcur = base;  L.rung = struct('name', {}, 'vars', {}, 'x', {}, 'P', {}, 'chain', {}, 'engine', {}, 'file', {}, 'merit', {});
    for k = opts.rungs
        rg = rungs{k+1};  V = rg.vars;
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
        file = sprintf('%s_s3_r%d.in', tag, k);
        M = spectrometer_rx(G, file, 'ngridpts', P.ngridpts, 'name', sprintf('%s_r%d', P.tag, k));
        macos.load_rx(file);
        Re = spectrometer_score(G, M, Pcur, 'nx', P.score_nx, 'nlam', P.score_nlam, 'quiet', true);
        L.rung(end+1) = struct('name', rg.name, 'vars', {V}, 'x', x, 'P', Pcur, 'chain', Rc, ...
                               'engine', Re, 'file', file, 'merit', rn);
        if ~opts.quiet
            fprintf('%s: merit %.4g | engine smile %.4f keystone %.4f CRF %.3f SRF %.3f EE %.3f | clear %.2f mm | R_g %.1f mm r %.1f mm face %.3f mm Kc %.3f A %s\n', ...
                rg.name, rn, Re.smile_max, Re.keystone_max, Re.crf_max, Re.srf_max, Re.ee_min, G.fpa.clear_to_slit*1e3, ...
                G.Rg*1e3, G.r*1e3, Pcur.face_offset*1e3, Pcur.block_Kc, mat2str(Pcur.block_asph, 4));
        end
    end
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
    wall = max(0, opts.clear_m - G.fpa.clear_to_slit)/opts.clear_m*100;
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
