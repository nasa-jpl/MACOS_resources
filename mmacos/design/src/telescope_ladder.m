function L = telescope_ladder(Pt0, GD, P, opts)
%TELESCOPE_LADDER  The telescope's departure ladder on the exact chain, under the slit scorer.
%   L = telescope_ladder(Pt0, GD, P) solves successive rungs of the three-
%   mirror telescope (telescope_geom) that feeds the Dyson GD, each by
%   lsqnonlin on the exact chain (chain_trace -- the engine reproduces it ray
%   for ray, gate tTelescopeRx), scoring every iterate with
%   telescope_score_chain at the slit in PIXEL units, with the pupil match
%   and the clearance wall in the merit from the first rung.  Rungs
%   (each warm-starts from the previous solution):
%     T0  the LAYOUT: bias, spacings, the three mirrors' FOLD angles and
%         M2/M3 decentre (the Bauer freedoms: a fold deviates the beam by
%         twice the angle, where a degree of field bias buys 2 mm) and the
%         flat fold's distance before the slit,
%         with the clearance wall dominant and the image terms weak -- a
%         70 mm beam folded inside a 110 mm box grazes every body at a
%         6-8 deg bias (seven pairs within 1.4 mm, 2026-10-01); the image
%         rungs start from an open layout;
%     T1  the conics, the three radii, the three spacings and the field bias
%         (the first-order seed holds f, a flat field and the exit pupil at
%         the spectrometer's; the conics start from spheres -- the Seidel
%         n-flip seed does not describe this geometry);
%     T2  + even aspheres h^4, h^6 on all three mirrors (Mouroulis & Green's
%         420 mm F/1.8 TMA needs three sixth-order aspheres);
%     T3  + M2 and M3 decentre (y) and tilt (about x): the Bauer freedoms,
%         off the coaxial parent.
%   Merit (lsqnonlin residual vector; px over the field set):
%     w_blur  * [su_i ; sv_i]            rms spot along / across the slit
%     w_v     * v_c,i                    the image line ON the slit
%     w_map   * [u_c(ends) - slit ends]  the field fills the 54 mm slit (EFL)
%     w_ftheta* (u_c - linear fit)       a near f-theta map (weak)
%     w_pupil * walk_i (mm)              the chief's miss of the grating vertex
%                                        when sent on into the spectrometer
%     w_flat  * zbf_i / 100 um           the best-focus offset along the slit
%                                        normal (a FLAT field is the spec; the
%                                        first solve without it traded a 3 mm
%                                        field-curvature swing for spot)
%     w_clear * max(0, clear - cmin) mm  the wall: telescope legs vs its own
%                                        mirrors, the fold and the
%                                        spectrometer's bodies (quick form)
%   Returns L.rung(k): .name .vars .x .P (the telescope parameter set),
%   .chain (the chain score), .cmin_mm/.worst (quick clearance), .merit,
%   .on_bounds (names).
    arguments
        Pt0 struct
        GD struct
        P struct
        opts.rungs (1,:) cell = {'T0', 'T1', 'T2', 'T3'}
        opts.nfield (1,1) double = 9
        opts.nring (1,1) double = 4
        opts.w_blur (1,1) double = 1
        opts.w_v (1,1) double = 1
        opts.w_map (1,1) double = 1
        opts.w_ftheta (1,1) double = 0.1
        opts.w_pupil (1,1) double = 1
        opts.w_clear (1,1) double = 10
        opts.w_flat (1,1) double = 1           % per field: the best-focus offset along the slit normal, in 100 um (~ a pixel of blur at F/1.8)
        opts.clear_m (1,1) double = 3e-3
        opts.max_iter (1,1) double = 60
        opts.quiet (1,1) logical = false
        opts.fold_gap (1,1) double = 15e-3     % the fold this far before the image (moves with t3)
        opts.npairs (1,1) double = 40          % wall residual slots (leg-body pairs; the rest zero-padded)
        opts.fd_step (1,1) double = 1e-4       % finite-difference step in scaled units (the field-line secant is noisy below)
    end
    mount = fld_(P, 'mount_margin_m', 5e-3);
    cloud = telescope_cloud(GD, P);
    % variable sets per rung: name, lower, upper, scale (the optimizer works in
    % scaled units so every variable is O(1))
    V1 = {'R1', 0.08, 1.5, 0.1;  'R2', 0.02, 0.8, 0.05;  'R3', 0.03, 1.0, 0.05; ...
          't1', 0.03, 0.4, 0.05;  't2', 0.02, 0.4, 0.05;  't3', 0.02, 0.4, 0.05; ...
          'bias', -0.4, 0.4, 0.05;  'Kc1', -20, 20, 1;  'Kc2', -20, 20, 1;  'Kc3', -20, 20, 1};
    % asphere scales: one scaled unit = 10 um of sag at h = 35 mm (the beam's
    % edge); at 1 and 300 a finite-difference step moved the sag 1e-11 m and
    % the rung returned its start untouched (2026-10-01)
    V2 = [V1; {'A1_4', -500, 500, 7;  'A2_4', -500, 500, 7;  'A3_4', -500, 500, 7; ...
               'A1_6', -5e5, 5e5, 5000;  'A2_6', -5e5, 5e5, 5000;  'A3_6', -5e5, 5e5, 5000}];
    VL = {'dec2', -0.04, 0.04, 2e-3;  'dec3', -0.04, 0.04, 2e-3;  'tilt1', -0.7, 0.7, 0.01;  'tilt2', -0.7, 0.7, 0.01;  'tilt3', -0.7, 0.7, 0.01;  'fold_gap', 0.012, 0.09, 0.01};
    V3 = [V2; VL];
    V0 = [{'bias', -0.4, 0.4, 0.05;  't1', 0.03, 0.4, 0.05;  't2', 0.02, 0.4, 0.05;  't3', 0.02, 0.4, 0.05}; VL];
    sets = struct('T0', {V0}, 'T1', {V1}, 'T2', {V2}, 'T3', {V3});
    % the layout rung's weights: the wall rules, the image terms only keep the
    % EFL / the map / the slit line from drifting
    wl = struct('w_blur', 0.05, 'w_v', 0.05, 'w_map', 0.1, 'w_ftheta', 0.02, 'w_pupil', 0.2, 'w_flat', 0.1, 'w_clear', 4*opts.w_clear);
    Pt = Pt0;
    for nm = {'Kc', 'dec', 'tilt'}, if ~isfield(Pt, nm{1}) || isempty(Pt.(nm{1})), Pt.(nm{1}) = [0 0 0]; end, end
    if ~isfield(Pt, 'A') || isempty(Pt.A), Pt.A = zeros(3, 2); end
    if ~isfield(Pt, 'bias'), Pt.bias = 0; end
    if ~isfield(Pt, 'fold_gap') || isempty(Pt.fold_gap), Pt.fold_gap = opts.fold_gap; end
    Pt.fold_d = max(1e-3, Pt.t(3) - Pt.fold_gap);
    rung = struct('name', {}, 'vars', {}, 'x', {}, 'P', {}, 'chain', {}, 'cmin_mm', {}, 'worst', {}, 'merit', {}, 'on_bounds', {});
    for k = 1:numel(opts.rungs)
        V = sets.(opts.rungs{k});
        x0 = cellfun(@(n) get_(Pt, n), V(:,1)) ./ cell2mat(V(:,4));
        lb = cell2mat(V(:,2)) ./ cell2mat(V(:,4));  ub = cell2mat(V(:,3)) ./ cell2mat(V(:,4));
        % a start outside its bounds is CLIPPED by lsqnonlin to a different
        % layout (the 24 deg folds of 2026-10-01 against a 22.9 deg bound:
        % the solver then descended from a 98k merit into the coaxial basin)
        out = find(x0(:) < lb(:) | x0(:) > ub(:));
        if ~isempty(out)
            warning('telescope_ladder:x0', '%s: start outside bounds for %s -- widen the bounds or move the seed', opts.rungs{k}, strjoin(V(out, 1)', ' '));
        end
        ok_ = opts;
        if strcmp(opts.rungs{k}, 'T0'), for fn = fieldnames(wl)', ok_.(fn{1}) = wl.(fn{1}); end, end
        f = @(x) resid_(set_(Pt, V, x), GD, ok_, cloud, mount);
        if ~opts.quiet
            r0 = f(x0(:));  nf = opts.nfield;
            fprintf('telescope_ladder %s start: merit %.4g = spots %.0f + v %.0f + ends %.0f + map %.0f + walk %.0f + flat %.0f + wall %.0f (weights blur %g v %g map %g ftheta %g pupil %g flat %g clear %g)\n', ...
                opts.rungs{k}, sum(r0.^2), sum(r0(1:2*nf).^2), sum(r0(2*nf+1:3*nf).^2), sum(r0(3*nf+1:3*nf+2).^2), sum(r0(3*nf+3:4*nf+2).^2), ...
                sum(r0(4*nf+3:5*nf+2).^2), sum(r0(5*nf+3:6*nf+2).^2), sum(r0(6*nf+3:end).^2), ok_.w_blur, ok_.w_v, ok_.w_map, ok_.w_ftheta, ok_.w_pupil, ok_.w_flat, ok_.w_clear);
        end
        o = optimoptions('lsqnonlin', 'Display', tern_(opts.quiet, 'off', 'iter'), 'MaxIterations', opts.max_iter, ...
                         'FunctionTolerance', 1e-10, 'StepTolerance', 1e-7, 'MaxFunctionEvaluations', 1e5, ...
                         'FiniteDifferenceStepSize', opts.fd_step);
        [x, rn] = lsqnonlin(f, x0(:), lb(:), ub(:), o);
        Pt = set_(Pt, V, x);
        GT = telescope_geom(Pt, GD);
        Rc = telescope_score_chain(GT, 'nfield', opts.nfield, 'nring', opts.nring, 'pixel_m', P.pixel_m, 'slit_px', P.slit_px);
        F = chain_footprints(GT, Rc.B);
        [cmin, worst] = telescope_clear_quick(GT, Rc.B, F, cloud, mount);
        hit = abs(x(:) - lb(:)) < 1e-6*max(1, abs(lb(:))) | abs(x(:) - ub(:)) < 1e-6*max(1, abs(ub(:)));
        rung(end+1) = struct('name', opts.rungs{k}, 'vars', {V(:,1)'}, 'x', x(:)'.*cell2mat(V(:,4))', 'P', Pt, 'chain', Rc, ...
                             'cmin_mm', cmin*1e3, 'worst', worst, 'merit', rn, 'on_bounds', {V(hit, 1)'});   %#ok<AGROW>
        if ~opts.quiet
            h = Rc.headline;
            fprintf('telescope_ladder %s: merit %.4g | spot max %.3f px (ee1 min %.3f, slit %.3f) | pupil walk max %.2f mm (chief err %.3f deg) | flat p-v %.1f um | ends %+.2f %+.2f px | clear %+.2f mm (%s)%s\n', ...
                opts.rungs{k}, rn, h.s_max_px, h.ee1_min, h.slit_min, h.walk_max_mm, h.err_max_deg, h.flat_pv_um, h.end_err_px, cmin*1e3, worst, ...
                tern_(any(hit), [' ON BOUNDS: ' strjoin(V(hit, 1)', ' ')], ''));
        end
    end
    L.rung = rung;  L.opts = opts;  L.cloud = cloud;
end

function r = resid_(Pt, GD, opts, cloud, mount)
    try
        GT = telescope_geom(Pt, GD);
        Rc = telescope_score_chain(GT, 'nfield', opts.nfield, 'nring', opts.nring, 'pixel_m', GD.P.pixel_m, 'slit_px', fld_(GD.P, 'slit_px', 2));
    catch
        r = 1e3*ones(6*opts.nfield + 2 + opts.npairs, 1);  return;
    end
    if any(isnan([Rc.SU Rc.SV Rc.V])) || any(Rc.nrays < 0.9*max(Rc.nrays))
        r = 1e3*ones(6*opts.nfield + 2 + opts.npairs, 1);  return;
    end
    walk = Rc.walk_m;  walk(isnan(walk)) = 0.05;              % a chief that cannot be sent into the spectrometer: a 50 mm miss
    F = chain_footprints(GT, Rc.B);
    [~, ~, cvec] = telescope_clear_quick(GT, Rc.B, F, cloud, mount);
    wall = opts.w_clear*max(0, opts.clear_m - cvec(:))*1e3;      % a hinge per leg-body pair (smoother than one on the minimum)
    wall(end+1:opts.npairs) = 0;  wall = wall(1:opts.npairs);
    zbf = Rc.zbf_m;  zbf(isnan(zbf)) = 0.01;
    r = [opts.w_blur*Rc.SU(:); opts.w_blur*Rc.SV(:); opts.w_v*Rc.V(:); ...
         opts.w_map*Rc.end_err_m(:)/Rc.pixel_m; opts.w_ftheta*Rc.map_lin(:)/Rc.pixel_m; ...
         opts.w_pupil*walk(:)*1e3; opts.w_flat*zbf(:)/1e-4; wall];
end

function v = get_(Pt, name)
    switch name
        case {'R1', 'R2', 'R3'}, v = Pt.R(str2double(name(2)));
        case {'t1', 't2', 't3'}, v = Pt.t(str2double(name(2)));
        case 'bias', v = Pt.bias;
        case 'fold_gap', v = Pt.fold_gap;
        case {'Kc1', 'Kc2', 'Kc3'}, v = Pt.Kc(str2double(name(3)));
        case {'dec2', 'dec3'}, v = Pt.dec(str2double(name(4)));
        case {'tilt1', 'tilt2', 'tilt3'}, v = Pt.tilt(str2double(name(5)));
        otherwise
            if regexp(name, '^A\d_\d$')
                v = Pt.A(str2double(name(2)), 1 + (name(4) == '6'));
            else
                error('telescope_ladder: unknown variable %s', name);
            end
    end
end

function Pt = set_(Pt, V, x)
    for i = 1:size(V, 1)
        name = V{i, 1};  v = x(i)*V{i, 4};
        switch name
            case {'R1', 'R2', 'R3'}, Pt.R(str2double(name(2))) = v;
            case {'t1', 't2', 't3'}, Pt.t(str2double(name(2))) = v;
            case 'bias', Pt.bias = v;
            case 'fold_gap', Pt.fold_gap = v;
            case {'Kc1', 'Kc2', 'Kc3'}, Pt.Kc(str2double(name(3))) = v;
            case {'dec2', 'dec3'}, Pt.dec(str2double(name(4))) = v;
            case {'tilt1', 'tilt2', 'tilt3'}, Pt.tilt(str2double(name(5))) = v;
            otherwise, Pt.A(str2double(name(2)), 1 + (name(4) == '6')) = v;
        end
    end
    if isfield(Pt, 'fold_d') && Pt.fold_d > 0, Pt.fold_d = max(1e-3, Pt.t(3) - Pt.fold_gap); end
end

function v = fld_(P, f, d)
    if isfield(P, f) && ~isempty(P.(f)), v = P.(f); else, v = d; end
end

function t = tern_(c, a, b), if c, t = a; else, t = b; end, end
