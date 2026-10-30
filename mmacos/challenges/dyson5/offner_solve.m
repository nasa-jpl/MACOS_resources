function O = offner_solve(P, tag, opts)
%OFFNER_SOLVE  The Offner sibling at a cleared ring radius, solved under the same operands.
%   O = offner_solve(P, tag) solves the Offner spectrometer (concave R used in
%   two zones, convex grating at the stop) at the ring radius P.y_slit_offner_m
%   -- chosen so the slit->M1 and M3->FPA beams pass BESIDE the grating body
%   (addendum 6) -- over the classical corrections: the convex grating's
%   radius factor (x R/2), the second concave zone's radius factor and centre
%   offsets (dy along the dispersion, dz), by lsqnonlin on the exact chain with
%   the same pixel-unit residuals as dyson_ladder (10 x smile/keystone, 1 x rms
%   spot, clearance wall).  Writes <tag>_s1_offner_solve.txt; returns the
%   solved parameter set O.P (what dyson5_params carries as the Offner's seed)
%   and the chain/clearance results.  Options: 'wall_mm' the clearance the
%   wall holds (3); 'ring_lb' > 0 frees the RING radius (P.y_slit_offner_m is
%   then the start) bounded below by ring_lb (m) and above by 0.45 R -- the
%   Offner's one layout freedom (addendum 46: the F/1.8 rows); 'warm' a
%   struct of start values for the four corrections (default: the seed);
%   'zone_wall' true adds a wall on the M1/M3 zone gap (the two clear
%   apertures must not overlap: two figures cannot share one area).
    arguments
        P struct
        tag (1,:) char
        opts.max_iter (1,1) double = 60
        opts.quiet (1,1) logical = false
        opts.wall_mm (1,1) double = 3
        opts.ring_lb (1,1) double = 0
        opts.warm struct = struct()          % start values for the corrections (e.g. a fixed-ring solve's O.P)
        opts.zone_wall (1,1) logical = false % wall on the M1/M3 zone gap (clear apertures must not overlap once their figures differ)
    end
    base = struct('Fno', P.Fno_offner, 'pixel_m', P.pixel_m, 'npix', P.npix, 'band_m', P.band_m, ...
                  'lambda_ref_m', P.lambda_ref_m, 'order', P.order, 'y_slit', P.y_slit_offner_m, ...
                  'offner_R', P.offner_R_m, 'block_r', P.block_r_m, 'glass', P.glass, 'face_offset', P.face_offset_m, ...
                  'Rg_factor', P.Rg_factor, 'grating_model', 'planes', 'slit_px', P.slit_px, ...
                  'offner_Rg_factor', 1.0, 'offner_M3_factor', 1.0, 'offner_M3_dy', 0, 'offner_M3_dz', 0);
    for nm = {'offner_Rg_factor', 'offner_M3_factor', 'offner_M3_dy', 'offner_M3_dz'}
        if isfield(opts.warm, nm{1}), base.(nm{1}) = opts.warm.(nm{1}); end
    end
    V = {'offner_Rg_factor', 0.90, 1.10, 1e-2;  'offner_M3_factor', 0.80, 1.10, 1e-2; ...
         'offner_M3_dy', -0.06, 0.06, 1e-2;  'offner_M3_dz', -0.06, 0.06, 1e-2};
    if opts.ring_lb > 0, V(end+1, :) = {'y_slit', opts.ring_lb, 0.45*P.offner_R_m, 1e-2}; end
    sc = cell2mat(V(:,4));
    x0 = cellfun(@(n) base.(n), V(:,1)) ./ sc;  lb = cell2mat(V(:,2)) ./ sc;  ub = cell2mat(V(:,3)) ./ sc;
    f = @(x) resid_(set_(base, V, x), P, opts.wall_mm, opts.zone_wall);
    o = optimoptions('lsqnonlin', 'Display', 'iter', 'MaxIterations', opts.max_iter, 'FunctionTolerance', 1e-10, ...
                     'StepTolerance', 1e-8, 'FiniteDifferenceStepSize', 1e-4);
    if opts.quiet, o.Display = 'off'; end
    r0 = sum(f(x0).^2);
    [x, rn] = lsqnonlin(f, x0(:), lb(:), ub(:), o);
    Ps = set_(base, V, x);  G = spectrometer_geom('offner', Ps);
    R = spectrometer_score_chain(G, Ps, 'nx', P.score_nx, 'nlam', P.score_nlam, 'nring', 6);
    C = spectrometer_clearance(G, P, 'quiet', true);
    O.P = Ps;  O.G = G;  O.chain = R;  O.clearance = C;  O.merit = [r0 rn];
    fid = fopen([tag '_s1_offner_solve.txt'], 'w');
    fprintf(fid, 'Offner solve at ring radius %.1f mm (%.3f R%s), F/%.1f, R %.3f m, wall %+.0f mm (%s): merit %.4g -> %.4g\n', Ps.y_slit*1e3, Ps.y_slit/P.offner_R_m, ...
        tern_(opts.ring_lb > 0, sprintf(', free from %.3f R, lower bound %.3f R', P.y_slit_offner_m/P.offner_R_m, opts.ring_lb/P.offner_R_m), ''), P.Fno_offner, P.offner_R_m, opts.wall_mm, datestr(now, 'yyyy-mm-dd HH:MM'), r0, rn);
    fprintf(fid, '  convex grating radius factor %.5f (R2 = %.3f mm); M3 radius factor %.5f (R3 = %.3f mm); M3 centre dy %+.3f mm, dz %+.3f mm\n', ...
        Ps.offner_Rg_factor, Ps.offner_Rg_factor*P.offner_R_m/2*1e3, Ps.offner_M3_factor, Ps.offner_M3_factor*P.offner_R_m*1e3, Ps.offner_M3_dy*1e3, Ps.offner_M3_dz*1e3);
    fprintf(fid, '  chain: keystone %.4f px, smile %.4f px, CRF %.3f px, SRF %.3f px, EE %.3f; clearance min %+.2f mm (%s vs %s)\n', ...
        R.keystone_max, R.smile_max, R.crf_max, R.srf_max, R.ee_min, C.min_mm, C.table.leg{1}, C.table.body{1});
    fclose(fid);
    if ~opts.quiet, type([tag '_s1_offner_solve.txt']); end
end

function r = resid_(Pc, P, wall_mm, zone_wall)
    try
        G = spectrometer_geom('offner', Pc);
    catch
        r = 1e3*ones(2*25*2 + 1 + zone_wall, 1);  return
    end
    R = spectrometer_score_chain(G, Pc, 'nx', 5, 'nlam', 5, 'nring', 4);
    smile = R.V - R.V(3, :);  keystone = R.U - R.U(:, 3);
    bad = isnan(R.SU) | isnan(R.SV);  smile(bad) = 10;  keystone(bad) = 10;  SU = R.SU;  SV = R.SV;  SU(bad) = 10;  SV(bad) = 10;
    C = spectrometer_clearance(G, P, 'quiet', true, 'nring', 1);
    wall = max(0, wall_mm - C.min_mm)/wall_mm*100;
    r = [10*smile(:); 10*keystone(:); SU(:); SV(:); wall];
    if zone_wall
        % the two concave zones' clear apertures (footprint + P.ap_margin_m)
        % must not share area: centre distance minus the two clear radii >= 0
        F = C.footprints;  am = P.ap_margin_m;
        cen = @(k) G.surf(k).vpt(:) + F(k).xap*F(k).xc + F(k).yap*F(k).yc;
        zg = norm(cen(1) - cen(3)) - (F(1).radius + am) - (F(3).radius + am);
        r(end+1) = max(0, -zg*1e3)/wall_mm*100;      % mm of overlap, on the clearance wall's scale
    end
end

function Pc = set_(Pc, V, x)
    for i = 1:size(V, 1), Pc.(V{i,1}) = x(i)*V{i,4}; end
end

function t = tern_(c, a, b), if c, t = a; else, t = b; end, end
