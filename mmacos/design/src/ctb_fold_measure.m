function M = ctb_fold_measure(R, opts)
%CTB_FOLD_MEASURE  What a CTB-behind-telescope train does: rays, wavefront, walk, AOI, packaging.
%   M = CTB_FOLD_MEASURE(R) measures the spliced train R.full_in that
%   CTB_FOLD_BUILD wrote:
%     .rays       rays reaching OAP1, DM1, DM2, the Lyot marker and the FPA
%                 (of .nrays), on the deck's own field
%     .beam_r     the beam radius (m) at those stations
%     .wfe_waves  FPA rms OPD over the valid pupil, in waves
%     .aoi        per CTB element: name, chief-ray AOI (deg) from the
%                 chief's directions in and out (AOI = 90 - acos(din.dout)/2)
%     .walk       the footprint-centroid walk on DM1 and DM2 for an 8-point
%                 ring of field radius opts.ring_as (arcsec) about the
%                 field centre, in metres and in DM pupil radii, and the
%                 rays through DM2 / to the FPA on that ring (the
%                 chapter-2 walk table's row)
%     .clear      deck_clearance on the train over the centre field and
%                 that ring (bodies: every element but the Return planes;
%                 mount opts.mount; a part list of overrides opts.bodies)
%     .shroud     shroud_deck's hardware union of the train (.D, .len) and
%                 the CTB-only extent from the clearance's bodies
%                 (.ctb_max_r), both about the launch axis
%     .zmin_ctb   the lowest z of any CTB body
%   Options: 'ring_as' (10), 'clear_as' (the ring radius, arcsec, the
%   CLEARANCE bodies and legs are sized over; NaN = ring_as -- give the
%   instrument's own field here, a wide walk ring sizes every focus body
%   by the field's image), 'mount' (0.008), 'bodies' (struct), 'holes'
%   (struct name -> through-hole radius, e.g. the primary's: a crossing
%   inside it is the design, not a piercing), 'leg_from' (score the legs
%   leaving this element index onward; 0 = every leg including the sky
%   leg, which the secondary obscures by design), 'shroud_D_m' (gate for
%   the printed verdict; NaN = none), 'quiet'.
%
%   See also CTB_FOLD_BUILD, DECK_CLEARANCE, SHROUD_DECK.

    arguments
        R struct
        opts.ring_as    (1,1) double = 10
        opts.clear_as   (1,1) double = NaN
        opts.mount      (1,1) double = 0.008
        opts.bodies     struct = struct()
        opts.shroud_D_m (1,1) double = NaN
        opts.holes      struct = struct()
        opts.leg_from   (1,1) double = 0
        opts.quiet      (1,1) logical = false
    end
    model = R.K.model;  mm = 1e-3;
    macos.init(model);
    nF = macos.load_rx(R.full_in);
    txt = fileread(R.full_in);
    st = {'OAP1', 'DM1', 'DM2', 'Lyot', 'FPA'};
    ie = [R.ix.OAP1 R.ix.DM1 R.ix.DM2 R.ix.Lyot nF];
    rays = zeros(1, numel(ie));  beam_r = rays;  nR = 0;
    for j = 1:numel(ie)
        s = macos.trace(ie(j));  ri = macos.get_ray_info(s.nRays);
        okk = ri.ok_pass & ri.ok_trace;  dc = ri.dir(:,1)/norm(ri.dir(:,1));
        w = ri.pos(:,okk) - ri.pos(:,1);  w = w - dc*(dc.'*w);
        rays(j) = nnz(okk);  beam_r(j) = max(vecnorm(w));  nR = s.nRays;
    end
    macos.trace(nF);  W = macos.opd();  Mk = macos.opd_mask();
    wfe = std(W(Mk)) / (R.P.WAVLEN*mm);

    % ---- AOI per CTB element from the chief ray's history ----------------
    macos.ray_hist('on');  macos.trace(nF);  h = macos.ray_hist(nR);  macos.ray_hist('off');
    k0 = R.n_base;
    aoi = struct('name', {}, 'elt', {}, 'aoi_deg', {});
    for k = k0+1:nF
        if ~(h.ok(1, k) && h.ok(1, k+1) && (k+2 <= nF+1) && h.ok(1, k+2)), continue; end
        din = h.P(:,1,k+1) - h.P(:,1,k);  dout = h.P(:,1,k+2) - h.P(:,1,k+1);
        if norm(din) < 1e-12 || norm(dout) < 1e-12, continue; end
        c = (din.'*dout)/(norm(din)*norm(dout));
        dev = acosd(max(-1, min(1, c)));
        kind = regexp(txt, sprintf('(?m)^\\s*iElt=\\s*%d\\s*\\n\\s*EltName=\\s*(\\S+)\\s*\\n\\s*Element=\\s*(\\S+)', k), 'tokens', 'once');
        if isempty(kind) || ~strcmpi(kind{2}, 'Reflector'), continue; end
        aoi(end+1) = struct('name', kind{1}, 'elt', k, 'aoi_deg', 90 - dev/2);  %#ok<AGROW>
    end

    % ---- the walk on the DMs over the ring --------------------------------
    dB = str2double(regexp(txt, 'ChfRayDir=\s*(\S+)\s+(\S+)\s+(\S+)', 'tokens', 'once')).';
    pB = str2double(regexp(txt, 'ChfRayPos=\s*(\S+)\s+(\S+)\s+(\S+)', 'tokens', 'once')).';
    dB = dB/norm(dB);  L0 = norm(pB);
    Fr = macos.design.field_ring(opts.ring_as/60, 'units', 'arcmin', 'n', 8, 'inner', 0);
    eD = [R.ix.DM1 R.ix.DM2];
    c0 = zeros(3, 2);
    macos.load_rx(R.full_in);
    for j = 1:2
        rj = macos.get_ray_info(macos.trace(eD(j)).nRays);  okj = rj.ok_pass & rj.ok_trace;
        c0(:,j) = mean(rj.pos(:,okj), 2);
    end
    wl = zeros(size(Fr,1), 2);  nk = wl;  nfpa = zeros(size(Fr,1), 1);
    Fx = zeros(2, size(Fr,1));
    for k = 1:size(Fr,1)
        dxy = Fr(k,:);  Fx(:,k) = dxy(:);            % field_ring returns RADIANS
        d = dB + [dxy(1); dxy(2); 0];  d = d/norm(d);
        t2 = regexprep(txt, '(?m)^(\s*ChfRayDir=)[^\n]*', sprintf('$1  %.16E  %.16E  %.16E', d));
        t2 = regexprep(t2, '(?m)^(\s*ChfRayPos=)[^\n]*', sprintf('$1  %.16E  %.16E  %.16E', -L0*d));
        tmp = [tempname '.in'];  fid = fopen(tmp, 'w');  fwrite(fid, t2);  fclose(fid);
        n = macos.load_rx(tmp);  delete(tmp);
        for j = 1:2
            rj = macos.get_ray_info(macos.trace(eD(j)).nRays);  okj = rj.ok_pass & rj.ok_trace;
            nk(k,j) = nnz(okj);
            if nk(k,j) == 0, wl(k,j) = NaN; else, wl(k,j) = norm(mean(rj.pos(:,okj), 2) - c0(:,j)); end
        end
        rn = macos.get_ray_info(macos.trace(n).nRays);  nfpa(k) = nnz(rn.ok_pass & rn.ok_trace);
    end
    rpup = R.Dpup/2;
    walk = struct('ring_as', opts.ring_as, 'dm1_m', max(wl(:,1)), 'dm2_m', max(wl(:,2)), ...
                  'dm1_rpup', max(wl(:,1))/rpup, 'dm2_rpup', max(wl(:,2))/rpup, ...
                  'rays_dm2', [min(nk(:,2)) max(nk(:,2))], 'rays_fpa', [min(nfpa) max(nfpa)]);

    % ---- clearance and shroud -----------------------------------------
    Fc = Fx;
    if ~isnan(opts.clear_as) && opts.clear_as ~= opts.ring_as
        Fc = macos.design.field_ring(opts.clear_as/60, 'units', 'arcmin', 'n', 8, 'inner', 0).';
    end
    C = deck_clearance(R.full_in, 'model', model, 'fields', [[0;0], Fc], 'mount', opts.mount, ...
                       'bodies', opts.bodies, 'holes', opts.holes, 'elts', opts.leg_from:nF, ...
                       'addon_from', k0, 'quiet', true);
    ctb = [C.bodies.elt] > k0;
    ext_ctb = [C.bodies(ctb).ext];  z_ctb = [C.bodies(ctb).z];
    Pd = struct('shroud_D_m', opts.shroud_D_m, 'model', model);
    if isnan(Pd.shroud_D_m), Pd.shroud_D_m = 0; end
    sh = shroud_deck(R.full_in, Pd, 'model', model);
    shroud = struct('D', sh.D, 'len', sh.len, 'ctb_max_r', max(ext_ctb), 'ctb_D', 2*max(ext_ctb), ...
                    'gate_D_m', opts.shroud_D_m);

    M = struct('stations', {st}, 'rays', rays, 'nrays', nR, 'beam_r', beam_r, 'wfe_waves', wfe, 'clear_as', opts.clear_as, ...
               'aoi', aoi, 'walk', walk, 'clear', C, 'shroud', shroud, 'zmin_ctb', min(z_ctb), ...
               'full_in', R.full_in);
    if ~opts.quiet
        fprintf('ctb_fold_measure: %s\n', R.full_in);
        for j = 1:numel(st), fprintf('   %-5s %4d of %4d rays, beam radius %6.2f mm\n', st{j}, rays(j), nR, beam_r(j)*1e3); end
        fprintf('   FPA rms wavefront %.4f waves\n', wfe);
        fprintf('   AOI (deg):');  for k = 1:numel(aoi), fprintf(' %s %.2f', aoi(k).name, aoi(k).aoi_deg); end;  fprintf('\n');
        fprintf('   walk at %g": DM1 %.3f mm (%.3f r_pup), DM2 %.3f mm (%.3f r_pup); rays DM2 %d..%d, FPA %d..%d\n', ...
                opts.ring_as, walk.dm1_m*1e3, walk.dm1_rpup, walk.dm2_m*1e3, walk.dm2_rpup, walk.rays_dm2, walk.rays_fpa);
        fprintf('   clearance min %+.1f mm (leg from %s vs %s); telescope-internal min %+.1f mm; shroud: train %.3f m, CTB bodies to r = %.3f m (D %.3f); CTB zmin %.3f m\n', ...
                C.min*1e3, C.worst.leg_name, C.worst.body_name, C.min_base*1e3, shroud.D, shroud.ctb_max_r, shroud.ctb_D, M.zmin_ctb);
    end
end
