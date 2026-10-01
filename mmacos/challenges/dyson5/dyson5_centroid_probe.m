function R = dyson5_centroid_probe(P, tag, opts)
%DYSON5_CENTROID_PROBE  Which centroid does the detector see?  (addendum 8)
%   R = dyson5_centroid_probe(P, tag) runs, on the s3 rung P.twin_rung's deck
%   (R4), at the slit centre and one slit end for lambda = 380 / 1440 / 2500
%   nm, the three discriminators of BRIEF_to_dyson5 addendum 8:
%
%   1. PUPIL-DOMAIN PREDICTION.  The engine's complex field on the reference
%      sphere (macos.complex_field at the ExitPupil element = the seeded
%      pupil, amplitudes carrying the Fresnel transmission of every
%      refraction) gives the local wavefront gradient grad(phi) by finite
%      differences of the wrapped field (angle(c(i+1) conj(c(i)))/dx -- modulo
%      lambda, addendum 3).  The far-field centroid theorem: the PSF
%      intensity centroid = L lambda/(2 pi) x the |a|^2-WEIGHTED mean of
%      grad(phi); the UNWEIGHTED mean (over the illuminated support) is the
%      ray centroid.  Both are predicted and compared with the measured PSF
%      centroid and the engine's ray centroid.  Agreement weighted <-> PSF and
%      unweighted <-> rays PROVES the amplitude-weighting mechanism.
%   2. THE WAVELENGTH LAW (numerics null).  opts.variant selects the twin's
%      grid: 'base' (model 512, 127 pts), 'window2' (model 1024, 255 pts:
%      the window doubled at the same pitch), 'pitch2' (model 1024, 127 pts:
%      the same window at half the pitch).  If the offset moves between
%      variants it is the twin's numerics; each variant runs in its OWN
%      MATLAB (model-size transitions in one process are the known hazard).
%   3. WINDOWS.  The PSF centroid over the full grid, over the smallest
%      centred square holding > 99 % of the energy, and over the 1-px box.
%
%   Writes <tag>_s2w_centroid_<variant>.txt and .mat.  Conventions as
%   spectrometer_wave (grid index 1 = X along the slit, 2 = Y dispersion,
%   centre pixel N/2+1; offsets in pixels from the chief).
    arguments
        P struct
        tag (1,:) char
        opts.variant (1,:) char {mustBeMember(opts.variant, {'base','window2','pitch2'})} = 'base'
        opts.lams (1,:) double = [380e-9 1440e-9 2500e-9]
        opts.xs (1,:) double = [0 0.027]
    end
    switch opts.variant
        case 'base',    model = 512;   ng = 127;
        case 'window2', model = 1024;  ng = 255;
        case 'pitch2',  model = 1024;  ng = 127;
    end
    s3 = load([tag '_s3.mat']);  L3 = s3.S;
    kk = find(strncmp({L3.rung.name}, [P.twin_rung ' '], numel(P.twin_rung) + 1), 1, 'last');
    G = spectrometer_geom('dyson', L3.rung(kk).P);
    file = sprintf('%s_s2w_centroid_%s_ff.in', tag, opts.variant);
    Mf = spectrometer_rx(G, file, 'ngridpts', ng, 'terminal', 'farfield', 'L_ref', P.wave_L_ref);
    macos.init(model);  macos.load_rx(file);
    iFP = Mf.iFPA;  iEP = Mf.iEP;  iFPr = Mf.iFPr;  iG = Mf.iG;  Lr = Mf.L_ref;  px = P.pixel_m;
    fid = fopen(sprintf('%s_s2w_centroid_%s.txt', tag, opts.variant), 'w');
    pr = @(varargin) dualprint_(fid, varargin{:});
    pr('dyson5 centroid probe on %s, variant %s: model %d, %d-pt grid (%s)\n', L3.rung(kk).name, opts.variant, model, ng, datestr(now, 'yyyy-mm-dd HH:MM'));
    pr('offsets in px from the chief (grid index 2 = dispersion v, 1 = slit u); pupil predictions from the field on the\n');
    pr('reference sphere: centroid = L lambda/(2pi) <grad phi>, weighted by |a|^2 (PSF) or unweighted (rays)\n');
    pr('grid axes: index 1 = -xGrid, index 2 = -yGrid of the SOURCE frame (the FFT inverts the pupil grid); sign map printed\n');
    pr('%6s %7s | %9s %9s | %9s %9s | %9s %9s %9s %9s | %9s %9s | %6s %6s | %s\n', 'x mm', 'nm', 'ray_dv', 'ray_du', 'pup_unw_v', 'pup_unw_u', 'psf_dv', 'psf99_dv', 'psf1px_dv', 'psf_du', 'pup_w_v', 'pup_w_u', 'E99 px', 'dx um', 'sign');
    rows = [];
    for xs = opts.xs
        slit = G.slit + [xs; 0; 0];
        for lam = opts.lams
            da = G.aim(slit, lam);
            macos.stop(iG);  macos.set_src_fov('src_pos', slit, 'src_dir', da, 'zSrc', -G.src.zsrc_gap);
            macos.set_src_wvl(lam);  macos.modify();
            s = macos.trace(iFP);  ri = macos.get_ray_info(s.nRays);  ok = ri.ok_trace & ri.ok_pass;
            pc = ri.pos(:,1);  pm = mean(ri.pos(:, ok), 2);  din = ri.dir(:,1)/norm(ri.dir(:,1));
            ray_du = (pm(1) - pc(1))/px;  ray_dv = (pm(2) - pc(2))/px;
            macos.set_xp(pc - Lr*din, din, -Lr);  macos.set_elt_vpt(iFPr, pc);  macos.set_elt_vpt(iFP, pc);
            % --- the pupil field on the reference sphere
            cE = macos.complex_field(iEP);  dxE = macos.dx_at(iEP);
            a2 = abs(cE).^2;  sup = a2 > 1e-6*max(a2(:));
            % gradient of the wrapped phase (index 1 = u, 2 = v)
            gu = angle(cE(2:end, :).*conj(cE(1:end-1, :)))/dxE;  su = sup(2:end, :) & sup(1:end-1, :);  wu = sqrt(a2(2:end, :).*a2(1:end-1, :));
            gv = angle(cE(:, 2:end).*conj(cE(:, 1:end-1)))/dxE;  sv = sup(:, 2:end) & sup(:, 1:end-1);  wv = sqrt(a2(:, 2:end).*a2(:, 1:end-1));
            k0 = Lr*lam/(2*pi)/px;                            % px per (rad/m) of phase gradient
            pup_unw = [mean(gu(su)), mean(gv(sv))]*k0;
            pup_w   = [sum(wu(su).*gu(su))/sum(wu(su)), sum(wv(sv).*gv(sv))/sum(wv(sv))]*k0;
            % --- the PSF
            cf = macos.complex_field(iFP, 'reset_trace', false);  I = abs(cf).^2;  dx = macos.dx_at(iFP);
            N = size(I, 1);  c0 = N/2 + 1;  [ii, jj] = ndgrid(1:N, 1:N);  tot = sum(I(:));
            % grid axes = the ExitPupil's aperture frame (cyclic permutation of
            % psi = the chief direction din; yObs = psi x xObs): map to +X/+Y
            sc = macos.get_src_csys();  sg = [-sign(sc.xDir(1)), -sign(sc.yDir(2))];   % FF grid = source grid, inverted
            cE = macos.complex_field(iEP);
            cen = @(m) sg.*([sum(I(m).*ii(m)), sum(I(m).*jj(m))]/sum(I(m)) - c0);
            full = cen(true(N));
            % smallest centred square holding > 99 %
            r = 1;  while r < N/2 && sum(I(abs(ii-c0) <= r & abs(jj-c0) <= r), 'all') < 0.99*tot, r = r + 1; end
            m99 = abs(ii-c0) <= r & abs(jj-c0) <= r;  w99 = cen(m99);
            mbox = abs((ii-c0)*dx) <= px/2 & abs((jj-c0)*dx) <= px/2;  w1 = cen(mbox);
            row = [xs, lam, ray_dv, ray_du, pup_unw(2), pup_unw(1), full(2)*dx/px, w99(2)*dx/px, w1(2)*dx/px, pup_w(2), pup_w(1), (2*r+1)*dx/px, dx*1e6, full(1)*dx/px, sg];
            rows(end+1, :) = row;  %#ok<AGROW>
            pr('%6.1f %7.0f | %+9.4f %+9.4f | %+9.4f %+9.4f | %+9.4f %+9.4f %+9.4f %+9.4f | %+9.4f %+9.4f | %6.2f %6.2f | %+d %+d\n', xs*1e3, lam*1e9, row([3:9 14 10:13]), row(15), row(16));
        end
    end
    fclose(fid);
    R.rows = rows;  R.variant = opts.variant;  R.model = model;  R.ng = ng;  R.rung = L3.rung(kk).name;
    save(sprintf('%s_s2w_centroid_%s.mat', tag, opts.variant), 'R', 'P');
end

function dualprint_(fid, varargin)
    fprintf(1, varargin{:});  fprintf(fid, varargin{:});
end
