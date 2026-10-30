function R = spectrometer_wave(G, M, P, opts)
%SPECTROMETER_WAVE  The propagation twin of spectrometer_score.
%   R = spectrometer_wave(G, M, P) turns the emitted geometric deck M.file
%   into a diffraction deck (macos.design.prop_layout: a far-field terminal
%   FP_return / ExitPupil / FPA replaces the detector, the exit pupil
%   MEASURED by FEX), then for every (slit position, wavelength) of the
%   scoring grid re-aims the chief, re-poses the exit pupil with FEX for
%   THAT field and wavelength, propagates the complex field to the FPA
%   (macos.complex_field) and reads the PSF back.
%
%   Conventions, stated before any number:
%   - the PSF grid is centred on the chief ray's FPA pierce; its FIRST
%     index runs along the ExitPupil element's xObs and its SECOND along
%     yObs = psi x xObs -- +X / +Y for a beam travelling +z (the Offner),
%     -X / -Y for one travelling -z (the Dyson): the harness maps both to
%     global X (slit, u) and Y (dispersion, v) with the signs the frame
%     implies (measured, addendum 8; R.grid_sign records them); both forms
%     put the FPA in the global x-y plane; the pitch is macos.dx_at(FPA) in SI metres, set by
%     the far-field FFT: dx_fpa = lambda R_ep / (N dx_ep), so it SCALES
%     WITH WAVELENGTH (0.3 um at 380 nm to 1.8 um at 2.5 um, model 512).
%   - WAVE centroid offset = intensity-weighted centroid of the PSF from
%     the grid centre, in pixels; RAY centroid offset = mean of the
%     engine's ray pierces minus the chief pierce, in pixels.  Their
%     difference per (s, lambda) is the twin's first product: where the
%     PSF is symmetric it is < 0.01 px; where it is not, that is the
%     finding.  The wave smile/keystone maps are the ray chief maps plus
%     the wave offsets.
%   - SRF_wave = FWHM of rect(slit_px) (x) LSF_v(PSF) (x) rect(1 px);
%     CRF_wave = LSF_u(PSF) (x) rect(1 px) -- the PSF replaces the
%     geometric spot (x) Airy of the analytic chain; LSFs are the PSF
%     integrated along the other axis, resampled to 0.01 px.
%   - R.energy = sum of the PSF per point (the far-field leg conserves it;
%     the engine's ray amplitudes carry the uncoated Fresnel losses, so
%     the Dyson's total sits below the Offner's).
%   - the slit is a POINT here; the slit-width diffraction loss is the
%     separate far-field measurement spectrometer_slit_loss.
%
%   Options: 'nx' (5), 'nlam' (5), 'model' (512), 'ngridpts' (127, odd --
%   the FPA window is ngridpts x lambda F, so 127 spans +-2.4 px at 380 nm),
%   'L_ref' (0.1 m, the reference-sphere radius), 'out', 'quiet'.
    arguments
        G struct
        M struct
        P struct
        opts.nx (1,1) double = 5
        opts.nlam (1,1) double = 5
        opts.model (1,1) double = 512
        opts.ngridpts (1,1) double = 127
        opts.out (1,:) char = ''
        opts.L_ref (1,1) double = 0.1
        opts.ap_margin (1,1) double = 5e-3
        opts.quiet (1,1) logical = false
    end
    % ---- 1) the diffraction deck: the far-field terminal on a REFERENCE
    % sphere L_ref upstream of the FPA (Rx_Cass_FarField idiom), emitted by
    % spectrometer_rx -- not prop_layout: its FEX-measured pupil lies PAST the
    % focus on the telecentric Offner and the reversed rays never reach it.
    if isempty(opts.out)
        [d, b] = fileparts(M.file);  opts.out = fullfile(d, [b '_ff.in']);
    end
    Mf = spectrometer_rx(G, opts.out, 'ngridpts', opts.ngridpts, 'terminal', 'farfield', 'L_ref', opts.L_ref, ...
                         'apertures', true, 'margin', opts.ap_margin);       % the optics carry their declared apertures here too
    R.prop_deck = opts.out;  R.M = Mf;
    macos.init(opts.model);
    macos.load_rx(opts.out);
    nE = macos.num_elt();  assert(nE == Mf.nElt, 'spectrometer_wave: deck loads %d of %d', nE, Mf.nElt);
    iFP = Mf.iFPA;  iEP = Mf.iEP;  iFPr = Mf.iFPr;  iG = Mf.iG;  L = Mf.L_ref;
    R.iFP = iFP;  R.iEP = iEP;

    W = P.npix(1)*P.pixel_m;  px = P.pixel_m;
    xs = linspace(-W/2, W/2, opts.nx);  lams = linspace(P.band_m(1), P.band_m(2), opts.nlam);
    nx = numel(xs);  nl = numel(lams);
    z = nan(nx, nl);
    R.xs = xs;  R.lams = lams;
    R.du_wave = z;  R.dv_wave = z;  R.du_ray = z;  R.dv_ray = z;
    R.su_wave = z;  R.sv_wave = z;  R.SRF = z;  R.CRF = z;  R.energy = z;  R.dx = z;  R.ee = z;
    R.ep_rad = z;  R.grid_sign = [];
    R.psf = cell(nx, nl);
    for i = 1:nx
        slit = G.slit + [xs(i); 0; 0];
        for j = 1:nl
            lam = lams(j);  da = G.aim(slit, lam);
            macos.stop(iG);                                   % stop FIRST
            macos.set_src_fov('src_pos', slit, 'src_dir', da, 'zSrc', -G.src.zsrc_gap);
            macos.set_src_wvl(lam);  macos.modify();
            s = macos.trace(iFP);  ri = macos.get_ray_info(s.nRays);
            ok = ri.ok_trace & ri.ok_pass;
            pc = ri.pos(:,1);  pm = mean(ri.pos(:, ok), 2);
            din = ri.dir(:,1);  din = din/norm(din);          % chief direction into the FPA
            R.du_ray(i,j) = (pm(1) - pc(1))/px;  R.dv_ray(i,j) = (pm(2) - pc(2))/px;
            % pose the reference sphere on THIS chief: centre = the chief's FPA
            % pierce, vertex L upstream, psi along the beam; and centre the
            % FP_return / FPA vertices on the pierce so the PSF grid is
            % centred on the chief
            macos.set_xp(pc - L*din, din, -L);
            macos.set_elt_vpt(iFPr, pc);  macos.set_elt_vpt(iFP, pc);
            R.ep_rad(i,j) = L;
            cf = macos.complex_field(iFP);                    % MODIFY + retrace + propagate
            I  = abs(cf).^2;  dx = macos.dx_at(iFP);          % SI metres
            if ~(sum(I(:)) > 0) || any(isnan(I(:)))
                warning('spectrometer_wave: empty field at x=%.3f lam=%.3g', xs(i), lam);  continue
            end
            R.dx(i,j) = dx;  R.energy(i,j) = sum(I(:));
            N = size(I, 1);  c0 = N/2 + 1;                     % the FFT grid's centre pixel (chief); measured on the order-0 relay
            [ii, jj] = ndgrid(1:N, 1:N);
            tot = sum(I(:));
            ci = sum(I(:).*ii(:))/tot;  cj = sum(I(:).*jj(:))/tot;
            % GRID ORIENTATION (measured 2026-10-01, addendum 8): see below.
            if isempty(R.grid_sign)
                % the far-field grid = the SOURCE grid's (xGrid, yGrid)
                % orientation carried in index space and INVERTED by the FFT:
                % index 1 runs along -xGrid, index 2 along -yGrid.  The emitter
                % writes xGrid = +X and yGrid = chief x X, i.e. +Y for a +z
                % chief (Dyson) and -Y for a -z chief (Offner) -- which is the
                % (-X,-Y) / (-X,+Y) pair measured 2026-10-01 (addendum 8).
                sc = macos.get_src_csys();
                R.grid_sign = [-sign(sc.xDir(1)), -sign(sc.yDir(2))];
                if ~opts.quiet, fprintf('  wave twin: far-field grid axes = (%+dX, %+dY) from the source frame\n', R.grid_sign); end
            end
            su = R.grid_sign(1);  sv = R.grid_sign(2);
            R.du_wave(i,j) = su*(ci - c0)*dx/px;  R.dv_wave(i,j) = sv*(cj - c0)*dx/px;
            R.su_wave(i,j) = sqrt(sum(I(:).*(ii(:)-ci).^2)/tot)*dx/px;
            R.sv_wave(i,j) = sqrt(sum(I(:).*(jj(:)-cj).^2)/tot)*dx/px;
            % ensquared energy in one pixel about the PSF centroid
            R.ee(i,j) = sum(I(abs((ii - ci)*dx) <= px/2 & abs((jj - cj)*dx) <= px/2))/tot;
            % LSFs along v (sum over u = index 1) and along u
            lsf_v = sum(I, 1);  lsf_u = sum(I, 2)';
            gv = ((1:N) - cj)*dx/px;  gu = ((1:N) - ci)*dx/px;
            R.SRF(i,j) = fwhm_from_lsf_(gv, lsf_v, P.slit_px);
            R.CRF(i,j) = fwhm_from_lsf_(gu, lsf_u, 0);
            R.psf{i,j} = struct('I', single(I), 'dx', dx);
            if ~opts.quiet
                fprintf('  wave twin: x=%+6.2f mm lam=%5.0f nm  dx %.2f um  EP R %.4f m  dv wave/ray %+.4f/%+.4f px  SRF %.3f CRF %.3f px  E %.4f\n', ...
                    xs(i)*1e3, lam*1e9, dx*1e6, L, R.dv_wave(i,j), R.dv_ray(i,j), R.SRF(i,j), R.CRF(i,j), R.energy(i,j));
            end
        end
    end
    R.d_du = R.du_wave - R.du_ray;  R.d_dv = R.dv_wave - R.dv_ray;
    if ~isfield(R, 'grid_sign'), R.grid_sign = [NaN NaN]; end
    R.d_max = max(abs([R.d_du(:); R.d_dv(:)]));
    R.srf_max = max(R.SRF(:));  R.crf_max = max(R.CRF(:));  R.ee_min = min(R.ee(:));
end

function w = fwhm_from_lsf_(g, lsf, slit_px)
%FWHM_FROM_LSF_  Resample the PSF's LSF onto 0.01 px, convolve with the slit
%   image (slit_px, 0 = none) and the pixel, return the FWHM in px.
    h = 0.01;  gg = -8:h:8;
    f = interp1(g, lsf, gg, 'linear', 0);  f = max(f, 0);  f = f/sum(f);
    if slit_px > 0, f = conv_(f, rect_(gg, slit_px)); end
    f = conv_(f, rect_(gg, 1));
    f = f/max(f);
    i = find(f >= 0.5);
    if isempty(i), w = NaN; return; end
    i1 = i(1);  i2 = i(end);
    w = interp_(gg, f, i2, i2+1) - interp_(gg, f, i1-1, i1);
end
function x = interp_(g, f, ia, ib)
    if ia < 1 || ib > numel(g), x = g(max(1, min(ia, numel(g)))); return; end
    x = g(ia) + (0.5 - f(ia))*(g(ib)-g(ia))/(f(ib)-f(ia));
end
function y = rect_(g, w), y = double(abs(g) <= w/2);  y = y/sum(y); end
function y = conv_(a, b)
    n = numel(a);  y = conv(a, b, 'full');  k0 = floor(numel(b)/2);  y = y(k0 + (1:n));
end
