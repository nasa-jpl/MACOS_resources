function out = tg96_pupilsim(varargin)
%TG96_PUPILSIM  Detailed pupil-image simulation of the Twyman-Green's detector leg (Dave, 2026-09-17).
%   out = TG96_PUPILSIM('rig','lens'|'oap', name/value ...)
%
%   Fang Shi's concern: pupil distortion, field curvature and the rest of the
%   detector leg's aberrations may limit how well the camera sees the DM's
%   modes.  The record's interferometer frames are the engine's ray-indexed
%   field on Geometric legs -- a PERFECT pupil relay by construction -- so none
%   of that is in the record.  This tool puts it in, in three stages:
%
%   STAGE 1 -- the leg's coherent PSF, zone by zone, from the engine's rays.
%     The bench as built, the DM declared the STOP (so every ray index stays on
%     one DM zone across tilts: measured 2 um), the field a lateral shift of the
%     source at the collimator's focus = a tilt theta about the DM = a spatial
%     frequency theta/lambda on its surface.  A two-dimensional set of tilts
%     over the actuator band (rings x azimuths) is traced; for every ray the
%     detector-plane INTERCEPT r(theta) and the exit angle a(theta) are read.
%     eps = r(theta) - r(0) is the zone's transverse ray aberration over the
%     band aperture, W = integral eps . da its wavefront, H(f) = exp(-ikW) the
%     zone's coherent transfer function and its inverse transform the zone's
%     complex PSF.  The quadratic part of W is the zone's image position vs the
%     detector plane (the pupil SURFACE), so this stage also answers "are we at
%     the best pupil image" and gives the compromise plane.
%   STAGE 2 -- the DM field through those PSFs ("convolve the DM field").
%     The DM field exp(i 4 pi h / lambda) on the DM's own grid is filtered zone
%     by zone (overlap-add, raised-cosine patches), the reference arm goes
%     through the same operator, and the four-step readout angle(Et conj(Er))
%     gives the recovered surface.  Test surfaces: sinusoids at the actuator
%     Nyquist, half, quarter, eighth (gain and amplitude cross-talk vs pupil
%     radius by demodulation); single pokes at six radii; the record's 30 nm
%     working surface (random actuator commands, seed_base 7).  Everything is
%     run at the detector plane as built AND at the compromise plane.
%   STAGE 3 -- the plane-to-plane cross-check is tg96_pupil_engine (the ENGINE's
%     own propagation through reference surfaces inserted in the .in file, the
%     CTB model; Dave 2026-09-17).  'fourier' true runs instead a standalone
%     paraxial MATLAB chain (exact thick-lens screens) as a standby; off by default.
%
%   Name/value: 'rig' ('lens'), 'deck' (the rig's deck of record), 'tag',
%   'model' (512), 'ngrid' (129), 'band' (3.2e-4 rad = lambda/(2 pitch)),
%   'rings' ([0.5 1 2 3.2]*1e-4), 'ring_out' (1e-3), 'naz' (8), 'lambda'
%   (6.328e-4 mm), 'dx' (0.125 mm: the DM grid), 'N' (1024), 'patch' (8 mm),
%   'pitch' (1 mm), 'nact' (96), 'infl_w' (0.85 pitch, dm_influence_map's),
%   'poke_nm' (100), 'work_nm' (30), 'seed' (7), 'fourier' (true), 'dm_ap' (48 mm:
%   the aperture put ON the DM = the actuator footprint; 0 keeps the deck's), 'overfill'
%   (1.06: the source cone is opened so the beam at the DM is this times dm_ap; 0 keeps
%   the deck's cone), 'outdir', 'stages' (2; 1 = stop after stage 1, which is what
%   tg96_tail's 'reading' objective calls per evaluation), 'figs' (true; false = no
%   PNGs, for the same reason).
%   THE STOP (Dave 2026-09-17): the deck of record's beam is the SOURCE CONE, sized by
%   the builder to the baffle (2 atan(R_BAFFLE/D_SB) x FILL, a full cone), 77 mm on the
%   lens rig and 82 on the mirrors -- neither the baffle nor the DM clips a ray, and the
%   outer actuator rings are unlit.  Here the baffle is opened (x3), the cone widened so
%   the beam overfills the DM, and an aperture is put on the DM at the actuator
%   footprint: the DM IS the stop, in fact and not only by declaration.
%   Writes runs/<tag>/<tag>_{report.txt, psf.png, surface.png, gain.png,
%   pokes.png, work.png, fourier.png, .mat}.
%   Run:  >> tg96_pupilsim('rig','lens');   >> tg96_pupilsim('rig','oap');
o = struct('rig','lens','deck','','tag','','model',512,'ngrid',129, ...
           'band',3.2e-4,'rings',[0.5 1 2 3.2]*1e-4,'ring_out',1e-3,'naz',8, ...
           'lambda',6.328e-4,'dx',0.125,'N',1024,'patch',8,'pitch',1,'nact',96, ...
           'infl_w',0.85,'poke_nm',100,'work_nm',30,'seed',7,'fourier',false,'dm_ap',48,'overfill',1.06,'outdir','', ...
           'stages',2,'figs',true);
for k = 1:2:numel(varargin), o.(varargin{k}) = varargin{k+1}; end
exdir = fileparts(mfilename('fullpath'));  if isempty(exdir), exdir = pwd; end
if isempty(which('macos.init')), run(fullfile(exdir,'..','..','..','mmacos_setup.m')); end
if isempty(o.deck)
    switch o.rig
        case 'lens', o.deck = fullfile(exdir,'runs','lensuw2','lensuw2_test.in');
        case 'oap',  o.deck = fullfile(exdir,'runs','oapifo2','oapifo2_test.in');
    end
end
if isempty(o.tag), o.tag = ['pupilsim_' o.rig]; end
if isempty(o.outdir), o.outdir = fullfile(exdir,'runs',o.tag); end
if ~exist(o.outdir,'dir'), mkdir(o.outdir); end
rep = fopen(fullfile(o.outdir,[o.tag '_report.txt']),'w');
say = @(varargin) say_(rep, varargin{:});
lam = o.lambda;  k0 = 2*pi/lam;
say('=== tg96_pupilsim: tag %s  rig %s  (%s) ===\n', o.tag, o.rig, datestr(now,'yyyy-mm-dd HH:MM'));
say('deck of record %s; model %d, %d rays across; the DM is the stop; the bench as built, the field a source shift at the collimator''s focus\n', o.deck, o.model, o.ngrid);

% ======================= the deck: the bench AS BUILT, the DM flat and the stop =======================
[hdr, blocks] = split_deck_(fileread(o.deck));
names = cellfun(@(b) getv_(b,'EltName'), blocks, 'uni', 0);
iDM = find(strcmp(names,'TestOptic'), 1);  assert(~isempty(iDM), 'no TestOptic in %s', o.deck);
blocks{iDM} = regexprep(blocks{iDM}, 'Surface=\s*GridData', 'Surface=  Flat');
for key = {'nGridMat','GridFile','GridSrfdx','pData','xData','yData','zData'}
    blocks{iDM} = regexprep(blocks{iDM}, ['[ \t]*' key{1} '=[^\n]*\n'], '');
end
iBaf = find(strcmp(names,'Baffle'), 1);
if ~isempty(iBaf), rb = getv_(blocks{iBaf},'ApVec');  blocks{iBaf} = regexprep(blocks{iBaf}, 'ApVec=[^\n]*', sprintf('ApVec=  %.10E  0.0D+00  0.0D+00', 3*rb(1))); end
if o.dm_ap > 0, blocks{iDM} = regexprep(blocks{iDM}, 'ApVec=[^\n]*', sprintf('ApVec=  %.10E  0.0D+00  0.0D+00', o.dm_ap)); end
psi_dm = getv_(blocks{iDM},'psiElt');  V_dm = getv_(blocks{iDM},'VptElt');  x_dm = getv_(blocks{iDM},'xObs');  y_dm = cross(psi_dm, x_dm);
R_dm  = getv_(blocks{iDM},'ApVec');  R_dm = R_dm(1);
iDet  = numel(blocks);  psi_det = getv_(blocks{iDet},'psiElt');  V_det = getv_(blocks{iDet},'VptElt');  x_det = getv_(blocks{iDet},'xObs');  y_det = cross(psi_det, x_det);
iFoc  = find(strcmp(names,'FocalMask'),1);
iL1   = find(strcmp(names,'L1pow') | strcmp(names,'L1') | strcmp(names,'OAP1'), 1);  V_L1 = getv_(blocks{iL1},'VptElt');
src0  = getv_(hdr,'ChfRayPos');  dir0 = unit_(getv_(hdr,'ChfRayDir'));  zS = getv_(hdr,'zSource');
srcpt = src0(:)' + zS*dir0(:)';  f1 = norm(V_L1(:)' - srcpt);
hdr   = regexprep(hdr, 'nGridpts=\s*[^\n]*', sprintf('nGridpts=  %d', o.ngrid));
e1 = [0 0 1];  e2 = unit_(cross(e1, dir0));            % the source-shift plane
A0 = getv_(hdr,'Aperture');  say('deck of record: source cone (full angle) %.5f rad, baffle %s, DM aperture now %.2f mm\n', A0, iff_(isempty(iBaf),'none','opened x3'), R_dm);
if o.overfill > 0
    % one trace of the record's cone to measure the beam at the DM, then widen the cone so the beam overfills the DM's aperture
    macos.init(o.model);
    f0 = write_deck_(hdr, blocks, src0(:)', fullfile(o.outdir, [o.tag '_cone.in']));
    blocks_noap = blocks;  blocks_noap{iDM} = regexprep(blocks_noap{iDM}, 'ApVec=[^\n]*', 'ApVec=  1.0E+03  0.0D+00  0.0D+00');   % DM aperture opened for the measurement
    f0 = write_deck_(hdr, blocks_noap, src0(:)', f0);  macos.load_rx(f0);  s0 = macos.trace(iDM);  r0i = macos.get_ray_info(s0.nRays);
    ok0 = r0i.ok_trace & r0i.ok_pass;  pc0 = r0i.pos(:,ok0) - V_dm(:);  Rc0 = max(hypot(x_dm(:)'*pc0, y_dm(:)'*pc0));
    A1 = A0 * o.overfill * R_dm / Rc0;
    hdr = regexprep(hdr, 'Aperture=\s*[^\n]*', sprintf('Aperture=  %.10E', A1));
    say('the record''s cone reaches %.2f mm at the DM; cone widened to %.5f rad so the beam is %.2f mm = %.2f x the DM aperture: THE DM IS THE STOP\n', Rc0, A1, o.overfill*R_dm, o.overfill);
    delete(f0);
end
deckS = @(sh, id) write_deck_(hdr, blocks, src0(:)' + f1*(sh(1)*e1 + sh(2)*e2), fullfile(o.outdir, sprintf('%s_t%03d.in', o.tag, id)));
copyfile(deckS([0 0], 0), fullfile(o.outdir,[o.tag '_deck.in']));
say('DM elt %d, clear radius %.2f mm, the STOP; detector elt %d; source-to-collimator %.1f mm\n', iDM, R_dm, iDet, f1);

% ======================= STAGE 1: the tilt set, traced =======================
th = [0 0];
for r = o.rings, for q = 0:o.naz-1, th(end+1,:) = r*[cos(2*pi*q/o.naz) sin(2*pi*q/o.naz)]; end, end %#ok<AGROW>
for q = 0:o.naz-1, th(end+1,:) = o.ring_out*[cos(2*pi*q/o.naz) sin(2*pi*q/o.naz)]; end %#ok<AGROW>
nT = size(th,1);
macos.init(o.model);
t0 = tic;
for t = 1:nT
    f = deckS(th(t,:), t);
    macos.load_rx(f);  macos.stop(iDM);
    s1 = macos.trace(iDM);  r1 = macos.get_ray_info(s1.nRays);
    if t == 1
        nR = s1.nRays;  uv = zeros(2,nR);  okA = true(1,nR);
        thd = zeros(2,nR,nT);  ra = zeros(2,nR,nT);  aa = zeros(2,nR,nT);  Pz = zeros(1,nR,nT);
        pc = r1.pos - V_dm(:);  uv = [x_dm(:)'*pc; y_dm(:)'*pc];
        d0 = r1.dir;
    end
    okA = okA & reshape(r1.ok_trace & r1.ok_pass, 1, []);
    % the tilt each ray actually has at the DM (its direction vs the theta=0 trace), in the DM frame
    dd = r1.dir;  thd(:,:,t) = [x_dm(:)'*dd; y_dm(:)'*dd] ./ (psi_dm(:)'*dd) - [x_dm(:)'*d0; y_dm(:)'*d0] ./ (psi_dm(:)'*d0);
    sD = macos.trace(iDet);  rD = macos.get_ray_info(sD.nRays);
    okA = okA & reshape(rD.ok_trace & rD.ok_pass, 1, []);
    pd = rD.pos - V_det(:);  ra(:,:,t) = [x_det(:)'*pd; y_det(:)'*pd];  Pz(1,:,t) = psi_det(:)'*pd;
    aa(:,:,t) = [x_det(:)'*rD.dir; y_det(:)'*rD.dir] ./ (psi_det(:)'*rD.dir);
    if t == 1, uv1 = uv; else
        pc = r1.pos - V_dm(:);  duv = [x_dm(:)'*pc; y_dm(:)'*pc] - uv1;
        if t == nT, say('DM-zone stability: ray-to-DM hit moves %.2e mm rms (%.2e max) at the outer ring vs theta=0\n', sqrt(mean(duv(:,okA).^2,'all')), max(abs(duv(:,okA)),[],'all')); end
    end
end
delete(fullfile(o.outdir, [o.tag '_t*.in']));
ok = okA & reshape(all(all(isfinite(ra),1),3), 1, []);
say('stage 1: %d tilts traced in %.0f s; %d of %d rays valid at every tilt\n', nT, toc(t0), nnz(ok), nR);
R_beam = max(hypot(uv(1,ok), uv(2,ok))) + 0.5*2*f1*tan(getv_(hdr,'Aperture'))/(o.ngrid-1);   % the outermost ray sits half a ray spacing inside the beam's edge
R_beam = min(R_beam, R_dm);
say('THE BEAM at the DM: rays reach %.1f mm; beam radius %.1f mm against the DM aperture of %.1f mm (%s); the outermost lit actuator ring is at %.1f mm\n', ...
    max(hypot(uv(1,ok), uv(2,ok))), R_beam, R_dm, iff_(nnz(ok) < nR, 'the DM clips: it is the stop', 'nothing clips: the source cone is the beam'), floor(R_beam - 0.5*o.pitch - 0.5)+0.5);

% ---- per ray: the DM tilt -> exit angle map A, and the wavefront W(a) from the intercept walk ----
% basis: monomials in a = (ax, ay)/ab of total degree 2..4 (12 terms); dW/da matched to eps
ab = 0;  for t = 2:nT, ab = max(ab, max(hypot(aa(1,ok,t)-aa(1,ok,1), aa(2,ok,t)-aa(2,ok,1)))); end
ab = ab * o.band / o.ring_out;                          % the band's exit angle (the outer ring is ring_out/band larger)
[pi_, pj_] = deal([]);
for d = 2:4, for i = d:-1:0, pi_(end+1) = i; pj_(end+1) = d - i; end, end %#ok<AGROW>
nP = numel(pi_);
A = zeros(2,2,nR);  Wc = zeros(nP,nR);  Wres = nan(1,nR);  Arot = nan(1,nR);
r0 = ra(:,:,1);  a0 = aa(:,:,1);
for j = find(ok)
    dth = squeeze(thd(:,j,2:end));  da = squeeze(aa(:,j,2:end)) - a0(:,j);  de = squeeze(ra(:,j,2:end)) - r0(:,j);
    A(:,:,j) = (da / dth);                              % a = A theta (2x2 least squares over the tilts)
    ax = da(1,:)'/ab;  ay = da(2,:)'/ab;
    Gx = zeros(nT-1,nP);  Gy = Gx;
    for m = 1:nP
        if pi_(m) > 0, Gx(:,m) = pi_(m) * ax.^(pi_(m)-1) .* ay.^pj_(m); end
        if pj_(m) > 0, Gy(:,m) = pj_(m) * ax.^pi_(m) .* ay.^(pj_(m)-1); end
    end
    G = [Gx; Gy] / ab;  e = [de(1,:)'; de(2,:)'];      % dW/da = eps, W in mm when a in rad
    c = G \ e;  Wc(:,j) = c;  Wres(j) = sqrt(mean((e - G*c).^2));
    % the antisymmetric (non-gradient) part of the linear response, a check: a rotation is not a wavefront
    L = de / da;  Arot(j) = (L(1,2) - L(2,1))/2;
end
% the quadratic part: W2 = 1/2 a' Q a with Q from the ab-normalized coefficients [ax^2, ax ay, ay^2]
Q11 = 2*Wc(1,:)/ab^2;  Q12 = Wc(2,:)/ab^2;  Q22 = 2*Wc(3,:)/ab^2;
zimg = -(Q11 + Q22)/2;                                  % the zone's image: z_img downstream of the detector plane (mm)
astg = hypot((Q11 - Q22)/2, Q12);                       % astigmatic split of the image (mm)
Wb   = W_eval_(Wc, ab, ab*[cosd(0:45:315); sind(0:45:315)], pi_, pj_);   % W at the band edge, 8 azimuths (mm)
Wb_hi = W_eval_(Wc(4:end,:), ab, ab*[cosd(0:45:315); sind(0:45:315)], pi_(4:end), pj_(4:end));  % the 3rd+4th order part alone
mag = sqrt(abs(det(mean(A(:,:,ok),3))));                % exit-angle per DM-tilt = DM-mm per detector-mm
say('\n---- stage 1: the leg per zone (all in the detector frame; a = exit angle; the band edge a_b = %.3e rad = %.1f x the DM band) ----\n', ab, ab/o.band);
say('angular magnification |det A|^1/2 %.4f (DM-mm per detector-mm); rotation %.2f deg; non-gradient (rotational) part of the ray response %.2e mm/rad rms (0 for a wavefront)\n', ...
    mag, atan2d(mean(squeeze(A(2,1,ok))), mean(squeeze(A(1,1,ok)))), rms_(Arot(ok)));
say('W fit residual (mm per rad, rms over rays) %.2e; the band-edge wavefront |W| rms over zones and azimuths %.2f nm, max %.2f nm; its 3rd+4th order part %.2f nm rms, %.2f max\n', ...
    rms_(Wres(ok)), rms_(Wb(:,ok))*1e6, max(abs(Wb(:,ok)),[],'all')*1e6, rms_(Wb_hi(:,ok))*1e6, max(abs(Wb_hi(:,ok)),[],'all')*1e6);
rho = hypot(uv(1,:), uv(2,:)) / R_dm;
Qs = [ones(nnz(ok),1) uv(1,ok)'/R_dm uv(2,ok)'/R_dm rho(ok)'.^2 (uv(1,ok).^2-uv(2,ok).^2)'/R_dm^2 (2*uv(1,ok).*uv(2,ok))'/R_dm^2];
cz = Qs \ zimg(ok)';
say('the image surface (each zone''s image, mm DOWNSTREAM of the detector plane): on axis %+.3f; over the pupil mean %+.3f, min %+.3f, max %+.3f; fit: piston %+.3f tilt [%+.3f %+.3f] defocus %+.3f astig [%+.3f %+.3f] (mm over the pupil radius)\n', ...
    cz(1), mean(zimg(ok)), min(zimg(ok)), max(zimg(ok)), cz(1), cz(2), cz(3), cz(4), cz(5), cz(6));
say('astigmatic split of the zone images: rms %.3f mm, max %.3f mm (outer third %.3f rms)\n', rms_(astg(ok)), max(astg(ok)), rms_(astg(ok & rho>2/3)));
% the compromise plane: the detector shift that minimizes the rms image offset over the lit pupil
zstar = mean(zimg(ok));   zmed = median(zimg(ok));
phi_edge = @(dz) k0 * abs(zimg(ok) - dz) * ab^2/2;      % the band-edge quadratic phase per zone (rad)
say('BEST PUPIL IMAGE: at the plane as built the band-edge (Nyquist) quadratic phase is %.3f rad rms / %.3f max over the pupil; moving the detector %+.2f mm downstream (mean) gives %.3f / %.3f; the astigmatic residual there %.3f rad rms\n', ...
    rms_(phi_edge(0)), max(phi_edge(0)), zstar, rms_(phi_edge(zstar)), max(phi_edge(zstar)), rms_(k0*astg(ok)*ab^2/2));
say('  (phase gain at the band edge = cos(phi): %.4f worst as built, %.4f worst at the compromise plane; amplitude cross-talk = sin(phi): %.3f / %.3f worst)\n', ...
    cos(max(phi_edge(0))), cos(max(phi_edge(zstar))), sin(max(phi_edge(0))), sin(max(phi_edge(zstar))));
% the distortion (as yesterday: the theta=0 intercept vs one global affine) for the record
M = [uv(1,ok); uv(2,ok); ones(1,nnz(ok))]';  Ax = M \ r0(1,ok)';  Ay = M \ r0(2,ok)';
magL = sqrt(abs(det([Ax(1:2)'; Ay(1:2)'])));  dist = [r0(1,ok)' - M*Ax, r0(2,ok)' - M*Ay] / magL;
say('pupil distortion vs one global affine (DM mm): rms %.3f, max %.3f; lateral magnification %.4f DM-mm per detector-mm\n', rms_(hypot(dist(:,1),dist(:,2))), max(hypot(dist(:,1),dist(:,2))), 1/magL);

% ---- the zone lattice for stage 2: patch centers, mean coefficients ----
pc_ = -ceil(R_beam/o.patch)*o.patch : o.patch : ceil(R_beam/o.patch)*o.patch;
[PU, PV] = meshgrid(pc_, pc_);  nZ = numel(PU);
Z = struct('u',{},'v',{},'A',{},'Wc',{},'n',{},'zimg',{});
for z = 1:nZ
    in = ok & abs(uv(1,:) - PU(z)) <= o.patch/2 & abs(uv(2,:) - PV(z)) <= o.patch/2;
    if nnz(in) < 3, continue; end
    Z(end+1) = struct('u',PU(z),'v',PV(z),'A',mean(A(:,:,in),3),'Wc',mean(Wc(:,in),2),'n',nnz(in),'zimg',mean(zimg(in))); %#ok<AGROW>
end
say('zone lattice for the field model: %d patches of %g mm (%d with rays)\n', nZ, o.patch, numel(Z));

% ---- the complex PSFs at three zones (the deliverable Dave asked for) ----
N = o.N;  dx = o.dx;  fx = ifftshift((-N/2:N/2-1)/(N*dx));  [FU, FV] = meshgrid(fx, fx);   % cycles/mm on the DM
xg = (-N/2:N/2-1)*dx;  [XG, YG] = meshgrid(xg, xg);
Hof = @(z, dz) H_zone_(z, dz, FU, FV, lam, k0, ab, pi_, pj_, o.ring_out);
[~, jc] = min(hypot([Z.u], [Z.v]));  [~, je] = max(hypot([Z.u], [Z.v]));  [~, jm] = min(abs(hypot([Z.u], [Z.v]) - 0.6*R_beam));
if o.figs
fp = figure('Visible','off','Position',[100 100 1500 900]);
for q = 1:3
    jz = [jc jm je];  jz = jz(q);  h = fftshift(ifft2(Hof(Z(jz), 0)));  hc = h(N/2+1, :);
    subplot(2,3,q); semilogy(xg, abs(hc)/max(abs(hc)), '-'); xlim([-6 6]); grid on; xlabel('DM mm'); ylabel('|PSF| / peak');
    title(sprintf('zone (%g, %g) mm: |h| cut; image %+.2f mm from the plane', Z(jz).u, Z(jz).v, Z(jz).zimg));
    Hz = fftshift(Hof(Z(jz), 0));
    subplot(2,3,3+q); imagesc(fftshift(fx), fftshift(fx), angle(Hz)); axis image; colorbar; xlim([-1 1]*1.2); ylim([-1 1]*1.2); hold on;
    tt = linspace(0,2*pi,100); plot(0.5*cos(tt), 0.5*sin(tt), 'w-'); xlabel('f_u, cycles/mm'); ylabel('f_v'); title('transfer phase arg H(f), rad (circle: actuator Nyquist)');
end
sgtitle(sprintf('%s: the detector leg''s complex PSF and transfer function at three DM zones (%s rig, as built)', o.tag, o.rig), 'Interpreter','none');
print(fp, fullfile(o.outdir,[o.tag '_psf.png']), '-dpng', '-r96');
fs = figure('Visible','off','Position',[100 100 1500 500]);
subplot(1,3,1); scatter(uv(1,ok), uv(2,ok), 8, zimg(ok), 'filled'); axis equal; colorbar; title('zone image vs the detector plane, mm (+ = downstream)'); xlabel('DM mm');
subplot(1,3,2); scatter(uv(1,ok), uv(2,ok), 8, astg(ok), 'filled'); axis equal; colorbar; title('astigmatic split of the image, mm');
subplot(1,3,3); scatter(uv(1,ok), uv(2,ok), 8, max(abs(Wb(:,ok)),[],1)*1e6, 'filled'); axis equal; colorbar; title('band-edge wavefront |W|, nm (max over azimuth)');
sgtitle(sprintf('%s: the pupil image surface (%s rig): the detector is not at the image', o.tag, o.rig), 'Interpreter','none');
print(fs, fullfile(o.outdir,[o.tag '_surface.png']), '-dpng', '-r96');
end

% ---- stop here when only the pupil SURFACE was asked for -------------
% tg96_tail's 'reading' objective (Dave 2026-09-17: the tail is tuned on
% the interferometer's own reading, not on the null) calls this per
% evaluation, so it wants stage 1 and no figures: the band-edge quadratic
% phase over the pupil is the stage-1 proxy for the working-surface error
% stage 2 measures, and the two track each other on every case run.
if o.stages < 2
    out = struct('o',o,'th',th,'uv',uv,'ok',ok,'r0',r0,'A',A,'Wc',Wc,'ab',ab,'pi',pi_,'pj',pj_, ...
                 'zimg',zimg,'astg',astg,'Wb',Wb,'cz',cz,'zstar',zstar,'Z',Z,'dist',dist,'magL',magL, ...
                 'phi_rms',rms_(phi_edge(0)), 'phi_max',max(phi_edge(0)), ...
                 'phi_rms_star',rms_(phi_edge(zstar)), 'phi_max_star',max(phi_edge(zstar)), ...
                 'dist_rms',rms_(hypot(dist(:,1),dist(:,2))), 'res',[]);
    say('stage 1 only (stages=1): band-edge phase %.4f rad rms / %.4f max as built; stopping before the field model\n', out.phi_rms, out.phi_max);
    fclose(rep);  return
end

% ======================= STAGE 2: the DM field through the zone PSFs =======================
R_ap = R_beam;  edge_mm = 0.7;                          % the beam of record; its edge is the baffle's, Fresnel-blurred over ~sqrt(lambda z) = 0.7 mm by the time it reaches the DM
rr = hypot(XG, YG);  ap = double(rr <= R_ap - edge_mm) + (rr > R_ap - edge_mm & rr <= R_ap + edge_mm) .* 0.5 .* (1 + cos(pi*(rr - R_ap + edge_mm)/(2*edge_mm)));
Rin = R_beam - 1.0;  lit = rr <= Rin;                      % scoring region: the lit actuators, one pitch inside the beam edge
R_lit = floor(Rin - 0.5*o.pitch)+0.5*o.pitch;              % the outermost lit actuator ring
win = @(u0, v0) max(0, cos(pi*(XG-u0)/(2*o.patch))).^2 .* max(0, cos(pi*(YG-v0)/(2*o.patch))).^2 .* (abs(XG-u0) < o.patch) .* (abs(YG-v0) < o.patch);
Wsum = zeros(N);  for z = 1:numel(Z), Wsum = Wsum + win(Z(z).u, Z(z).v); end
say('partition of unity over the lit pupil: min %.3f, max %.3f (the windows are renormalized)\n', min(Wsum(ap>0)), max(Wsum(ap>0)));
Wsum(Wsum == 0) = 1;
apply = @(E, dz) apply_zones_(E, Z, dz, Hof, win, Wsum);
dzs = [0 zstar];  dznm = {'as built', sprintf('detector %+.2f mm downstream', zstar)};
% test surfaces (mm)
freqs = [0.5 0.25 0.125 0.0625];  amp_s = 5e-6;
% the influence function (dm_influence_map's Gaussian, 1/e radius infl_w x pitch)
w_mm = o.infl_w * o.pitch;  infl = @(u0, v0) exp(-((XG-u0).^2 + (YG-v0).^2)/w_mm^2);
sites = [0.5 0.5; 0.6*R_lit 0.5; 0.9*R_lit 0.5; R_lit 0.5; 0.5 R_lit; R_lit/sqrt(2) R_lit/sqrt(2)];  sites = round(sites - 0.5)+0.5;   % actuator centres: the last three on the outermost LIT ring
rng(o.seed);  cmd = o.work_nm*1e-6*randn(o.nact);  xa = ((1:o.nact) - (o.nact+1)/2)*o.pitch;
hw = zeros(N);  for ia = 1:o.nact, for ja = 1:o.nact, if hypot(xa(ia), xa(ja)) <= R_dm, hw = hw + cmd(ia,ja)*infl(xa(ia), xa(ja)); end, end, end
say('working surface: %d x %d actuators at %g mm, %g nm rms commands (seed %d), influence 1/e radius %.2f mm -> surface %.1f nm rms in the lit pupil\n', o.nact, o.nact, o.pitch, o.work_nm, o.seed, w_mm, std(hw(lit))*1e6);
res = struct();
for id = 1:numel(dzs)
    dz = dzs(id);
    Er = apply(ap, dz);                                     % the reference arm (a flat) through the same leg
    read = @(h) angle(apply(ap.*exp(1i*4*pi*h/lam), dz) .* conj(Er)) * lam/(4*pi);
    ampl = @(h) abs(apply(ap.*exp(1i*4*pi*h/lam), dz)) ./ max(abs(Er), 1e-6) - 1;
    say('\n---- stage 2, %s: the DM field through the zone PSFs ----\n', dznm{id});
    say('flat DM: |E| over the lit actuators %.4f mean, %.4f rms variation; outermost lit ring %.4f rms (the beam edge''s Fresnel ringing)\n', mean(abs(Er(lit))), std(abs(Er(lit))), std(abs(Er(lit & rr > Rin - o.pitch))));
    % sinusoids: gain and amplitude cross-talk vs radius by demodulation
    G = struct('f',{},'dir',{},'gain_r',{},'xt_r',{},'gmin',{},'gmean',{},'xtmax',{});
    rb = 0:0.1:1;  rc = rho_grid_(XG, YG, Rin);
    for f = freqs, for dirn = 1:2
        if dirn == 1, h = amp_s*sin(2*pi*f*XG); car = exp(-1i*2*pi*f*XG); else, h = amp_s*sin(2*pi*f*YG); car = exp(-1i*2*pi*f*YG); end
        ho = read(h);  am = ampl(h);
        gmap = abs(lpf_(ho.*car.*ap, FU, FV, 2/f)) ./ max(abs(lpf_(h.*car.*ap, FU, FV, 2/f)), 1e-12);   % both sides masked by the pupil: the ratio is fair at the edge
        xmap = abs(lpf_(am.*car.*ap, FU, FV, 2/f)) ./ max(abs(lpf_(h.*car.*ap, FU, FV, 2/f)), 1e-12) * lam/(4*pi);   % amplitude modulation per unit phase modulation
        gr = zeros(1,numel(rb)-1);  xr = gr;
        for b = 1:numel(rb)-1, m = lit & rc >= rb(b) & rc < rb(b+1); gr(b) = mean(gmap(m)); xr(b) = mean(xmap(m)); end
        G(end+1) = struct('f',f,'dir',dirn,'gain_r',gr,'xt_r',xr,'gmin',min(gmap(lit)),'gmean',mean(gmap(lit)),'xtmax',max(xmap(lit))); %#ok<AGROW>
        say('  sinusoid f %.4f cyc/mm (%5.1f mm period) along %s: gain mean %.4f, min %.4f in the lit pupil; by radius [%s]; amplitude cross-talk per unit phase max %.3f\n', ...
            f, 1/f, iff_(dirn==1,'u','v'), mean(gmap(lit)), min(gmap(lit)), sprintf('%.3f ', gr), max(xmap(lit)));
    end, end
    % single pokes
    PK = struct('site',{},'peak',{},'width',{},'shift',{});
    say('  single pokes (%g nm, influence 1/e %.2f mm): recovered peak / true, 1/e width / true, centroid shift (DM mm)\n', o.poke_nm, w_mm);
    for s = 1:size(sites,1)
        h = o.poke_nm*1e-6*infl(sites(s,1), sites(s,2));  ho = read(h);
        m = hypot(XG-sites(s,1), YG-sites(s,2)) <= 4*o.pitch;
        pk = max(ho(m))/max(h(m));  m2 = hypot(XG-sites(s,1), YG-sites(s,2)) <= 2*o.pitch;  wgt = max(ho,0).*m2;  cx = sum(XG(:).*wgt(:))/sum(wgt(:));  cy = sum(YG(:).*wgt(:))/sum(wgt(:));
        wd = sqrt(nnz(ho.*m > max(ho(m))/exp(1))*dx^2/pi) / w_mm;   % 1/e radius from the area above 1/e of the peak, over the true 1/e radius
        PK(end+1) = struct('site',sites(s,:),'peak',pk,'width',wd,'shift',hypot(cx-sites(s,1), cy-sites(s,2))); %#ok<AGROW>
        say('    site (%5.1f, %5.1f) rho %.2f: peak %.4f, width %.3f, shift %.4f mm\n', sites(s,1), sites(s,2), hypot(sites(s,1),sites(s,2))/Rin, pk, wd, PK(end).shift);
    end
    % the working surface
    ho = read(hw);  d = (ho - hw);  Ap = [ones(nnz(lit),1) XG(lit) YG(lit)];  d(lit) = d(lit) - Ap*(Ap\d(lit));  d(~lit) = 0;   % piston and tilt removed: the interferometer never reads them
    dsp = abs(fft2(d.*lit));  hsp = abs(fft2(hw.*lit));  fr = hypot(FU, FV);
    say('  working surface %.1f nm rms: recovered - true %.3f nm rms in the lit pupil, piston and tilt removed (%.4f of the surface); error / surface by band: %s\n', std(hw(lit))*1e6, std(d(lit))*1e6, std(d(lit))/std(hw(lit)), ...
        sprintf('%s', band_str_(dsp, hsp, fr)));
    res(id).dz = dz;  res(id).G = G;  res(id).PK = PK;  res(id).work_err_nm = std(d(lit))*1e6;  res(id).work_map = d;  res(id).Er = Er;
    res(id).work_out = ho;
end
% figures
fg = figure('Visible','off','Position',[100 100 1500 500]);
rmid = (rb(1:end-1)+rb(2:end))/2;  cols = lines(numel(freqs));
for id = 1:2
    subplot(1,3,id); hold on;
    for g = res(id).G, if g.dir == 1, plot(rmid, g.gain_r, '-o', 'Color', cols(freqs==g.f,:), 'DisplayName', sprintf('%.3f cyc/mm', g.f)); else, plot(rmid, g.gain_r, '--', 'Color', cols(freqs==g.f,:), 'HandleVisibility','off'); end, end
    xlabel('radius / lit radius'); ylabel('phase gain (recovered / true)'); title(sprintf('gain vs radius, %s', dznm{id})); legend('Location','southwest'); grid on; ylim([min(0.9, min([res(id).G.gmin])-0.01) 1.01]);
end
subplot(1,3,3); hold on;
for g = res(1).G, if g.dir == 1, plot(rmid, g.xt_r, '-o', 'Color', cols(freqs==g.f,:), 'DisplayName', sprintf('%.3f cyc/mm', g.f)); end, end
xlabel('radius / lit radius'); ylabel('amplitude cross-talk per unit phase'); title('amplitude modulation (as built; solid u, dashed v)'); legend('Location','northwest'); grid on;
sgtitle(sprintf('%s: DM mode observability through the detector leg (%s rig)', o.tag, o.rig), 'Interpreter','none');
print(fg, fullfile(o.outdir,[o.tag '_gain.png']), '-dpng', '-r96');
fw = figure('Visible','off','Position',[100 100 1500 480]);
subplot(1,3,1); imagesc(xg, xg, hw*1e6.*lit); axis image; colorbar; title(sprintf('working surface, nm (%.1f rms)', std(hw(lit))*1e6));
subplot(1,3,2); imagesc(xg, xg, res(1).work_map*1e6); axis image; colorbar; title(sprintf('recovered - true, nm (%.3f rms), as built', res(1).work_err_nm));
subplot(1,3,3); imagesc(xg, xg, res(2).work_map*1e6); axis image; colorbar; title(sprintf('recovered - true, nm (%.3f rms), compromise plane', res(2).work_err_nm));
sgtitle(sprintf('%s: the record''s 30 nm working surface through the leg (%s rig)', o.tag, o.rig), 'Interpreter','none');
print(fw, fullfile(o.outdir,[o.tag '_work.png']), '-dpng', '-r96');

out = struct('o',o,'th',th,'uv',uv,'ok',ok,'r0',r0,'A',A,'Wc',Wc,'ab',ab,'pi',pi_,'pj',pj_,'zimg',zimg,'astg',astg,'Wb',Wb,'cz',cz,'zstar',zstar,'Z',Z,'res',res,'dist',dist,'mag',mag,'magL',magL);

% ======================= STAGE 3: the Fourier cross-check =======================
if o.fourier
    try
        out.F3 = fourier_tail_(o, blocks, names, iDM, iFoc, iDet, R_dm, R_beam, hw, sites, freqs, amp_s, lam, say, out);
    catch ME
        say('stage 3 (Fourier cross-check) FAILED: %s\n', ME.message);
        for s = ME.stack(1:min(3,end))', say('   at %s line %d\n', s.name, s.line); end
    end
end
save(fullfile(o.outdir,[o.tag '.mat']), 'out');
say('run complete\n');  fclose(rep);
end

% ============================================================================================
function E2 = apply_zones_(E, Z, dz, Hof, win, Wsum)
E2 = zeros(size(E));
for z = 1:numel(Z)
    w = win(Z(z).u, Z(z).v) ./ Wsum;
    if ~any(w(:)), continue; end
    E2 = E2 + ifft2(Hof(Z(z), dz) .* fft2(w .* E));
end
end
function H = H_zone_(z, dz, FU, FV, lam, k0, ab, pi_, pj_, thmax)
% the zone's coherent transfer function on the DM's frequency grid: a DM sinusoid at f
% diffracts at theta = lambda f (DM frame); the leg maps it to the exit angle a = A theta;
% H = exp(-i k W(a)) with W the fitted wavefront (+ dz |a|^2/2 for a detector moved dz downstream)
th = lam * [FU(:)'; FV(:)'];  a = z.A * th;
W = W_eval_(z.Wc, ab, a, pi_, pj_) + dz * sum(a.^2, 1)/2;
H = exp(-1i * k0 * W);   % a pure phase on the whole grid: beyond the traced tilt range (1e-3 rad = 1.6 cyc/mm) the quadratic
                         % part extrapolates as physics does and the (sub-nm) higher orders are harmless; a hard cut would ring the pupil edge
H = reshape(H, size(FU));
end
function W = W_eval_(Wc, ab, a, pi_, pj_)
% W(a) = sum_m c_m (ax/ab)^i (ay/ab)^j ; Wc is nP x nZ, a is 2 x nA -> W is nA x nZ (or 1 x nA for one zone)
ax = a(1,:)'/ab;  ay = a(2,:)'/ab;  P = zeros(numel(ax), numel(pi_));
for m = 1:numel(pi_), P(:,m) = ax.^pi_(m) .* ay.^pj_(m); end
W = P * Wc;  if size(Wc,2) == 1, W = W'; end
end
function m = lpf_(x, FU, FV, sig)
m = ifft2(fft2(x) .* exp(-2*pi^2*sig^2*(FU.^2 + FV.^2)));
end
function r = rho_grid_(XG, YG, R), r = hypot(XG, YG)/R; end
function s = band_str_(dsp, hsp, fr)
edges = [0 0.0625 0.125 0.25 0.5 10];  s = '';
for b = 1:numel(edges)-1
    m = fr >= edges(b) & fr < edges(b+1);  s = [s sprintf('[%.3f-%.3f cyc/mm: %.4f] ', edges(b), min(edges(b+1),0.5), sqrt(sum(dsp(m).^2)/max(sum(hsp(m).^2),1e-30)))]; %#ok<AGROW>
end
end
function s = iff_(c, a, b), if c, s = a; else, s = b; end, end

% ============================================================================================
function F3 = fourier_tail_(o, blocks, names, iDM, iFoc, iDet, R_dm, R_beam, hw, sites, freqs, amp_s, lam, say, S1)
%FOURIER_TAIL_  Plane-to-plane Fourier propagation of the detector leg (the cross-check).
%   Paraxial legs (single-FFT Fresnel / angular spectrum, Sziklas-Siegman scaled where the
%   beam diverges), the focuser and the field lens as EXACT thick-lens phase screens from a
%   meridional trace of the deck's own surfaces.  Distances along the chief ray from the deck.
%   The DM field on a finer grid (0.125 mm, 1024) so the field-lens plane is sampled at the
%   chirp's Nyquist.  Output: the detector field on the detector grid, resampled onto the DM
%   frame through the ray map of stage 1, read out as in stage 2 and compared.
k0 = 2*pi/lam;
V = @(i) getv_(blocks{i},'VptElt');  chief = @(i, j) norm(V(j) - V(i));
% the tail's elements along the chief: focuser (L2pow/L2flat or OAP2), FocalMask, FLpow, FLflat, Detector
iFLp = find(strcmp(names,'FLpow'),1);  iFLf = find(strcmp(names,'FLflat'),1);
if strcmp(o.rig,'lens')
    iL2p = find(strcmp(names,'L2pow'),1);  iL2f = find(strcmp(names,'L2flat'),1);
    z_dm_L2 = chief(iDM, iL2p);  tL2 = chief(iL2p, iL2f);  z_L2_foc = chief(iL2f, iFoc);
    L2 = struct('Kr',getv_(blocks{iL2p},'KrElt'),'Kc',getv_(blocks{iL2p},'KcElt'),'n',getv_(blocks{iL2p},'IndRef'),'t',tL2);
else
    iL2p = find(strcmp(names,'OAP2'),1);  z_dm_L2 = chief(iDM, iL2p);  z_L2_foc = chief(iL2p, iFoc);
    L2 = struct('Kr',getv_(blocks{iL2p},'KrElt'),'Kc',-1,'n',NaN,'t',0);
end
tFL = chief(iFLp, iFLf);  z_foc_FL = chief(iFoc, iFLp);  z_FL_det = chief(iFLf, iDet);
FL = struct('Kr',getv_(blocks{iFLp},'KrElt'),'Kc',getv_(blocks{iFLp},'KcElt'),'n',getv_(blocks{iFLp},'IndRef'),'t',tFL);
% best focus from the stage-1 rays is not carried here; the FocalMask marker is the deck's focus reference and
% the exact L2 screen puts the focus where the lens does.  Distances are all from the deck.
say('\n---- stage 3: Fourier propagation of the tail (paraxial legs, exact thick-lens screens) ----\n');
say('chief-ray legs (mm): DM -> focuser %.2f; focuser thickness %.2f; focuser -> mask marker %.2f; mask -> FL %.2f; FL thickness %.2f; FL -> detector %.2f\n', z_dm_L2, L2.t, z_L2_foc, z_foc_FL, FL.t, z_FL_det);
if ~strcmp(o.rig,'lens'), say('  (OAP rig: the focuser is a parabola; the Fourier model treats it as an ideal focuser of the OAP''s focal length, so its field-linear coma is NOT in this check)\n'); end
% ---- the grids ----
N = 1024;  dx = 0.125;  xg = (-N/2:N/2-1)*dx;  [XG, YG] = meshgrid(xg, xg);
fx = ifftshift((-N/2:N/2-1)/(N*dx));  [FU, FV] = meshgrid(fx, fx);
rr = hypot(XG, YG);  edge_mm = 0.7;
ap = double(rr <= R_beam - edge_mm) + (rr > R_beam - edge_mm & rr <= R_beam + edge_mm) .* 0.5 .* (1 + cos(pi*(rr - R_beam + edge_mm)/(2*edge_mm)));
Rin = R_beam - 1.0;  lit = rr <= Rin;
w_mm = o.infl_w * o.pitch;  infl = @(u0, v0) exp(-((XG-u0).^2 + (YG-v0).^2)/w_mm^2);   % on this stage's own grid
% ---- leg A: DM -> focuser front face, collimated, angular spectrum (exact kernel) ----
Hfree = @(z) exp(1i*k0*z*real(sqrt(1 - (lam*FU).^2 - (lam*FV).^2)));
% ---- the focuser: exact OPL screen for a collimated beam through the thick lens; f_par from the trace ----
if strcmp(o.rig,'lens')
    [phiL2, f2, bfd] = lens_screen_collimated_(L2, lam, max(abs(xg)));
else
    f2 = z_L2_foc;  bfd = z_L2_foc;  phiL2 = @(r) zeros(size(r));   % ideal parabola for the on-axis beam; the marker IS the focus on this rig
end
say('focuser: paraxial focal length %.2f mm from the exact trace (thin-lens R/(n-1) %.2f); the marker sits %.2f mm from the powered face\n', f2, abs(L2.Kr)/max(L2.n-1,0.5), z_L2_foc + L2.t);
% the field right after the focuser, relative to the converging sphere of radius f2 (paraxial), = pupil field x aberration screen
% then a single-FFT Fresnel to the FOCAL PLANE (at f2 from the rear principal plane): E_f = FFT[E_pupil], sampling lam f2/(N dx)
dfoc = lam*bfd/(N*dx);  xf = (-N/2:N/2-1)*dfoc;  [XF, YF] = meshgrid(xf, xf);
% ---- leg C: focus -> FL: single-FFT Fresnel from the focal plane over z1 (the field there relative to the diverging sphere) ----
z1 = (z_L2_foc + z_foc_FL) - bfd;                            % the exact focus (bfd behind the focuser's exit face) to the FL's front face; the deck's marker need not be the focus
say('the exact focus sits %.2f mm behind the focuser''s exit face; the deck''s mask marker at %.2f (%+.2f mm from the focus); focus -> FL %.2f mm\n', bfd, z_L2_foc, z_L2_foc - bfd, z1);
dFL = lam*z1/(N*dfoc);  xFLg = (-N/2:N/2-1)*dFL;  [XL, YL] = meshgrid(xFLg, xFLg);
say('sampling: focal plane %.2f um over %.2f mm; field-lens plane %.1f um over %.2f mm (beam there %.1f mm)\n', dfoc*1e3, N*dfoc, dFL*1e3, N*dFL, 2*R_beam*z1/f2);
% ---- the field lens: exact OPL screen for a point source at distance z1 through the thick lens, relative to the paraxial sphere ----
[phiFL, fFL, zimgFL] = lens_screen_point_(FL, lam, z1, max(abs(xFLg)));
s_obj2 = f2*f2/(z_dm_L2 - f2) - z1;                          % the DM's image by the focuser sits f2^2/(z-f2) behind the focus: a virtual object this far behind the FL
say('field lens: paraxial focal length %.2f mm; the on-axis image of the focus %.1f mm behind its rear face; thin-lens DM image %.2f mm behind the FL (virtual object %.0f mm behind it) vs the detector %.2f behind its rear face\n', ...
    fFL, zimgFL, 1/(1/fFL + 1/s_obj2), s_obj2, z_FL_det);
% ---- leg E: after the FL, to the detector.  The beam after the FL is referenced to the sphere through the on-axis image of the
%      focus (radius Rout = zimgFL, negative = virtual/diverging): Sziklas-Siegman scaled angular-spectrum over z2 ----
z2 = z_FL_det;  Rout = zimgFL;  m_ss = 1 - z2/Rout;  zeff = z2/m_ss;   % Rout > 0: converging to a real point; < 0: diverging from a virtual one
say('after the FL: reference sphere radius %+.1f mm, scale to the detector %.4f (pupil image %.2f mm wide; stage 1 lateral magnification gives %.2f)\n', Rout, m_ss, 2*R_beam*z1/f2*m_ss, 2*R_beam/(1/S1.magL));
ddet = dFL*m_ss;  xdet = (-N/2:N/2-1)*ddet;
Hdet = exp(1i*k0*zeff*real(sqrt(1 - (lam*ifftshift((-N/2:N/2-1)/(N*dFL))').^2 - (lam*ifftshift((-N/2:N/2-1)/(N*dFL))).^2)));
prop = @(Edm) tail_prop_(Edm, Hfree(z_dm_L2), phiL2(hypot(XG,YG)), XF, YF, z1, bfd, lam, phiFL(hypot(XL,YL)), Hdet);
% ---- the ray map from stage 1: DM (u,v) -> detector (x,y), a cubic fit, to resample the detector field onto the DM frame ----
ok = S1.ok;  u = S1.uv(1,ok)'/R_dm;  v = S1.uv(2,ok)'/R_dm;
Pm = [ones(size(u)) u v u.^2 u.*v v.^2 u.^3 u.^2.*v u.*v.^2 v.^3];
cx = Pm \ S1.r0(1,ok)';  cy = Pm \ S1.r0(2,ok)';
ug = XG/R_dm;  vg = YG/R_dm;  Pg = @(f) f(1) + f(2)*ug + f(3)*vg + f(4)*ug.^2 + f(5)*ug.*vg + f(6)*vg.^2 + f(7)*ug.^3 + f(8)*ug.^2.*vg + f(9)*ug.*vg.^2 + f(10)*vg.^3;
XD = Pg(cx);  YD = Pg(cy);
% the Fourier model's own image centre and scale are its own; align by the flat-pupil field's centroid and the ray affine's scale
Eflat = prop(ap);  I = abs(Eflat).^2;  [XDg, YDg] = meshgrid(xdet, xdet);
cxF = sum(XDg(:).*I(:))/sum(I(:));  cyF = sum(YDg(:).*I(:))/sum(I(:));
% the model's magnification: the flat pupil's equivalent radius vs R_dm
rF = sqrt(sum(I(:) > 0.5*median(I(I > 0.05*max(I(:)))))*ddet^2/pi);  magF = R_beam/rF;
say('Fourier flat pupil at the detector: centroid [%+.3f %+.3f] mm, equivalent radius %.3f mm -> magnification %.4f DM-mm per detector-mm (stage 1: %.4f)\n', cxF, cyF, rF, magF, 1/S1.magL);
% resample: detector field at the ray-mapped positions of each DM grid point; the ray map is in the detector's own frame
% whose origin/scale may differ from the model's by a shift and the frame's handedness -- use the model's own scale and the
% ray map's shape (distortion) about the ray affine
Lm = [S1.magL 0; 0 S1.magL];  %#ok<NASGU>
sh = [mean(S1.r0(1,ok)) mean(S1.r0(2,ok))];
XDm = (XD - sh(1)) * (1/S1.magL) / magF + cxF;  YDm = (YD - sh(2)) * (1/S1.magL) / magF + cyF;   % model frame: detector mm
% handedness: compare the ray-map's orientation (sign of the affine's determinant) with the model's (the FFT chain
% inverts the image ONCE per Fourier leg; two legs -> upright... determined empirically from the flat pupil's response to a poke)
samp = @(E) interp2(XDg, YDg, E, XDm, YDm, 'linear', 0);
Er = samp(Eflat);
% orientation check with an off-centre poke: the recovered peak must land on the poke
h = o.poke_nm*1e-6*infl(sites(3,1), sites(3,2));
Et = samp(prop(ap.*exp(1i*4*pi*h/lam)));  ho = angle(Et.*conj(Er))*lam/(4*pi);
[~, im] = max(ho(:).*lit(:));  say('orientation: poke at (%.1f, %.1f) recovered peak at (%.1f, %.1f) DM mm', sites(3,1), sites(3,2), XG(im), YG(im));
flipu = sign(XG(im)*sites(3,1));  flipv = 1;
if flipu < 0 || abs(XG(im) - sites(3,1)) > 3
    % try the mirrored map
    for fu = [1 -1], for fv = [1 -1]
        XDm2 = (fu*(XD - sh(1))) * (1/S1.magL) / magF + cxF;  YDm2 = (fv*(YD - sh(2))) * (1/S1.magL) / magF + cyF;
        Et2 = interp2(XDg, YDg, prop(ap.*exp(1i*4*pi*h/lam)), XDm2, YDm2, 'linear', 0);  Er2 = interp2(XDg, YDg, Eflat, XDm2, YDm2, 'linear', 0);
        ho2 = angle(Et2.*conj(Er2))*lam/(4*pi);  [~, im2] = max(ho2(:).*lit(:));
        if hypot(XG(im2)-sites(3,1), YG(im2)-sites(3,2)) < 1.5, XDm = XDm2; YDm = YDm2; samp = @(E) interp2(XDg, YDg, E, XDm, YDm, 'linear', 0); Er = samp(Eflat); say(' -> map flipped [%d %d]', fu, fv); break; end
    end, end
end
say('\n');
read = @(h) angle(samp(prop(ap.*exp(1i*4*pi*h/lam))) .* conj(Er)) * lam/(4*pi);
say('flat DM (Fourier): |E| over the lit pupil %.4f mean, %.4f rms variation\n', mean(abs(Er(lit))), std(abs(Er(lit))));
G = struct('f',{},'dir',{},'gain_r',{},'gmin',{},'gmean',{});  rb = 0:0.1:1;  rc = hypot(XG,YG)/Rin;
for f = freqs, for dirn = 1:2
    if dirn == 1, h = amp_s*sin(2*pi*f*XG); car = exp(-1i*2*pi*f*XG); else, h = amp_s*sin(2*pi*f*YG); car = exp(-1i*2*pi*f*YG); end
    ho = read(h);
    gmap = abs(lpf_(ho.*car.*ap, FU, FV, 2/f)) ./ max(abs(lpf_(h.*car.*ap, FU, FV, 2/f)), 1e-12);
    gr = zeros(1,numel(rb)-1);  for b = 1:numel(rb)-1, m = lit & rc >= rb(b) & rc < rb(b+1); gr(b) = mean(gmap(m)); end
    G(end+1) = struct('f',f,'dir',dirn,'gain_r',gr,'gmin',min(gmap(lit)),'gmean',mean(gmap(lit))); %#ok<AGROW>
    say('  Fourier sinusoid f %.4f along %s: gain mean %.4f, min %.4f; by radius [%s]\n', f, iff_(dirn==1,'u','v'), mean(gmap(lit)), min(gmap(lit)), sprintf('%.3f ', gr));
end, end
say('  single pokes (Fourier): recovered peak / true\n');
PK = [];
for s = 1:size(sites,1)
    h = o.poke_nm*1e-6*infl(sites(s,1), sites(s,2));  ho = read(h);  m = hypot(XG-sites(s,1), YG-sites(s,2)) <= 4*o.pitch;
    PK(s) = max(ho(m))/max(h(m));  say('    site (%5.1f, %5.1f): peak %.4f\n', sites(s,1), sites(s,2), PK(s)); %#ok<AGROW>
end
hwF = interp2(S1.o.dx*(-S1.o.N/2:S1.o.N/2-1), S1.o.dx*(-S1.o.N/2:S1.o.N/2-1)', hw, XG, YG, 'linear', 0);
ho = read(hwF);  d = ho - hwF;  Ap = [ones(nnz(lit),1) XG(lit) YG(lit)];  d(lit) = d(lit) - Ap*(Ap\d(lit));  d(~lit) = 0;
say('  working surface (Fourier): recovered - true %.3f nm rms in the lit pupil, piston and tilt removed\n', std(d(lit))*1e6);
% comparison figure: stage-2 vs Fourier gain vs radius
ff = figure('Visible','off','Position',[100 100 1000 450]);
rmid = (rb(1:end-1)+rb(2:end))/2;  cols = lines(numel(freqs));  hold on;
for g = S1.res(1).G, if g.dir == 1, plot(rmid, g.gain_r, '-o', 'Color', cols(freqs==g.f,:), 'DisplayName', sprintf('zone PSFs %.3f cyc/mm', g.f)); end, end
for g = G, if g.dir == 1, plot(rmid, g.gain_r, ':s', 'Color', cols(freqs==g.f,:), 'DisplayName', sprintf('Fourier %.3f cyc/mm', g.f)); end, end
xlabel('pupil radius / DM radius'); ylabel('phase gain'); grid on; legend('Location','southwest'); title(sprintf('%s: zone-PSF model vs plane-to-plane Fourier propagation (as built, u direction)', o.tag), 'Interpreter','none');
print(ff, fullfile(o.outdir,[o.tag '_fourier.png']), '-dpng', '-r96');
F3 = struct('G',G,'PK',PK,'work_err_nm',std(d(lit))*1e6,'work_map',d,'magF',magF,'z1',z1,'z2',z2,'f2',f2,'fFL',fFL,'zimgFL',zimgFL);
end

function E = tail_prop_(Edm, HA, phiL2, XF, YF, z1, bfd, lam, phiFL, Hdet)
% DM -> focuser (angular spectrum), the focuser's screen, FFT to the focus (relative to the converging sphere),
% single-FFT Fresnel focus -> FL (relative to the diverging sphere), the FL's screen, scaled angular spectrum -> detector
E = ifft2(fft2(Edm) .* HA);
E = E .* exp(1i*phiL2);
Ef = fftshift(fft2(ifftshift(E)));                          % the focal field (up to the focal chirp, which the next leg carries)
Ef = Ef .* exp(1i*pi*(XF.^2 + YF.^2)/lam*(1/z1 + 1/bfd));   % the focal plane's own chirp (from the converging leg) + the Fresnel chirp of the focus -> FL leg; both slow here
E = fftshift(fft2(ifftshift(Ef)));                          % the field at the FL relative to the sphere of radius z1
E = E .* exp(1i*phiFL);
E = ifft2(fft2(E) .* Hdet);                                 % to the detector, on the scaled grid
end

function [phi, f_par, bfd] = lens_screen_collimated_(L, lam, xmax)
% exact OPL of a collimated beam through a plano-convex conic singlet (powered face first, KrElt = -|R| facing the beam),
% relative to the paraxial thin lens of the traced focal length: phi(r) = k (OPL(r) - OPL(0) + r^2/(2 f))
n = L.n;  R = abs(L.Kr);  K = L.Kc;  t = L.t;
r = linspace(0, xmax, 4001);
sag = @(r) r.^2 ./ (R*(1 + sqrt(1 - (1+K)*r.^2/R^2)));     % conic sag, vertex at z = 0, opening toward +z (the beam)
% ray at height r, direction +z: hits the surface at z = sag(r); the outward normal has slope dsag/dr
ds = (r/R) ./ sqrt(1 - (1+K)*r.^2/R^2);                     % dsag/dr
th_i = atan(ds);                                            % angle of incidence (surface normal vs +z)
th_t = asin(sin(th_i)/n);  dev = th_i - th_t;               % ray bends toward the axis by dev
% inside the glass from (r, sag) to the flat at z = t: travels t - sag along z, moving inward by (t - sag) tan(dev)
zin = t - sag(r);  rin = r - zin.*tan(dev);
% exit through the flat: angle dev in glass -> asin(n sin dev) in air
th_o = asin(n*sin(dev));
OPL = sag(r) + n*zin./cos(dev);                             % to the flat's exit point
% continue in air to the rear principal plane?  Reference everything to the flat's plane: the exit ray at height rin with angle th_o
% converges to the axis at distance rin/tan(th_o) -> paraxial focal length from the smallest rays
f_par = rin(2)/tan(th_o(2)) - 0;                            % from the flat (back focal distance)
% the wave at the flat's plane, as a function of the exit height rin: OPL(rin); the ideal converging sphere to the focus at f_bfd:
% phase relative to -rin^2/(2 f_bfd) (paraxial); resample onto the exit height
OPLx = interp1(rin, OPL, r, 'spline', 'extrap');
phi_r = 2*pi/lam * (OPLx - OPLx(1) + sqrt(f_par^2 + r.^2) - f_par);   % relative to the EXACT converging sphere: the FFT to the focus is the Debye relation for the envelope on that sphere
% the principal-plane focal length: add the flat-to-vertex offset: f = bfd + t/n (plano-convex, thick-lens)
bfd = f_par;  f_par = f_par + t/n;
phi = @(rr) interp1(r, phi_r, min(rr, xmax), 'linear', 0);
end

function [phi, f_thin, zimg] = lens_screen_point_(L, lam, z1, xmax)
% exact OPL from a point source at distance z1 ahead of the powered (front) face of a plano-convex conic singlet,
% to the flat's exit plane, as a function of the exit height; relative to the paraxial diverging sphere of radius z1
% AND the paraxial thin lens (so what remains is the aberration + the lens's non-paraxial focusing).  The screen is
% applied to a field carried relative to the sphere of radius z1; after it the field is relative to the sphere of
% radius zimg (the on-axis image of the source point, from the exact trace of the smallest rays).
n = L.n;  R = abs(L.Kr);  K = L.Kc;  t = L.t;
u = linspace(0, atan(xmax/z1)*1.05, 4001);                   % launch angles
sag = @(r) r.^2 ./ (R*(1 + sqrt(1 - (1+K)*r.^2/R^2)));
% intersect the ray from (0, -z1) at angle u with z = sag(r): solve z1 + z = r/tan(u) with z = sag(r) by iteration on r
r = z1*tan(u);  for it = 1:30, r = (z1 + sag(r)).*tan(u); end
z = sag(r);  ds = (r/R) ./ sqrt(1 - (1+K)*r.^2/R^2);
th_n = atan(ds);                                             % surface normal tilt vs +z
th_i = u + th_n;                                             % incidence angle vs the inward normal, which tilts by -th_n (toward the axis) at height r
th_t = asin(sin(th_i)/n);  ang_in = th_t - th_n;            % ray angle in the glass vs +z (negative = bending toward the axis)
zin = t - z;  rin = r + zin.*tan(ang_in);
th_o = asin(n*sin(ang_in));                                  % exit angle vs +z
OPL = sqrt((z1 + z).^2 + r.^2) + n*zin./cos(ang_in);         % source -> the front face at (r, sag) -> flat exit point
% paraxial: the image of the source point behind the flat
zimg = -rin(2)/tan(th_o(2));                                 % + = real image behind the flat; - = virtual ahead
f_thin = R/(n-1);
OPLx = interp1(rin, OPL, r, 'spline', 'extrap');            % OPL vs exit height on the exit-height axis r
% relative to the EXACT spherical wave from the source and the exact sphere to/from the image (sign: converging subtracts, diverging adds):
phi_r = 2*pi/lam * (OPLx - OPLx(1) - (sqrt(z1^2 + r.^2) - z1) + r.^2/(2*z1) + sign(zimg)*(sqrt(zimg^2 + r.^2) - abs(zimg)));   % the lens maps the EXACT incoming sphere to its exact exit wave; the model's incoming envelope rides the PARAXIAL sphere r^2/(2 z1) (its Fresnel step), and the exit envelope is referenced to the exact image sphere
phi = @(rr) interp1(r, phi_r, min(rr, max(r)), 'linear', 0);
end

% ---------------------------------------------------------------------------
function [hdr, blocks] = split_deck_(txt)
idx = regexp(txt, '\n[ \t]*iElt=');
hdr = txt(1:idx(1));  blocks = cell(1, numel(idx));
for i = 1:numel(idx)
    e = numel(txt); if i < numel(idx), e = idx(i+1); end
    blocks{i} = txt(idx(i)+1:e);
end
end
function v = getv_(blk, key)
m = regexp(blk, ['(?m)^[ \t]*' key '=[ \t]*([^\n]*)'], 'tokens', 'once');
if isempty(m), v = []; return; end
v = str2num(regexprep(m{1}, '[DdEe]([+-]?\d)', 'e$1')); %#ok<ST2NM>
if isempty(v), v = strtrim(m{1}); end
end
function u = unit_(v), u = v / norm(v); end
function f = write_deck_(hdr, blocks, pos, f)
hdr = regexprep(hdr, 'ChfRayPos=\s*[^\n]*', sprintf('ChfRayPos=  %.15g  %.15g  %.15g', pos));
fid = fopen(f,'w'); fwrite(fid, [hdr blocks{:}]); fclose(fid);
end
function r = rms_(x), x = x(isfinite(x)); r = sqrt(mean(x(:).^2)); end
function say_(rep, varargin), fprintf(varargin{:}); fprintf(rep, varargin{:}); end
