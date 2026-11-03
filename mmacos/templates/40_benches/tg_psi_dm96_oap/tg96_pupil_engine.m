function out = tg96_pupil_engine(varargin)
%TG96_PUPIL_ENGINE  The ENGINE's plane-to-plane propagation of the detector leg (Dave, 2026-09-17).
%   out = TG96_PUPIL_ENGINE('rig','lens'|'oap', name/value ...)
%
%   The cross-check of tg96_pupilsim's zone-PSF model, done the way the CTB does its
%   physical optics: reference surfaces inserted into the .in file, and the field read
%   where the engine propagates it.  From the simulation's deck (the bench as built, the
%   baffle opened, the cone widened, the 48 mm aperture on the DM) with the DM's GridData
%   surface restored:
%     * the mask sandwich (the CTB quartet as the twyman_green 'nf' emission has it):
%       MaskSphereIn (Reference, Conic Kr=-R, NF1, zElt R) one R before the mask marker,
%       the FocalMask carrying NF2, MaskSphereOut (Conic Kr=-R, Geometric, zElt R) after
%       it -- symmetric, so the unmasked round trip is the identity and the exit sphere
%       holds the DM-conjugate pupil field per ray index;
%     * PupilSphere (Reference, Conic) just after the field lens, CONCENTRIC with the
%       diverging beam there (its center is the field lens's image of the focus, R_s
%       upstream, measured from the rays), carrying PropType NFS1surf with zElt R_s; the
%       Detector carries zElt R_s + its distance.  The engine's NFPROP then propagates
%       the field from the sphere to the detector plane as a scaled Fresnel step (the
%       effective distance (Z2-Z1) Z1/Z2, the pitch scaled by Z2/Z1) -- the field lands
%       on the DETECTOR's regular grid, with the leg's field curvature, astigmatism and
%       the pupil edge's diffraction in it, which the record's Geometric leg (per ray
%       index) can never show.
%   The test surfaces of tg96_pupilsim (flat, sinusoids at the actuator Nyquist and
%   below, single pokes, the 30 nm working surface) are written as GridData files onto
%   the DM (256 x 0.4 mm, model 512), the four-step readout angle(Et conj Er) is taken on
%   the detector grid, mapped to the DM frame through the ray affine, and scored as stage
%   2 scores -- at the plane as built and at the compromise plane the simulation found.
%
%   Name/value: 'rig' ('lens'), 'sim' (the tg96_pupilsim run to compare with; default
%   runs/pupilsim_<rig>), 'tag' ('pupileng_<rig>'), 'model' (512), 'ngrid' (385, the
%   record's ray grid), 'kr_sign' (0 = try both and keep the one whose flat-DM OPD on the
%   pupil sphere is smaller), 'freqs', 'amp_nm' (5), 'outdir'.
o = struct('rig','lens','sim','','tag','','model',512,'ngrid',385,'kr_sign',0, ...
           'freqs',[0.5 0.25 0.125 0.0625],'amp_nm',5,'outdir','','n_g',256,'dx_g',0.4);
for k = 1:2:numel(varargin), o.(varargin{k}) = varargin{k+1}; end
exdir = fileparts(mfilename('fullpath'));  if isempty(exdir), exdir = pwd; end
if isempty(which('macos.init')), run(fullfile(exdir,'..','..','..','mmacos_setup.m')); end
if isempty(o.sim), o.sim = fullfile(exdir,'runs',['pupilsim_' o.rig]); end
if isempty(o.tag), o.tag = ['pupileng_' o.rig]; end
if isempty(o.outdir), o.outdir = fullfile(exdir,'runs',o.tag); end
if ~exist(o.outdir,'dir'), mkdir(o.outdir); end
rep = fopen(fullfile(o.outdir,[o.tag '_report.txt']),'w');
say = @(varargin) say_(rep, varargin{:});
S = load(fullfile(o.sim, ['pupilsim_' o.rig '.mat']));  S = S.out;   % the simulation of record: ray map, lattice, results
lam = S.o.lambda;  k0 = 2*pi/lam;  pitch = S.o.pitch;
say('=== tg96_pupil_engine: tag %s  rig %s  (%s) ===\n', o.tag, o.rig, datestr(now,'yyyy-mm-dd HH:MM'));
say('the engine''s propagation through reference surfaces in the .in file (the CTB model); compared with %s\n', o.sim);

% ---- the deck: the simulation's (bench as built, baffle opened, cone widened, aperture on the DM) with the DM's GridData restored ----
[hdr, blocks] = split_deck_(fileread(fullfile(o.sim, ['pupilsim_' o.rig '_deck.in'])));
[~, rblocks] = split_deck_(fileread(S.o.deck));
names = cellfun(@(b) getv_(b,'EltName'), blocks, 'uni', 0);
iDM = find(strcmp(names,'TestOptic'),1);  iFoc = find(strcmp(names,'FocalMask'),1);  iFLf = find(strcmp(names,'FLflat'),1);  iDet = numel(blocks);
rDM = rblocks{find(cellfun(@(b) strcmp(getv_(b,'EltName'),'TestOptic'), rblocks),1)};
grid_keys = regexp(rDM, '(?m)^[ \t]*(pData|xData|yData|zData)=[^\n]*\n', 'match');
blocks{iDM} = regexprep(blocks{iDM}, 'Surface=[^\n]*', 'Surface=  GridData');
blocks{iDM} = regexprep(blocks{iDM}, '([ \t]*ApType=)', ['         nGridMat=  ' num2str(o.n_g) '\n         GridFile=  GRIDFILE\n        GridSrfdx=  ' sprintf('%.10E', o.dx_g) '\n' [grid_keys{:}] '$1'], 'once');
psi_dm = getv_(blocks{iDM},'psiElt');  V_dm = getv_(blocks{iDM},'VptElt');  x_dm = getv_(blocks{iDM},'xObs');  y_dm = cross(psi_dm, x_dm);
R_dm = getv_(blocks{iDM},'ApVec');  R_dm = R_dm(1);
psi_f = getv_(blocks{iFoc},'psiElt');  V_f = getv_(blocks{iFoc},'VptElt');
psi_det = getv_(blocks{iDet},'psiElt');  V_det = getv_(blocks{iDet},'VptElt');  x_det = getv_(blocks{iDet},'xObs');  y_det = cross(psi_det, x_det);
V_flf = getv_(blocks{iFLf},'VptElt');
hdr = regexprep(hdr, 'nGridpts=\s*[^\n]*', sprintf('nGridpts=  %d', o.ngrid));
% ---- the mask sandwich (twyman_green 'nf': d_in = 0.85 x the focuser-to-mask distance, d_out = 0.6 x the mask-to-FL distance) ----
iPrev = iFoc - 1;  V_prev = getv_(blocks{iPrev},'VptElt');  iFLp = find(strcmp(names,'FLpow'),1);  V_flp = getv_(blocks{iFLp},'VptElt');
d_in = 0.85 * norm(V_f(:) - V_prev(:));  d_out = 0.6 * norm(V_flp(:) - V_f(:));
sphIn  = sphere_block_(blocks{iFoc}, 'MaskSphereIn',  V_f(:)' - d_in*psi_f(:)', -d_in, 'NF1', d_in);
sphOut = sphere_block_(blocks{iFoc}, 'MaskSphereOut', V_f(:)' + d_out*psi_f(:)', -d_in, 'Geometric', d_in);
blocks{iFoc} = set_prop_(blocks{iFoc}, 'NF2', 1e22);
blocks = [blocks(1:iFoc-1) {sphIn} blocks(iFoc) {sphOut} blocks(iFoc+1:end)];
names = cellfun(@(b) getv_(b,'EltName'), blocks, 'uni', 0);  iFLf = find(strcmp(names,'FLflat'),1);  iDet = numel(blocks);  iDM = find(strcmp(names,'TestOptic'),1);
say('mask sandwich inserted: spheres of radius %.2f mm (Kr -R, zElt +R) at %.2f before and %.2f after the mask marker; the mask carries NF2\n', d_in, d_in, d_out);
% ---- the pupil sphere after the field lens: concentric with the beam there (R_s from the rays), NFS1surf to the detector ----
macos.init(o.model);
macos.write_grid_file(fullfile(o.outdir,'eng_flat.txt'), zeros(o.n_g));
deck0 = write_(hdr, blocks, fullfile(o.outdir, [o.tag '_probe.in']), 'eng_flat.txt');
cd(o.outdir);  macos.load_rx(deck0);  macos.stop(iDM);
sF = macos.trace(iFLf);  rF = macos.get_ray_info(sF.nRays);  okF = reshape(rF.ok_trace & rF.ok_pass, 1, []);
pF = rF.pos(:,okF) - V_flf(:);  dF = rF.dir(:,okF);
xF = x_det(:)'*pF;  yF = y_det(:)'*pF;  aX = (x_det(:)'*dF)./(psi_det(:)'*dF);  aY = (y_det(:)'*dF)./(psi_det(:)'*dF);
r_ = hypot(xF, yF);  a_ = (xF.*aX + yF.*aY)./max(r_, 1e-9);            % radial angle
sel = r_ > 0.2*max(r_);  Rc = -median(r_(sel)./a_(sel));                  % the crossing of the exit rays with the axis: <0 = upstream (virtual)
d_sph = 1.0;  R_s = abs(Rc) + d_sph;  d_det = norm(V_det(:) - V_flf(:));
say('exit rays after the field lens cross the axis %.1f mm from its exit face (negative = upstream, a diverging beam); pupil sphere placed %.1f mm after the face, radius %.2f mm; detector %.2f mm after the face -> zElt %.2f\n', Rc, d_sph, R_s, d_det, R_s + d_det - d_sph);
diverging = Rc < 0;
% Kr sign: the sphere must be concentric with the beam; try both and keep the one whose flat-DM OPD on it is smaller
signs = [-1 1];  if o.kr_sign ~= 0, signs = o.kr_sign; end
best = [];
for sg = signs
    sph = sphere_block_(blocks{iFLf}, 'PupilSphere', V_flf(:)' + d_sph*psi_det(:)', sg*R_s, 'NFS1surf', R_s);
    bl = [blocks(1:iFLf) {sph} blocks(iFLf+1:end)];
    bl{end} = set_prop_(bl{end}, 'Geometric', R_s + d_det - d_sph);
    f = write_(hdr, bl, fullfile(o.outdir, sprintf('%s_kr%+d.in', o.tag, sg)), 'eng_flat.txt');
    macos.load_rx(f);  macos.stop(iDM);
    s = macos.trace(numel(bl)-1);  W = macos.opd();  W = W(isfinite(W) & W ~= 0);
    say('  Kr %+.2f: flat-DM OPD on the pupil sphere %.3f nm rms (rays %d)\n', sg*R_s, std(W)*1e6, s.nRays);
    if isempty(best) || std(W) < best.w, best = struct('sg',sg,'w',std(W),'bl',{bl},'f',f); end
end
blocks = best.bl;  iSph = iFLf + 1;  iDet = numel(blocks);
say('pupil sphere Kr %+.2f kept (OPD %.3f nm rms: the sphere is concentric with the beam)\n', best.sg*R_s, best.w*1e6);
deck = write_(hdr, blocks, fullfile(o.outdir, [o.tag '_deck.in']), 'GRIDFILE');   % the token stays in the deck of record; per-surface decks substitute it
delete(fullfile(o.outdir, [o.tag '_probe.in']));  delete(fullfile(o.outdir, [o.tag '_kr*.in']));

% ---- the DM frame, the test surfaces, and the map from the detector grid to the DM frame ----
N = S.o.N;  dx = S.o.dx;  xg = (-N/2:N/2-1)*dx;  [XG, YG] = meshgrid(xg, xg);  rr = hypot(XG, YG);
R_beam = min(R_dm, max(hypot(S.uv(1,S.ok), S.uv(2,S.ok))) + 0.5);  Rin = R_beam - 1.0;  lit = rr <= Rin;  R_lit = floor(Rin - 0.5*pitch) + 0.5*pitch;
fx = ifftshift((-N/2:N/2-1)/(N*dx));  [FU, FV] = meshgrid(fx, fx);
xa_g = ((1:o.n_g) - (o.n_g+1)/2)*o.dx_g;  [UG, VG] = meshgrid(xa_g, xa_g);              % the GridData lattice (first index = +x)
w_mm = S.o.infl_w*pitch;  infl = @(u0, v0, U, V) exp(-((U-u0).^2 + (V-v0).^2)/w_mm^2);
sites = vertcat(S.res(1).PK.site);                                        % the simulation's own poke sites
rng(S.o.seed);  cmd = S.o.work_nm*1e-6*randn(S.o.nact);  xa = ((1:S.o.nact) - (S.o.nact+1)/2)*pitch;
work = @(U, V) work_(cmd, xa, R_dm, infl, U, V);
% surfaces as functions of (U,V): written onto the GridData lattice; the truth on the DM frame is the SAME lattice
% linearly interpolated (what the engine's GridData surface is)
surf_list = {};
surf_list{end+1} = struct('name','flat', 'h', @(U,V) zeros(size(U)));
for f = o.freqs, surf_list{end+1} = struct('name',sprintf('sin_f%.4f', f), 'h', @(U,V) o.amp_nm*1e-6*sin(2*pi*f*U), 'f', f); end %#ok<AGROW>
for s = 1:size(sites,1), surf_list{end+1} = struct('name',sprintf('poke_%d', s), 'h', @(U,V) S.o.poke_nm*1e-6*infl(sites(s,1), sites(s,2), U, V), 'site', sites(s,:)); end %#ok<AGROW>
surf_list{end+1} = struct('name','work', 'h', work);
mg_ = @(sf) sf.h(UG', VG');                                                % GridData M(i,j) = h(u_i, v_j): first index = +x
truth = @(sf) interp2(UG, VG, mg_(sf)', XG, YG, 'linear', 0);              % the same lattice, linearly interpolated: what the engine's surface is
% the ray affine DM -> detector (from the simulation's stage 1), the detector's array frame (dmg_frame: row <-> u2, col <-> v2)
u2 = macos.design.Bench.perp(psi_det(:));  v2 = cross(psi_det(:), u2);
Rrot = [u2'; v2'] * [x_det(:) y_det(:)];                                  % (x_det,y_det) components -> (u2,v2)
ok = S.ok;  M = [S.uv(1,ok); S.uv(2,ok); ones(1,nnz(ok))]';  Ax = M \ S.r0(1,ok)';  Ay = M \ S.r0(2,ok)';
XD = Ax(1)*XG + Ax(2)*YG + Ax(3);  YD = Ay(1)*XG + Ay(2)*YG + Ay(3);        % detector mm in (x_det, y_det), for every DM-frame point
UD = Rrot(1,1)*XD + Rrot(1,2)*YD;  VD = Rrot(2,1)*XD + Rrot(2,2)*YD;      % in (u2, v2)
planes = [0 S.zstar];  plnm = {'as built', sprintf('detector %+.2f mm downstream (the simulation''s compromise plane)', S.zstar)};
res = struct();
for ip = 1:numel(planes)
    dz = planes(ip);
    bl = blocks;  bl{iDet} = regexprep(bl{iDet}, 'VptElt=[^\n]*', sprintf('VptElt=  %.12g  %.12g  %.12g', V_det(:)' + dz*psi_det(:)'));
    bl{iDet} = regexprep(bl{iDet}, 'RptElt=[^\n]*', sprintf('RptElt=  %.12g  %.12g  %.12g', V_det(:)' + dz*psi_det(:)'));
    bl{iDet} = set_prop_(bl{iDet}, 'Geometric', R_s + d_det - d_sph + dz);
    deckp = fullfile(o.outdir, sprintf('%s_plane%d_SURF.in', o.tag, ip));
    say('\n---- engine, %s ----\n', plnm{ip});
    E = cell(1, numel(surf_list));  dxd = NaN;
    for is = 1:numel(surf_list)
        sf = surf_list{is};  Mg = mg_(sf);
        gf = sprintf('eng_%s.txt', sf.name);  macos.write_grid_file(fullfile(o.outdir, gf), Mg);
        fdeck = write_(hdr, bl, strrep(deckp, 'SURF', sf.name), gf);
        macos.load_rx(fdeck);  macos.stop(iDM);  delete(fdeck);
        E{is} = macos.complex_field(iDet);  dxd = abs(macos.dx_at(iDet, 'mm'));
    end
    Nd = size(E{1},1);  cen = (Nd+1)/2;
    I = abs(E{1}).^2;  [C1, C2] = meshgrid(1:Nd, 1:Nd);  c1 = sum(C1(:).*I(:))/sum(I(:));  c2 = sum(C2(:).*I(:))/sum(I(:));
    rF = sqrt(sum(I(:) > 0.5*median(I(I > 0.05*max(I(:)))))*dxd^2/pi);
    say('flat DM at the detector: grid %d px at %.4f mm; pupil image centroid at pixel (%.1f, %.1f) of centre %.1f; equivalent radius %.3f mm -> %.4f DM-mm per detector-mm (the rays: %.4f)\n', Nd, dxd, c1, c2, cen, rF, R_beam/rF, 1/S.magL);
    % the sample map: DM-frame point -> array (row, col); orientation resolved on an off-centre poke (8 variants)
    ipk = find(cellfun(@(s) strcmp(s.name,'poke_3'), surf_list), 1);  sp = surf_list{ipk}.site;
    variants = {@(u,v) [u v], @(u,v) [-u v], @(u,v) [u -v], @(u,v) [-u -v], @(u,v) [v u], @(u,v) [-v u], @(u,v) [v -u], @(u,v) [-v -u]};
    bestv = [];
    for iv = 1:numel(variants)
        q = variants{iv}(UD, VD);  ROWm = q(:,1:N)/dxd + c2;  COLm = q(:,N+1:end)/dxd + c1;
        samp = @(A) interp2(A, COLm, ROWm, 'linear', 0);
        ho = angle(samp(E{ipk}) .* conj(samp(E{1}))) * lam/(4*pi);  ho(~lit) = 0;
        [pk, im] = max(ho(:));  d = hypot(XG(im) - sp(1), YG(im) - sp(2));
        if isempty(bestv) || d < bestv.d, bestv = struct('iv',iv,'d',d,'pk',pk,'samp',samp); end
    end
    samp = bestv.samp;  say('array orientation: variant %d (poke at (%.1f, %.1f) lands %.2f mm off; peak %.3f of %d nm)\n', bestv.iv, sp, bestv.d, bestv.pk*1e6, S.o.poke_nm);
    Er = samp(E{1});  read = @(is) angle(samp(E{is}) .* conj(Er)) * lam/(4*pi);
    say('flat DM: |E| over the lit actuators %.4f mean (normalized), %.4f relative rms variation; outermost lit ring %.4f\n', mean(abs(Er(lit)))/mean(abs(Er(lit))), std(abs(Er(lit)))/mean(abs(Er(lit))), std(abs(Er(lit & rr > Rin - pitch)))/mean(abs(Er(lit))));
    G = struct('f',{},'gain_r',{},'gmin',{},'gmean',{},'xt',{});  rb = 0:0.1:1;  rc = rr/Rin;
    for is = 2:numel(surf_list)
        sf = surf_list{is};  h = truth(sf);  ho = read(is);
        if isfield(sf,'f')
            car = exp(-1i*2*pi*sf.f*XG);  ap = double(rr <= R_beam);
            gmap = abs(lpf_(ho.*car.*ap, FU, FV, 2/sf.f)) ./ max(abs(lpf_(h.*car.*ap, FU, FV, 2/sf.f)), 1e-12);
            am = abs(samp(E{is}))./max(abs(Er),1e-9) - 1;
            xmap = abs(lpf_(am.*car.*ap, FU, FV, 2/sf.f)) ./ max(abs(lpf_(h.*car.*ap, FU, FV, 2/sf.f)), 1e-12) * lam/(4*pi);
            gr = zeros(1,numel(rb)-1);  for b = 1:numel(rb)-1, m = lit & rc >= rb(b) & rc < rb(b+1); gr(b) = mean(gmap(m)); end
            G(end+1) = struct('f',sf.f,'gain_r',gr,'gmin',min(gmap(lit)),'gmean',mean(gmap(lit)),'xt',max(xmap(lit))); %#ok<AGROW>
            jz = find([S.res(ip).G.f] == sf.f & [S.res(ip).G.dir] == 1, 1);
            say('  sinusoid f %.4f: ENGINE gain mean %.4f, min %.4f, by radius [%s]; cross-talk max %.3f  |  zone-PSF model: mean %.4f, min %.4f, by radius [%s]\n', ...
                sf.f, mean(gmap(lit)), min(gmap(lit)), sprintf('%.3f ', gr), max(xmap(lit)), S.res(ip).G(jz).gmean, S.res(ip).G(jz).gmin, sprintf('%.3f ', S.res(ip).G(jz).gain_r));
        elseif isfield(sf,'site')
            m = hypot(XG-sf.site(1), YG-sf.site(2)) <= 4*pitch;  pk = max(ho(m))/max(h(m));
            js = find(all(abs(vertcat(S.res(ip).PK.site) - sf.site) < 1e-6, 2), 1);  pks = NaN;  if ~isempty(js), pks = S.res(ip).PK(js).peak; end
            say('  poke at (%5.1f, %5.1f): ENGINE peak %.4f  |  zone-PSF model %.4f\n', sf.site, pk, pks);
        else
            d = ho - h;  Ap = [ones(nnz(lit),1) XG(lit) YG(lit)];  d(lit) = d(lit) - Ap*(Ap\d(lit));  d(~lit) = 0;
            say('  working surface %.1f nm rms: ENGINE recovered - true %.3f nm rms (piston, tilt removed)  |  zone-PSF model %.3f nm\n', std(h(lit))*1e6, std(d(lit))*1e6, S.res(ip).work_err_nm);
            res(ip).work_map = d;
        end
    end
    res(ip).G = G;  res(ip).dz = dz;  res(ip).Er = Er;  res(ip).dxd = dxd;
end
% figure: engine vs zone-PSF model, gain vs radius, both planes
fg = figure('Visible','off','Position',[100 100 1300 480]);  rmid = (rb(1:end-1)+rb(2:end))/2;  cols = lines(numel(o.freqs));
for ip = 1:2
    subplot(1,2,ip); hold on;
    for g = res(ip).G, plot(rmid, g.gain_r, '-o', 'Color', cols(o.freqs==g.f,:), 'DisplayName', sprintf('engine %.3f cyc/mm', g.f)); end
    for g = S.res(ip).G, if g.dir == 1, plot(rmid, g.gain_r, ':', 'Color', cols(o.freqs==g.f,:), 'LineWidth', 1.5, 'DisplayName', sprintf('zone PSFs %.3f', g.f)); end, end
    xlabel('radius / lit radius'); ylabel('phase gain'); grid on; legend('Location','southwest'); title(plnm{ip});
end
sgtitle(sprintf('%s: the engine''s propagation through reference surfaces vs the zone-PSF model (%s rig)', o.tag, o.rig), 'Interpreter','none');
print(fg, fullfile(o.outdir,[o.tag '_gain.png']), '-dpng', '-r96');
fw = figure('Visible','off','Position',[100 100 900 420]);
for ip = 1:2, subplot(1,2,ip); imagesc(xg, xg, res(ip).work_map*1e6); axis image; colorbar; title(sprintf('engine: recovered - true, nm (%s)', plnm{ip}), 'Interpreter','none'); end
print(fw, fullfile(o.outdir,[o.tag '_work.png']), '-dpng', '-r96');
out = struct('o',o,'res',res,'R_s',R_s,'kr_sign',best.sg,'deck',deck,'d_in',d_in,'d_out',d_out);
save(fullfile(o.outdir,[o.tag '.mat']), 'out');
say('run complete\n');  fclose(rep);
end

% ---------------------------------------------------------------------------
function b = sphere_block_(tmpl, name, vpt, kr, ptype, zelt)
b = tmpl;
b = regexprep(b, 'EltName=[^\n]*', ['EltName=  ' name]);
b = regexprep(b, 'Element=[^\n]*', 'Element=  Reference');
b = regexprep(b, 'Surface=[^\n]*', 'Surface=  Conic');
b = regexprep(b, 'KrElt=[^\n]*', sprintf('KrElt=  %.10E', kr));
b = regexprep(b, 'KcElt=[^\n]*', 'KcElt=  0.0000000000E+00');
b = regexprep(b, 'VptElt=[^\n]*', sprintf('VptElt=  %.12g  %.12g  %.12g', vpt));
b = regexprep(b, 'RptElt=[^\n]*', sprintf('RptElt=  %.12g  %.12g  %.12g', vpt));
b = regexprep(b, '[ \t]*ApType=[^\n]*\n', '         ApType=  None\n');
b = regexprep(b, '[ \t]*ApVec=[^\n]*\n', '');
b = set_prop_(b, ptype, zelt);
end
function b = set_prop_(b, ptype, zelt)
b = regexprep(b, 'PropType=[^\n]*', ['PropType=  ' ptype]);
if isempty(regexp(b, '(?m)^[ \t]*zElt=', 'once')), b = regexprep(b, '(PropType=[^\n]*\n)', sprintf('$1             zElt=  %.10E\n', zelt));
else, b = regexprep(b, 'zElt=[^\n]*', sprintf('zElt=  %.10E', zelt)); end
end
function f = write_(hdr, blocks, f, gridfile)
for i = 1:numel(blocks), blocks{i} = regexprep(blocks{i}, '(?m)^([ \t]*iElt=[ \t]*)\d+', sprintf('$1%d', i)); blocks{i} = strrep(blocks{i}, 'GRIDFILE', gridfile); end
hdr = regexprep(hdr, 'nElt=\s*\d+', sprintf('nElt=  %d', numel(blocks)));
fid = fopen(f,'w'); fwrite(fid, [hdr blocks{:}]); fclose(fid);
end
function h = work_(cmd, xa, R_dm, infl, U, V)
h = zeros(size(U));  n = numel(xa);
for ia = 1:n, for ja = 1:n, if hypot(xa(ia), xa(ja)) <= R_dm, h = h + cmd(ia,ja)*infl(xa(ia), xa(ja), U, V); end, end, end
end
function m = lpf_(x, FU, FV, sig), m = ifft2(fft2(x) .* exp(-2*pi^2*sig^2*(FU.^2 + FV.^2))); end
function [hdr, blocks] = split_deck_(txt)
idx = regexp(txt, '\n[ \t]*iElt=');
hdr = txt(1:idx(1));  blocks = cell(1, numel(idx));
for i = 1:numel(idx), e = numel(txt); if i < numel(idx), e = idx(i+1); end; blocks{i} = txt(idx(i)+1:e); end
end
function v = getv_(blk, key)
m = regexp(blk, ['(?m)^[ \t]*' key '=[ \t]*([^\n]*)'], 'tokens', 'once');
if isempty(m), v = []; return; end
v = str2num(regexprep(m{1}, '[DdEe]([+-]?\d)', 'e$1')); %#ok<ST2NM>
if isempty(v), v = strtrim(m{1}); end
end
function say_(rep, varargin), fprintf(varargin{:}); fprintf(rep, varargin{:}); end
