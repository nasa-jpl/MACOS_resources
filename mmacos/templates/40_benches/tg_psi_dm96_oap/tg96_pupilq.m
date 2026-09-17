function out = tg96_pupilq(varargin)
%TG96_PUPILQ  Pupil image quality of the Twyman-Green's detector leg (Fang Shi, 2026-09-16).
%   out = TG96_PUPILQ('rig','lens'|'oap', name/value ...)
%
%   THE DM IS THE STOP (Dave's ruling).  From the rig's test-arm deck of record
%   everything upstream of the DM is removed, the source becomes a COLLIMATED
%   beam launched at the DM with the DM's aperture as the stop, and the beam is
%   tilted about the DM by a field angle theta (two azimuths).  A tilt theta at
%   the DM is a spatial frequency theta/lambda on its surface, so:
%     * at the CAMERA (the DM's exit pupil) the index-matched rays of two
%       neighbouring tilts cross at the image of that DM zone: the crossing
%       cloud gives the pupil SURFACE (defocus / astigmatism of the DM's image),
%       the pupil DISTORTION (each zone's image against its DM coordinate times
%       the magnification, i.e. against the runner's single global affine) and
%       the pupil BLUR (the crossing's walk over the actuator tilt band);
%     * at the FOCAL PLANE (the mask seat) the rodgers2 set per tilt: spot,
%       wavefront with piston, tilt and focus removed, centroid against F x theta.
%   Every number in DM millimetres where it is about the DM, against the 1 mm
%   pitch and the detector pixel (0.2 DM-mm).
%
%   Name/value: 'rig' ('lens'), 'deck' (default: the rig's deck of record),
%   'tag' ('pupilq_<rig>'), 'model' (512), 'ngrid' (129), 'tilts' (rad, one-
%   sided list; default [0.5 1 2 3.2 10]*1e-4), 'band' (3.2e-4: the actuator
%   Nyquist as a tilt), 'dtheta' (5e-6, the crossing differential), 'lambda'
%   (6.328e-4 mm), 'aperture' (110 mm: the launched beam, wider than the DM
%   so the DM clips = is the stop), 'zones' (12: the DM grid the maps are
%   binned to), 'outdir'.
%   Writes runs/<tag>/<tag>_{deck.in, report.txt, pupil.png, focal.png, .mat}.
%   Run:  >> tg96_pupilq('rig','lens');   >> tg96_pupilq('rig','oap');
o = struct('rig','lens','deck','','tag','','model',512,'ngrid',129, ...
           'tilts',[0.5 1 2 3.2 10]*1e-4,'band',3.2e-4,'dtheta',5e-6, ...
           'lambda',6.328e-4,'aperture',110,'zones',12,'outdir','');
for k = 1:2:numel(varargin), o.(varargin{k}) = varargin{k+1}; end
exdir = fileparts(mfilename('fullpath'));  if isempty(exdir), exdir = pwd; end
if isempty(which('macos.init')), run(fullfile(exdir,'..','..','..','mmacos_setup.m')); end
if isempty(o.deck)
    switch o.rig
        case 'lens', o.deck = fullfile(exdir,'runs','lensuw2','lensuw2_test.in');
        case 'oap',  o.deck = fullfile(exdir,'runs','oapifo2','oapifo2_test.in');
    end
end
if isempty(o.tag), o.tag = ['pupilq_' o.rig]; end
if isempty(o.outdir), o.outdir = fullfile(exdir,'runs',o.tag); end
if ~exist(o.outdir,'dir'), mkdir(o.outdir); end
rep = fopen(fullfile(o.outdir,[o.tag '_report.txt']),'w');
say = @(varargin) say_(rep, varargin{:});
say('=== tg96_pupilq: tag %s  rig %s  (%s) ===\n', o.tag, o.rig, datestr(now,'yyyy-mm-dd HH:MM'));
say('deck of record %s; model %d, %d rays across; the DM is the stop; collimated source at the DM, tilted about it\n', o.deck, o.model, o.ngrid);

% ---- the deck: the bench AS BUILT, the DM declared the stop, the field a lateral shift of the source ----
% (A collimated source AT the DM measures a bench that does not exist: the lens
% rig's detector leg is tuned to the collimator's actual beam, fed 25 mm inside
% its focus, and lands 16 lam/D of blur on an ideal collimated beam.)
[hdr, blocks] = split_deck_(fileread(o.deck));
names = cellfun(@(b) getv_(b,'EltName'), blocks, 'uni', 0);
iDM = find(strcmp(names,'TestOptic'), 1);  assert(~isempty(iDM), 'no TestOptic in %s', o.deck);
blocks{iDM} = regexprep(blocks{iDM}, 'Surface=\s*GridData', 'Surface=  Flat');   % the DM flat: the leg, not the surface
for key = {'nGridMat','GridFile','GridSrfdx','pData','xData','yData','zData'}
    blocks{iDM} = regexprep(blocks{iDM}, ['[ \t]*' key{1} '=[^\n]*\n'], '');
end
psi_dm = getv_(blocks{iDM},'psiElt');  V_dm = getv_(blocks{iDM},'VptElt');  x_dm = getv_(blocks{iDM},'xObs');  y_dm = cross(psi_dm, x_dm);
R_dm  = getv_(blocks{iDM},'ApVec');  R_dm = R_dm(1);
iDet  = numel(blocks);  psi_det = getv_(blocks{iDet},'psiElt');  V_det = getv_(blocks{iDet},'VptElt');  x_det = getv_(blocks{iDet},'xObs');  y_det = cross(psi_det, x_det);
iFoc  = find(strcmp(names,'FocalMask'),1);  psi_f = getv_(blocks{iFoc},'psiElt');  V_f = getv_(blocks{iFoc},'VptElt');  x_f = getv_(blocks{iFoc},'xObs');  y_f = cross(psi_f, x_f);
iL1   = find(strcmp(names,'L1pow') | strcmp(names,'L1'), 1);  V_L1 = getv_(blocks{iL1},'VptElt');
src0  = getv_(hdr,'ChfRayPos');  dir0 = unit_(getv_(hdr,'ChfRayDir'));  zS = getv_(hdr,'zSource');
srcpt = src0(:)' + zS*dir0(:)';                      % the point source
f1    = norm(V_L1(:)' - srcpt);                       % the collimator's conjugate: a shift d at the source = a tilt d/f1 at the DM
hdr   = regexprep(hdr, 'nGridpts=\s*[^\n]*', sprintf('nGridpts=  %d', o.ngrid));
say('DM elt %d at [%.2f %.2f %.2f], clear radius %.2f mm, declared the STOP; focal plane elt %d; detector elt %d; source-to-collimator %.1f mm (shift = f1 x tilt)\n', iDM, V_dm, R_dm, iFoc, iDet, f1);
axes_ = {[0 0 1], unit_(cross([0 0 1], dir0))};      % the field's two azimuths, as directions of the source shift
aznm  = {'in-plane','out-of-plane'};
deckA = @(th, a, d) write_deck_(hdr,  blocks,  src0(:)' + f1*th*axes_{a}, fullfile(o.outdir, sprintf('%s_A_t%+.0e_a%d_d%d.in', o.tag, th, a, d)));
copyfile(deckA(0, 1, 0), fullfile(o.outdir,[o.tag '_deck.in']));

macos.init(o.model);
tilts = [-fliplr(o.tilts) 0 o.tilts];
nT = numel(tilts);  nA = 2;
frame = struct('V_dm',V_dm,'x_dm',x_dm,'y_dm',y_dm);
% ---- per tilt: rays at the DM, the detector (nominal + differential), the focal plane ----
S = struct('theta',{},'az',{},'uv',{},'cross',{},'okc',{},'spot_um',{},'spot_seat_um',{},'zbest',{},'cen',{},'wfe_nm',{},'wfe_raw_nm',{});
for a = 1:nA
    for t = 1:nT
        th = tilts(t);
        [uv, ok1, Pd, Dd, okd] = trace_(deckA(th, a, 0), iDM, iDet, frame);
        [~, ~, Pd2, Dd2, okd2] = trace_(deckA(th + o.dtheta, a, 1), iDM, iDet, frame);
        C = cross_lines_(Pd, Dd, Pd2, Dd2);            % 3xN crossing points = the DM zone's image
        okc = ok1 & okd & okd2;
        % focal plane (the mask seat): the spot, and the wavefront with piston, tilt and focus removed
        macos.load_rx(deckA(th, a, 0));  macos.stop(iDM);
        sf = macos.trace(iFoc);  rf = macos.get_ray_info(sf.nRays);  okf = reshape(rf.ok_trace & rf.ok_pass, 1, []);
        pf = rf.pos(:,okf) - V_f(:);  df = rf.dir(:,okf);  xf = x_f(:)'*pf;  yf = y_f(:)'*pf;
        cen = [mean(xf) mean(yf)];  spot_seat = sqrt(mean((xf-cen(1)).^2 + (yf-cen(2)).^2)) * 1e3;
        % the seat marker is where the runner parks the mask seat, not necessarily the focus: find the best focus along
        % the chief from the rays themselves (pure geometry), and score the spot there
        zn = psi_f(:)'*df;  sx = (x_f(:)'*df)./zn;  sy = (y_f(:)'*df)./zn;  z0 = -(psi_f(:)'*pf)./zn;   % ray -> plane at offset z: x + sx (z - z0)
        spotz = @(z) sqrt(var(xf + sx.*(z - z0), 1) + var(yf + sy.*(z - z0), 1)) * 1e3;
        zb = fminbnd(spotz, -60, 60);  spot = spotz(zb);
        cen = [mean(xf + sx.*(zb - z0)) mean(yf + sy.*(zb - z0))];
        W = macos.opd();  W = W(:);  Wok = isfinite(W) & W ~= 0;
        [gx, gy] = meshgrid(1:sqrt(numel(W)));  gx = gx(:) - mean(gx(:));  gy = gy(:) - mean(gy(:));
        A = [ones(nnz(Wok),1) gx(Wok) gy(Wok) gx(Wok).^2+gy(Wok).^2];   % piston, tilt, focus (the seat is a plane through a converging beam)
        Wres = W(Wok) - A * (A \ W(Wok));
        S(end+1) = struct('theta',th,'az',a,'uv',uv,'cross',C,'okc',okc,'spot_um',spot,'spot_seat_um',spot_seat,'zbest',zb,'cen',cen, ...
                          'wfe_nm',std(Wres)*1e6,'wfe_raw_nm',sf.rmsWFE*1e6); %#ok<AGROW>
        say('  az %-12s theta %+9.2e rad: rays ok %5d; spot at best focus %6.2f um (%+7.2f mm from the seat marker; at the marker %6.1f um); centroid [%+9.4f %+9.4f] mm; WFE piston/tilt/focus removed %6.2f nm\n', ...
            aznm{a}, th, nnz(okc), spot, zb, spot_seat, cen, S(end).wfe_nm);
    end
end
delete(fullfile(o.outdir, [o.tag '_A_t*_a*_d*.in']));

% ---- the pupil image: the nominal crossing cloud vs the DM coordinate ----
i0 = find([S.theta]==0 & [S.az]==1, 1);
uv = S(i0).uv;  C = S(i0).cross;  ok = S(i0).okc(:)' & all(isfinite(C),1);
Pc = C(:,ok) - V_det(:);  X = x_det(:)'*Pc;  Y = y_det(:)'*Pc;  Z = psi_det(:)'*Pc;
u = uv(1,ok);  v = uv(2,ok);
M = [u; v; ones(1,numel(u))]';  Ax = M \ X';  Ay = M \ Y';        % the global affine
Aff = [Ax(1:2)'; Ay(1:2)'];  mag = sqrt(abs(det(Aff)));            % detector-mm per DM-mm
rX = X' - M*Ax;  rY = Y' - M*Ay;  dist_dm = [rX rY] / mag;           % distortion in DM mm
rho = hypot(u, v) / R_dm;
Q = [ones(numel(u),1) u'/R_dm v'/R_dm rho'.^2 (u.^2-v.^2)'/R_dm^2 (2*u.*v)'/R_dm^2];
cz = Q \ Z';                                                          % pupil surface: sag terms, mm
say('\n---- the pupil image at the camera (theta = 0, in-plane pair; the bench as built, the DM the stop) ----\n');
say('global affine: magnification %.4f detector-mm per DM-mm (the runner''s ray affine 9.88 lens / 10.10 mirrors), rotation %.3f deg\n', 1/mag, atan2d(Aff(2,1), Aff(1,1)));
say('pupil distortion (image vs the affine, DM mm): rms %.4f, max %.4f; outer third of the pupil rms %.4f  [pitch 1.0, detector px %.3f]\n', ...
    rms_(hypot(dist_dm(:,1),dist_dm(:,2))), max(hypot(dist_dm(:,1),dist_dm(:,2))), rms_(hypot(dist_dm(rho>2/3,1),dist_dm(rho>2/3,2))), 1/mag*0.0417/1);
say('pupil surface (sag of the DM''s image along the camera normal, mm over the pupil radius): defocus %+.4f, astig 0 %+.4f, astig 45 %+.4f, tilt [%+.4f %+.4f]\n', cz(4), cz(5), cz(6), cz(2), cz(3));
% blur: each ray's crossing walk over the tilt band, both azimuths
blur = nan(1, numel(ok));  blur_full = blur;
inband = abs([S.theta]) <= o.band + 1e-12;
for r = find(ok)
    cc = cell2mat(arrayfun(@(s) s.cross(:,r), S(inband), 'uni', 0));  cf = cell2mat(arrayfun(@(s) s.cross(:,r), S, 'uni', 0));
    if all(isfinite(cc(:))), pc = cc - mean(cc,2); blur(r) = sqrt(mean(sum(([x_det(:)'; y_det(:)']*pc).^2,1))) / mag; end
    if all(isfinite(cf(:))), pc = cf - mean(cf,2); blur_full(r) = sqrt(mean(sum(([x_det(:)'; y_det(:)']*pc).^2,1))) / mag; end
end
say('pupil blur (the zone''s image walk over the tilt band, DM mm rms): |theta| <= %.1e rad (the actuator band): rms %.4f, max %.4f; over all tilts to %.1e: rms %.4f, max %.4f\n', ...
    o.band, rms_(blur(ok)), max(blur(ok)), max(o.tilts), rms_(blur_full(ok)), max(blur_full(ok)));
% wander per tilt (the cloud's mean shift), DM mm
say('pupil wander vs tilt (mean image shift, DM mm; axial shift, mm):\n');
for s = S
    okk = s.okc(:)' & all(isfinite(s.cross),1);  pc = s.cross(:,okk) - V_det(:);
    say('  az %-12s theta %+9.2e: lateral [%+.4f %+.4f]  axial %+.3f\n', aznm{s.az}, s.theta, (x_det(:)'*mean(pc,2) - mean(X))/mag, (y_det(:)'*mean(pc,2) - mean(Y))/mag, psi_det(:)'*mean(pc,2) - mean(Z));
end
% ---- the focal plane: the rodgers2 set per tilt, the edge-only rule ----
say('\n---- the focal plane (the mask seat), per tilt ----\n');
F_D = o.lambda / (2*R_dm);                                            % lambda/D as an angle
for a = 1:nA
    sa = S([S.az]==a);  th = [sa.theta];  cen = vertcat(sa.cen);
    [~, jc] = max((max(cen, [], 1) - min(cen, [], 1)));  p = polyfit(th, cen(:,jc), 1);  F_eff = abs(p(1));  resid = (cen(:,jc) - polyval(p, th')) * 1e3;
    say('  %s: effective focal length from the centroid %.1f mm (F2 429 nominal); best focus %+.2f mm from the seat marker; centroid residual from linear (distortion) rms %.3f um, max %.3f um\n', aznm{a}, F_eff, mean([sa.zbest]), rms_(resid), max(abs(resid)));
    for s = sa
        say('    theta %+9.2e rad (%5.2f lam/D): spot %7.2f um = %6.2f lam F/D; WFE (piston/tilt/focus removed) %7.2f nm (raw %7.2f)\n', ...
            s.theta, s.theta/F_D, s.spot_um, s.spot_um*1e-3/(o.lambda*F_eff/(2*R_dm)), s.wfe_nm, s.spot_seat_um);
    end
    ib = abs(th) <= o.band + 1e-12;
    say('  %s, worst in the actuator band (|theta| <= %.1e): spot %.2f um, WFE %.2f nm, centroid residual %.2f um\n', aznm{a}, o.band, max([sa(ib).spot_um]), max([sa(ib).wfe_nm]), max(abs(resid(ib))));
end
% ---- figures: the runner's own ----
f1 = figure('Visible','off','Position',[100 100 1500 700]);
subplot(1,2,1); quiver(u(1:4:end), v(1:4:end), dist_dm(1:4:end,1)', dist_dm(1:4:end,2)', 3, 'Color',[0.1 0.3 0.6]); axis equal; hold on;
th_ = linspace(0,2*pi,200); plot(R_dm*cos(th_), R_dm*sin(th_), 'k-');
title(sprintf('pupil distortion vs the global affine (arrows x3, every 4th ray), DM mm: rms %.3f, max %.3f', rms_(hypot(dist_dm(:,1),dist_dm(:,2))), max(hypot(dist_dm(:,1),dist_dm(:,2)))));
xlabel('DM x, mm'); ylabel('DM y, mm');
subplot(1,2,2); scatter(u, v, 12, blur(ok), 'filled'); axis equal; colorbar; hold on; plot(R_dm*cos(th_), R_dm*sin(th_), 'k-');
title(sprintf('pupil blur over |theta| <= %.1e rad, DM mm rms (pitch 1.0, pixel %.2f)', o.band, 1/mag*0.0417)); xlabel('DM x, mm'); ylabel('DM y, mm');
sgtitle(sprintf('%s: the DM''s image at the camera (%s rig, the DM as the stop)', o.tag, o.rig), 'Interpreter','none');
print(f1, fullfile(o.outdir,[o.tag '_pupil.png']), '-dpng', '-r96');
f2 = figure('Visible','off','Position',[100 100 1500 500]);
for a = 1:nA
    sa = S([S.az]==a);  th = [sa.theta];
    subplot(1,3,1); plot(th*1e3, [sa.spot_um], '-o'); hold on; xlabel('tilt at the DM, mrad'); ylabel('spot rms at best focus, um'); title('focal-plane spot');
    subplot(1,3,2); plot(th*1e3, [sa.wfe_nm], '-o'); hold on; xlabel('tilt at the DM, mrad'); ylabel('WFE, nm rms (piston, tilt, focus removed)'); title('focal-plane wavefront');
    cen = vertcat(sa.cen); [~, jc] = max((max(cen, [], 1) - min(cen, [], 1))); p = polyfit(th, cen(:,jc), 1);
    subplot(1,3,3); plot(th*1e3, (cen(:,jc)-polyval(p,th'))*1e3, '-o'); hold on; xlabel('tilt at the DM, mrad'); ylabel('centroid residual from linear, um'); title('focal-plane distortion');
end
for k = 1:3, subplot(1,3,k); legend(aznm, 'Location','best'); xline(o.band*1e3,':'); xline(-o.band*1e3,':'); end
sgtitle(sprintf('%s: the focal plane per tilt (dotted: the actuator band)', o.tag), 'Interpreter','none');
print(f2, fullfile(o.outdir,[o.tag '_focal.png']), '-dpng', '-r96');
out = struct('o',o,'S',S,'Aff',Aff,'mag',mag,'dist_dm',dist_dm,'blur',blur,'blur_full',blur_full,'cz',cz,'u',u,'v',v);
save(fullfile(o.outdir,[o.tag '.mat']), 'out');
say('run complete\n');  fclose(rep);
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
v = str2num(regexprep(m{1}, '[DdEe]([+-]?\d)', 'e$1')); %#ok<ST2NM>
if isempty(v), v = strtrim(m{1}); end
end
function u = unit_(v), u = v / norm(v); end
function w = rot_(u, ax, th)
ax = unit_(ax); w = u*cos(th) + cross(ax, u)*sin(th) + ax*dot(ax,u)*(1-cos(th)); w = unit_(w);
end
function f = write_deck_(hdr, blocks, pos, f)
hdr = regexprep(hdr, 'ChfRayPos=\s*[^\n]*', sprintf('ChfRayPos=  %.15g  %.15g  %.15g', pos));
fid = fopen(f,'w'); fwrite(fid, [hdr blocks{:}]); fclose(fid);
end
function [uv, ok1, P, D, ok] = trace_(deck, iDM, iDet, fr)
macos.load_rx(deck);  macos.stop(iDM);
s1 = macos.trace(iDM);  r1 = macos.get_ray_info(s1.nRays);
pc = r1.pos - fr.V_dm(:);  uv = [fr.x_dm(:)'*pc; fr.y_dm(:)'*pc];   % each ray's DM coordinate
ok1 = reshape(r1.ok_trace & r1.ok_pass, 1, []);
sD = macos.trace(iDet);  rD = macos.get_ray_info(sD.nRays);
P = rD.pos;  D = rD.dir;  ok = reshape(rD.ok_trace & rD.ok_pass, 1, []);
end
function C = cross_lines_(P1, D1, P2, D2)
% closest point between lines P1 + s D1 and P2 + t D2, per column
n = size(P1,2); C = nan(3,n);
for i = 1:n
    d1 = D1(:,i); d2 = D2(:,i); w = P1(:,i) - P2(:,i);
    a = d1'*d1; b = d1'*d2; c = d2'*d2; d = d1'*w; e = d2'*w; den = a*c - b*b;
    if den < 1e-14, continue; end
    s = (b*e - c*d)/den; t = (a*e - b*d)/den;
    C(:,i) = (P1(:,i) + s*d1 + P2(:,i) + t*d2)/2;
end
end
function r = rms_(x), x = x(isfinite(x)); r = sqrt(mean(x(:).^2)); end
function say_(rep, varargin), fprintf(varargin{:}); fprintf(rep, varargin{:}); end
