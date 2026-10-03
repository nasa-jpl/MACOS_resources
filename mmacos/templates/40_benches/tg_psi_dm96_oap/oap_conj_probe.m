function out = oap_conj_probe(varargin)
%OAP_CONJ_PROBE  Is the reflective rig's "fold coma" a property of the FOLD,
%   or of a 25 mm CONJUGATE ERROR that the fold merely amplifies?
%
%   A paraboloid fed exactly at its focus collimates PERFECTLY at any off-axis
%   distance -- the off-axis angle costs nothing on axis.  So a blur that GROWS
%   with the fold angle (CCMac measured 0.17 / 0.31 / 0.47 / 0.65 / 0.82
%   lambda F/D at 1 / 3 / 5 / 7 / 9 deg, REPORT_oap) cannot be the fold alone:
%   a longitudinal conjugate error delta turns into coma proportional to the
%   off-axis angle, which is exactly that signature.
%
%   Where the conjugate error comes from: Bench emits zSource (default 25) and
%   the engine puts the real point source at ChfRayPos + zSource*ChfRayDir
%   (sourcsub.F:38), so the source sits 25 mm DOWNSTREAM of the point
%   twyman_green's front_end computes -- while add_oap builds the parabola for
%   a focus at that computed point.  The collimator is therefore fed 25 mm
%   inside its focus.  The LENS rig has the identical error; an on-axis lens
%   turns it into pure defocus, which the tail tune absorbs, and only the OAP
%   converts it into an angle-dependent coma.
%
%   This probe isolates it on a bare two-mirror train (no node, no splitter):
%   source -> baffle -> OAP1 (collimate) -> fold -> OAP2 (focus) -> detector,
%   at the rig's own scale, with the source at the record's distance and at
%   the corrected one, over a sweep of fold angles.  Reports, per angle:
%     - the collimated beam's angular spread after OAP1 (rad, rms about the
%       chief) and the equivalent focus distance;
%     - the best-focus ray blur at the OAP2 focus, in lambda F/D, and the
%       longitudinal trim it needed.
%
%   Usage:  out = oap_conj_probe                       % 1..45 deg, both cases
%           out = oap_conj_probe('AOI',[5 9 20 45])

exdir = fileparts(mfilename('fullpath'));  if isempty(exdir), exdir = pwd; end
if isempty(which('macos.init'))
    run(fullfile(exdir, '..', '..', '..', 'mmacos_setup.m'));
end
cd(exdir);
o = struct('AOI', [1 3 5 7 9 15 20 25 30 37 45], 'MODEL', 512, 'NGRID', 65, ...
           'tag', 'conj', 'zsrc', 25);
for i = 1:2:numel(varargin), o.(varargin{i}) = varargin{i+1}; end

P = tg96_params();  s = P.dm(1).nact/56;  b = P.bench;
F1 = s*b.F1;  F2 = s*b.F2;  RAP = s*b.R_TO_AP;  RB = s*b.R_BAFFLE;  DSB = s*b.D_SB;
LAM = P.LAM;
lamFD = LAM * F2 / (2*RAP);                    % lambda F/D at the OAP2 focus, mm
outdir = fullfile(exdir, 'runs', o.tag);
if ~exist(outdir,'dir'), mkdir(outdir); end
rep = fopen(fullfile(outdir, [o.tag '_conj.txt']), 'w');
cleaner = onCleanup(@() fclose(rep));
say = @(varargin) say_(rep, varargin{:});

macos.init(o.MODEL);
say('=== OAP conjugate probe: is the fold coma the FOLD, or a 25 mm conjugate error? ===\n');
say('bare two-mirror train at the rig scale: F1 %.1f, F2 %.1f, beam radius %.1f mm, lambda %.4g mm\n', ...
    F1, F2, RAP, LAM);
say('lambda F/D at the OAP2 focus = %.4f mm (%.2f um); zSource = %g mm\n\n', lamFD, lamFD*1e3, o.zsrc);
say('%5s | %11s %10s | %11s %10s | %9s %9s\n', 'AOI', ...
    'spread rec', 'focus m', 'spread cor', 'focus m', 'blur rec', 'blur cor');
say('%5s | %11s %10s | %11s %10s | %9s %9s\n', 'deg', ...
    'urad rms', '(rec)', 'urad rms', '(cor)', 'lam F/D', 'lam F/D');

R = nan(numel(o.AOI), 6);   % spread_rec, foc_rec, spread_cor, foc_cor, blur_rec, blur_cor
T = nan(numel(o.AOI), 2);   % trim_rec, trim_cor
for k = 1:numel(o.AOI)
    a = o.AOI(k);
    for c = 1:2                                  % 1 = record, 2 = corrected
        extra = (c-1) * o.zsrc;                  % push the source back by zSource
        [sp, foc] = collimation_(a, F1, extra, RAP, RB, DSB, o);
        f = @(t) blur_(a, F1, F2, extra, RAP, RB, DSB, o, t);
        [tb, bb] = fminbnd(f, -25, 25, optimset('TolX',0.02,'MaxFunEvals',40));
        R(k, 2*c-1) = sp*1e6;  R(k, 2*c) = foc/1e3;  R(k, 4+c) = bb/lamFD;
        T(k, c) = tb;
    end
    say('%5g | %11.2f %10.1f | %11.2f %10.1f | %9.3f %9.3f\n', a, ...
        R(k,1), R(k,2), R(k,3), R(k,4), R(k,5), R(k,6));
end
say('\nlongitudinal trim the focus needed (mm): record ');
say('%.2f ', T(:,1));  say('\n                                       corrected ');
say('%.2f ', T(:,2));  say('\n');
out = struct('AOI', o.AOI, 'R', R, 'T', T, 'lamFD', lamFD, 'o', o);
save(fullfile(outdir, [o.tag '_conj.mat']), 'out');
draw_(o, R, outdir);
say('\nwrote %s_conj.{txt,mat,png} in %s\n', o.tag, outdir);
end

% =====================================================================
function b = front_(a, F1, extra, RAP, RB, DSB, o)
%FRONT_  source -> baffle -> OAP1, the front_end geometry of twyman_green's
%   'oap' mode.  EXTRA pushes the bench origin back so that, after the engine
%   applies zSource, the real point source lands on the parabola's focus.
d_out = [1;0;0];
dev = deg2rad(180 - 2*a);
c = cos(-dev);  sn = sin(-dev);
d_in = [c; sn; 0];
pole = [F1; 0; 0];
src  = pole - (F1 + extra)*d_in;
AP = 2*atan(RB/DSB)*0.9;
b = macos.design.Bench('conj', 'aperture', AP, 'ngridpts', o.NGRID, ...
                       'pos', src, 'dir', d_in);
b.add_baffle(DSB, RB);
b.add_oap(F1 + extra - DSB, d_out, 'mode','collimate', 'focus_dist', F1, ...
          'name','OAP1', 'aprad', RAP);
end

function [spread, foc] = collimation_(a, F1, extra, RAP, RB, DSB, o)
%COLLIMATION_  rms angle of the collimated bundle about its chief, and the
%   distance at which the marginal ray would cross it.
bb = front_(a, F1, extra, RAP, RB, DSB, o);
bb.add_reference(300, 'Probe');
rxf = [tempname '.in'];  bb.emit(rxf);  macos.load_rx(rxf);
t = macos.trace(macos.num_elt());
ri = macos.get_ray_info(t.nRays);
ok = ri.ok_trace(:) & ri.ok_pass(:);
D = ri.dir(:,ok);  D = D ./ vecnorm(D);
dch = ri.dir(:,1)/norm(ri.dir(:,1));
ang = acos(max(min(dch.'*D, 1), -1));
spread = sqrt(mean(ang.^2));
% marginal-ray crossing: max height / max angle
P3 = ri.pos(:,ok);  h = vecnorm(P3 - ri.pos(:,1));
if max(ang) > 0, foc = max(h)/max(ang); else, foc = inf; end
delete(rxf);
end

function blur = blur_(a, F1, F2, extra, RAP, RB, DSB, o, trim)
%BLUR_  rms ray blur at the OAP2 focus, TRIM mm from the nominal focus.
try
    bb = front_(a, F1, extra, RAP, RB, DSB, o);
    d_rc = bb.dir;
    dev = deg2rad(180 - 2*a);
    c = cos(dev);  sn = sin(dev);
    out = [c*d_rc(1) - sn*d_rc(2); sn*d_rc(1) + c*d_rc(2); 0];
    bb.add_oap(400, out, 'mode','focus', 'focus_dist', F2, 'name','OAP2', 'aprad', RAP);
    bb.add_detector(F2 + trim, 'Detector');
    rxf = [tempname '.in'];  bb.emit(rxf);  macos.load_rx(rxf);
    t = macos.trace(macos.num_elt());
    ri = macos.get_ray_info(t.nRays);
    ok = ri.ok_trace(:) & ri.ok_pass(:);
    p = ri.pos(:,ok);  cm = mean(p,2);
    blur = sqrt(mean(sum((p-cm).^2,1)));
    delete(rxf);
catch
    blur = 1;                      % 1 mm penalty on a failed build
end
end

function draw_(o, R, outdir)
f = figure('Color','w','Position',[40 40 900 380],'Visible','off');
tl = tiledlayout(f,1,2,'Padding','compact','TileSpacing','compact');
ax = nexttile(tl);
semilogy(ax, o.AOI, R(:,1), 'o-', o.AOI, R(:,3), 's-', 'LineWidth',1.4);
grid(ax,'on');  xlabel(ax,'fold AOI, deg');  ylabel(ax,'collimation spread, urad rms');
legend(ax, {'source at the record''s distance','source at the parabola''s focus'}, 'Location','best');
title(ax,'After OAP1','FontWeight','normal');
ax = nexttile(tl);
plot(ax, o.AOI, R(:,5), 'o-', o.AOI, R(:,6), 's-', 'LineWidth',1.4);
grid(ax,'on');  xlabel(ax,'fold AOI, deg');  ylabel(ax,'best-focus blur, \lambda F/D');
legend(ax, {'record','corrected conjugate'}, 'Location','best');
title(ax,'At the OAP2 focus','FontWeight','normal');
print(f, fullfile(outdir,[o.tag '_conj.png']), '-dpng','-r140');  close(f);
end

function say_(fid, varargin)
fprintf(varargin{:});  fprintf(fid, varargin{:});
end
