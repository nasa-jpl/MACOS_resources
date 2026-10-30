function out = tg96_pupil_options(varargin)
%TG96_PUPIL_OPTIONS  Ways to improve the pupil image, assessed on surgered decks (Dave, 2026-09-17).
%   out = TG96_PUPIL_OPTIONS(name/value ...)
%   Runs tg96_pupilsim (stages 1-2) on variants of the lens rig's deck of record:
%     'record'      the bench as built (the tuned tail: field lens ~f past the focus)
%     'seed'        the geometric seed tail (field lens 10.8 mm past the mask, f 42.9, conic -2.11,
%                   detector at the thin-lens conjugate): the mirror rig's tail geometry on the lens rig
%     'sph'         the record with a SPHERICAL field lens (conic 0): the asphere's zonal power isolated
%     'coll'        TRUE COLLIMATION: the collimator's powered face made the exact hyperbola (Kc = -n^2)
%                   and the source moved to its focus (found from the exit rays); the tuned tail kept
%     'coll_seed'   true collimation + the seed tail
%   Each variant's numbers come from tg96_pupilsim's own report (image surface, band-edge phase,
%   gains at the plane as built and at the compromise plane).  A summary table is written to
%   runs/pupil_options/pupil_options_report.txt.
%   Name/value: 'variants' (cell; default all), 'rig' ('lens'), 'stages' (2: stages 1-2; 1: stage 1 only).
o = struct('variants',{{'record','seed','sph','coll','coll_seed'}},'rig','lens','stages',2);
for k = 1:2:numel(varargin), o.(varargin{k}) = varargin{k+1}; end
exdir = fileparts(mfilename('fullpath'));  if isempty(exdir), exdir = pwd; end
if isempty(which('macos.init')), run(fullfile(exdir,'..','..','..','mmacos_setup.m')); end
outdir = fullfile(exdir,'runs','pupil_options');  if ~exist(outdir,'dir'), mkdir(outdir); end
rep = fopen(fullfile(outdir,'pupil_options_report.txt'),'w');  say = @(varargin) say_(rep, varargin{:});
deck0 = fullfile(exdir,'runs','lensuw2','lensuw2_test.in');
say('=== tg96_pupil_options (%s): variants of %s ===\n', datestr(now,'yyyy-mm-dd HH:MM'), deck0);
[hdr0, blocks0] = split_deck_(fileread(deck0));
names = cellfun(@(b) getv_(b,'EltName'), blocks0, 'uni', 0);
iFoc = find(strcmp(names,'FocalMask'),1);  iFLp = find(strcmp(names,'FLpow'),1);  iFLf = find(strcmp(names,'FLflat'),1);  iDet = numel(blocks0);
iL1p = find(strcmp(names,'L1pow'),1);  iL1f = find(strcmp(names,'L1flat'),1);  iL2p = find(strcmp(names,'L2pow'),1);  iDM = find(strcmp(names,'TestOptic'),1);
V = @(b) getv_(b,'VptElt');  psi_f = getv_(blocks0{iFoc},'psiElt');  dirb = psi_f(:)';   % the beam direction after the focuser
t_fl = norm(V(blocks0{iFLf}) - V(blocks0{iFLp}));
s = 96/56;  FL_F = 25.021*s;  FL_Kc = -2.11278288;  D_MASK_FL = 6.277463741*s;  DET_TRIM = 1.085330067*s;  F2 = 250*s;  n_g = 1.5;
R = struct();
for v = o.variants
    vn = v{1};  hdr = hdr0;  blocks = blocks0;
    switch vn
        case 'record'
        case {'seed','coll_seed'}
            % the seed tail: field lens D_MASK_FL past the mask marker, f FL_F (Kr = -f (n-1)), conic FL_Kc, detector at the thin-lens conjugate + DET_TRIM
            Vm = V(blocks{iFoc});  Vp = Vm + D_MASK_FL*dirb;  Vf = Vp + t_fl*dirb;
            blocks{iFLp} = setv_(blocks{iFLp}, 'VptElt', Vp);  blocks{iFLp} = setv_(blocks{iFLp}, 'RptElt', Vp);
            blocks{iFLp} = regexprep(blocks{iFLp}, 'KrElt=[^\n]*', sprintf('KrElt=  %.10E', -FL_F*(n_g-1)));
            blocks{iFLp} = regexprep(blocks{iFLp}, 'KcElt=[^\n]*', sprintf('KcElt=  %.10E', FL_Kc));
            blocks{iFLf} = setv_(blocks{iFLf}, 'VptElt', Vf);  blocks{iFLf} = setv_(blocks{iFLf}, 'RptElt', Vf);
            s_o = norm(V(blocks{iL2p}) - V(blocks{iDM}));  s_i1 = 1/(1/F2 - 1/s_o);  d12 = norm(Vp - V(blocks{iL2p}));  s_o2 = d12 - s_i1;  s_i2 = 1/(1/FL_F - 1/s_o2);
            det_leg = s_i2 - t_fl + DET_TRIM;  Vd = Vf + det_leg*dirb;
            blocks{iDet} = setv_(blocks{iDet}, 'VptElt', Vd);  blocks{iDet} = setv_(blocks{iDet}, 'RptElt', Vd);
            say('[%s] seed tail: field lens %.2f mm past the marker (f %.1f, conic %.2f), detector %.2f mm past its exit face (thin-lens image %.2f + trim %.2f)\n', vn, D_MASK_FL, FL_F, FL_Kc, det_leg, s_i2 - t_fl, DET_TRIM);
        case 'sph'
            blocks{iFLp} = regexprep(blocks{iFLp}, 'KcElt=[^\n]*', 'KcElt=  0.0000000000E+00');
            say('[%s] spherical field lens (conic 0 instead of %.3f)\n', vn, getv_(blocks0{iFLp},'KcElt'));
    end
    if any(strcmp(vn, {'coll','coll_seed'}))
        % true collimation: the collimator's powered face the exact hyperbola, the source at the focus found from the exit rays
        blocks{iL1p} = regexprep(blocks{iL1p}, 'KcElt=[^\n]*', sprintf('KcElt=  %.10E', -n_g^2));
        src0 = getv_(hdr,'ChfRayPos');  d0 = getv_(hdr,'ChfRayDir');  d0 = d0(:)'/norm(d0);
        macos.init(512);
        tmp = fullfile(outdir, 'coll_probe.in');
        spread = @(zs) coll_spread_(hdr, blocks, zs, tmp, iL1p, d0);
        zs0 = getv_(hdr,'zSource');  zbest = fminbnd(spread, zs0 - 40, zs0 + 40);
        say('[%s] collimator conic %.4f -> %.2f (the exact hyperbola); source moved from zSource %.2f to %.2f mm: exit-ray angular spread %.2e -> %.2e rad rms\n', vn, getv_(blocks0{iL1p},'KcElt'), -n_g^2, zs0, zbest, spread(zs0), spread(zbest));
        hdr = regexprep(hdr, 'zSource=\s*[^\n]*', sprintf('zSource=  %.10g', zbest));
        delete(tmp);
    end
    f = fullfile(outdir, sprintf('opt_%s.in', vn));  write_(hdr, blocks, f);
    say('---- variant %s ----\n', vn);
    r = tg96_pupilsim('rig', o.rig, 'deck', f, 'tag', ['popt_' vn], 'fourier', false);
    R.(vn) = r;
    G1 = r.res(1).G(find([r.res(1).G.f]==0.5 & [r.res(1).G.dir]==1,1));  G2 = r.res(2).G(find([r.res(2).G.f]==0.5 & [r.res(2).G.dir]==1,1));
    say('[%s] image surface vs the detector: on axis %+.2f, mean %+.2f, min %+.2f, max %+.2f mm; astig split %.2f rms / %.2f max; band-edge phase %.3f rms / %.3f max rad\n', ...
        vn, r.cz(1), mean(r.zimg(r.ok)), min(r.zimg(r.ok)), max(r.zimg(r.ok)), rms_(r.astg(r.ok)), max(r.astg(r.ok)), rms_(2*pi/r.o.lambda*abs(r.zimg(r.ok))*r.ab^2/2), max(2*pi/r.o.lambda*abs(r.zimg(r.ok))*r.ab^2/2));
    say('[%s] Nyquist gain: as built mean %.4f min %.4f; at the compromise plane (%+.2f mm) mean %.4f min %.4f; working surface %.3f -> %.3f nm; distortion %.3f rms\n', ...
        vn, G1.gmean, G1.gmin, r.zstar, G2.gmean, G2.gmin, r.res(1).work_err_nm, r.res(2).work_err_nm, rms_(hypot(r.dist(:,1), r.dist(:,2))));
end
say('\n==== summary (lens rig unless stated) ====\n');
say('%-10s | %-28s | %-14s | %-22s | %-22s | %-13s | %s\n', 'variant', 'image vs detector (mm): ctr/mean/min/max', 'astig rms/max', 'Nyquist gain as built', 'at the compromise plane', 'surface err nm', 'distortion');
for v = o.variants
    r = R.(v{1});  G1 = r.res(1).G(find([r.res(1).G.f]==0.5 & [r.res(1).G.dir]==1,1));  G2 = r.res(2).G(find([r.res(2).G.f]==0.5 & [r.res(2).G.dir]==1,1));
    say('%-10s | %+6.2f / %+6.2f / %+6.2f / %+6.2f | %5.2f / %5.2f | mean %.4f min %.4f | %+5.2f mm: %.4f / %.4f | %.2f -> %.2f | %.2f mm\n', v{1}, r.cz(1), mean(r.zimg(r.ok)), min(r.zimg(r.ok)), max(r.zimg(r.ok)), ...
        rms_(r.astg(r.ok)), max(r.astg(r.ok)), G1.gmean, G1.gmin, r.zstar, G2.gmean, G2.gmin, r.res(1).work_err_nm, r.res(2).work_err_nm, rms_(hypot(r.dist(:,1), r.dist(:,2))));
end
out = R;  save(fullfile(outdir,'pupil_options.mat'), 'R');  say('run complete\n');  fclose(rep);
end

function sp = coll_spread_(hdr, blocks, zs, tmp, iL1p, d0)
hdr = regexprep(hdr, 'zSource=\s*[^\n]*', sprintf('zSource=  %.10g', zs));
write_(hdr, blocks, tmp);  macos.load_rx(tmp);
s = macos.trace(iL1p);  r = macos.get_ray_info(s.nRays);  ok = r.ok_trace & r.ok_pass;
d = r.dir(:,ok);  a = d - d0(:);  sp = sqrt(mean(sum((a - mean(a,2)).^2, 1)));   % angular spread of the exit rays about their mean
end
function b = setv_(b, key, v), b = regexprep(b, [key '=[^\n]*'], sprintf('%s=  %.12g  %.12g  %.12g', key, v)); end
function write_(hdr, blocks, f), fid = fopen(f,'w'); fwrite(fid, [hdr blocks{:}]); fclose(fid); end
function [hdr, blocks] = split_deck_(txt)
idx = regexp(txt, '\n[ \t]*iElt=');  hdr = txt(1:idx(1));  blocks = cell(1, numel(idx));
for i = 1:numel(idx), e = numel(txt); if i < numel(idx), e = idx(i+1); end; blocks{i} = txt(idx(i)+1:e); end
end
function v = getv_(blk, key)
m = regexp(blk, ['(?m)^[ \t]*' key '=[ \t]*([^\n]*)'], 'tokens', 'once');
if isempty(m), v = []; return; end
v = str2num(regexprep(m{1}, '[DdEe]([+-]?\d)', 'e$1')); %#ok<ST2NM>
if isempty(v), v = strtrim(m{1}); end
end
function r = rms_(x), x = x(isfinite(x)); r = sqrt(mean(x(:).^2)); end
function say_(rep, varargin), fprintf(varargin{:}); fprintf(rep, varargin{:}); end
