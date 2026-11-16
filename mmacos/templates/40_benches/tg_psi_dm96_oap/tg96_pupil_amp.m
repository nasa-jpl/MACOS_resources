function out = tg96_pupil_amp(varargin)
%TG96_PUPIL_AMP  The seeded pupil's AMPLITUDE at the entrance sphere, against a
%   top hat -- and whether it explains the focal spot's excess (REPORT 9.2).
%
%   CHECK 2a says the through-focus quartet's focal field has an 83.8 %
%   encircled-energy radius of 3.96 um where a uniformly illuminated pupil of
%   that diameter gives 3.34 (3.42 on the pupil's measured radius): 16 % large,
%   and NOT quantisation -- halving the focal pitch leaves the number alone.
%   The ray spot at that plane is 0.17 um rms, so the WAVEFRONT is not the
%   cause.  That leaves the amplitude, which is what this measures:
%
%     1. |E| at the entrance sphere, flat DM, against a top hat of the same
%        50 %-radius: the edge width, the ripple inside the clear aperture,
%        and the fraction of the energy outside the hat.
%     2. what that amplitude ALONE does to the focal spot.  The measured
%        amplitude is given a FLAT phase and transformed; so is the top hat.
%        If the measured pupil's own 83.8 % radius lands on the engine's 3.96
%        um, the amplitude is the whole story and nothing else need be
%        chased.  If it lands on the top hat's 3.3, something else is.
%
%   Usage:  tg96_pupil_amp                      % the redo lens rig
%           tg96_pupil_amp('sim',<pupilsim dir>,'tag',...)
%   Name/value: 'rig' ('lens'), 'sim', 'tag', 'model' (1024), 'n_g' (384),
%   'dx_g' (0.28), 'ngrid' (385), 'outdir'.
%   Writes runs/<tag>/<tag>_{report.txt,amp.png}.

exdir = fileparts(mfilename('fullpath'));  if isempty(exdir), exdir = pwd; end
if isempty(which('macos.init')), run(fullfile(exdir,'..','..','..','mmacos_setup.m')); end
o = struct('rig','lens','sim','','tag','','model',1024,'n_g',384,'dx_g',0.28,'ngrid',385,'outdir','');
for k = 1:2:numel(varargin), o.(varargin{k}) = varargin{k+1}; end
if isempty(o.sim),    o.sim    = fullfile(exdir,'runs',['pupilsim_redo_' o.rig]); end
if isempty(o.tag),    o.tag    = ['pupilamp_' o.rig]; end
if isempty(o.outdir), o.outdir = fullfile(exdir,'runs',o.tag); end
if ~exist(o.outdir,'dir'), mkdir(o.outdir); end
rep = fopen(fullfile(o.outdir,[o.tag '_report.txt']),'w');
say = @(varargin) say_(rep, varargin{:});
say('=== tg96_pupil_amp: %s (%s) ===\n', o.tag, datestr(now,'yyyy-mm-dd HH:MM'));

% ---- the chain's own deck, built by the tool that owns it ----------------
G = tg96_pupil_s2s('rig',o.rig, 'sim',o.sim, 'tag',[o.tag '_build'], ...
                   'model',o.model, 'n_g',o.n_g, 'dx_g',o.dx_g, 'ngrid',o.ngrid, ...
                   'build_only',true);
cd(o.outdir);
macos.init(o.model);
macos.write_grid_file('amp_flat.txt', zeros(o.n_g));
txt = fileread(G.deck);  txt = strrep(txt, 'GRIDFILE', 'amp_flat.txt');
f = fullfile(o.outdir,[o.tag '_deck.in']);  fid = fopen(f,'w');  fwrite(fid, txt);  fclose(fid);
names = cellfun(@(b) getv_(b,'EltName'), G.B, 'uni', 0);
iS1 = find(strcmp(names,'S1'),1);  iF = find(strcmp(names,'F'),1);  iDM = find(strcmp(names,'TestOptic'),1);
macos.load_rx(f);  macos.stop(iDM);
E1 = macos.complex_field(iS1);  dx1 = abs(macos.dx_at(iS1,'mm'));
EF = macos.complex_field(iF);   dxF = abs(macos.dx_at(iF,'mm'));
A  = abs(E1);  N = size(A,1);

% ---- 1. the pupil against a top hat ------------------------------------
[CC, RR] = meshgrid(1:N, 1:N);
w = A.^2;  c1 = sum(RR(:).*w(:))/sum(w(:));  c2 = sum(CC(:).*w(:))/sum(w(:));
rr = hypot(RR-c1, CC-c2)*dx1;
A0 = median(A(rr < 0.3*max(rr(A(:) > 0.5*max(A(:))))));      % the plateau
an = A/A0;
prof = radial_(an, rr, dx1);
r50 = interp_cross_(prof.r, prof.v, 0.5);
r90 = interp_cross_(prof.r, prof.v, 0.9);  r10 = interp_cross_(prof.r, prof.v, 0.1);
inr = rr <= 0.9*r50;
ripple = std(an(inr))/mean(an(inr));
hat = double(rr <= r50);
e_out = sum(w(rr > r50))/sum(w(:));
say('pupil at the entrance sphere: 50%% radius %.3f mm, edge 90->10%% over %.3f mm (%.2f of the pitch %.4f mm)\n', ...
    r50, r10 - r90, (r10-r90)/dx1, dx1);
say('   inside 0.9 of that radius: amplitude ripple %.4f rms of the mean; energy outside the 50%% radius %.4f of the total\n', ...
    ripple, e_out);

% ---- 2. what the amplitude alone does to the focal spot ----------------
% flat phase, so nothing but the amplitude can act; the top hat through the
% identical transform is the control
psf_ = @(P) fftshift(abs(fft2(ifftshift(P))).^2);
dxf_pred = lam_()*G.R1/(N*dx1);
[ee_meas, prof_m] = ee_radius_(psf_(A),   dxf_pred, 0.838);
[ee_hat,  prof_h] = ee_radius_(psf_(hat*A0), dxf_pred, 0.838);
airy = 1.22*lam_()*G.R1/(2*r50);
% and the engine's own focal field, for the comparison that started this
[ee_eng, prof_e] = ee_radius_(abs(EF).^2, dxF, 0.838);
say('\nthe focal spot, 83.8%% encircled-energy radius:\n');
say('   the measured pupil amplitude, flat phase   %.3f um\n', 1e3*ee_meas);
say('   a TOP HAT of the same 50%% radius          %.3f um   (Airy 1.22 lam R1 / D = %.3f um)\n', 1e3*ee_hat, 1e3*airy);
say('   the ENGINE''s own focal field               %.3f um   (pitch %.3f um)\n', 1e3*ee_eng, 1e3*dxF);
say('\nverdict: the pupil amplitude accounts for %.0f%% of the engine''s excess over the top hat\n', ...
    100*(ee_meas - ee_hat)/max(ee_eng - ee_hat, eps));
out = struct('o',o,'r50',r50,'edge',r10-r90,'ripple',ripple,'e_out',e_out, ...
             'ee_meas',ee_meas,'ee_hat',ee_hat,'ee_eng',ee_eng,'airy',airy, ...
             'prof',prof,'dx1',dx1,'dxF',dxF);

% ---- the figure --------------------------------------------------------
fh = figure('Visible','off','Position',[100 100 1500 460]);
subplot(1,3,1); imagesc(((1:N)-c2)*dx1, ((1:N)-c1)*dx1, an); axis image; colorbar;
title('|E| at the entrance sphere / plateau'); xlabel('mm');
subplot(1,3,2); plot(prof.r, prof.v, '-', 'LineWidth',1.2); hold on;
plot([0 r50 r50 max(prof.r)], [1 1 0 0], 'k--'); grid on;
xlim([0 1.15*r50]); ylim([-0.05 1.15]); xlabel('pupil radius, mm'); ylabel('|E| / plateau');
title(sprintf('against a top hat: edge %.3f mm = %.1f px, ripple %.3f', r10-r90, (r10-r90)/dx1, ripple));
legend({'measured','top hat'}, 'Location','southwest');
subplot(1,3,3); semilogy(prof_m.r*1e3, prof_m.v/max(prof_m.v), '-', 'LineWidth',1.2); hold on;
semilogy(prof_h.r*1e3, prof_h.v/max(prof_h.v), 'k--');
semilogy(prof_e.r*1e3, prof_e.v/max(prof_e.v), 'r:', 'LineWidth',1.2);
grid on; xlim([0 12]); ylim([1e-5 1.2]); xlabel('focal radius, um'); ylabel('PSF / peak');
title(sprintf('83.8%% EE: measured %.2f, hat %.2f, engine %.2f um', 1e3*ee_meas, 1e3*ee_hat, 1e3*ee_eng));
legend({'measured amplitude','top hat','engine'}, 'Location','northeast');
sgtitle(sprintf('%s: the seeded pupil''s amplitude, and what it does to the focal spot', o.tag), 'Interpreter','none');
print(fh, fullfile(o.outdir,[o.tag '_amp.png']), '-dpng', '-r96');  close(fh);
say('\nwrote %s_amp.png\nrun complete\n', o.tag);
fclose(rep);
end

% =========================================================================
function l = lam_(), l = 6.328e-4; end
function p = radial_(A, rr, dx)
edges = 0:dx:max(rr(:));  v = zeros(1, numel(edges)-1);
for k = 1:numel(edges)-1, m = rr >= edges(k) & rr < edges(k+1); if any(m(:)), v(k) = mean(A(m)); end, end
p = struct('r', (edges(1:end-1)+edges(2:end))/2, 'v', v);
end
function r = interp_cross_(x, y, lev)
% the largest radius at which the falling profile crosses lev
k = find(y >= lev, 1, 'last');
if isempty(k) || k >= numel(y), r = x(end); return; end
r = x(k) + (y(k)-lev)/max(y(k)-y(k+1), eps)*(x(k+1)-x(k));
end
function [r, prof] = ee_radius_(I, dx, frac)
N = size(I,1);  [CC, RR] = meshgrid(1:N, 1:N);
c = N/2 + 1;  rr = hypot(RR-c, CC-c)*dx;
[rs, is] = sort(rr(:));  cum = cumsum(I(is));  cum = cum/cum(end);
k = find(cum >= frac, 1);  r = rs(max(k,1));
prof = radial_(I, rr, dx);
end
function say_(fid, varargin)
fprintf(varargin{:});  if fid > 2, fprintf(fid, varargin{:}); end
end
function v = getv_(b, key)
t = regexp(b, [key '=\s*([^\n]*)'], 'tokens', 'once');
if isempty(t), v = []; return; end
s = strtrim(t{1});  n = str2double(strsplit(s));
if all(~isnan(n)), v = n; else, v = strtrim(strtok(s)); end
end
