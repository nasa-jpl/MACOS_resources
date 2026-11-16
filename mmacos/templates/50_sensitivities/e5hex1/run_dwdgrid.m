% run_dwdgrid.m -- graphical dw/d(grid-data) sensitivity demo + verification.
% =====================================================================
%  GRID-DATA SENSITIVITY (the GMI pgrid channel) in mmacos, on e5hex1.
%
%  Each grid "poke" -- an influence-function map ADDED to a surface's grid
%  data via macos.elt_grid_add (ANY GridData-enabled SrfType, IN PLACE, so
%  the element keeps its conic/Zernike/monomial components -- unlike GMI,
%  which forces SrfType->9 and clobbers a composite surface) -- has a
%  wavefront sensitivity dW/d(poke).  macos.dw_dgrid builds the Jacobian.
%
%  This shows EVERY poke's wavefront sensitivity in parallel, and verifies
%  the channel two ways:
%    (a) FD-consistency  -- a manual central difference of one poke
%        reproduces its Jacobian column exactly (convention-free);
%    (b) dw_dz_zernike cross-check -- a Zernike-shaped grid poke and the same
%        Zernike via the coefficient channel are the SAME physical sag, so
%        they produce the same wavefront SHAPE (cosine ~1, scale-robust).
%
%  Run:  >> run('.../templates/50_sensitivities/e5hex1/run_dwdgrid.m')
% =====================================================================

here = fileparts(mfilename('fullpath'));  if isempty(here), here = pwd; end
addpath(fullfile(here, '..', '..', '..', 'src'));   % +macos
rx = fullfile(here, 'e5hex1_grid.in');   % pData (grid frame) = RptElt per segment
                                          % (the stock e5hex1.in has pData at the PM
                                          % vertex, so segment Zernikes mis-center)

% ====================  KNOBS  ========================================
MODES = [4 5 6 7 8 9];  % poke modes = MACOS ANSI indices (= MonZernModes in the
                        % .in): 4=astig45 5=defocus 6=astig0 7=trefoil-y
                        % 8=coma-y 9=coma-x.  zernike_grid_basis builds them in
                        % the engine's own convention, so grid poke j pairs with
                        % MonZern mode j directly.
ELTS  = 1:7;            % the 7 hex Segments (Seg1..Seg7)
DELTA = 1e-6;           % FD step (grid-map amplitude, BaseUnits)
MODEL = 256;            % >= e5hex1's nGridMat(256) so mGridMat=256 (else the
                        % grid-add overflows the slot -- cross-segment corruption)
% =====================================================================

m = macos.Session(MODEL);  m.load_rx(rx);
g = macos.find_grid_elts();
if ~isempty(ELTS), g = intersect(g, ELTS(:)); end
fprintf('grid-bearing elements (any GridData type): %s\n', mat2str(g(:).'));
N    = double(mmacos('elt_srf_grid_size', g(1), 1));
% Confine each Zernike to the segment APERTURE (lMon), not the oversized grid:
% the grid is harmless past the segment boundary (rays there hit other segments,
% which see only their own data), but a grid-sized Zernike is a flat sliver
% inside lMon.  ap_frac = aperture radius / grid half-width.
txt  = fileread(rx);
% str2num (not str2double) so Fortran D-exponents parse, e.g. 1.0D-02
lMon = str2num(regexp(txt, '(?<=lMon=)\s*[\d.eEdD+-]+',      'match', 'once'));  %#ok<ST2NM>
gdx  = str2num(regexp(txt, '(?<=GridSrfdx=)\s*[\d.eEdD+-]+', 'match', 'once'));  %#ok<ST2NM>
ap_frac = lMon / (((N-1)/2) * gdx);
fprintf('lMon=%.1f  GridSrfdx=%.3f  -> aperture = %.0f%% of grid half-width\n', ...
        lMon, gdx, 100*ap_frac);
infl = macos.zernike_grid_basis(N, MODES, ap_frac);

res = macos.dw_dgrid(m, rx, 'influence', infl, 'delta', DELTA, ...
                     'reload_rx', false);
% keep only the requested elements (dw_dgrid covers all grid elts)
keep = ismember(res.iElt, g);
res.dwdg = res.dwdg(:, keep);
res.channel_names = res.channel_names(keep);
res.iElt = res.iElt(keep);  res.map_idx = res.map_idx(keep);
ng = numel(g);  nm = numel(MODES);
fprintf('dw_dgrid: %d channels (%d elts x %d pokes), wavefront @ elt %d\n', ...
        numel(res.channel_names), ng, nm, res.wf_elt);

%% -- FIGURE 1: each poke's wavefront sensitivity, in parallel --------
f1 = figure('Name','dw/d(grid) -- poke sensitivities', ...
            'Position',[40 40 max(640,170*nm) max(480,150*ng)]);
for k = 1:numel(res.channel_names)
    subplot(ng, nm, k);
    macos.mimg(macos.v2m(res.dwdg(:,k), res.indx), -1);
    title(res.channel_names{k}, 'FontSize', 8, 'Interpreter','none');
end
colormap(parula);
sgtitle('dW / d(grid poke) -- each influence-function poke in parallel');
print(f1, fullfile(here,'dwdgrid_pokes_e5hex1.png'), '-dpng','-r140');

%% -- MonZern Z4-Z9 sensitivities: the validated reference channel ----
%  NOTE: dw_dz_zernike's 'n_zcoef' is the END mode (target_modes = start:n_zcoef),
%  so 9 gives modes 4..9 -- INCLUDING the comas (7,8,9).  (6 stops at astig0,
%  which is why the coma column was missing before.)
resz = macos.dw_dz_zernike(m, rx, 'kinds',{'monzern'}, 'zmode_start',4, ...
                           'n_zcoef',9, 'reload_rx',true);   % modes 4..9

%% -- FIGURE 2: PAIRED grid-vs-MonZern (DIRECT mode-for-mode) ---------
%  Left  = dW/dZernike via the GRID-DATA channel (dw_dgrid).
%  Right = dW/dZernike via the MONZERN coef channel (dw_dz_zernike), SAME mode.
%  Both build the IDENTICAL ANSI sag on the same per-segment frame, so the pair
%  overlays (signed cosine ~+1) -- for the ORIENTED modes (astig + coma) too,
%  which only align once the grid-array orientation (ndgrid: 1st index = +x)
%  matches the engine's GridMat.  DIRECT pairing (no best-correlation), so the
%  right column is the SAME MonZern mode at every segment.
showm = intersect([6 8 9], MODES, 'stable');   % astig0 (6) + coma-y (8) + coma-x (9)
ccs = [];  ns = numel(showm);
f2 = figure('Name','grid vs MonZern','Position',[60 60 360*ns max(420,150*ng)]);
for r = 1:ng
    ie = g(r);
    for q = 1:ns
        j  = showm(q);
        gk = (r-1)*nm + find(MODES == j, 1);            % grid poke (seg r, mode j)
        zk = find(resz.iElt==ie & resz.mode==j, 1);     % MonZern col, SAME mode j
        A  = macos.v2m(res.dwdg(:,gk),  res.indx);
        B  = macos.v2m(resz.dwdz(:,zk), resz.indx);
        v  = (A~=0) & (B~=0);
        c  = (A(v).'*B(v)) / max(norm(A(v))*norm(B(v)), eps);   % SIGNED cosine
        ccs(end+1) = c; %#ok<SAGROW>
        base = (r-1)*2*ns + (q-1)*2;
        subplot(ng, 2*ns, base+1); macos.mimg(A, -1);
            title(sprintf('grid Z%d S%d', j, r), 'FontSize',7);
        subplot(ng, 2*ns, base+2); macos.mimg(B, -1);
            title(sprintf('MonZern Z%d  %.3f', j, c), 'FontSize',7);
    end
end
colormap(parula);
sgtitle(sprintf('dW/dZernike:  left = grid data,  right = MonZern   (mean cosine %.3f)', mean(ccs)));
print(f2, fullfile(here,'dwdgrid_vs_monzern_e5hex1.png'), '-dpng','-r140');

%% -- VERIFY: FD-consistency (manual central diff vs Jacobian column) -
M1 = infl(:,:,1);  e1 = g(1);  wf = res.wf_elt;
m.load_rx(rx);                       m.trace(wf);  W0 = m.opd();
macos.elt_grid_add(e1, +DELTA*M1);   m.modify();  m.trace(wf);  Wp = m.opd();
macos.elt_grid_add(e1, -2*DELTA*M1); m.modify();  m.trace(wf);  Wm = m.opd();
macos.elt_grid_add(e1, +DELTA*M1);   m.modify();                    % restore
col1 = find(res.iElt==e1 & res.map_idx==1, 1);
dW_manual = macos.m2v((Wp - Wm)/(2*DELTA), res.indx);
rel = norm(dW_manual - res.dwdg(:,col1)) / max(norm(res.dwdg(:,col1)), eps);
fprintf('FD-consistency (manual vs Jacobian, Elt %d poke 1): rel = %.3e  [%s]\n', ...
        e1, rel, tern_(rel < 1e-9, 'OK', '** FAIL'));
fprintf('\n=== dw_dgrid example: %d pokes; FD-consistency %s; grid-vs-MonZern mean cosine %.4f ===\n', ...
        numel(res.channel_names), tern_(rel<1e-9,'PASS','FAIL'), mean(ccs));

% ---- helper --------------------------------------------------------
function s = tern_(c,a,b), if c, s=a; else, s=b; end, end
