%S9_DIFFRACTION  e2e stage 9: the coupled telescope + CTB diffraction model, and its dark zone.
%
%   Consumes s8_fold_full.in (the telescope with the CTB folded behind it,
%   stage 8's pick) and produces the diffraction model of that train: the
%   CTB's own generator (templates/30_instruments/bench_ctb/ctb_prop_layout,
%   its 'lead' option) inserts the three focal-mask quartets, the DM1->DM2
%   near-field leg and the far-field terminal behind the telescope, every
%   reference sphere FEX'd through the whole train.  Then the CTB's masked
%   chain, DM layer, Jacobian and EFC run on it unchanged (ctb_chain /
%   ctb_dm_rx / ctb_dm_jacobian / ctb_efc, each given the coupled deck and
%   its station map) -- the coronagraph analyses of the mmacos manual's
%   Section 8, on the telescope instead of the bench's point source
%   (PLAN_CONSOLIDATION 3a, Dave 2026-10-28: the coupled model AFTER the
%   fold; then pairwise probing on it).
%
%   [1] the decks: s9_coupled.in (compact, 40 elements) and
%       s9_coupled_s2s.in (full, 53); the generator's own verification
%       (chief ray at every optic against the bare train, PSF on the
%       centre pixel), the quartet audit (the two spheres' zElt equal to
%       the digit, Return count even) and the NF1/NF2 round-trip identity
%       on every quartet -- the three conventions tCtbProp pins on the
%       bench, here on the coupled train.
%   [2] the DM deck s9_coupled_dm.in (grid-data DMs; flat256.txt beside it,
%       regenerated) and the station map.
%   [3] the static coronagraph, coupled against the bench: bare peak,
%       lambda/D per pixel, the mask scales, the dark-zone contrast
%       3-15 lambda/D (hard occulter 2.70 lambda/D, Lyot 0.50, the shipped
%       config), the exit-pupil footprint (the secondary's shadow and the
%       anamorphic beam) and what the engine makes of it: the PSF core's
%       axis ratio against the pupil's.  The engine carries the field on
%       the ray lattice with ONE spacing, so a pupil that is anamorphic by
%       4 % gives a round PSF -- a model limit, measured here and bracketed
%       by the static contrast at the occulter radius scaled by that ratio.
%   [3b] what sets the static halo.  Not the telescope's wavefront: on DM1
%       it is 0.0024 waves rms, and a DM1 flattening (a least-squares fit
%       of the per-ray optical path on the influence functions, measured in
%       [4a]) leaves the dark zone unchanged.  It is the secondary's shadow:
%       0.4 % of the pupil area, whose diffraction the occulter throws into
%       the Lyot plane around the shadow's image, where the bench's plain
%       disk stop passes it.  Measured as a counterfactual (the same deck
%       with the obscuration removed) and cured as a testbed would: an
%       annular Lyot stop (ctb_chain 'r_lyot_inner_frac'), its blocker
%       radius scanned over P.coro.lyot_inner and the best static picked.
%   [4] the dark zone dug on that stop: the engine-measured Jacobian
%       (central differences; reused when its fingerprint truth holds and
%       its chain stamp is this chain's, else re-measured) and EFC from
%       flat mirrors.  The EFC step the bench's regularization grid
%       proposes is outside the linear range on a brighter halo (DM2's
%       edge actuators at 65 nm peaks), so the grid runs to 1.  Then one
%       relinearization round, the bench's depth lever (ch08b): the
%       Jacobian re-measured about the dug state and the loop continued
%       from it (tag e2e_coupled_r1, the same reuse rule).
%   [5] products: s9_coupled.in, s9_coupled_s2s.in, s9_coupled_dm.in,
%       s9_noobs_dm.in (the counterfactual),
%       ctb_dm_jacobian_N512_e2e_coupled{,_r1}.{mat,fp.json},
%       ctb_efc_e2e_coupled{,_r1}.{mat,png}, s9_fields.png, s9_report.txt,
%       s9_run.mat.
%
%   See also S8_CORONAGRAPH, CTB_PROP_LAYOUT, CTB_CHAIN, CTB_DM_JACOBIAN, CTB_EFC.

exdir = fileparts(mfilename('fullpath'));
root  = fullfile(exdir, '..', '..', '..');
run(fullfile(root, 'mmacos_setup.m'));
bench = fullfile(root, 'templates', '30_instruments', 'bench_ctb');
addpath(bench);  addpath(exdir);
P = e2e_params();  Q = P.coro;
cd(exdir);
t0 = tic;
say = @(varargin) fprintf('%s\n', sprintf(varargin{:}));
MODEL = 512;                                                   % the CTB's diffraction model size (nGridpts 255)

src = fullfile(exdir, 's8_fold_full.in');
assert(isfile(src), 's9_diffraction: run s8_coronagraph first (%s missing)', src);
txt = fileread(src);
names = regexp(txt, '(?m)^\s*EltName=\s*(\S+)', 'tokens');  names = [names{:}];
lead = find(strcmp(names, 'OAP1'), 1) - 1;                     % the telescope and its fold F0
assert(~isempty(lead) && lead > 0, 's9_diffraction: no OAP1 in %s', src);

%% -- [1] the coupled diffraction decks ------------------------------------
say('\n==================== e2e s9 -- the coupled telescope + CTB diffraction model');
say('[1] %s: %d elements, %d before the CTB (%s .. %s)', src, numel(names), lead, names{1}, names{lead});
info = ctb_prop_layout('src', src, 'lead', lead, 'stem', 's9_coupled', 'suffix', '', ...
                       'outdir', exdir, 'model', MODEL, 'verify', true);
G = struct();
for v = 1:numel(info)
    c = info(v).chk;
    say('    %-8s %s: %d elements; chief-ray match %.2e mm (%s); PSF peak %.4e at [%d,%d] (centre %d)', ...
        info(v).name, info(v).out, info(v).nElt, c.chief_max_mm, c.chief_worst, c.psf_peak, c.psf_row, c.psf_col, c.psf_centre);
    G.(['chief_' info(v).name]) = struct('v', c.chief_max_mm, 'ok', c.chief_max_mm < 1e-9, 'say', sprintf('%s: chief ray at every optic within %.1e mm of the bare train', info(v).name, c.chief_max_mm));
    G.(['psf_' info(v).name])   = struct('v', c.psf_centred, 'ok', logical(c.psf_centred), 'say', sprintf('%s: PSF peak on the centre pixel [%d,%d]', info(v).name, c.psf_row, c.psf_col));
    % the quartet audit and the round trip
    q = quartets_(info(v).out);
    nret = count_(info(v).out, 'Return');
    zeq = all(arrayfun(@(k) strcmp(q(k).zEPstr, q(k).zEP2str), 1:numel(q)));
    G.(['quartet_' info(v).name]) = struct('v', numel(q), 'ok', numel(q) == 3 && zeq && mod(nret, 2) == 0, ...
        'say', sprintf('%s: %d quartets, sphere zElts equal to the digit (%d), %d Return surfaces (even)', info(v).name, numel(q), zeq, nret));
    macos.init(MODEL);  macos.load_rx(info(v).out);
    first = true;  relmax = 0;
    for k = 1:numel(q)
        a = macos.complex_field(q(k).iEP,  'reset_trace', first);  first = false;
        b = macos.complex_field(q(k).iEP2, 'reset_trace', false);
        relmax = max(relmax, max(abs(b(:) - a(:))) / max(abs(a(:))));
    end
    G.(['roundtrip_' info(v).name]) = struct('v', relmax, 'ok', relmax < 1e-10, ...
        'say', sprintf('%s: NF1/NF2 round trip is the identity on every quartet (max rel %.1e)', info(v).name, relmax));
    say('    %-8s spheres: %s', info(v).name, strjoin(cellfun(@(f) sprintf('%s %.6f m', f, info(v).R.(f)), fieldnames(info(v).R), 'UniformOutput', false), ', '));
end
compact = info(strcmp({info.name}, 'compact'));
ix = compact.ix;
e = struct('DM1', ix.DM1, 'DM2', ix.DM2, 'Apodizer', ix.Apodizer, 'FPM', ix.FPM, ...
           'Lyot', ix.Lyot, 'ExitPupil', ix.ExitPupil, 'FPA', ix.FPA);

%% -- [2] the DM deck -----------------------------------------------------
r = ctb_dm_rx('rx_in', compact.out, 'rx_out', fullfile(exdir, 's9_coupled_dm.in'));
say('[2] %s: DMs at %s, base unit %g m, grid %d (%.4g m)', r.rx_out, mat2str(r.ielt), r.cbm, r.ng, r.gdx_mm(1));
say('    station map: DM1 %d DM2 %d Apodizer %d FPM %d Lyot %d ExitPupil %d FPA %d', e.DM1, e.DM2, e.Apodizer, e.FPM, e.Lyot, e.ExitPupil, e.FPA);

%% -- [3] the static coronagraph, coupled against the bench -------------------
S = struct();
for which = {'coupled', 'bench'}
    w = which{1};
    if strcmp(w, 'coupled'), ch = ctb_chain('rx', r.rx_out, 'elt', e, 'model_size', MODEL);
    else,                    ch = ctb_chain('model_size', MODEL); end
    Eb = ch.run_bare();  E = ch.run();  M = ch.dz_mask(3, 15);
    c = mean(abs(E(M)).^2) / ch.peak_bare;
    Ib = abs(Eb).^2;  N = ch.N;  c0 = ch.center_px;
    [xx, yy] = meshgrid((1:N) - c0, (1:N) - c0);
    core = hypot(xx, yy) <= 1.5*ch.lamD_px;
    w2 = Ib .* core;  s = sum(w2(:));
    mx = sum(w2(:).*xx(:))/s;  my = sum(w2(:).*yy(:))/s;
    Cxx = sum(w2(:).*(xx(:)-mx).^2)/s;  Cyy = sum(w2(:).*(yy(:)-my).^2)/s;  Cxy = sum(w2(:).*(xx(:)-mx).*(yy(:)-my))/s;
    ev = eig([Cxx Cxy; Cxy Cyy]);  core_ratio = sqrt(max(ev)/min(ev));
    % the ray footprint on the FPM's feeding sphere: the pupil's own axis ratio
    j = ch.elt.FPM - 1;
    ri = macos.get_ray_info(macos.trace(j).nRays);  okk = ri.ok_pass & ri.ok_trace;
    qq = ri.pos(:, okk) - mean(ri.pos(:, okk), 2);  [~, Sv, ~] = svd(qq.', 'econ');
    ax = sqrt(2)*diag(Sv)/sqrt(nnz(okk)) * macos.cbm();  pupil_ratio = ax(1)/ax(2);   % metres
    macos.intensity(ch.elt.FPA);  Iep = macos.intensity(ch.elt.ExitPupil, 'reset_trace', false);
    % the model-error bracket: the occulter at the radius the other pupil axis would see
    cb = zeros(1, 2);  rr = 2.70 * [1/pupil_ratio, pupil_ratio];
    if strcmp(w, 'coupled')
        for b = 1:2
            chb = ctb_chain('rx', r.rx_out, 'elt', e, 'model_size', MODEL, 'r_fpm_lamD', rr(b));
            Ebb = chb.run();  cb(b) = mean(abs(Ebb(M)).^2) / chb.peak_bare;
        end
    end
    S.(w) = struct('peak_bare', ch.peak_bare, 'lamD_px', ch.lamD_px, 'lamD_fpm_um', ch.masks.lamD_fpm_m*1e6, ...
                   'dx_f_um', ch.masks.dx_f*1e6, 'r_lyot_geom_mm', ch.masks.r_lyot_geom_m*1e3, 'contrast', c, ...
                   'core_ratio', core_ratio, 'pupil_ratio', pupil_ratio, 'pupil_axes_mm', ax(1:2).'*1e3, 'rays', nnz(okk), ...
                   'bracket', cb, 'r_fpm_bracket', rr, 'Ib', Ib, 'I', abs(E).^2/ch.peak_bare, 'Iep', Iep, 'E0', E);
    say('[3] %-8s bare peak %.4e, lambda/D %.3f px, FPM scale %.2f um/(lambda/D) (pitch %.3f um), Lyot geometric radius %.3f mm', ...
        w, ch.peak_bare, ch.lamD_px, S.(w).lamD_fpm_um, S.(w).dx_f_um, S.(w).r_lyot_geom_mm);
    say('    %-8s dark zone 3-15 lambda/D, static: %.3e; pupil footprint %.3f x %.3f mm (ratio %.4f, %d rays), PSF core axis ratio %.4f', ...
        w, c, S.(w).pupil_axes_mm, pupil_ratio, nnz(okk), core_ratio);
    if strcmp(w, 'coupled')
        say('    %-8s occulter at 2.70/%.4f and 2.70x%.4f lambda/D: %.3e .. %.3e (the model-error bracket of the one-spacing lattice)', w, pupil_ratio, pupil_ratio, cb);
    end
end
G.static = struct('v', S.coupled.contrast, 'ok', S.coupled.contrast < 1e-4 && S.coupled.contrast > S.bench.contrast, ...
    'say', sprintf('static dark zone %.3e on the telescope with the bench''s plain Lyot disk (bench alone %.3e)', S.coupled.contrast, S.bench.contrast));
G.lattice = struct('v', S.coupled.core_ratio, 'ok', true, ...
    'say', sprintf('MODEL LIMIT (reported, not gated): pupil axis ratio %.4f, PSF core axis ratio %.4f -- the field rides the ray lattice with one spacing; static contrast %.3e .. %.3e over the bracket', ...
        S.coupled.pupil_ratio, S.coupled.core_ratio, S.coupled.bracket));

%% -- [3b] what sets the halo: the secondary's shadow, and the Lyot stop against it ----
txtdm = fileread(r.rx_out);
noobs = regexprep(txtdm, '(?m)^\s*nObs=\s*1\s*\n\s*ObsType=\s*Circle\s*\n\s*ObsVec=[^\n]*\n', '             nObs=  0\n', 'once');
assert(~strcmp(noobs, txtdm), 's9_diffraction: no circular obscuration found to remove');
fnoobs = fullfile(exdir, 's9_noobs_dm.in');  fid = fopen(fnoobs, 'w');  fwrite(fid, noobs);  fclose(fid);
ch = ctb_chain('rx', fnoobs, 'elt', e, 'model_size', MODEL);
En = ch.run();  M = ch.dz_mask(3, 15);  c_noobs = mean(abs(En(M)).^2) / ch.peak_bare;
rin = macos.get_ray_info(macos.trace(e.FPM - 1).nRays);  n_noobs = nnz(rin.ok_pass & rin.ok_trace);
shadow = 1 - S.coupled.rays / n_noobs;
say('[3b] the same deck without the secondary''s obscuration: static dark zone %.3e (with it %.3e): the shadow, %.2f %% of the pupil area (%d of %d rays), sets the halo', ...
    c_noobs, S.coupled.contrast, 100*shadow, n_noobs - S.coupled.rays, n_noobs);
li = Q.lyot_inner;  c_li = zeros(size(li));                            % the Lyot stop against it: a central blocker, scanned
for k = 1:numel(li)
    chk = ctb_chain('rx', r.rx_out, 'elt', e, 'model_size', MODEL, 'r_lyot_inner_frac', li(k));
    Ek = chk.run();  c_li(k) = mean(abs(Ek(M)).^2) / chk.peak_bare;
    say('     Lyot inner blocker %.2f of the pupil radius (annulus throughput %3.0f %% of the plain stop): static %.3e', li(k), 100*(1 - (li(k)/0.5)^2), c_li(k));
end
[c_lyot, kpick] = min(c_li);  li_pick = li(kpick);
say('     pick: inner %.2f, static %.3e (plain stop %.3e, no obscuration %.3e)', li_pick, c_lyot, S.coupled.contrast, c_noobs);
S.coupled.noobs = c_noobs;  S.coupled.shadow_frac = shadow;  S.coupled.lyot_inner = li;  S.coupled.lyot_scan = c_li;  S.coupled.lyot_pick = li_pick;  S.coupled.contrast_lyot = c_lyot;
G.shadow = struct('v', c_noobs, 'ok', c_noobs < S.coupled.contrast/5, 'say', sprintf('the halo is the secondary''s shadow: without it %.3e, with it %.3e (gate x5)', c_noobs, S.coupled.contrast));
G.lyot = struct('v', c_lyot, 'ok', c_lyot < S.coupled.contrast/5 && li_pick > 0, 'say', sprintf('a central Lyot blocker of %.2f pupil radii takes it: static %.3e (gate x5 below the plain stop''s %.3e)', li_pick, c_lyot, S.coupled.contrast));
chain_pick = {'elt', e, 'r_lyot_inner_frac', li_pick};
ch = ctb_chain('rx', r.rx_out, 'model_size', MODEL, chain_pick{:});   % the picked stop: the chain owns the session from here
Ep = ch.run();  S.coupled.I_lyot = abs(Ep).^2 / ch.peak_bare;

%% -- [4a] DM1 takes the telescope's wavefront ---------------------------------
cbm = macos.cbm();
ri = macos.get_ray_info(macos.trace(e.DM1).nRays);                   % per-ray OPL at DM1, deck units
okr = ri.ok_pass(:) & ri.ok_trace(:);
blk = regexp(fileread(r.rx_out), '(?s)EltName=\s*DM1.*?nECoord', 'match', 'once');
v3 = @(key) sscanf(strrep(upper(char(regexp(blk, ['(?m)^\s*' key '=\s*([^\n]+)'], 'tokens', 'once'))), 'D', 'E'), '%g');
Vp = v3('VptElt');  zx = v3('psiElt');  zx = zx(1:3)/norm(zx(1:3));  xx = v3('xObs');  xx = xx(1:3)/norm(xx(1:3));  yx = cross(zx, xx);
pr = ri.pos(:, okr) - Vp(1:3);  px = xx.'*pr;  py = yx.'*pr;            % ray hits in DM1's grid frame
detr = @(w, pos) w - [ones(size(w)), pos.'] * ([ones(size(w)), pos.'] \ w);   % piston and tilt out (a plane in position)
w0 = detr(ri.opl(okr), ri.pos(:, okr));                                 % the telescope's wavefront on DM1
q1 = J_dm_geom_(r, 1);                                                  % ctb_dm's lattice for DM1 (the Jacobian's)
act = find(q1.active);
A = exp(log(q1.coupling) * (hypot(px(:) - q1.acx(act).', py(:) - q1.acy(act).') / q1.pitch_mm).^2);
aoi1 = 5;
coef = [A, ones(size(w0)), px(:), py(:)] \ (-w0 / (2*cosd(aoi1)));     % surface that cancels it (piston/tilt free)
a1 = zeros(q1.nact^2, 1);  a1(act) = coef(1:numel(act));
resid_ = @() std(detr_after_(e.DM1 + 1, detr));
r_before = std(w0);  q1.apply(a1);  r_after = resid_();
if r_after > r_before, a1 = -a1;  q1.apply(a1);  r_after = resid_(); end   % the grid's displacement sign, settled by measurement
macos.trace(e.FPA);  Wf = macos.opd() / cbm;  Mf = macos.opd_mask();
Ef = ch.run();  M = ch.dz_mask(3, 15);  c_flat = mean(abs(Ef(M)).^2) / ch.peak_bare;
lam = ch.lambda_m / cbm;
say('[4a] DM1 flattening (a measurement): wavefront on DM1 %.4f waves rms over %d rays -> %.4f after it (fit on %d actuators, %.2f nm rms, %.1f nm peak); FPA %.4f waves; static dark zone %.3e (before %.3e): the halo is amplitude, not phase', ...
    r_before/lam, nnz(okr), r_after/lam, numel(act), rms(a1(act))*1e9*cbm, max(abs(a1))*1e9*cbm, std(Wf(Mf))/lam, c_flat, c_lyot);
S.coupled.flat = struct('opd_before_waves', r_before/lam, 'opd_after_waves', r_after/lam, 'fpa_waves', std(Wf(Mf))/lam, ...
                        'a1', a1, 'stroke_rms_nm', rms(a1(act))*1e9*cbm, 'stroke_peak_nm', max(abs(a1))*1e9*cbm, 'contrast', c_flat, 'I', abs(Ef).^2/ch.peak_bare);
G.flat = struct('v', c_flat, 'ok', true, 'say', sprintf('REPORTED, not gated: the wavefront on DM1 is %.4f waves; flattening it (%.2f nm rms on DM1) moves the static dark zone %.3e -> %.3e -- the halo is amplitude', r_before/lam, rms(a1(act))*1e9*cbm, c_lyot, c_flat));
q1.clear();

%% -- [4] the dark zone dug ---------------------------------------------------
% resumable: a Jacobian measured earlier on THIS deck and chain is reused when its fingerprint truth
% beside it holds and its chain stamp is this chain's (the truth is committed, the .mat is not --
% delete the .mat to re-measure); a stale pair is removed and the Jacobian re-measured
J = jacobian_(exdir, 'e2e_coupled', r, e, ch, chain_pick, {}, MODEL, say);
say('[4] Jacobian: %d columns, %d dark-zone pixels, contrast0 %.3e, %.1f min', size(J.G, 2), numel(J.dz_idx), J.contrast0, J.timing_s/60);
% the telescope's halo is 20x the bench's, so the first step the bench's alpha grid (1e-6..1e-2 of
% s_max^2) proposes is 4.5 nm rms on DM2 and already outside the linear range (the achieved field
% change correlates 0.34 with the prediction at gain 1, 0.99 at gain 0.1); the grid runs to 1 here
out = ctb_efc('jac', J, 'tag', 'e2e_coupled', 'outdir', exdir, 'niter', 20, 'alphas', logspace(-6, 0, 7), 'visible', false);
cf = out.contrast;
say('    EFC on the annular stop: %.3e -> %.3e in %d iterations (x%.0f); strokes %s nm rms', cf(1), cf(end), numel(cf)-1, cf(1)/cf(end), mat2str(out.stroke_rms_nm, 3));
%% -- [4b] one relinearization round: the Jacobian about the dug state --------
J2 = jacobian_(exdir, 'e2e_coupled_r1', r, e, ch, chain_pick, out.a, MODEL, say);
out2 = ctb_efc('jac', J2, 'a0', out.a, 'tag', 'e2e_coupled_r1', 'outdir', exdir, 'niter', 20, 'alphas', logspace(-6, 0, 7), 'visible', false);
cf2 = out2.contrast;
say('    relinearized: %.3e -> %.3e in %d iterations; strokes %s nm rms; overall x%.0f from the annular stop''s static', cf2(1), cf2(end), numel(cf2)-1, mat2str(out2.stroke_rms_nm, 3), cf(1)/cf2(end));
G.dig = struct('v', cf(1)/cf2(end), 'ok', cf2(end) < cf(1)/10, 'say', sprintf('EFC digs the coupled dark zone %.3e -> %.3e fixed-G, -> %.3e relinearized (x%.0f, gate x10)', cf(1), cf(end), cf2(end), cf(1)/cf2(end)));
G.stroke = struct('v', max(out2.stroke_rms_nm), 'ok', max(out2.stroke_rms_nm) < 50, 'say', sprintf('DM strokes %s nm rms (gate 50)', mat2str(out2.stroke_rms_nm, 3)));

%% -- [5] products ---------------------------------------------------------
gn = fieldnames(G);  allok = true;
for k = 1:numel(gn), say('    [%s] %s', tern_(G.(gn{k}).ok, 'PASS', 'FAIL'), G.(gn{k}).say);  allok = allok && G.(gn{k}).ok; end
try
    f = figure('Visible', 'off', 'Position', [0 0 1500 900], 'Color', 'w');  k = 0;
    for which = {'coupled', 'bench'}
        w = which{1};  R = S.(w);  N = size(R.Ib, 1);  c0 = floor(N/2) + 1;  hw = round(20*R.lamD_px);  rg = c0-hw:c0+hw;  lx = ((rg) - c0)/R.lamD_px;
        k = k+1;  subplot(2, 3, k);  imagesc(lx, lx, log10(R.Ib(rg, rg)/max(R.Ib(:))));  axis image;  colorbar;  caxis([-8 0]);  title(sprintf('%s: bare PSF (log10, peak 1)', w));  xlabel('\lambda/D');
        k = k+1;  subplot(2, 3, k);
        if isfield(R, 'I_lyot')
            imagesc(lx, lx, log10(R.I_lyot(rg, rg)));  axis image;  colorbar;  caxis([-10 -4]);  xlabel('\lambda/D');
            title(sprintf('%s: coronagraph, Lyot blocker %.2f: %.2e (plain stop %.2e)', w, R.lyot_pick, R.contrast_lyot, R.contrast));
        else
            imagesc(lx, lx, log10(R.I(rg, rg)));  axis image;  colorbar;  caxis([-10 -4]);  title(sprintf('%s: coronagraph, dark zone %.2e', w, R.contrast));  xlabel('\lambda/D');
        end
        k = k+1;  subplot(2, 3, k);  imagesc(R.Iep);  axis image;  colorbar;  title(sprintf('%s: exit pupil (axis ratio %.3f)', w, R.pupil_ratio));
    end
    sgtitle(f, 'e2e s9: the CTB on the telescope against the bench alone (model 512, 500 nm)');
    print(f, fullfile(exdir, 's9_fields.png'), '-dpng', '-r110');  close(f);
catch ME, say('    (figure: %s)', ME.message);
end
rep = {};
rep{end+1} = sprintf('e2e s9 -- the coupled telescope + CTB diffraction model (%s)', datestr(now, 'yyyy-mm-dd'));
rep{end+1} = sprintf('source: %s (%d elements, %d before the CTB); decks: %s', src, numel(names), lead, strjoin({info.out}, ', '));
for v = 1:numel(info)
    rep{end+1} = sprintf('   %-8s %d elements; spheres %s', info(v).name, info(v).nElt, strjoin(cellfun(@(f) sprintf('%s %.6f m', f, info(v).R.(f)), fieldnames(info(v).R), 'UniformOutput', false), ', '));
end
rep{end+1} = sprintf('station map (compact): DM1 %d DM2 %d Apodizer %d FPM %d Lyot %d ExitPupil %d FPA %d', e.DM1, e.DM2, e.Apodizer, e.FPM, e.Lyot, e.ExitPupil, e.FPA);
for which = {'coupled', 'bench'}
    w = which{1};  R = S.(w);
    rep{end+1} = sprintf('%-8s bare peak %.4e, lambda/D %.3f px, FPM scale %.2f um, Lyot geometric radius %.3f mm; static dark zone %.3e; pupil %.3f x %.3f mm (ratio %.4f), PSF core ratio %.4f', ...
        w, R.peak_bare, R.lamD_px, R.lamD_fpm_um, R.r_lyot_geom_mm, R.contrast, R.pupil_axes_mm, R.pupil_ratio, R.core_ratio);
end
rep{end+1} = sprintf('model-error bracket: occulter %.3f / %.3f lambda/D -> %.3e / %.3e', S.coupled.r_fpm_bracket, S.coupled.bracket);
rep{end+1} = sprintf('the shadow: %.2f %% of the pupil area; without it the static dark zone is %.3e; Lyot inner blocker scan %s -> %s; pick %.2f: %.3e', ...
    100*S.coupled.shadow_frac, S.coupled.noobs, mat2str(S.coupled.lyot_inner), mat2str(S.coupled.lyot_scan, 4), S.coupled.lyot_pick, S.coupled.contrast_lyot);
fl = S.coupled.flat;
rep{end+1} = sprintf('DM1 flattening: pupil OPD %.4f -> %.4f waves rms (FPA %.4f); %.2f nm rms, %.1f nm peak on DM1; static dark zone %.3e', fl.opd_before_waves, fl.opd_after_waves, fl.fpa_waves, fl.stroke_rms_nm, fl.stroke_peak_nm, fl.contrast);
rep{end+1} = sprintf('EFC (central-difference Jacobian at flat mirrors, %d columns, %.1f min): %s', size(J.G, 2), J.timing_s/60, mat2str(cf, 4));
rep{end+1} = sprintf('   strokes %s nm rms', mat2str(out.stroke_rms_nm, 3));
rep{end+1} = sprintf('relinearized (Jacobian about the dug state, %.1f min): %s', J2.timing_s/60, mat2str(cf2, 4));
rep{end+1} = sprintf('   strokes %s nm rms; bench reference (ch08b, plain stop): 2.934e-7 -> 8.30e-9 fixed-G, 3.82e-9 relinearized', mat2str(out2.stroke_rms_nm, 3));
for k = 1:numel(gn), rep{end+1} = sprintf('   [%s] %s', tern_(G.(gn{k}).ok, 'PASS', 'FAIL'), G.(gn{k}).say); end
rep{end+1} = sprintf('products: s9_coupled.in s9_coupled_s2s.in s9_coupled_dm.in s9_noobs_dm.in ctb_dm_jacobian_N512_e2e_coupled{,_r1}.{mat,fp.json} ctb_efc_e2e_coupled{,_r1}.{mat,png} s9_fields.png s9_run.mat; %.0f s', toc(t0));
fid = fopen(fullfile(exdir, 's9_report.txt'), 'w');  fprintf(fid, '%s\n', rep{:});  fclose(fid);
S9 = struct('G', G, 'S', rmfield_(S), 'info', info, 'e', e, 'efc_contrast', cf, 'strokes_nm', out.stroke_rms_nm, ...
            'efc_contrast_r1', cf2, 'strokes_nm_r1', out2.stroke_rms_nm, 'lead', lead);
save(fullfile(exdir, 's9_run.mat'), '-struct', 'S9');
say('[5] report s9_report.txt, s9_fields.png, s9_run.mat; %s; %.0f s', tern_(allok, 'ALL GATES PASS', 'A GATE FAILED'), toc(t0));

% =================================================================================================
function s = tern_(c, a, b)
    if c, s = a; else, s = b; end
end

function S = rmfield_(S)
    for f = fieldnames(S).'
        S.(f{1}) = rmfield(S.(f{1}), {'Ib', 'I', 'Iep', 'E0'});
        if isfield(S.(f{1}), 'flat'), S.(f{1}).flat = rmfield(S.(f{1}).flat, 'I'); end
        if isfield(S.(f{1}), 'I_lyot'), S.(f{1}) = rmfield(S.(f{1}), 'I_lyot'); end
    end
end

function J = jacobian_(exdir, tag, r, e, ch, chain_pick, a0, MODEL, say)
%JACOBIAN_  The tagged Jacobian: reused from disk when its truth holds and its stamp and linearization
%   point are this run's, else measured (a stale .mat and its truth are removed first).
    jp = fullfile(exdir, sprintf('ctb_dm_jacobian_N%d_%s.mat', MODEL, tag));  fpp = regexprep(jp, '\.mat$', '.fp.json');
    if isfile(jp)
        J = load(jp);
        try
            assert(isfile(fpp), 'no fingerprint truth beside it');
            [okfp, repfp] = jac_fingerprint('check', J, fpp);  assert(okfp, 'fingerprint: %s', repfp);
            assert(isfield(J, 'elt') && isequal(J.elt, e) && strcmp(J.rx, r.rx_out), 'another deck');
            assert(isfield(J, 'chain_opts') && isequal(J.chain_opts, ch.config), 'another chain stamp (the Lyot blocker among it)');
            if isempty(a0), assert(isempty(J.a0) || all(cellfun(@(x) ~any(x), J.a0)), 'another linearization point');
            else, assert(~isempty(J.a0) && isequal(J.a0, a0), 'another linearization point'); end
            say('[4] Jacobian %s reused: %s (fingerprint holds, chain stamp and linearization point are this run''s; measured %s in %.1f min)', tag, jp, J.date, J.timing_s/60);
            return
        catch ME
            say('[4] cached Jacobian %s is not this run''s (%s): re-measuring; the stale .mat and its truth are removed', jp, ME.message);
            delete(jp);  if isfile(fpp), delete(fpp); end
        end
    end
    args = {'dmrx', r, 'chain', chain_pick, 'tag', tag, 'outdir', exdir, 'model_size', MODEL, 'verbose', true};
    if ~isempty(a0), args = [args, {'a0', a0}]; end
    J = ctb_dm_jacobian(args{:});
end

function q = J_dm_geom_(r, k)
%J_DM_GEOM_  ctb_dm's lattice model for DM k of the DM deck r, at the Jacobian's defaults (beam 21.3 mm).
    u_mm = 1e-3 / r.cbm;
    q = ctb_dm('ielt', r.ielt(k), 'ng', r.ng, 'gdx_mm', r.gdx_mm(k), 'nact', 32, 'beam_d_mm', 21.3 * u_mm, 'coupling', 0.12);
end

function w = detr_after_(j, detr)
%DETR_AFTER_  The per-ray OPL at element j, piston and tilt removed, valid rays only.
    ri = macos.get_ray_info(macos.trace(j).nRays);  ok = ri.ok_pass(:) & ri.ok_trace(:);
    w = detr(ri.opl(ok), ri.pos(:, ok));
end

function n = count_(deck, element)
    n = numel(regexp(fileread(deck), ['(?m)^\s*Element=\s*' element '\s*$'], 'match'));
end

function q = quartets_(deck)
%QUARTETS_  Every NF1 / NF2 / sphere triple in light order (tCtbProp's audit).
    ln = regexp(fileread(deck), '\r?\n', 'split');
    ie = find(~cellfun('isempty', regexp(ln, '^\s*iElt=', 'once')));
    fi = find(~cellfun('isempty', regexp(ln, '^\s*nOutCord=', 'once')), 1);
    if isempty(fi), fi = numel(ln) + 1; end
    E = struct('name', {}, 'prop', {}, 'zstr', {});
    for k = 1:numel(ie)
        if k < numel(ie), b = ie(k+1)-1; else, b = fi-1; end
        blk = strjoin(ln(ie(k):b), newline);
        g = @(key) strtrim(char(regexp(blk, ['^\s*' key '=\s*(.*)$'], 'tokens', 'once', 'lineanchors', 'dotexceptnewline')));
        E(k) = struct('name', g('EltName'), 'prop', g('PropType'), 'zstr', g('zElt'));
    end
    q = struct('name', {}, 'iEP', {}, 'iMask', {}, 'iEP2', {}, 'zEPstr', {}, 'zEP2str', {});
    for k = 1:numel(E)-2
        if ~strcmp(E(k).prop, 'NF1'), continue; end
        assert(strcmp(E(k+1).prop, 'NF2'), 'NF1 at %s is not followed by an NF2 plane', E(k).name);
        q(end+1) = struct('name', E(k).name, 'iEP', k, 'iMask', k+1, 'iEP2', k+2, 'zEPstr', E(k).zstr, 'zEP2str', E(k+2).zstr); %#ok<AGROW>
    end
end
