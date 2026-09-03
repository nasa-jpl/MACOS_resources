function out = tg_widen(AT, QWP, Sr, p_null, THETAS, LAM, msk, map, N_G, DX_G, NACT, PITCH, Mdm)
%TG_WIDEN  Re-run the beat-8 calibration on the DM's OWN actuators.
%   Dave's basis: the actuator lattice, not Zernikes.  The gauge error is a
%   spatial-frequency effect (the detector images the DM, which smooths), so
%   the modes that diagonalize it are the actuator lattice's own spatial
%   frequencies -- separable cosines on the 16x16 command grid.  The beat-7
%   checkerboard is exactly the Nyquist member of that set, so it lands IN
%   the span by construction; a RANDOM command set is the honest held-out.
na  = NACT;  AMPa = 50e-6;
axs = ((1:N_G)-(N_G+1)/2)*DX_G;
ii  = (1:na)';
PQ  = [1 1; 2 2; 4 4; 6 6; 8 8; 10 10; 12 12; 14 14; 16 16; 16 0; 0 16; 8 0];
nM  = size(PQ,1);
Tt = zeros(nnz(msk), nM);  Mm = zeros(nnz(msk), nM);  frq = zeros(1,nM);
t0 = tic;
for k = 1:nM
    p = PQ(k,1);  q = PQ(k,2);
    Ak = cos(pi*p*ii/na) * cos(pi*q*ii/na)';
    Ak = Ak / max(abs(Ak(:)));
    Mk = dm_influence_map(N_G, DX_G, 'nact',na, 'pitch',PITCH, 'act', AMPa*Ak);
    hk = meas_surface(AT, QWP, Mk, Sr, p_null, THETAS, LAM);
    tk = interpn(axs, axs, Mk, map.Xt, map.Yt, 'spline', 0);
    Mm(:,k) = hk(msk) - mean(hk(msk));
    Tt(:,k) = tk(msk) - mean(tk(msk));
    frq(k)  = hypot(p,q)/2;                 % cycles across the pupil
end
G = Tt \ Mm;
fprintf('  actuator-lattice basis: %d cosine modes on the %dx%d command grid\n', nM, na, na);
fprintf('  %d modes x 3 traces in %.1f s\n', nM, toc(t0));
fprintf('  %-9s %8s %8s %11s\n', 'mode(p,q)', 'cyc/pup', 'gain', 'cross-talk');
for k = 1:nM
    x = G(:,k);  x(k) = 0;
    fprintf('  %3d,%-5d %8.1f %8.4f %11.4f\n', PQ(k,1), PQ(k,2), frq(k), G(k,k), norm(x));
end
%  held-out validation: the checkerboard (in span) and a RANDOM command set
Mrnd = dm_influence_map(N_G, DX_G, 'nact',na, 'pitch',PITCH, 'poke',50e-6, ...
                        'pattern','random', 'seed',11);
val = {'beat-7 checkerboard', Mdm; 'random command (HELD OUT)', Mrnd};
out = struct('PQ',PQ, 'frq',frq, 'gain',diag(G).', 'G',G);
for v = 1:2
    hv = meas_surface(AT, QWP, val{v,2}, Sr, p_null, THETAS, LAM);
    tv = interpn(axs, axs, val{v,2}, map.Xt, map.Yt, 'spline', 0);
    hv = hv(msk) - mean(hv(msk));   tv = tv(msk) - mean(tv(msk));
    cc = G \ (Tt \ hv);
    hc = hv - Tt*(Tt\hv) + Tt*cc;
    b  = 1e6*std(hv - tv);   a = 1e6*std(hc - tv);
    rc = corrcoef(hc - tv, tv);
    fprintf('  %-26s %.4f -> %.4f nm rms  (%.2f%% of a %.2f nm input)  resid/truth corr %+.2f\n', ...
            val{v,1}, b, a, 100*a/(1e6*std(tv)), 1e6*std(tv), rc(1,2));
    out.val(v) = struct('name',val{v,1}, 'before',b, 'after',a, 'corr',rc(1,2));
end
end
function beat(n, ttl, interactive)
%BEAT  Banner, optional pause, and SELF-TIMING.  The rehearsal question for a
%   live demo is "does any beat lose the room", so measure it rather than
%   estimate: the timer stops on the NEXT banner, and the total prints at the
%   end (call beat(0,...) to close the last one).
%
%   If TG_DEMO_MARKER names a directory, each finished beat also drops an
%   empty beat<N>.done file there.  A driver outside MATLAB needs a
%   completion signal it can trust, and stdout to a redirected file is
%   block-buffered -- the filesystem is not.  The marker is written BEFORE
%   the next beat's input() blocks, so it means "beat N is finished and the
%   script is waiting", which is exactly what the driver has to know.
    persistent t0 last
    if ~isempty(t0) && ~isempty(last)
        fprintf('\n   [beat %d took %.2f s]\n', last, toc(t0));
        md = getenv('TG_DEMO_MARKER');
        if ~isempty(md)
            fid = fopen(fullfile(md, sprintf('beat%d.done', last)), 'w');
            if fid > 0, fclose(fid); end
        end
    end
    if n == 0, t0 = [];  last = [];  return; end
    fprintf('\n');
    fprintf('#########################################################\n');
    fprintf('#  BEAT %d -- %s\n', n, ttl);
    fprintf('#########################################################\n');
    if interactive && n > 1
        input('   [enter to run this beat] ', 's');
    end
    t0 = tic;  last = n;
end

function print_train(ttl, b)
    fprintf('\n  %s\n', ttl);
    for k = 1:numel(b.E)
        e = b.E(k);
        mark = '';
        if ismember(e.element, {'TrPolarizer','WavePlate'}), mark = '   <-- polarization'; end
        if ~isempty(e.gridfile), mark = '   <-- the DM'; end
        fprintf('   %2d  %-11s %-9s %-10s s=%8.3f mm%s\n', ...
            k, e.name, e.element, e.surface, e.s, mark);
    end
end

function A = arm_desc(rx, b, ix, base_deg)
    nm = {b.E.name};
    A = struct('rx', rx, 'b', b, 'iPol', find(strcmp(nm,'PolIn'),1), ...
        'iQ', find(contains(nm,'QWP') & ~strcmp(nm,'OutQWP')), ...
        'base', base_deg, 'qwp_deg', base_deg, 'oq_deg', 0, 'iTO', [], ...
        'iRC', ix.iRC, 'iOQ', ix.iOutQWP, 'iAn', ix.iAnalyzer, 'iDET', ix.iDET);
    if isfield(ix,'iTO'), A.iTO = ix.iTO; end
end

function A = set_pol_align(A, qwp_deg, oq_deg)
    A.qwp_deg = qwp_deg;  A.oq_deg = oq_deg;
end

function a = lax(psi, deg)
    u1 = macos.design.Bench.perp(psi(:));  u2 = cross(psi(:), u1);
    a = cosd(deg)*u1 + sind(deg)*u2;  a = a(:).';
end

function load_arm(A, QWP, an_deg, grid)
%LOAD_ARM  Load the deck, optionally rewrite the DM grid IN THE LOADED MODEL
%   (the live poke), then set every polarizing element.
    macos.load_rx(A.rx);  b = A.b;
    if nargin >= 4 && ~isempty(grid)
        macos.set_elt_grid(A.iTO, macos.get_elt_grid_spacing(A.iTO), grid);
    end
    macos.polarizer(A.iPol, 'axis', lax(b.E(A.iPol).psi, 45));
    qa = lax(b.E(A.iQ(1)).psi, A.qwp_deg);
    for j = 1:2, macos.waveplate(A.iQ(j), 'axis', qa, 'retardance', QWP); end
    macos.waveplate(A.iOQ, 'axis', lax(b.E(A.iOQ).psi, A.oq_deg), 'retardance', QWP);
    macos.polarizer(A.iAn, 'axis', lax(b.E(A.iAn).psi, an_deg));
    macos.polarization('on', 'Ex',[1/sqrt(2) 0], 'Ey',[1/sqrt(2) 0]);
    macos.vector_diffraction(true);
end

function E = arm_field(A, QWP, an_deg, grid)
    load_arm(A, QWP, an_deg, grid);
    E = cat(3, macos.complex_field(A.iDET,'plane',1), ...
               macos.complex_field(A.iDET,'plane',2), ...
               macos.complex_field(A.iDET,'plane',3));
end

function S = analyzer_basis(A, QWP, grid)
    E0  = arm_field(A, QWP,  0, grid);
    E45 = arm_field(A, QWP, 45, grid);
    E90 = arm_field(A, QWP, 90, grid);
    S = struct('A', E0, 'C', E90, 'B', 2*E45 - E0 - E90);
end

function E = synth(S, th)
    c = cosd(th);  s = sind(th);
    E = c^2*S.A + c*s*S.B + s^2*S.C;
end

function R = probe_rt(rx_t, iT, rx_r, iR)
%PROBE_RT  The engine's R and T of the cemented diagonal, per polarization.
%   The amplitude ratio ACROSS the diagonal -- the field just after the
%   entrance face vs just after the diagonal.  The glass between is lossless,
%   so the propagation phase has unit modulus and drops out of |ratio|.  The
%   source frame for the +x chief is xGrid = yhat (the diagonal's p axis) and
%   yGrid = zhat (its s axis); PolIn is set to the probe axis so it passes
%   the state instead of projecting it.
    P = {rx_t, iT, 'T'; rx_r, iR, 'R'};  Q = struct();
    for k = 1:2
        for pol = {'s','p'}
            macos.load_rx(P{k,1});
            if strcmp(pol{1},'s'), ax = [0 0 1];  Ex = [0 0];  Ey = [1 0];
            else,                  ax = [0 1 0];  Ex = [1 0];  Ey = [0 0]; end
            ip = P{k,2};
            macos.polarizer(ip(1)-1, 'axis', ax);
            macos.polarization('on','Ex',Ex,'Ey',Ey);
            macos.vector_diffraction(true);
            macos.trace(ip(1));  f1 = macos.ray_field(ip(1));
            macos.trace(ip(2));  f2 = macos.ray_field(ip(2));
            ok = (f1.status == 0) & (f2.status == 0);
            a1 = sqrt(abs(f1.Ex).^2 + abs(f1.Ey).^2 + abs(f1.Ez).^2);
            a2 = sqrt(abs(f2.Ex).^2 + abs(f2.Ey).^2 + abs(f2.Ez).^2);
            Q.([P{k,3} pol{1}]) = median(a2(ok)./a1(ok))^2;
        end
    end
    R = struct('Ts',Q.Ts, 'Tp',Q.Tp, 'Rs',Q.Rs, 'Rp',Q.Rp);
end

function h = meas_surface(A, QWP, M, Sr, p_null, THETAS, LAM)
%MEAS_SURFACE  Put shape M on the DM, run the four-step measurement against
%   the reference arm, and return the recovered SURFACE in mm.  Differential
%   against the flat-DM phase, so every static term cancels.  Three traces.
    d = angle(exp(1i*(fourstep(analyzer_basis(A, QWP, M), Sr, THETAS) - p_null)));
    h = d * LAM/(4*pi);
end

function I = frame(Sx, Sr, th)
    I = sum(abs(synth(Sx,th) + synth(Sr,th)).^2, 3);
end

function p = fourstep(Sx, Sr, th)
    I1 = frame(Sx,Sr,th(1));  I2 = frame(Sx,Sr,th(2));
    I3 = frame(Sx,Sr,th(3));  I4 = frame(Sx,Sr,th(4));
    p  = atan2(I2-I4, I1-I3);
end

function e = arm_state(A, QWP, iElt)
    load_arm(A, QWP, 0, []);
    macos.trace(iElt);  f = macos.ray_field(iElt);
    ok = f.status == 0;
    psi = A.b.E(iElt).psi(:);
    u1 = macos.design.Bench.perp(psi);  u2 = cross(psi, u1);
    e1 = f.Ex*u1(1) + f.Ey*u1(2) + f.Ez*u1(3);
    e2 = f.Ex*u2(1) + f.Ey*u2(2) + f.Ez*u2(3);
    r  = e2(ok)./e1(ok);  a = median(abs(e1(ok)));
    e  = [a; a*(median(real(r)) + 1i*median(imag(r)))];
end

function az = arm_azimuth(A, QWP, qwp_deg)
    A.qwp_deg = qwp_deg;
    e = arm_state(A, QWP, A.iRC);
    az = 0.5*atan2d(2*real(conj(e(1))*e(2)), abs(e(1))^2 - abs(e(2))^2);
end

function [best, map] = register_to_dm(A, ix, Mdm, N_G, DX_G, h, msk)
%REGISTER_TO_DM  The instrument's pupil mapping, measured from the trace: one
%   (DM position, detector position) pair per surviving ray.
    macos.load_rx(A.rx);
    s1 = macos.trace(ix.iTO);   ito  = macos.get_ray_info(s1.nRays);
    s2 = macos.trace(ix.iDET);  idet = macos.get_ray_info(s2.nRays);
    okr = ito.ok_trace(:) & ito.ok_pass(:) & idet.ok_trace(:) & idet.ok_pass(:);
    psi1 = macos.get_elt_psi(ix.iTO);  vpt1 = macos.get_elt_vpt(ix.iTO);
    u1 = macos.design.Bench.perp(psi1);  v1 = cross(psi1, u1);
    xy_to = [u1.'; v1.'] * (ito.pos - vpt1);
    psi2 = macos.get_elt_psi(ix.iDET);
    u2 = macos.design.Bench.perp(psi2);  v2 = cross(psi2, u2);
    xy_d = [u2.'; v2.'] * (idet.pos - idet.pos(:,1));
    xy_to = xy_to(:,okr);  xy_d = xy_d(:,okr);
    Aaf = [xy_d.' ones(nnz(okr),1)] \ xy_to.';
    Lm  = Aaf(1:2,:).';
    [~,Ss,~] = svd(Lm);  sm = diag(Ss);
    nl  = xy_to - (Lm*xy_d + Aaf(3,:).');
    map = struct('mag', sqrt(abs(det(Lm))), 'anam_pct', 100*(sm(1)/sm(2)-1), ...
                 'nonlin_mm', sqrt(mean(sum(nl.^2,1))));
    Fx = scatteredInterpolant(xy_d(1,:).', xy_d(2,:).', xy_to(1,:).', 'linear','linear');
    Fy = scatteredInterpolant(xy_d(1,:).', xy_d(2,:).', xy_to(2,:).', 'linear','linear');
    N = size(h,1);  [cg, rg] = meshgrid(1:N, 1:N);
    cx = sum(cg(msk))/nnz(msk);  cy = sum(rg(msk))/nnz(msk);
    dxp = macos.dx_at(ix.iDET, 'mm');
    a1 = (cg-cx)*dxp;  a2 = (rg-cy)*dxp;
    c_d = mean(xy_d, 2);
    axs = ((1:N_G)-(N_G+1)/2)*DX_G;
    hm = h - mean(h(msk));
    cands = {a1,a2; a1,-a2; -a1,a2; -a1,-a2; a2,a1; a2,-a1; -a2,a1; -a2,-a1};
    best = struct('c',-inf, 'i',1);
    for c = 1:size(cands,1)
        [cc, ht] = reg_corr([0 0 0 0], cands{c,1}, cands{c,2}, c_d, Fx, Fy, axs, Mdm, hm, msk);
        if cc > best.c, best = struct('c',cc, 'ht',ht, 'i',c); end
    end
    A1 = cands{best.i,1};  A2 = cands{best.i,2};
    p = fminsearch(@(q) -reg_corr(q,A1,A2,c_d,Fx,Fy,axs,Mdm,hm,msk), [0 0 0 0], ...
                   optimset('TolX',1e-7,'TolFun',1e-10,'Display','off'));
    [c2, ht2, Xt, Yt] = reg_corr(p, A1, A2, c_d, Fx, Fy, axs, Mdm, hm, msk);
    if c2 > best.c, best.c = c2;  best.ht = ht2; end
    best.hm = hm;
    map.Xt = Xt;  map.Yt = Yt;     % the reusable resampler (see reg_corr)
end

function [c, ht, Xt, Yt] = reg_corr(p, A1, A2, c_d, Fx, Fy, axs, Mdm, hm, msk)
%   XT/YT are each detector pixel's position ON THE DM, in mm.  Returned so
%   the solved registration can be REUSED to resample any other shape (beat
%   8 injects a dozen of them) instead of re-solving it every time.
    s = exp(p(4));  ct = cos(p(3));  st = sin(p(3));
    X = s*(ct*A1 - st*A2) + c_d(1) + p(1);
    Y = s*(st*A1 + ct*A2) + c_d(2) + p(2);
    Xt = Fx(X,Y);  Yt = Fy(X,Y);
    ht = interpn(axs, axs, Mdm, Xt, Yt, 'spline', 0);
    ht = ht - mean(ht(msk));
    cc = corrcoef(hm(msk), ht(msk));  c = cc(1,2);
end

function show(Z, msk, N, box, cl, ttl)
    q = nan(N);  q(msk) = Z(msk);  q = sub(q, box);
    nexttile; imagesc(q, 'AlphaData', ~isnan(q)); axis image off;
    if ~isempty(cl), clim(cl); end
    colorbar; title(ttl);
end

function b = beam_box(msk, pad)
%BEAM_BOX  Padded bounding box of the illuminated pixels -- the beam is a
%   small disc on the padded diffraction array.
    [rr, cc] = find(msk);  N = size(msk,1);
    b = [max(1,min(rr)-pad) min(N,max(rr)+pad) max(1,min(cc)-pad) min(N,max(cc)+pad)];
end

function Z = sub(Z, b), Z = Z(b(1):b(2), b(3):b(4)); end

function m = erode_disc(msk, frac)
%ERODE_DISC  Keep the inner FRAC of the illuminated disc (no toolbox needed).
    N = size(msk,1);  [cg, rg] = meshgrid(1:N, 1:N);
    cx = mean(cg(msk));  cy = mean(rg(msk));
    r  = sqrt((cg-cx).^2 + (rg-cy).^2);
    m  = msk & (r <= frac*max(r(msk)));
end
