function out = tg_aoi_ladder(aois)
%TG_AOI_LADDER  Option 3: the PLATE rig at shallower beam-splitter incidence.
%   Dave's suggestion (2026-09-03, from the VSG2 bench layout, whose plate BS
%   works near ~10-15 deg): plate diattenuation -- the mechanism behind v1's
%   7.48 deg arm rotation and 11.7% scale error at 45 deg -- falls with
%   incidence angle, so a shallow plate may buy back most of the cube's error
%   budget while staying a plate.  This ladder MEASURES that: at each BS_AOI
%   it builds the polarizing plate rig (design azimuths, NO alignment step)
%   and records
%     az_t, az_r   arm polarization azimuths at the recombination plane
%     dep          departure from orthogonality (v1@45: +7.48 deg)
%     gain         UNALIGNED four-step PSI scale on a 20 nm grid piston
%                  (v1@45: 1.117), differential against the flat-DM phase
%     power        median fringe intensity in the beam (relative across AOI)
%   Writes tg_aoi_ladder.mat + tg_aoi_ladder.png (rotation and scale error
%   vs AOI).  New file only -- the frozen v1 example/demo are untouched;
%   helpers are copied verbatim from the v1 example (the tg_widen pattern).
%
%   Run:  cd <this dir>;  matlab -batch "tg_aoi_ladder"

if nargin < 1, aois = [45 30 20 15 12 8 5]; end

exdir = fileparts(mfilename('fullpath'));  if isempty(exdir), exdir = pwd; end
if isempty(which('macos.init'))
    run(fullfile(exdir, '..', '..', '..', 'mmacos_setup.m'));
end
assert(~isempty(getenv('MACOS_HOME')), 'MACOS_HOME must be set.');
cd(exdir);

MODEL = 256;  NGRID = 63;  N_G = 256;  DX_G = 0.35;
LAM = 6.328e-4;  QWP = 0.25;  THETAS = [0 45 90 135];
DZ  = 20e-6;                       % 20 nm piston, in mm of surface
TAIL = {'tail_arch','fieldlens', 'FL_F',25.02100857, 'FL_Kc',-2.11278288, ...
        'D_MASK_FL',6.277463741, 'DET_TRIM',1.085330067};
macos.init(MODEL);

macos.write_grid_file('aoi_flat.txt', zeros(N_G));
macos.write_grid_file('aoi_pist.txt', DZ*ones(N_G));
expect = 4*pi*DZ/LAM;              % rad of fringe phase for DZ of surface

n = numel(aois);
out = struct('aoi',aois, 'az_t',nan(1,n), 'az_r',nan(1,n), 'dep',nan(1,n), ...
             'gain',nan(1,n), 'power',nan(1,n));
fprintf('\n=== plate rig vs BS incidence angle (design azimuths, unaligned) ===\n');
fprintf('%6s %10s %10s %10s %10s %10s\n', ...
        'AOI', 'az_test', 'az_ref', 'dep(deg)', 'PSI gain', 'power');
t0 = tic;
for k = 1:n
    a = aois(k);
    mk = @(gf) macos.design.twyman_green('polarizing',true, 'ngridpts',NGRID, ...
        'to_grid_file',gf, 'to_grid_n',N_G, 'to_grid_dx',DX_G, ...
        'qwp_ret',QWP, 'pol_in_deg',45, 'qwp_test_deg',0, 'qwp_ref_deg',45, ...
        'out_qwp_deg',0, 'analyzer_deg',0, 'BS_AOI',a, TAIL{:});
    Gf = mk('aoi_flat.txt');  Gp = mk('aoi_pist.txt');
    ft = sprintf('aoi%02d_test.in',round(a));  fr = sprintf('aoi%02d_ref.in',round(a));
    fp = sprintf('aoi%02d_pist.in',round(a));
    Gf.bt.emit(ft);  Gf.br.emit(fr);  Gp.bt.emit(fp);

    AT = arm_desc(ft, Gf.bt, Gf.T, 0);
    AR = arm_desc(fr, Gf.br, Gf.R, 45);
    AP = arm_desc(fp, Gp.bt, Gp.T, 0);

    out.az_t(k) = arm_azimuth(AT, QWP, 0);
    out.az_r(k) = arm_azimuth(AR, QWP, 45);
    out.dep(k)  = wrap180(out.az_t(k) - out.az_r(k) - 90);

    Sr = analyzer_basis(AR, QWP);
    S0 = analyzer_basis(AT, QWP);
    S1 = analyzer_basis(AP, QWP);
    I0 = frame(S0, Sr, 0);
    msk = I0 > 0.1*max(I0(:));
    d  = angle(exp(1i*(fourstep(S1,Sr,THETAS) - fourstep(S0,Sr,THETAS))));
    out.gain(k)  = median(d(msk))/expect;
    out.power(k) = median(I0(msk));
    fprintf('%6.1f %+10.4f %+10.4f %+10.4f %10.5f %10.4g\n', a, ...
            out.az_t(k), out.az_r(k), out.dep(k), out.gain(k), out.power(k));
end
fprintf('  [%d angles, %.1f s]\n', n, toc(t0));
save('tg_aoi_ladder.mat', 'out');

fig = figure('Visible','off','Position',[80 80 900 380]);
subplot(1,2,1);
plot(out.aoi, abs(out.dep), 'o-', 'LineWidth',1.5);  grid on;
xlabel('BS incidence angle (deg)');  ylabel('arm rotation from orthogonal (deg)');
title('the v1 systematic vs plate angle');
subplot(1,2,2);
semilogy(out.aoi, abs(100*(out.gain-1)), 's-', 'LineWidth',1.5);  grid on;
xlabel('BS incidence angle (deg)');  ylabel('|PSI scale error| (%)');
title('unaligned gauge scale error vs plate angle');
print(fig, 'tg_aoi_ladder.png', '-dpng', '-r140');
fprintf('  wrote tg_aoi_ladder.mat + tg_aoi_ladder.png\n');
end

% ==== helpers, copied verbatim from example_tg_psi_dm.m (v1, frozen) =====
function A = arm_desc(rx, b, ix, base_deg)
    nm = {b.E.name};
    A = struct('rx', rx, 'b', b, ...
        'iPol', find(strcmp(nm,'PolIn'),1), ...
        'iQ',   find(contains(nm,'QWP') & ~strcmp(nm,'OutQWP')), ...
        'base', base_deg, 'qwp_deg', base_deg, 'oq_deg', 0, ...
        'iRC', ix.iRC, 'iOQ', ix.iOutQWP, 'iAn', ix.iAnalyzer, ...
        'iDET', ix.iDET, 'shift', []);
    assert(numel(A.iQ) == 2, 'arm_desc: expected a double-passed arm QWP');
end

function a = lax(psi, deg)
    u1 = macos.design.Bench.perp(psi(:));  u2 = cross(psi(:), u1);
    a = cosd(deg)*u1 + sind(deg)*u2;  a = a(:).';
end

function x = wrap180(x)
    x = mod(x + 90, 180) - 90;
end

function load_arm(A, QWP, an_deg)
    macos.load_rx(A.rx);  b = A.b;
    macos.polarizer(A.iPol, 'axis', lax(b.E(A.iPol).psi, 45));
    qa = lax(b.E(A.iQ(1)).psi, A.qwp_deg);
    for j = 1:2, macos.waveplate(A.iQ(j), 'axis', qa, 'retardance', QWP); end
    macos.waveplate(A.iOQ, 'axis', lax(b.E(A.iOQ).psi, A.oq_deg), 'retardance', QWP);
    macos.polarizer(A.iAn, 'axis', lax(b.E(A.iAn).psi, an_deg));
    macos.polarization('on', 'Ex',[1/sqrt(2) 0], 'Ey',[1/sqrt(2) 0]);
    macos.vector_diffraction(true);
end

function E = arm_field(A, QWP, an_deg)
    load_arm(A, QWP, an_deg);
    E = cat(3, macos.complex_field(A.iDET,'plane',1), ...
               macos.complex_field(A.iDET,'plane',2), ...
               macos.complex_field(A.iDET,'plane',3));
end

function S = analyzer_basis(A, QWP)
    E0  = arm_field(A, QWP, 0);
    E45 = arm_field(A, QWP, 45);
    E90 = arm_field(A, QWP, 90);
    S = struct('A', E0, 'C', E90, 'B', 2*E45 - E0 - E90);
end

function E = synth(S, th)
    c = cosd(th);  s = sind(th);
    E = c^2*S.A + c*s*S.B + s^2*S.C;
end

function e = arm_state(A, QWP, iElt)
    load_arm(A, QWP, 0);
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

function I = frame(Sx, Sr, th)
    I = sum(abs(synth(Sx,th) + synth(Sr,th)).^2, 3);
end

function p = fourstep(Sx, Sr, th)
    I1 = frame(Sx,Sr,th(1));  I2 = frame(Sx,Sr,th(2));
    I3 = frame(Sx,Sr,th(3));  I4 = frame(Sx,Sr,th(4));
    p  = atan2(I2-I4, I1-I3);
end
