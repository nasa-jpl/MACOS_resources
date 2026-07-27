% example_bench_ifo_pol.m
% ========================================================================
%  BENCH IFO POLARIZATION: RAY-LEVEL JONES SCORING (SLICE 1 of 3)
% ========================================================================
%  Polarization-honest Twyman-Green: a real Al coating on the beam-splitter
%  face, ifPol on, scored with the GATED Phase-2 layer (jones_pupil +
%  pol_maps).  The coated BS face is hit as an EXTERNAL reflection in the
%  test arm (air->Al, element 'BSrefl') and as an INTERNAL reflection in
%  the reference arm (glass->Al, two glass transits, element 'BScrefr') --
%  genuinely different Jones -- so the arm-differential diattenuation /
%  retardance at recombination is a MEASURED result, and its pupil
%  variation is the polarization-induced PSI phase-error contribution.
%
%  SCORING RULE (PLAN_POLARIZATION 2c/3, Tranche 1): ray-level Jones ONLY.
%  The BS sits past the first propagation leg, where Tranche 1 caps the
%  diffraction grid, so we NEVER score with vector-diffraction intensities
%  -- everything here comes from macos.ray_field via jones_pupil.
%
%  Gates:
%    1. Coating machinery pinned vs a hand-computed single-surface 45-deg
%       bare-interface Fresnel analytic on the test-arm BS reflection
%       (textbook Born&Wolf r_s/r_p; mirrors tJonesPupil's proven,
%       non-circular Fresnel gate).
%    2. pol-off bit-identity: with polarization OFF the coating is inert,
%       so the coated train's OPD equals the uncoated train's to round-off.
%
%  Run: cd ~/dev/MACOS_resources/mmacos/examples/design/bench_ifo_pol
%       matlab -batch "run('example_bench_ifo_pol.m')"
% ========================================================================

addpath('~/dev/MACOS_resources/mmacos/src');
exdir = fileparts(mfilename('fullpath'));
if isempty(exdir), exdir = pwd; end
assert(~isempty(getenv('MACOS_HOME')), 'MACOS_HOME must be set.');
cd(exdir);

MODEL = 256;
macos.init(MODEL);

% ---- parameters ---------------------------------------------------------
% Al coating at 632.8 nm (HeNe); Bench BaseUnits = mm, so a physical
% optically-thick layer thickness is given in mm (matches tJonesPupil).
nAl  = 1.45;  kAl = 7.54;  thkAl = 2.0e-4;   % mm
Ex_in = 1.0;  Ey_in = 0.0;                   % x-polarized source

coat = @(i) macos.coating(i, 'index', nAl, 'extinc', kAl, 'thickness', thkAl);

% ---- 1. Build the (uncoated) polarization-honest Twyman-Green ----------
fprintf('Building Twyman-Green rig...\n');
G = macos.design.twyman_green('ngridpts', 63);
G.bt.emit('ifo_test_pol.in');
G.br.emit('ifo_ref_pol.in');
% NB: transmitting Refractors now carry Extinc=0 (transparent glass) by
% Bench default -- the earlier Extinc=1e22 Reflector idiom that leaked onto
% them made the glass perfectly ABSORBING under ifPol (invisible on the
% scalar path).  Fixed in Bench.m (add_bs_transmit / add_bs_reflect_return);
% Reflectors keep 1e22.

% coated-face element indices, read from the in-memory bench (the element
% index IS its position in b.E; no engine name-query API is needed)
iBS_test = find(strcmp({G.bt.E.name}, 'BSrefl'),  1);
iBS_ref  = find(strcmp({G.br.E.name}, 'BScrefr'), 1);
assert(~isempty(iBS_test), 'test-arm BS reflect element not found');
assert(~isempty(iBS_ref),  'ref-arm  BS internal-reflect element not found');
iRC_t = G.T.iRC;  iRC_r = G.R.iRC;
iDET_t = G.T.iDET; iDET_r = G.R.iDET;
fprintf('  test arm: coated BSrefl = elt %d, Recomb = elt %d\n', iBS_test, iRC_t);
fprintf('  ref  arm: coated BScrefr = elt %d, Recomb = elt %d\n', iBS_ref,  iRC_r);

% =========================================================================
%  GATE 1 -- coating machinery vs single-surface 45-deg Fresnel analytic
% =========================================================================
%  Trace the test arm with the BS coated, s+p lit, and pull the per-ray
%  reflected s/p amplitudes at the BS.  Compare RS/RP against the textbook
%  bare-interface (optically-thick single-layer) Fresnel coefficients.
%  The analytic is written from Born&Wolf (NOT transcribed from the engine)
%  so the r_p SIGN is pinned non-circularly, exactly as tJonesPupil does.
fprintf('\n=== GATE 1: single-surface 45-deg Fresnel analytic (BS reflect) ===\n');
macos.load_rx('ifo_test_pol.in');
coat(iBS_test);
macos.polarization('on', 'Ex', [1/sqrt(2) 0], 'Ey', [1/sqrt(2) 0]);

% The field arriving at the BS has already passed L1's two glass
% refractions (real index -> a ~1e-3 s/p DIATTENUATION, zero retardance).
% To isolate the BS surface Fresnel we must use the field INCIDENT on the
% BS as the reference input, not the source launch frame -- otherwise L1's
% diattenuation contaminates the magnitude (the retardance is untouched,
% which is exactly the observed 1.3e-3-magnitude / round-off-phase split).
% RayE is overwritten per element, so read the incident field from a
% SEPARATE trace to the element just before the BS (collimated leg; E is
% unchanged in air transit and the common propagation phase cancels in the
% s/p ratio).
macos.trace(iBS_test-1);
ri = macos.ray_field(iBS_test-1);     % field incident on the BS
macos.trace(iBS_test);
rf = macos.ray_field(iBS_test);       % field reflected off the coated BS
m  = (rf.status == 0) & (ri.status == 0);
fprintf('  %d rays good at BS\n', nnz(m));
assert(nnz(m) > 100, 'GATE 1: too few rays at BS');

% geometry from the reflected trace: incident dir = reflect exit thru flat
kox=rf.kx(m); koy=rf.ky(m); koz=rf.kz(m);
nx=rf.nx(m);  ny=rf.ny(m);  nz=rf.nz(m);
kd  = kox.*nx + koy.*ny + koz.*nz;
kix = kox-2*kd.*nx;  kiy = koy-2*kd.*ny;  kiz = koz-2*kd.*nz;
% engine s/p frames: s = ki x n; pi = s x ki (incident); pr = s x ko (refl)
sx=kiy.*nz-kiz.*ny; sy=kiz.*nx-kix.*nz; sz=kix.*ny-kiy.*nx;
sm=sqrt(sx.^2+sy.^2+sz.^2); sx=sx./sm; sy=sy./sm; sz=sz./sm;
pix=sy.*kiz-sz.*kiy; piy=sz.*kix-sx.*kiz; piz=sx.*kiy-sy.*kix;
prx=sy.*koz-sz.*koy; pry=sz.*kox-sx.*koz; prz=sx.*koy-sy.*kox;

% reflected field decomposed in the reflected s/p frame
Es = rf.Ex(m).*sx  + rf.Ey(m).*sy  + rf.Ez(m).*sz;
Ep = rf.Ex(m).*prx + rf.Ey(m).*pry + rf.Ez(m).*prz;
% MEASURED incident field decomposed in the incident s/p frame
qs = ri.Ex(m).*sx  + ri.Ey(m).*sy  + ri.Ez(m).*sz;
qp = ri.Ex(m).*pix + ri.Ey(m).*piy + ri.Ez(m).*piz;
ratio_meas = (Es./Ep).*(qp./qs);      % convention-independent RS/RP

% textbook bare-interface Fresnel (Born&Wolf, ray-following p-hat):
%   r_p = (N2 c_i - N1 c_t)/(N2 c_i + N1 c_t)
%   r_s = (N1 c_i - N2 c_t)/(N1 c_i + N2 c_t)
N1 = 1.0;  N2 = complex(nAl, -kAl);
cthi = abs(kix.*nx + kiy.*ny + kiz.*nz);
ctht = sqrt(1 - (N1/N2)^2*(1 - cthi.^2));
RPa = (N2*cthi - N1*ctht)./(N2*cthi + N1*ctht);
RSa = (N1*cthi - N2*ctht)./(N1*cthi + N2*ctht);

g1_mag = max(abs(abs(ratio_meas) - abs(RSa./RPa)));
g1_ph  = max(abs(angle(ratio_meas./(RSa./RPa))));
aoi_deg = mean(acosd(cthi));
fprintf('  BS mean AOI            : %.3f deg\n', aoi_deg);
fprintf('  RS/RP magnitude resid  : %.3e\n', g1_mag);
fprintf('  RS/RP phase resid (rad): %.3e\n', g1_ph);
assert(g1_mag < 1e-12 && g1_ph < 1e-12, ...
    'GATE 1 FAILED: engine s/p disagree with textbook Fresnel');
fprintf('GATE 1 PASS: coated BS reflection matches Fresnel to < 1e-12\n');

% =========================================================================
%  Arm-differential Jones at recombination  (ray-level, double-pole basis)
% =========================================================================
fprintf('\nBuilding arm Jones pupils at recombination...\n');
% -- test arm --
macos.load_rx('ifo_test_pol.in');       % load_rx clears coating state
coat(iBS_test);
jt = macos.jones_pupil(iRC_t);          % default double-pole basis
% -- ref arm, forced into the SAME exit basis so the differential is honest
macos.load_rx('ifo_ref_pol.in');
coat(iBS_ref);
jr = macos.jones_pupil(iRC_r, 'axis', jt.axis, 'xref', jt.xref);
fprintf('  test leak=%.2e  ref leak=%.2e (longitudinal residual)\n', ...
    jt.leak, jr.leak);

mask = jt.mask & jr.mask;
fprintf('  %d rays good in both arms\n', nnz(mask));
assert(nnz(mask) > 100, 'too few common rays at recombination');

% arm-to-arm transfer  M = J_test * inv(J_ref), per ray (2x2 vectorized)
a = jr.J(:,:,1,1); b = jr.J(:,:,1,2); c = jr.J(:,:,2,1); d = jr.J(:,:,2,2);
det = a.*d - b.*c;
inv11 =  d./det; inv12 = -b./det; inv21 = -c./det; inv22 = a./det;
t11 = jt.J(:,:,1,1); t12 = jt.J(:,:,1,2); t21 = jt.J(:,:,2,1); t22 = jt.J(:,:,2,2);
M = nan(size(jt.J));  M = complex(M, M);
M(:,:,1,1) = t11.*inv11 + t12.*inv21;
M(:,:,1,2) = t11.*inv12 + t12.*inv22;
M(:,:,2,1) = t21.*inv11 + t22.*inv21;
M(:,:,2,2) = t21.*inv12 + t22.*inv22;
for aa=1:2, for bb=1:2
    Mab = M(:,:,aa,bb); Mab(~mask) = NaN+1i*NaN; M(:,:,aa,bb) = Mab;
end, end

% decompose the differential through the SAME gated layer used everywhere
pm = macos.pol_maps(struct('J', M, 'mask', mask));

D   = pm.D(mask);
ret = pm.ret(mask);
% mean = geometric common-mode (a state change, not an aberration);
% variation = the polarization aberration (CLAUDE.md pol_maps convention)
fprintf('\nArm-differential polarization at recombination (%d rays):\n', nnz(mask));
fprintf('  Diattenuation D : mean=%.4e  var(rms)=%.4e  max=%.4e\n', ...
    mean(D), std(D), max(D));
fprintf('  Retardance (rad): mean=%.4e  var(rms)=%.4e  max=%.4e\n', ...
    mean(ret), std(ret), max(ret));

% =========================================================================
%  PSI phase-error contribution  (co-polarized fringe, x-pol input)
% =========================================================================
%  A phase-shifting interferometer measures arg(<E_test, E_ref>) per pupil
%  point.  With balanced arms and no polarization aberration this is flat;
%  the coating's arm-differential Jones tilts/curves it.  The PUPIL
%  VARIATION of the co-pol fringe phase (mean removed = piston) is the
%  polarization-induced PSI phase error.
ein = [Ex_in; Ey_in];  ein = ein/norm(ein);
ft1 = t11*ein(1) + t12*ein(2);  ft2 = t21*ein(1) + t22*ein(2);   % test field
fr1 = a  *ein(1) + b  *ein(2);  fr2 = c  *ein(1) + d  *ein(2);   % ref  field
fringe = conj(ft1).*fr1 + conj(ft2).*fr2;                        % <E_t,E_r>
phi = angle(fringe(mask));
phi = phi - angle(mean(exp(1i*phi)));      % remove piston (circular mean)
phi = mod(phi + pi, 2*pi) - pi;            % rewrap about 0
lam_nm = 632.8;                            % HeNe
psi_rms_rad = std(phi);
psi_rms_wav = psi_rms_rad/(2*pi);
psi_rms_nm  = psi_rms_wav*lam_nm;
fprintf('\nPSI phase-error contribution (co-pol fringe, x-pol input):\n');
fprintf('  RMS phase : %.4e rad = %.4e waves = %.4e nm @ %.1f nm\n', ...
    psi_rms_rad, psi_rms_wav, psi_rms_nm, lam_nm);

% =========================================================================
%  GATE 2 -- pol-off bit-identity (coating inert without polarization)
% =========================================================================
fprintf('\n=== GATE 2: pol-off bit-identity (coated vs uncoated) ===\n');
g2 = zeros(1,2);  arms = {{'ifo_test_pol.in', iBS_test, iDET_t}, ...
                          {'ifo_ref_pol.in',  iBS_ref,  iDET_r}};
for k = 1:2
    rx = arms{k}{1};  iBS = arms{k}{2};  iDET = arms{k}{3};
    macos.load_rx(rx);
    macos.polarization('off');
    macos.trace(iDET);  w_uncoated = macos.opd();
    coat(iBS);                              % coat, still pol OFF
    macos.trace(iDET);  w_coated = macos.opd();
    both = isfinite(w_uncoated) & isfinite(w_coated);
    g2(k) = max(abs(w_uncoated(both) - w_coated(both)));
end
fprintf('  test-arm OPD max-diff: %.3e mm\n', g2(1));
fprintf('  ref-arm  OPD max-diff: %.3e mm\n', g2(2));
assert(all(g2 < 1e-12), 'GATE 2 FAILED: coating perturbs OPD with pol OFF');
fprintf('GATE 2 PASS: coating is inert when polarization is off (<1e-12 mm)\n');

% ---- persist results ----------------------------------------------------
results = struct('nAl',nAl,'kAl',kAl,'thkAl',thkAl, ...
    'iBS_test',iBS_test,'iBS_ref',iBS_ref, ...
    'gate1_mag',g1_mag,'gate1_phase',g1_ph,'bs_aoi_deg',aoi_deg, ...
    'D_mean',mean(D),'D_var',std(D),'D_max',max(D), ...
    'ret_mean',mean(ret),'ret_var',std(ret),'ret_max',max(ret), ...
    'psi_rms_rad',psi_rms_rad,'psi_rms_nm',psi_rms_nm, ...
    'gate2_test',g2(1),'gate2_ref',g2(2));
save('bench_ifo_pol_results.mat', 'results', 'pm', 'jt', 'jr', 'M');

fprintf('\n=== ALL GATES PASS ===\n');
fprintf('Slice 1 complete. Deferred: slice 2 (BS AOI vs clearance trade),\n');
fprintf('slice 3 (polarizing-PSI variant + comparison).\n');
