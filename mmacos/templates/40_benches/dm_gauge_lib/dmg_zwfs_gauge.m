function ZW = dmg_zwfs_gauge(iTO, iMASK, iDET, opt)
%DMG_ZWFS_GAUGE  Zernike-sensor measurement factory (dimple at FocalMask).
%   ZW = dmg_zwfs_gauge(iTO, iMASK, iDET, opt) builds the flat-DM
%   reference frames and masks on the LOADED bench and returns:
%     ZW.measL(M)      frozen-reference LINEAR height map (mm) -- one
%                      masked frame; small-differential workhorse
%     ZW.measI(M)      ITERATED-reference EXACT height map (mm) -- the
%                      SAME one masked frame: per-pixel exact solve
%                      (Ruane 2020 eq 36-37 / N'Diaye 2013 eq 6-7) with the
%                      reference wave b re-propagated from the estimate
%                      through the FFT surrogate of the mask model, NITER
%                      times (Doelman 2019 / Chambouleyron 2024 / Haffert
%                      2024).  A = |E0| (the flat's amplitude; exact on a
%                      DM-conjugate pupil, where a phase-only DM state
%                      leaves |E| unchanged -- gated by ZW.gate.roundtrip).
%                      Principal branch (phi - Theta) in [-pi, 0] = the
%                      quarter-wave sensor's -pi/4 .. 3pi/4; the other
%                      branch per pixel via a prior (ZW.reconI(Ia,[],plus),
%                      plus from ZW.priorS(Ia, Fr): the stepped frames of
%                      the same state, refined with the iterated |b|^2).
%     ZW.measV(M)      VECTOR (polarized-dimple) exact height map (mm):
%                      TWO simultaneous pupil images, one through a +PHI_M
%                      dimple and one through -PHI_M (a geometric-phase
%                      metasurface splits the two circular polarizations;
%                      Doelman 2019).  Per pixel the pair gives cos and sin
%                      of (phi - beta) at once, so the solve is exact with
%                      NO branch choice (no quarter-wave fold); the
%                      reference wave b is iterated as for measI.  Ideal
%                      metasurface: each channel is the scalar sensor with
%                      its own dimple sign (dmg_zwfs_gauge:V, 2026-09-11).
%                      ZW.frameV(M) -> [Ip, Im]; ZW.reconV(Ip, Im, I0, b0,
%                      niter); ZW.solveV -> [phi, info] (info.rcons = rms
%                      of the amplitude-consistency |cos^2+sin^2 - 1|).
%     ZW.steppedX(M)   rank-2 phase-stepped complex retrieval X (S2b:
%                      |c|^2 = -2 Re(c) identically, so depth steps give
%                      TWO observables/px; |Eb|^2 from the one-time
%                      flat disk-frame calibration b2cal)
%     ZW.stepdiff(X1,X0)  differential height (mm) = S_CONV *
%                      angle(X1 conj X0) * LAM/(4 pi) -- range +-pi
%     fields: msk, den, I_flat, b2cal, N_WF, ctr, dia_mm, E0, Eb0, D,
%             cc, bsur (the surrogate b operator), gate (roundtrip,
%             bsur = surrogate-vs-engine Eb on msk)
%   opt fields: LAM, F2 (mask-leg focal, mm), R_BEAM (mm), DIA_LAMD,
%   PHI_M, PHIS (3 depths), S_CONV, NITER (default 5); V2: V_RET_ERR,
%   V_LEAK_PHASE, V_CAL ('ideal' | 'fit' | 'map'); V3: V_ARM ('none' |
%   'engine' | 'synthetic' | a struct with qL, qR), V_LASER_DEG,
%   V_ARM_DPHASE, V_ARM_DAMP.  Requires zwfs_mask on the path (run from
%   zwfs_dm96/).  L/S verbatim from zwfs_s3 @ 10cf593; the iterated
%   reading added 2026-09-09 (zwfs_s7iter).
%
% V3 (2026-09-12): the ARM's polarization aberration per channel.  The
% metasurface converts L -> R with the +phi dimple and R -> L with -phi, so
% the two images are of DIFFERENT pupil fields, qL.*E and qR.*E, where
% qL, qR are the laser state's circular components through the arm's Jones
% pupil (dmg_arm_maps: the engine's two polarized vector traces, common
% scalar stripped, normalized to the ideal 50/50 split; or synthetic
% astigmatic maps of a given rms: V_ARM_DPHASE rad of differential PHASE
% between the channels -- the diattenuation-type term -- and V_ARM_DAMP of
% differential AMPLITUDE -- the retardance-type term).  Frames: each
% channel's pupil map is applied at the mask sandwich's entrance sphere
% (identical to the detector field, gate G1) and the dimple follows -- the
% engine's chained apodization, gated against the surrogate.  With
% retardance error the leaked (unconverted) light in one output channel
% comes from the OTHER input channel: I+ = |sqrt(eta)(qL E)_masked +
% sqrt(1-eta) e^{i alpha} qR E|^2, and vice versa.  The solver carries
% per-channel amplitude maps (V_CAL 'amp': |qL|, |qR| from the per-channel
% UNMASKED reference frames every bench takes; the polarization phases
% unknown), plus constants kappa+, kappa-, eta (V_CAL 'fit': fitted on the
% flat's two masked images, 5 real parameters), or the true maps (V_CAL
% 'map': a polarimetrically calibrated bench); 'ideal' knows nothing of
% the arm (kappa 1: the raw size of the term).
LAM = opt.LAM;  PHIS = opt.PHIS;  S_CONV = opt.S_CONV;
if isfield(opt, 'NITER'), NITER = opt.NITER; else, NITER = 5; end
% V2 (2026-09-12): a REAL geometric-phase metasurface has retardance pi +
% V_RET_ERR; it converts eta = cos^2(err/2) of the light (with the +-phi
% geometric phase) and leaks the rest unshifted.  With a linearly
% polarized laser the leaked and converted light in one output channel
% are COHERENT, so each channel's field is sqrt(eta)*E_masked +
% sqrt(1-eta)*exp(i*V_LEAK_PHASE)*E_unmasked -- one complex constant
% kappa = sqrt(eta) + sqrt(1-eta) e^{i alpha} on E0 in the per-pixel model.
% V_CAL 'ideal' solves with the ideal model (kappa 1, eta 1: the bias of
% an uncalibrated metasurface); 'fit' fits (|kappa|, arg kappa, eta) on
% the flat DM's two images (ZW.calV) and solves with them.
vre = 0;  vla = 0;  vcal = 'ideal';
if isfield(opt, 'V_RET_ERR'),    vre = opt.V_RET_ERR; end
if isfield(opt, 'V_LEAK_PHASE'), vla = opt.V_LEAK_PHASE; end
if isfield(opt, 'V_CAL'),        vcal = opt.V_CAL; end
eta_true = cos(vre/2)^2;  leak = struct('eta', eta_true, 'alpha', vla);
varm = 'none';  vlaser = 45;  vdph = 0;  vdam = 0;
if isfield(opt, 'V_ARM'),        varm = opt.V_ARM; end
if isfield(opt, 'V_LASER_DEG'),  vlaser = opt.V_LASER_DEG; end
if isfield(opt, 'V_ARM_DPHASE'), vdph = opt.V_ARM_DPHASE; end
if isfield(opt, 'V_ARM_DAMP'),   vdam = opt.V_ARM_DAMP; end
E0f = macos.complex_field(iDET);  N_WF = size(E0f,1);
dx_mask_m = abs(macos.dx_at(iMASK));
lamD_mm = LAM*opt.F2/(2*opt.R_BEAM);  dia_mm = opt.DIA_LAMD*lamD_mm;
If = abs(macos.complex_field(iMASK)).^2;
assert(max(If(:))/sum(If(:)) >= 1e-2, 'dmg_zwfs_gauge: mask plane not focused');
[~, ipk] = max(If(:));  [pr, pc] = ind2sub(size(If), ipk);
w = 12;  rows = max(1,pr-w):min(N_WF,pr+w);  cols = max(1,pc-w):min(N_WF,pc+w);
Iw = If(rows, cols);
ctr = [sum(sum(Iw,1).*cols)/sum(Iw(:)), sum(sum(Iw,2).'.*rows)/sum(Iw(:))];
[V, D] = zwfs_mask(N_WF, dx_mask_m*1e3, dia_mm, opt.PHI_M, ctr);
cc = exp(1i*opt.PHI_M) - 1;
Vm = zwfs_mask(N_WF, dx_mask_m*1e3, dia_mm, -opt.PHI_M, ctr);     % the -phi channel (vector reading)
ccm = exp(-1i*opt.PHI_M) - 1;
macos.intensity(iMASK);  macos.apodize_complex(iMASK, D);
Ebf = macos.complex_field(iDET, 'reset_trace', false);
b2cal = abs(Ebf).^2;
macos.intensity(iMASK);  macos.apodize_complex(iMASK, V);
I_flat = abs(macos.complex_field(iDET, 'reset_trace', false)).^2;
Kmap = cc*Ebf.*conj(E0f);  den = 2*imag(Kmap);
I0 = abs(E0f).^2;  supp = I0 > 0.1*max(I0(:));
msk = supp & (abs(den) > 0.05*max(abs(den(:))));
VK = cell(1,3);
for k = 1:3
    VK{k} = zwfs_mask(N_WF, dx_mask_m*1e3, dia_mm, PHIS(k), ctr);
end
M2 = [(2-2*cos(PHIS)).', 2*sin(PHIS).'];
M2i = pinv(M2);

% ---- the mask-model surrogate (iterated reading) ----------------------
% The engine's NF2 leg (PL2SPH) is fftshift(fft2(fftshift(.)))/N and its
% geometric tail is the identity on the grid (measured: |E_det - E_out|
% = 0, zwfs_s7 probe), so the field through the dimple-support disk at
% the detector is  b(E) = T( D .* Ti(E) )  for ANY detector field E.
% Gated below against the engine's own Eb on the flat (1e-15 class).
T  = @(x) fftshift(fft2(fftshift(x)))/N_WF;
Ti = @(x) fftshift(ifft2(fftshift(x)))*N_WF;
bsur = @(E) T(D .* Ti(E));
nrm = @(x) norm(x(:));
gate = struct();
gate.bsur = nrm((bsur(E0f) - Ebf).*msk) / nrm(Ebf.*msk);
% Sandwich round trip: the element before the mask (NF1 entrance sphere)
% to the element after it (NF2 exit sphere) must be the identity on the
% unmasked field, else the tail sees a Fresnel-DEFOCUSED pupil (the
% asymmetric 'nf_legacy' emission: 0.159 on zwfs_dm96) and A = |E0| no
% longer holds under a DM state.  Warn once; the stage asserts.
if iMASK > 1
    E_in  = macos.complex_field(iMASK-1);
    E_out = macos.complex_field(iMASK+1);
    gate.roundtrip = nrm(E_out - E_in) / nrm(E_in);
    if gate.roundtrip > 1e-9
        warning('dmg_zwfs_gauge:roundtrip', ['mask sandwich round trip is not the ' ...
            'identity (rel %.2e): the detector pupil is NOT DM-conjugate ' ...
            '(mask_prop nf_legacy?) -- the iterated reading assumes it is.'], gate.roundtrip);
    end
else
    gate.roundtrip = NaN;
end
% ---- V3: the arm's per-channel pupil maps -----------------------------
arm = struct('mode', 'none', 'qL', [], 'qR', [], 'info', struct());
if isstruct(varm)
    arm.mode = 'given';  arm.qL = varm.qL;  arm.qR = varm.qR;
elseif strcmp(varm, 'engine')
    arm.mode = 'engine';
    [arm.qL, arm.qR, arm.info] = dmg_arm_maps(iMASK-1, msk, vlaser);   % at the sandwich's entrance sphere: every arm optic, not the mask or the field lens
elseif strcmp(varm, 'synthetic')
    arm.mode = 'synthetic';
    [cy, cx] = find(msk);  c0 = [mean(cx) mean(cy)];
    [XX, YY] = meshgrid(1:N_WF, 1:N_WF);
    rr = hypot(XX - c0(1), YY - c0(2));  rr = rr / prctile(rr(msk), 99);
    Z = rr.^2 .* cos(2*atan2(YY - c0(2), XX - c0(1)));  Z = Z / std(Z(msk));   % astigmatic, unit rms on msk
    dph = vdph*Z;  dam = vdam*Z;
    arm.qL = (1 - dam/2) .* exp(-1i*dph/2);  arm.qR = (1 + dam/2) .* exp(+1i*dph/2);
    arm.qL(~msk) = 1;  arm.qR(~msk) = 1;
    arm.info = struct('dphase_rms', vdph, 'damp_rms', vdam);
elseif ~strcmp(varm, 'none')
    error('dmg_zwfs_gauge: V_ARM must be ''none'', ''engine'', ''synthetic'' or a struct with qL, qR');
end
if ~strcmp(arm.mode, 'none')
    % the chained apodization (pupil map at iMASK-1, dimple at iMASK) must
    % reproduce the plain frame with a unit map and the surrogate with the
    % channel map: the frames of the vector reading go through it
    I1 = chain_(zeros(macos.get_elt_grid_size(iTO)), ones(N_WF), V, iTO, iMASK, iDET);
    gate.chain = nrm(abs(I1).^2 - I_flat) / nrm(I_flat);
    Eq = arm.qL .* E0f;
    IL = chain_(zeros(macos.get_elt_grid_size(iTO)), arm.qL, V, iTO, iMASK, iDET);
    gate.chain_sur = nrm((abs(IL).^2 - abs(Eq + cc*bsur(Eq)).^2).*msk) / nrm(abs(Eq + cc*bsur(Eq)).^2.*msk);
end
C = struct('E0',E0f, 'Eb0',Ebf, 'cc',cc, 'ccm',ccm, 'msk',msk, 'N_WF',N_WF, ...
           'NITER',NITER, 'bsur',bsur, 'S_CONV',S_CONV, 'LAM',LAM, ...
           'kapP', 1, 'kapM', 1, 'eta', 1, 'qL', [], 'qR', [], ...     % the solver's model: ideal metasurface, ideal arm
           'EbP0', Ebf, 'EbM0', Ebf);                                   % per-channel flat reference waves

ZW = struct();
ZW.msk = msk;  ZW.den = den;  ZW.I_flat = I_flat;  ZW.b2cal = b2cal;
ZW.N_WF = N_WF;  ZW.ctr = ctr;  ZW.dia_mm = dia_mm;
ZW.E0 = E0f;  ZW.Eb0 = Ebf;  ZW.D = D;  ZW.cc = cc;  ZW.bsur = bsur;
ZW.gate = gate;  ZW.NITER = NITER;
ZW.measL    = @(M) measL_(M, iTO, iMASK, iDET, V, I_flat, den, msk, ...
                          S_CONV, LAM, N_WF);
ZW.steppedX = @(M) steppedX_(M, iTO, iMASK, iDET, VK, M2i, b2cal, N_WF);
ZW.stepdiff = @(X1, X0) S_CONV * angle(X1 .* conj(X0)) * LAM/(4*pi);
% ---- frame-level access (noise stage): raw intensity frames + the
% pure-MATLAB reconstructions, so shot noise can be injected between
% capture and reconstruction without re-tracing.
ZW.frameL   = @(M) frameL_(M, iTO, iMASK, iDET, V, N_WF);
ZW.framesS  = @(M) framesS_(M, iTO, iMASK, iDET, VK, N_WF);
ZW.reconL   = @(Ia) reconL_(Ia, I_flat, den, msk, S_CONV, LAM, N_WF);
ZW.reconS   = @(Fr) reconS_(Fr, M2i, b2cal);
% ---- iterated-reference exact reading (the SAME single masked frame)
ZW.frameI   = ZW.frameL;
ZW.reconI   = @(Ia, varargin) reconI_(Ia, C, varargin{:});   % (Ia, I0, plus, b0, niter)
ZW.solveI   = @(Ia, varargin) solveI_(Ia, C, varargin{:});   % -> [phi, info]
ZW.measI    = @(M) reconI_(frameL_(M, iTO, iMASK, iDET, V, N_WF), C);
% branch prior from a stepped retrieval X = E conj(Eb): the branch
% variable is (phi - Theta) = angle(X) - arg c; TRUE selects the plus branch
ZW.plusFromX = @(X) plusFromX_(X, C);
% the REFINED prior: stepped frames of the same state re-solved with the
% iterated reading's own |b|^2 (the 'I+' definition since zwfs_s7iter)
ZW.priorS   = @(Ia, Fr, varargin) priorS_(Ia, Fr, C, M2i, varargin{:});   % (Ia, Fr, nref) -> [plus, info]
% ---- vector (polarized-dimple) reading: the +phi / -phi image pair -----
ZW.Vm = Vm;  ZW.ccm = ccm;  ZW.leak = leak;  ZW.arm = arm;
ZW.v_scalar_equiv = (eta_true == 1) && strcmp(arm.mode, 'none');   % the +phi image IS the scalar frame
ZW.frameV   = @(M) frameV_(M, iTO, iMASK, iDET, V, Vm, N_WF, leak, arm);  % -> [Ip, Im]
ZW.calV     = @(Ip, Im) calV_(Ip, Im, C);                            % -> [kapP, kapM, eta, info]: the flat's images
kap_true = sqrt(eta_true) + sqrt(1-eta_true)*exp(1i*vla);
if any(strcmp(vcal, {'amp', 'fit'})) && ~strcmp(arm.mode, 'none')
    % the per-channel UNMASKED reference frames (the mask substrate's clear
    % area, what every Zernike-sensor bench takes): the amplitude maps |qL|,
    % |qR| are measured, the polarization phases are not
    C.qL = abs(arm.qL);  C.qR = abs(arm.qR);
    C.kapP = abs(arm.qL);  C.kapM = abs(arm.qR);
    C.EbP0 = bsur(C.qL .* E0f);  C.EbM0 = bsur(C.qR .* E0f);
end
switch vcal
    case 'ideal'
    case 'amp'
    case 'fit'
        % calibrate on the FLAT DM (what a bench does): fit the per-channel
        % constants kappa+, kappa- (complex) and eta from the flat's two
        % masked images, on top of the unmasked amplitude maps
        [Ipf, Imf] = frameV_(zeros(macos.get_elt_grid_size(iTO)), iTO, iMASK, iDET, V, Vm, N_WF, leak, arm);
        [kP, kM, C.eta, ZW.calV_info] = calV_(Ipf, Imf, C);
        C.kapP = kP * C.kapP;  C.kapM = kM * C.kapM;
    case 'map'
        % the solver is told the truth: the arm maps and the metasurface
        % constants (a polarimetrically calibrated bench)
        C.eta = eta_true;
        if strcmp(arm.mode, 'none')
            C.kapP = kap_true;  C.kapM = kap_true;
        else
            lk = sqrt(1-eta_true)*exp(1i*vla);  se = sqrt(eta_true);
            C.qL = arm.qL;  C.qR = arm.qR;
            C.kapP = se*arm.qL + lk*arm.qR;  C.kapM = se*arm.qR + lk*arm.qL;
            C.EbP0 = bsur(arm.qL .* E0f);  C.EbM0 = bsur(arm.qR .* E0f);
        end
    otherwise
        error('dmg_zwfs_gauge: V_CAL must be ''ideal'', ''amp'', ''fit'' or ''map''');
end
ZW.vcal = struct('mode', vcal, 'kapP', C.kapP, 'kapM', C.kapM, 'eta', C.eta, 'eta_true', eta_true, ...
                 'kap_true', kap_true);
ZW.reconV   = @(Ip, Im, varargin) reconV_(Ip, Im, C, varargin{:});   % (Ip, Im, I0, b0, niter)
ZW.solveV   = @(Ip, Im, varargin) solveV_(Ip, Im, C, varargin{:});   % -> [phi, info]
ZW.measV    = @(M) measV_(M, iTO, iMASK, iDET, V, Vm, N_WF, C, leak, arm);
% differential height between two states, the phase DIFFERENCE wrapped
% (as stepdiff does): the absolute maps wrap at +-pi individually, so on a
% large working surface a differential of two maps carries 2 pi jumps
% where the base sits near the wrap; the wrapped difference does not
ZW.diffV    = @(Ip1, Im1, Ip0, Im0) diffV_(Ip1, Im1, Ip0, Im0, C);
end

function d = diffV_(Ip1, Im1, Ip0, Im0, C)
p1 = solveV_(Ip1, Im1, C);  p0 = solveV_(Ip0, Im0, C);
d = C.S_CONV * atan2(sin(p1 - p0), cos(p1 - p0)) * C.LAM/(4*pi);
end

% ---- vector reading: frames + solve ------------------------------------
function [Ip, Im] = frameV_(M, iTO, iMASK, iDET, V, Vm, N_WF, leak, arm) %#ok<INUSD>
% the two pupil images of one DM state: +phi dimple (== frameL's frame
% when the metasurface is ideal) and -phi dimple, from the same trace.
% With retardance error (leak.eta < 1) each output channel is the COHERENT
% sum sqrt(eta) E_masked + sqrt(1-eta) e^{i alpha} E_unmasked (linear
% laser).  With arm maps (V3) the +phi channel's masked light is the L
% input's, qL.*E, its leaked light the R input's, qR.*E (unconverted, no
% dimple), and vice versa; the maps are applied at the sandwich's entrance
% sphere before the dimple (chain_).
if nargin < 8 || isempty(leak), leak = struct('eta', 1, 'alpha', 0); end
if nargin < 9 || isempty(arm), arm = struct('mode', 'none', 'qL', [], 'qR', []); end
macos.set_elt_grid(iTO, macos.get_elt_grid_spacing(iTO), M);
if leak.eta < 1
    E0s = macos.complex_field(iDET);                       % the state's unmasked field
    lk = sqrt(1 - leak.eta) * exp(1i*leak.alpha);  se = sqrt(leak.eta);
else
    E0s = 0;  lk = 0;  se = 1;
end
if strcmp(arm.mode, 'none')
    macos.intensity(iMASK);
    macos.apodize_complex(iMASK, V);
    Ip = abs(se*macos.complex_field(iDET, 'reset_trace', false) + lk*E0s).^2;
    macos.intensity(iMASK);
    macos.apodize_complex(iMASK, Vm);
    Im = abs(se*macos.complex_field(iDET, 'reset_trace', false) + lk*E0s).^2;
else
    Ip = abs(se*chain_(M, arm.qL, V,  iTO, iMASK, iDET) + lk*arm.qR.*E0s).^2;
    Im = abs(se*chain_(M, arm.qR, Vm, iTO, iMASK, iDET) + lk*arm.qL.*E0s).^2;
end
end

function E = chain_(M, q, VV, iTO, iMASK, iDET)
% the detector field of DM state M with the pupil map q applied at the
% mask sandwich's entrance sphere (iMASK-1, identical to the detector's
% unmasked field: gate G1) and the dimple VV at the mask
macos.set_elt_grid(iTO, macos.get_elt_grid_spacing(iTO), M);
macos.intensity(iMASK-1);  macos.apodize_complex(iMASK-1, q);
macos.intensity(iMASK, 'reset_trace', false);  macos.apodize_complex(iMASK, VV);
E = macos.complex_field(iDET, 'reset_trace', false);
end

function [kapP, kapM, eta, info] = calV_(Ip, Im, C)
% fit the per-channel constants on the flat DM's two masked images: model
% I+ = |kappa+ aP E0 + sqrt(eta) c+ bP0|^2, I- = |kappa- aM E0 + sqrt(eta) c- bM0|^2
% per pixel, 5 real parameters (|kappa+|, arg kappa+, |kappa-|, arg kappa-,
% eta) on top of the solver's base amplitude maps aP, aM (1, or the
% unmasked per-channel reference frames' |q|) and their reference waves.
% With an ideal arm kappa+ = kappa- = kappa (V2's 3-parameter fit).
m = C.msk;  ip = Ip(m);  im = Im(m);
aP = C.kapP;  aM = C.kapM;  if isscalar(aP), aP = aP*ones(size(C.E0)); end;  if isscalar(aM), aM = aM*ones(size(C.E0)); end
EP = aP(m) .* C.E0(m);  EM = aM(m) .* C.E0(m);  bP = C.EbP0(m);  bM = C.EbM0(m);   % the base model's fields + reference waves
sc = mean(ip);
f = @(q) sum((abs(q(1)*exp(1i*q(2))*EP + sqrt(max(q(5),0))*C.cc*bP).^2 - ip).^2 + ...
             (abs(q(3)*exp(1i*q(4))*EM + sqrt(max(q(5),0))*C.ccm*bM).^2 - im).^2) / sc^2;
[q, fv] = fminsearch(f, [1 0 1 0 1], optimset('TolX', 1e-10, 'TolFun', 1e-14, 'MaxFunEvals', 8000, 'MaxIter', 8000, 'Display', 'off'));
kapP = q(1)*exp(1i*q(2));  kapM = q(3)*exp(1i*q(4));  eta = q(5);
info = struct('resid', sqrt(fv/numel(ip)), 'q', q);
end

function h = measV_(M, iTO, iMASK, iDET, V, Vm, N_WF, C, leak, arm)
[Ip, Im] = frameV_(M, iTO, iMASK, iDET, V, Vm, N_WF, leak, arm);
h = reconV_(Ip, Im, C);
end

function [h, info] = reconV_(Ip, Im, C, varargin)
[phi, info] = solveV_(Ip, Im, C, varargin{:});
h = C.S_CONV*phi*C.LAM/(4*pi);
end

function [phi, info] = solveV_(Ip, Im, C, I0, b0, niter)
%SOLVEV_  Per-pixel exact solve from the +phi / -phi image pair.
%   Same model as solveI_ for each image, I+- = |E + c+- b|^2 with
%   c- = conj(c+) (a real dimple phase of either sign), so with
%   x+- = (I+- - A^2 - |c|^2|b|^2) / (2 A |c| |b|) = cos(u -/+ tc),
%   u = phi - (arg b - th0), tc = arg c+:
%       cos u = (x+ + x-) / (2 cos tc),   sin u = (x+ - x-) / (2 sin tc),
%   u = atan2(sin u, cos u): the full circle, no branch, no clamp.  b is
%   iterated from the estimate as in solveI_ (NITER; 0 = frozen b).
%   General per-channel model (V2 metasurface constants, V3 arm maps):
%       I+ = |kappa+ A e^{i phi'} + sqrt(eta) c+ b+|^2,   b+ = bsur(qL E)
%       I- = |kappa- A e^{i phi'} + sqrt(eta) c- b-|^2,   b- = bsur(qR E)
%   with phi' = th0 + phi, E = A0 e^{i phi'}; kappa+-, qL, qR scalars or
%   per-pixel maps (C.kapP, C.kapM, C.qL, C.qR; 1, 1, [], [] = the ideal
%   sensor).  Then x+ = cos(psi - t), x- = cos(psi + t) with
%       s+- = arg kappa+- - arg b+-,  m = (s+ + s-)/2,  d = (s+ - s-)/2,
%       psi = phi' + m,  t = tc - d,
%   the same two-image algebra per pixel with (psi, t) for (u, tc).
%   (V2 wrote the constant-kappa shift as 2 arg kappa: a piston, invisible
%   to every mean-referenced number; it is arg kappa, corrected here.)
%   info: dphi (rms update per iteration), rcons (rms of sqrt(cos^2 +
%   sin^2) - 1 on msk: the two images' consistency with the model; 0 for
%   an exact model), b (final, the + channel's), bM (the - channel's).
if nargin < 4, I0 = []; end
if nargin < 5, b0 = []; end
if nargin < 6 || isempty(niter), niter = C.NITER; end
N = C.N_WF;  m = C.msk;
wrap = @(p) atan2(sin(p), cos(p));
if isempty(I0), A0 = abs(C.E0); else, A0 = sqrt(max(I0, 0)); end
th0 = angle(C.E0);  tc = angle(C.cc);
if isempty(C.qL), qL = 1; else, qL = C.qL; end
if isempty(C.qR), qR = 1; else, qR = C.qR; end
if isempty(b0), bP = C.EbP0;  bM = C.EbM0; else, bP = b0;  bM = b0; end
aP = abs(C.kapP) .* A0;  aM = abs(C.kapM) .* A0;  ac = sqrt(C.eta) * abs(C.cc);
sP0 = angle(C.kapP);  sM0 = angle(C.kapM);
phi = zeros(N);
info = struct('dphi', zeros(1, niter+1), 'rcons', zeros(1, niter+1));
for it = 0:niter
    xp = (Ip - aP.^2 - ac^2*abs(bP).^2) ./ max(2*aP.*ac.*abs(bP), realmin);
    xm = (Im - aM.^2 - ac^2*abs(bM).^2) ./ max(2*aM.*ac.*abs(bM), realmin);
    sP = sP0 - angle(bP);  sM = sM0 - angle(bM);
    mm = (sP + sM)/2;  t = tc - (sP - sM)/2;
    cu = (xp + xm) ./ (2*cos(t));  su = (xp - xm) ./ (2*sin(t));
    r = hypot(cu, su);  info.rcons(it+1) = sqrt(mean((r(m) - 1).^2));
    ph = wrap(atan2(su, cu) - mm - th0);
    ph(~m) = 0;
    info.dphi(it+1) = sqrt(mean((ph(m) - phi(m)).^2));
    phi = ph;
    if it < niter
        E = A0 .* exp(1i*(th0 + phi));                   % the reference waves from the true-amplitude field
        bP = C.bsur(qL .* E);  bM = C.bsur(qR .* E);
    end
end
info.b = bP;  info.bM = bM;
end

function plus = plusFromX_(X, C)
d = angle(X) - angle(C.cc);
plus = C.msk & (atan2(sin(d), cos(d)) > 0);
end

function [plus, info] = priorS_(Ia, Fr, C, M2i, nref)
%PRIORS_  Branch prior for the iterated reading from the stepped frames of
%   the SAME state, refined.  The stepped retrieval's |Eb|^2 is the flat's
%   b2cal, 14% off under a 30 nm state (the 'moving core'), so its branch
%   map misses ~3% of pixels -- enough to sign-flip a single-actuator
%   differential whose footprint sits beyond the fold (zwfs_s7iter,
%   48x48).  Re-solving the stepped retrieval with the iterated reading's
%   own |b|^2 and re-deriving the branch converges to the TRUE branch in
%   two passes (fold_site probe: agreement 0.970 -> 0.9999 -> 1.0000 on
%   the pupil; the sign-flipped row 1.6232 -> +1.0026 = the oracle
%   branch).  NREF default 2.  info.frac = beyond-fold fraction per pass.
if nargin < 5 || isempty(nref), nref = 2; end
plus = plusFromX_(reconS_(Fr, M2i, abs(C.Eb0).^2), C);
info = struct('frac', zeros(1, nref+1));  info.frac(1) = mean(plus(C.msk));
for r = 1:nref
    [~, inf1] = solveI_(Ia, C, [], plus);
    plus = plusFromX_(reconS_(Fr, M2i, abs(inf1.b).^2), C);
    info.frac(r+1) = mean(plus(C.msk));
end
end

% ---- frame capture + pure-MATLAB reconstructions (noise stage) -------
function Ia = frameL_(M, iTO, iMASK, iDET, V, N_WF) %#ok<INUSD>
macos.set_elt_grid(iTO, macos.get_elt_grid_spacing(iTO), M);
macos.intensity(iMASK);
macos.apodize_complex(iMASK, V);
Ia = abs(macos.complex_field(iDET, 'reset_trace', false)).^2;
end

function Fr = framesS_(M, iTO, iMASK, iDET, VK, N_WF)
% Fr(:,:,1) = unmasked; Fr(:,:,2:4) = the three depth frames.
macos.set_elt_grid(iTO, macos.get_elt_grid_spacing(iTO), M);
Fr = zeros(N_WF, N_WF, 4);
Fr(:,:,1) = abs(macos.complex_field(iDET)).^2;
for k = 1:3
    macos.intensity(iMASK);
    macos.apodize_complex(iMASK, VK{k});
    Fr(:,:,k+1) = abs(macos.complex_field(iDET, 'reset_trace', false)).^2;
end
end

function h = reconL_(Ia, I_flat, den, msk, S_CONV, LAM, N_WF)
phi = zeros(N_WF);
phi(msk) = (Ia(msk) - I_flat(msk)) ./ den(msk);
h = S_CONV*phi*LAM/(4*pi);
end

function X = reconS_(Fr, M2i, b2)
d1 = Fr(:,:,2)-Fr(:,:,1);  d2 = Fr(:,:,3)-Fr(:,:,1);  d3 = Fr(:,:,4)-Fr(:,:,1);
p = M2i(1,1)*d1 + M2i(1,2)*d2 + M2i(1,3)*d3;
q = M2i(2,1)*d1 + M2i(2,2)*d2 + M2i(2,3)*d3;
X = (b2 - p) + 1i*q;
end

% ---- iterated-reference exact solve ----------------------------------
function [h, info] = reconI_(Ia, C, varargin)
[phi, info] = solveI_(Ia, C, varargin{:});
h = C.S_CONV*phi*C.LAM/(4*pi);
end

function [phi, info] = solveI_(Ia, C, I0, plus, b0, niter)
%SOLVEI_  Per-pixel exact ZWFS solve with an iterated reference wave.
%   Model: I = |E + c b|^2, E = A exp(i(th0 + phi)), b = T(D Ti(E)) (the
%   field through the dimple-support disk alone).  Per pixel, for general
%   complex E0, b, c:
%       cos(phi - Theta) = (I - A^2 - |c|^2 |b|^2) / (2 A |c| |b|),
%       Theta = arg c + arg b - th0,
%   principal branch phi = Theta - acos(.), i.e. (phi - Theta) in [-pi, 0]
%   (the quarter-wave sensor's -pi/4 .. 3pi/4 about a flat pupil); PLUS
%   (logical, per pixel) selects the other branch where a prior says so.
%   b starts at the flat's engine Eb and is re-propagated from the estimate
%   NITER times; the un-iterated case (NITER = 0) is the exact solve with a
%   FROZEN b.  A = |E0| by default (exact on a DM-conjugate pupil); pass
%   I0 = the state's own unmasked (dimple-offset) frame to use sqrt(I0).
%   B0 (default the flat's engine Eb) seeds b -- pass the engine's Eb of
%   the state itself for an ORACLE solve; NITER overrides the factory's.
%   info: dphi (rms update per iteration on msk), nclamp (px with the
%   cosine argument clamped to [-1,1] -- model inconsistency), b (final).
if nargin < 3, I0 = []; end
if nargin < 4 || isempty(plus), plus = false(C.N_WF); end
if nargin < 5 || isempty(b0), b0 = C.Eb0; end
if nargin < 6 || isempty(niter), niter = C.NITER; end
N = C.N_WF;  m = C.msk;
wrap = @(p) atan2(sin(p), cos(p));
if isempty(I0), A = abs(C.E0); else, A = sqrt(max(I0, 0)); end
th0 = angle(C.E0);  ac = abs(C.cc);  tc = angle(C.cc);
b = b0;  phi = zeros(N);
info = struct('dphi', zeros(1, niter+1), 'nclamp', zeros(1, niter+1));
anyplus = any(plus(:));
for it = 0:niter
    den = 2*A.*ac.*abs(b);
    x = (Ia - A.^2 - ac^2*abs(b).^2) ./ max(den, realmin);
    info.nclamp(it+1) = nnz(m & abs(x) > 1);
    x = min(max(x, -1), 1);
    Th = tc + angle(b) - th0;
    ph = wrap(Th - acos(x));
    if anyplus
        pp = wrap(Th + acos(x));  ph(plus) = pp(plus);
    end
    ph(~m) = 0;
    info.dphi(it+1) = sqrt(mean((ph(m) - phi(m)).^2));
    phi = ph;
    if it < niter
        b = C.bsur(A .* exp(1i*(th0 + phi)));
    end
end
info.b = b;
end

% ---- frozen-linear measurement (verbatim) ----------------------------
function h = measL_(M, iTO, iMASK, iDET, V, I_flat, den, msk, S_CONV, LAM, N_WF)
macos.set_elt_grid(iTO, macos.get_elt_grid_spacing(iTO), M);
macos.intensity(iMASK);
macos.apodize_complex(iMASK, V);
Ia = abs(macos.complex_field(iDET, 'reset_trace', false)).^2;
phi = zeros(N_WF);
phi(msk) = (Ia(msk) - I_flat(msk)) ./ den(msk);
h = S_CONV*phi*LAM/(4*pi);
end

% ---- stepped retrieval (rank-2, calibrated b2) (verbatim) ------------
function X = steppedX_(M, iTO, iMASK, iDET, VK, M2i, b2, N_WF)
macos.set_elt_grid(iTO, macos.get_elt_grid_spacing(iTO), M);
I0m = abs(macos.complex_field(iDET)).^2;
Ik = zeros(N_WF, N_WF, 3);
for k = 1:3
    macos.intensity(iMASK);
    macos.apodize_complex(iMASK, VK{k});
    Ik(:,:,k) = abs(macos.complex_field(iDET, 'reset_trace', false)).^2;
end
d1 = Ik(:,:,1)-I0m;  d2 = Ik(:,:,2)-I0m;  d3 = Ik(:,:,3)-I0m;
p = M2i(1,1)*d1 + M2i(1,2)*d2 + M2i(1,3)*d3;
q = M2i(2,1)*d1 + M2i(2,2)*d2 + M2i(2,3)*d3;
X = (b2 - p) + 1i*q;
end
