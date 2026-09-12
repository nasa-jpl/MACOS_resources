function PD = dmg_pdi_gauge(iTO, iMASK, iDET, opt)
%DMG_PDI_GAUGE  Point-diffraction-interferometer measurement factory (FocalMask).
%   PD = dmg_pdi_gauge(iTO, iMASK, iDET, opt) builds a phase-shifting
%   point-diffraction interferometer on the LOADED bench (the ZWFS test arm:
%   collimated beam off the DM, focus at the FocalMask inside the NF
%   sandwich, the reimaged pupil on the detector) and returns a reading in
%   the same shape as dmg_zwfs_gauge's:
%     PD.frames(M)        the K (or K+1) intensity frames of DM grid M
%     PD.solve(Fr)        -> [phi, amp, info]: per-pixel phase of the
%                         detector field RELATIVE TO THE FLAT DM (rad,
%                         wrapped) and its amplitude; exact, no branch
%     PD.height(Fr)       absolute height map (mm) = S_CONV*phi*LAM/(4 pi)
%     PD.diff(Fr1, Fr0)   differential height (mm): the WRAPPED phase
%                         difference of two states = angle(X1 conj X0), the
%                         complex-division form of Dube 2024 eq. 15-19 (no
%                         unwrapping; absolute maps wrap individually)
%     PD.meas(M)          frames + height in one call
%     PD.refstab(E, dias) the reference's motion under a state (see below)
%     PD.nframes, PD.throughput, PD.vis, PD.msk, PD.E0, PD.Eb0, PD.R,
%     PD.D, PD.t, PD.a, PD.f, PD.eta_c, PD.gate (bsur, flat, amp)
%
%   THE INSTRUMENT.  A point-diffraction interferometer (Smartt 1972)
%   interferes the beam with a reference wave diffracted from a PINHOLE at
%   its own focus: the pinhole passes only the core of the spot, so what
%   comes out is a clean sphere whose shape barely depends on the
%   aberration that made the spot.  The Zernike sensor is the same
%   instrument with a PHASE dimple instead of a pinhole and unit
%   transmission everywhere; what the PDI trades is throughput (the
%   surround is attenuated to match the reference's amplitude) for a
%   reference that does not move with the state and a phase-stepped solve
%   that is exact and linear, with no quarter-wave fold (Medecki 1996
%   phase-shifting PDI; Naulleau 1999 EUV PS/PDI).  Two forms, opt.MODE:
%
%   'pinhole' -- COMMON-PATH, the mask at the FocalMask:
%       V_k = t + (e^{i theta_k} - t) D,   D = the pinhole disk,
%       t = amplitude transmission of the surround (opt.T_SURR; 'auto'
%       matches the reference's rms amplitude over the pupil), theta_k the
%       phase steps (opt.THETAS: a stepped pinhole substrate).  Detector
%       field  t E + c_k b,  c_k = e^{i th_k} - t,  b = T(D Ti(E)) the
%       pinhole-diffracted reference of THIS state (the FFT surrogate of the
%       mask leg, gated against the engine).  Per pixel, with X = E conj(b):
%         I_k = A + B cos th_k + C sin th_k,
%         B = 2t (Re X - |b|^2),  C = 2t Im X,
%         A = t^2 |E|^2 + (1+t^2)|b|^2 - 2 t^2 Re X,
%       then Re X needs |b|^2: opt.B2 'flat' takes the flat DM's pinhole-
%       only frame (one calibration frame, K frames per state; |b|^2 then
%       iterated with the phase) and 'state' adds a pinhole-only frame to
%       every state (K+1, exact).  The reference's PHASE is the flat's, then
%       iterated from the estimate NITER times exactly as the ZWFS exact
%       readings do (opt.NITER; 0 = frozen).  With t = 1 and the ZWFS
%       dimple diameter this IS the phase-stepped Zernike reading S.
%
%   'fiber' -- NON-COMMON-PATH: the phase-shifting self-referenced
%       interferometer (P/SRI) of Dube, Nejadriahi, Sidick, Jewell, Redding,
%       Lou, Basinger, Proc. SPIE 13092, 130926F (2024), modeled as their
%       `coupleThroughPhotonicChipRecollimate`: a beamsplitter sends the
%       fraction f of the beam power (opt.PICKOFF; their R = 0.6) to a
%       reference arm that focuses it onto a single-mode waveguide in a
%       photonic chip; the waveguide's LP01 mode is the only thing that comes
%       back out, scaled by the coupling of the state's own focal field into
%       it, phase-shifted thermo-optically, recollimated and recombined with
%       the test beam (1 - f).  Detector field
%           s E + a kappa e^{i th_k} R,   s = sqrt(1 - f),
%       R = T(mode) the recollimated mode at unit rms over the pupil
%       (Gaussian-like, opt.REF_SHAPE 'fiber'; 'pinhole' = the pinhole-
%       diffracted flat field instead), mode = the step-index LP01 field on
%       the focal grid (opt.FIB_V 2.3, FIB_B 0.5, FIB_A_LAMD 0.5 = core
%       radius in lam/D: their Thorlabs UV-fiber set), a = the reference
%       amplitude (opt.A_REF; 'auto' = min of the visibility-1 match
%       a = s rms|E| and the pickoff BUDGET a^2 sum|R|^2 <= f |c0|^2, c0 =
%       the flat's coupled complex amplitude), kappa = c(E)/c0 the state's
%       coupling relative to the flat -- the ONE way the state reaches the
%       reference: a complex SCALAR, so the reference's shape is fixed by
%       construction (Strehl-class amplitude drop + a piston).  Solve:
%       X = E conj(R),  B = 2 a s Re X,  C = 2 a s Im X,  A = s^2|E|^2 +
%       a^2|kappa|^2|R|^2 -- exact in one pass, no iteration; the solver
%       takes kappa = 1 (opt.B2 'flat': the paper's differential mode, the
%       coupling change reads as an amplitude scale + piston) or |kappa| from
%       a shutter frame of the reference alone ('state', K+1 frames, their
%       "shutter that blocks the test arm").
%
%   Step schemes (opt.SCHEME): 'ls' = the 3-parameter least-squares fit
%   [1 cos sin] over any K >= 3 steps (default; opt.THETAS, default the
%   four-step 0 pi/2 pi 3pi/2); 'sh5' = the five-frame Schwider-Hariharan
%   scan of the paper, steps -pi -pi/2 0 pi/2 pi with de Groot's weights
%   c = {-1 0 2 0 -1}/4, s = {0 -2 0 2 0}/4 (both sum to zero: a constant
%   bias pattern subtracts; first-order immune to a step-size error).
%   opt.STEP_ERR applies a fractional step miscalibration to the FRAMES only
%   (the solve keeps the nominal steps) -- the scheme trade.
%
%   Photons are counted at the DETECTOR (the campaign's currency); the
%   surround attenuation / pickoff loss is reported as PD.throughput =
%   detected / incident on the flat, so a cost can be restated in incident
%   photons.  Height convention: S_CONV*phi*LAM/(4 pi), single reflection.
%   Requires zwfs_mask on the path (run from zwfs_dm96/).
LAM = opt.LAM;  S_CONV = opt.S_CONV;
mode = 'pinhole';  if isfield(opt, 'MODE'), mode = opt.MODE; end
assert(any(strcmp(mode, {'pinhole', 'fiber'})), 'dmg_pdi_gauge: MODE must be ''pinhole'' or ''fiber''');
scheme = 'ls';  if isfield(opt, 'SCHEME') && ~isempty(opt.SCHEME), scheme = opt.SCHEME; end
th = [0 pi/2 pi 3*pi/2];  if isfield(opt, 'THETAS') && ~isempty(opt.THETAS), th = opt.THETAS(:).'; end
if strcmp(scheme, 'sh5'), th = [-pi -pi/2 0 pi/2 pi]; end
K = numel(th);  assert(K >= 3, 'dmg_pdi_gauge: at least 3 phase steps');
sterr = 0;  if isfield(opt, 'STEP_ERR'), sterr = opt.STEP_ERR; end
NITER = 3;  if isfield(opt, 'NITER'), NITER = opt.NITER; end
b2mode = 'flat';  if isfield(opt, 'B2'), b2mode = opt.B2; end
tsurr = 'auto';  if isfield(opt, 'T_SURR'), tsurr = opt.T_SURR; end
f = 0.6;  if isfield(opt, 'PICKOFF'), f = opt.PICKOFF; end
aref = 'auto';  if isfield(opt, 'A_REF'), aref = opt.A_REF; end
refshape = 'fiber';  if isfield(opt, 'REF_SHAPE') && ~isempty(opt.REF_SHAPE), refshape = opt.REF_SHAPE; end
fibV = 2.3;  fibB = 0.5;  fibA = 0.5;
if isfield(opt, 'FIB_V'), fibV = opt.FIB_V; end
if isfield(opt, 'FIB_B'), fibB = opt.FIB_B; end
if isfield(opt, 'FIB_A_LAMD'), fibA = opt.FIB_A_LAMD; end

% ---- the flat DM's fields, the spot, the pinhole ---------------------------
E0f = macos.complex_field(iDET);  N_WF = size(E0f, 1);
dx_mask_m = abs(macos.dx_at(iMASK));
lamD_mm = LAM*opt.F2/(2*opt.R_BEAM);  dia_mm = opt.DIA_LAMD*lamD_mm;
If = abs(macos.complex_field(iMASK)).^2;
assert(max(If(:))/sum(If(:)) >= 1e-2, 'dmg_pdi_gauge: mask plane not focused');
[~, ipk] = max(If(:));  [pr, pc] = ind2sub(size(If), ipk);
w = 12;  rows = max(1,pr-w):min(N_WF,pr+w);  cols = max(1,pc-w):min(N_WF,pc+w);
Iw = If(rows, cols);
ctr = [sum(sum(Iw,1).*cols)/sum(Iw(:)), sum(sum(Iw,2).'.*rows)/sum(Iw(:))];
[~, D] = zwfs_mask(N_WF, dx_mask_m*1e3, dia_mm, 0, ctr);
macos.intensity(iMASK);  macos.apodize_complex(iMASK, D);
Ebf = macos.complex_field(iDET, 'reset_trace', false);       % the flat's pinhole reference (engine)
T  = @(x) fftshift(fft2(fftshift(x)))/N_WF;
Ti = @(x) fftshift(ifft2(fftshift(x)))*N_WF;
bsur = @(E) T(D .* Ti(E));
nrm = @(x) norm(x(:));
I0 = abs(E0f).^2;  supp = I0 > 0.1*max(I0(:));
msk = supp & (abs(Ebf) > 0.05*max(abs(Ebf(supp))));
gate = struct();
gate.bsur = nrm((bsur(E0f) - Ebf).*msk) / nrm(Ebf.*msk);
eta_pin = sum(abs(Ebf(:)).^2) / sum(I0(:));                 % pinhole coupling (core fraction)
rmsE = sqrt(mean(abs(E0f(msk)).^2));  rmsB = sqrt(mean(abs(Ebf(msk)).^2));
th0 = angle(E0f);

C = struct('mode',mode, 'th',th, 'thf',th*(1+sterr), 'K',K, 'N',N_WF, 'msk',msk, 'E0',E0f, 'Eb0',Ebf, ...
           'th0',th0, 'bsur',bsur, 'Ti',Ti, 'S_CONV',S_CONV, 'LAM',LAM, 'NITER',NITER, 'b2mode',b2mode, ...
           'scheme',scheme);
% the step weights: rows [A; B; C] of I_k = A + B cos th_k + C sin th_k
Mk = [ones(K,1) cos(th(:)) sin(th(:))];  Mi = pinv(Mk);
if strcmp(scheme, 'sh5')
    Mi(2,:) = [-1 0 2 0 -1]/4;  Mi(3,:) = [0 -2 0 2 0]/4;   % de Groot weights (Schwider-Hariharan)
end
C.Mi = Mi;
C.mode_f = [];  C.c0 = NaN;  C.eta_c = NaN;
switch mode
case 'pinhole'
    if ischar(tsurr) || isstring(tsurr)
        assert(strcmpi(tsurr, 'auto'), 'dmg_pdi_gauge: T_SURR ''auto'' or a number');
        t = min(1, rmsB / rmsE);                                % t |E| ~ |b|: visibility ~ 1
    else
        t = tsurr;
    end
    assert(t > 0 && t <= 1, 'dmg_pdi_gauge: T_SURR must be in (0, 1]');
    C.t = t;  C.a = NaN;  C.s = NaN;  C.f = NaN;  C.R = [];  C.refshape = 'pinhole';
    VK = cell(1, K);
    for k = 1:K, VK{k} = t + (exp(1i*C.thf(k)) - t) * D; end
    C.VK = VK;  C.b2cal = abs(Ebf).^2;
    C.nframes = K + strcmp(b2mode, 'state');
    PD.frames = @(M) framesPin_(M, iTO, iMASK, iDET, C, D);
case 'fiber'
    assert(f > 0 && f < 1, 'dmg_pdi_gauge: PICKOFF in (0, 1)');
    s = sqrt(1 - f);
    switch refshape
    case 'fiber'
        % the LP01 mode on the focal grid (lam/D units; Dube's singleModeField)
        [cc_, rr_] = meshgrid((1:N_WF) - ctr(1), (1:N_WF) - ctr(2));
        r_lamd = hypot(cc_, rr_) * (dx_mask_m*1e3) / lamD_mm;
        modef = lp01_(fibV, fibB, fibA, r_lamd);
        modef = modef / sqrt(sum(abs(modef(:)).^2));            % unit power
        Rraw = T(modef);                                         % recollimated: the pupil field
        c0 = sum(conj(modef(:)) .* reshape(Ti(E0f), [], 1));    % the flat's coupled amplitude (complex)
        Efoc0 = Ti(E0f);  eta_c = abs(c0)^2 / sum(abs(Efoc0(:)).^2);
        C.mode_f = modef;  C.c0 = c0;  C.eta_c = eta_c;
        pref = f*abs(c0)^2;                                      % reference power available
    case 'pinhole'
        Rraw = Ebf;  pref = f*eta_pin*sum(I0(:));  C.eta_c = eta_pin;
    otherwise
        error('dmg_pdi_gauge: REF_SHAPE must be ''fiber'' or ''pinhole''');
    end
    R = Rraw / sqrt(mean(abs(Rraw(msk)).^2));                   % unit rms over msk
    a_match = s*rmsE;
    a_budget = sqrt(pref / sum(abs(R(:)).^2));
    if ischar(aref) || isstring(aref)
        assert(strcmpi(aref, 'auto'), 'dmg_pdi_gauge: A_REF ''auto'' or a number');
        a = min(a_match, a_budget);
    else
        a = aref;
    end
    C.t = NaN;  C.a = a;  C.s = s;  C.f = f;  C.R = R;  C.a_match = a_match;  C.a_budget = a_budget;
    C.refshape = refshape;
    C.nframes = K + strcmp(b2mode, 'state');
    PD.frames = @(M) framesFib_(M, iTO, iDET, C);
end
PD.solve  = @(Fr) solve_(Fr, C);
PD.height = @(Fr) height_(Fr, C);
PD.diff   = @(Fr1, Fr0) diff_(Fr1, Fr0, C);
PD.meas   = @(M) height_(PD.frames(M), C);
% the reference's motion under a state, from a detector field E:
%   pinhole-shaped reference: [total, shape, scale] of the pinhole-diffracted
%   reference for pinhole diameter(s) dias (lam/D; default this one);
%   fiber reference: total = |kappa - 1|, shape = 0 by construction, scale
%   = kappa (the coupling relative to the flat)
PD.refstab = @(E, varargin) refstab_(E, E0f, C, N_WF, dx_mask_m*1e3, lamD_mm, ctr, msk, opt.DIA_LAMD, varargin{:});
% ---- the flat's frames: throughput, visibility, the flat identity ---------
Fr0 = PD.frames(zeros(macos.get_elt_grid_size(iTO)));
inc = sum(I0(:));
PD.throughput = mean(squeeze(sum(sum(Fr0(:,:,1:K), 1), 2))) / inc;
Imax = max(Fr0(:,:,1:K), [], 3);  Imin = min(Fr0(:,:,1:K), [], 3);
vis = (Imax - Imin) ./ max(Imax + Imin, realmin);
PD.vis = mean(vis(msk));
[phi0, amp0] = solve_(Fr0, C);
gate.flat = sqrt(mean(phi0(msk).^2));                          % the flat reads zero (rad rms)
gate.amp  = sqrt(mean((amp0(msk)./abs(E0f(msk)) - 1).^2));    % and its amplitude (rel rms)
PD.frames0 = Fr0;  PD.gate = gate;
PD.msk = msk;  PD.E0 = E0f;  PD.Eb0 = Ebf;  PD.R = C.R;  PD.D = D;  PD.ctr = ctr;  PD.dia_mm = dia_mm;
PD.mode = mode;  PD.K = K;  PD.th = th;  PD.nframes = C.nframes;  PD.eta_pin = eta_pin;  PD.eta_c = C.eta_c;
PD.t = C.t;  PD.a = C.a;  PD.f = C.f;  PD.NITER = NITER;  PD.b2mode = b2mode;  PD.N_WF = N_WF;
PD.scheme = scheme;  PD.step_err = sterr;  PD.refshape = C.refshape;  PD.mode_f = C.mode_f;
if strcmp(mode, 'fiber'), PD.a_match = C.a_match;  PD.a_budget = C.a_budget; end
PD.bsur = bsur;  PD.C = C;
end

% ==== the fiber mode ===================================================
function field = lp01_(v, b, a, r)
% the (single) LP01 field of a step-index fiber: J0 inside the core, K0
% outside, continuous at r = a (Dube's fibers.singleModeField; r, a in the
% same units)
U = v*sqrt(1-b);  W = v*sqrt(b);
rn = r / a;  in_ = rn < 1;
field = zeros(size(r));
field(in_)  = besselj(0, U*rn(in_)) / besselj(1, U);
field(~in_) = besselk(0, W*rn(~in_)) / besselk(1, W);
end

% ==== frames ==========================================================
function Fr = framesPin_(M, iTO, iMASK, iDET, C, D)
% the K stepped frames (+ the pinhole-only frame when B2 'state') of one DM state
macos.set_elt_grid(iTO, macos.get_elt_grid_spacing(iTO), M);
Fr = zeros(C.N, C.N, C.nframes);
for k = 1:C.K
    macos.intensity(iMASK);
    macos.apodize_complex(iMASK, C.VK{k});
    Fr(:,:,k) = abs(macos.complex_field(iDET, 'reset_trace', false)).^2;
end
if C.nframes > C.K
    macos.intensity(iMASK);
    macos.apodize_complex(iMASK, D);
    Fr(:,:,C.K+1) = abs(macos.complex_field(iDET, 'reset_trace', false)).^2;
end
end

function Fr = framesFib_(M, iTO, iDET, C)
% ONE trace of the state (no mask); the K frames are the interference with
% the reference at the K photonic phase steps, the reference scaled by the
% state's coupling into the waveguide (kappa; 1 for the pinhole shape)
macos.set_elt_grid(iTO, macos.get_elt_grid_spacing(iTO), M);
E = macos.complex_field(iDET);
kap = 1;
if ~isempty(C.mode_f)
    Efoc = C.Ti(E);
    kap = sum(conj(C.mode_f(:)) .* Efoc(:)) / C.c0;
end
Fr = zeros(C.N, C.N, C.nframes);
for k = 1:C.K
    Fr(:,:,k) = abs(C.s*E + C.a*kap*exp(1i*C.thf(k))*C.R).^2;
end
if C.nframes > C.K
    Fr(:,:,C.K+1) = abs(C.a*kap*C.R).^2;                     % the shutter frame: reference alone
end
end

% ==== the solve =========================================================
function [phi, amp, info] = solve_(Fr, C)
%SOLVE_  Per-pixel step fit, then the exact phase (relative to the flat).
N = C.N;  K = C.K;  m = C.msk;
wrap = @(p) atan2(sin(p), cos(p));
Y = reshape(Fr(:,:,1:K), N*N, K).';                 % K x N^2
Q = C.Mi * Y;                                        % 3 x N^2: [A; B; C]
A = reshape(Q(1,:), N, N);  B = reshape(Q(2,:), N, N);  Cc = reshape(Q(3,:), N, N);
info = struct('dphi', [], 'b', [], 'kap', 1);
switch C.mode
case 'fiber'
    aeff = C.a;
    if C.nframes > K                                 % shutter frame -> |kappa|
        Ish = Fr(:,:,K+1);
        info.kap = sqrt(mean(Ish(m)) / mean(C.a^2*abs(C.R(m)).^2));
        aeff = C.a*info.kap;
    end
    as = aeff*C.s;
    X = (B + 1i*Cc) / (2*as);                        % E conj(R)
    amp2 = (A - aeff^2*abs(C.R).^2) / C.s^2;
    phi = wrap(angle(X) + angle(C.R) - C.th0);
    amp = sqrt(max(amp2, 0));
    phi(~m) = 0;
case 'pinhole'
    t = C.t;
    if C.nframes > K, b2 = Fr(:,:,K+1); else, b2 = C.b2cal; end
    ImX = Cc / (2*t);
    b = C.Eb0;  phi = zeros(N);
    niter = C.NITER;  info.dphi = zeros(1, niter+1);
    for it = 0:niter
        ReX = B/(2*t) + b2;
        X = ReX + 1i*ImX;                            % E conj(b)
        amp2 = (A - (1+t^2)*b2 + 2*t^2*ReX) / t^2;
        amp = sqrt(max(amp2, 0));
        ph = wrap(angle(X) + angle(b) - C.th0);
        ph(~m) = 0;
        info.dphi(it+1) = sqrt(mean((ph(m) - phi(m)).^2));
        phi = ph;
        if it < niter
            % re-propagate the reference from the estimate: the true-amplitude
            % field (a phase-only DM state on a DM-conjugate pupil keeps |E0|)
            b = C.bsur(abs(C.E0) .* exp(1i*(C.th0 + phi)));
            if C.nframes == K, b2 = abs(b).^2; end   % 'flat' mode: |b|^2 iterated too
        end
    end
    info.b = b;
end
end

function h = height_(Fr, C)
phi = solve_(Fr, C);
h = C.S_CONV * phi * C.LAM/(4*pi);
end

function d = diff_(Fr1, Fr0, C)
p1 = solve_(Fr1, C);  p0 = solve_(Fr0, C);
d = C.S_CONV * atan2(sin(p1 - p0), cos(p1 - p0)) * C.LAM/(4*pi);
end

function [r, rshape, scl] = refstab_(E, E0, C, N, dx_mm, lamD_mm, ctr, msk, dia0, dia_lamd)
% the reference's change between the flat (E0) and a state (E):
%   pinhole-diffracted reference, for pinhole diameter dia_lamd (default this one):
%     r      = |b1 - b0| / |b0|              the total relative change
%     scl    = (b0' b1) / (b0' b0)           the best complex SCALE (a Strehl-
%                                            class amplitude drop + a piston:
%                                            calibrated away by the |b|^2 frame
%                                            / the iteration; harmless)
%     rshape = |b1 - scl b0| / |b0|          what is left: the SHAPE change,
%                                            the part the solve must iterate out
%   fiber reference: scl = kappa = c(E)/c0 (the coupling relative to the flat),
%     r = |kappa - 1|, rshape = 0 -- the shape is the mode's, by construction
if ~isempty(C.mode_f)
    Efoc = C.Ti(E);
    scl = sum(conj(C.mode_f(:)) .* Efoc(:)) / C.c0;
    r = abs(scl - 1);  rshape = 0;
    return
end
if nargin < 10 || isempty(dia_lamd), dia_lamd = dia0; end
T  = @(x) fftshift(fft2(fftshift(x)))/N;
Ti = @(x) fftshift(ifft2(fftshift(x)))*N;
r = zeros(size(dia_lamd));  rshape = r;  scl = complex(r);
for i = 1:numel(dia_lamd)
    [~, Dd] = zwfs_mask(N, dx_mm, dia_lamd(i)*lamD_mm, 0, ctr);
    b0 = T(Dd .* Ti(E0));  b1 = T(Dd .* Ti(E));
    b0m = b0(msk);  b1m = b1(msk);
    r(i) = norm(b1m - b0m) / norm(b0m);
    scl(i) = (b0m' * b1m) / (b0m' * b0m);
    rshape(i) = norm(b1m - scl(i)*b0m) / norm(b0m);
end
end
