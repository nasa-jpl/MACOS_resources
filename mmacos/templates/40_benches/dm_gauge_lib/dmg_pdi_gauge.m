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
%                         difference of two states (the V1 lesson: absolute
%                         maps wrap individually at +-pi)
%     PD.meas(M)          frames + height in one call
%     PD.nframes, PD.throughput, PD.vis, PD.msk, PD.E0, PD.Eb0, PD.R,
%     PD.D, PD.t, PD.a, PD.f, PD.gate (bsur, flat)
%
%   THE INSTRUMENT.  A point-diffraction interferometer (Smartt 1972)
%   interferes the beam with a reference wave diffracted from a PINHOLE at
%   its own focus: the pinhole passes only the core of the spot, so what
%   comes out is a clean sphere whose shape barely depends on the
%   aberration that made the spot.  The Zernike sensor is the same
%   instrument with a PHASE dimple instead of a pinhole and unit
%   transmission everywhere; what the PDI trades is throughput (the
%   surround is attenuated to match the reference's amplitude) for a
%   reference that does not move with the state and a four-step solve that
%   is exact and linear, with no quarter-wave fold (Medecki 1996 phase-
%   shifting PDI; Naulleau 1999 EUV PS/PDI).  Two forms, opt.MODE:
%
%   'pinhole' -- COMMON-PATH, the mask at the FocalMask:
%       V_k = t + (e^{i theta_k} - t) D,   D = the pinhole disk,
%       t = amplitude transmission of the surround (opt.T_SURR; 'auto'
%       matches the reference's rms amplitude over the pupil), theta_k the
%       phase steps (opt.THETAS, default 0 pi/2 pi 3pi/2: a stepped
%       pinhole substrate).  Detector field  t E + c_k b,  c_k = e^{i th_k}
%       - t,  b = T(D Ti(E)) the pinhole-diffracted reference of THIS
%       state (the FFT surrogate of the mask leg, gated against the
%       engine).  Per pixel, with X = E conj(b):
%         I_k = A + B cos th_k + C sin th_k,
%         B = 2t (Re X - |b|^2),  C = 2t Im X,
%         A = t^2 |E|^2 + (1+t^2)|b|^2 - 2 t^2 Re X,
%       a 3-parameter least-squares fit over the K steps, then Re X needs
%       |b|^2: opt.B2 'flat' takes the flat DM's pinhole-only frame (one
%       calibration frame, K frames per state) and 'state' adds a pinhole-
%       only frame to every state (K+1, exact).  The reference's PHASE is
%       the flat's, then iterated from the estimate NITER times exactly as
%       the ZWFS exact readings do (opt.NITER; 0 = frozen).  A small
%       pinhole makes both corrections small -- that is the PDI's
%       argument, and PD.refstab measures it.  With t = 1 and the ZWFS
%       dimple diameter this IS the phase-stepped Zernike reading S.
%
%   'fiber' -- NON-COMMON-PATH, the reference in its own arm (Dube et al.
%       2024, SPIE 13092-178: a pinhole / single-mode-fiber reference with a
%       PHOTONIC phase shifter, recombined with the unattenuated beam):
%       detector field  s E + a e^{i th_k} R,  s = sqrt(1 - f) (opt.PICKOFF
%       f = fraction of the beam power sent to the reference arm), R = the
%       flat DM's pinhole-diffracted field normalized to unit rms over the
%       pupil (the shape a fiber-launched reference has after the relay),
%       a = the reference amplitude (opt.A_REF; 'auto' = the smaller of
%       the visibility-1 match a s rms|E| and the pickoff BUDGET a^2 sum
%       |R|^2 <= f eta_pin sum|E|^2, eta_pin = the pinhole's coupling).
%       X = E conj(R):  B = 2 a s Re X,  C = 2 a s Im X,  A = s^2|E|^2 +
%       a^2|R|^2 -- exact in one pass, no |b|^2 degeneracy, no iteration,
%       the reference by construction independent of the state.  The
%       model is IDEAL: the arm has no drift (what the photonic modulation
%       buys on hardware) -- a drift knob is the loop stage's business.
%
%   Photons are counted at the DETECTOR (the campaign's currency); the
%   surround attenuation / pickoff loss is reported as PD.throughput =
%   detected / incident on the flat, so a cost can be restated in incident
%   photons.  Height convention: S_CONV*phi*LAM/(4 pi), single reflection.
%   Requires zwfs_mask on the path (run from zwfs_dm96/).
%
%   opt: LAM, F2, R_BEAM (mm), DIA_LAMD (pinhole diameter, lam/D), MODE,
%   T_SURR, PICKOFF, A_REF, THETAS, B2, S_CONV, NITER.
LAM = opt.LAM;  S_CONV = opt.S_CONV;
mode = 'pinhole';  if isfield(opt, 'MODE'), mode = opt.MODE; end
assert(any(strcmp(mode, {'pinhole', 'fiber'})), 'dmg_pdi_gauge: MODE must be ''pinhole'' or ''fiber''');
th = [0 pi/2 pi 3*pi/2];  if isfield(opt, 'THETAS') && ~isempty(opt.THETAS), th = opt.THETAS(:).'; end
K = numel(th);  assert(K >= 3, 'dmg_pdi_gauge: at least 3 phase steps');
NITER = 3;  if isfield(opt, 'NITER'), NITER = opt.NITER; end
b2mode = 'flat';  if isfield(opt, 'B2'), b2mode = opt.B2; end
tsurr = 'auto';  if isfield(opt, 'T_SURR'), tsurr = opt.T_SURR; end
f = 0.5;  if isfield(opt, 'PICKOFF'), f = opt.PICKOFF; end
aref = 'auto';  if isfield(opt, 'A_REF'), aref = opt.A_REF; end

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

C = struct('mode',mode, 'th',th, 'K',K, 'N',N_WF, 'msk',msk, 'E0',E0f, 'Eb0',Ebf, 'th0',th0, ...
           'bsur',bsur, 'S_CONV',S_CONV, 'LAM',LAM, 'NITER',NITER, 'b2mode',b2mode);
% the 3-parameter step fit [1 cos sin] over the K steps (pseudo-inverse)
Mk = [ones(K,1) cos(th(:)) sin(th(:))];  C.Mi = pinv(Mk);
switch mode
case 'pinhole'
    if ischar(tsurr) || isstring(tsurr)
        assert(strcmpi(tsurr, 'auto'), 'dmg_pdi_gauge: T_SURR ''auto'' or a number');
        t = min(1, rmsB / rmsE);                                % t |E| ~ |b|: visibility ~ 1
    else
        t = tsurr;
    end
    assert(t > 0 && t <= 1, 'dmg_pdi_gauge: T_SURR must be in (0, 1]');
    C.t = t;  C.a = NaN;  C.s = NaN;  C.f = NaN;  C.R = [];
    VK = cell(1, K);
    for k = 1:K, VK{k} = t + (exp(1i*th(k)) - t) * D; end
    C.VK = VK;  C.b2cal = abs(Ebf).^2;
    C.nframes = K + strcmp(b2mode, 'state');
    PD.frames = @(M) framesPin_(M, iTO, iMASK, iDET, C, D);
case 'fiber'
    assert(f > 0 && f < 1, 'dmg_pdi_gauge: PICKOFF in (0, 1)');
    s = sqrt(1 - f);  R = Ebf / rmsB;                           % unit-rms reference over msk
    a_match = s*rmsE;
    a_budget = sqrt(f*eta_pin*sum(I0(:)) / sum(abs(R(:)).^2));
    if ischar(aref) || isstring(aref)
        assert(strcmpi(aref, 'auto'), 'dmg_pdi_gauge: A_REF ''auto'' or a number');
        a = min(a_match, a_budget);
    else
        a = aref;
    end
    C.t = NaN;  C.a = a;  C.s = s;  C.f = f;  C.R = R;  C.a_match = a_match;  C.a_budget = a_budget;
    C.nframes = K;
    PD.frames = @(M) framesFib_(M, iTO, iDET, C);
end
PD.solve  = @(Fr) solve_(Fr, C);
PD.height = @(Fr) height_(Fr, C);
PD.diff   = @(Fr1, Fr0) diff_(Fr1, Fr0, C);
PD.meas   = @(M) height_(PD.frames(M), C);
% the reference's motion under a state: rel change of the pinhole-diffracted
% reference (surrogate) between the flat and a detector field E, on msk,
% for THIS pinhole or any diameter (lam/D) -- the PDI's argument, measured
PD.refstab = @(E, varargin) refstab_(E, E0f, N_WF, dx_mask_m*1e3, lamD_mm, ctr, msk, opt.DIA_LAMD, varargin{:});
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
PD.mode = mode;  PD.K = K;  PD.th = th;  PD.nframes = C.nframes;  PD.eta_pin = eta_pin;
PD.t = C.t;  PD.a = C.a;  PD.f = C.f;  PD.NITER = NITER;  PD.b2mode = b2mode;  PD.N_WF = N_WF;
if strcmp(mode, 'fiber'), PD.a_match = C.a_match;  PD.a_budget = C.a_budget; end
PD.bsur = bsur;  PD.C = C;
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
% the fixed reference at the K photonic phase steps
macos.set_elt_grid(iTO, macos.get_elt_grid_spacing(iTO), M);
E = macos.complex_field(iDET);
Fr = zeros(C.N, C.N, C.K);
for k = 1:C.K
    Fr(:,:,k) = abs(C.s*E + C.a*exp(1i*C.th(k))*C.R).^2;
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
info = struct('dphi', [], 'b', []);
switch C.mode
case 'fiber'
    as = C.a*C.s;
    X = (B + 1i*Cc) / (2*as);                        % E conj(R)
    amp2 = (A - C.a^2*abs(C.R).^2) / C.s^2;
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

function [r, rshape, scl] = refstab_(E, E0, N, dx_mm, lamD_mm, ctr, msk, dia0, dia_lamd)
% the pinhole-diffracted reference's change between the flat (E0) and a
% state (E), on msk, for pinhole diameter dia_lamd (default: this one):
%   r      = |b1 - b0| / |b0|              the total relative change
%   scl    = (b0' b1) / (b0' b0)           the best complex SCALE (a Strehl-
%                                          class amplitude drop + a piston:
%                                          calibrated away by the |b|^2 frame
%                                          / the iteration; harmless)
%   rshape = |b1 - scl b0| / |b0|          what is left: the SHAPE change,
%                                          the part the solve must iterate out
if nargin < 9 || isempty(dia_lamd), dia_lamd = dia0; end
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
