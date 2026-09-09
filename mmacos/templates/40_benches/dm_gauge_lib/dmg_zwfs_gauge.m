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
%   PHI_M, PHIS (3 depths), S_CONV, NITER (default 5).  Requires zwfs_mask
%   on the path (run from zwfs_dm96/).  L/S verbatim from zwfs_s3 @
%   10cf593; the iterated reading added 2026-09-09 (zwfs_s7iter).
LAM = opt.LAM;  PHIS = opt.PHIS;  S_CONV = opt.S_CONV;
if isfield(opt, 'NITER'), NITER = opt.NITER; else, NITER = 5; end
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
C = struct('E0',E0f, 'Eb0',Ebf, 'cc',cc, 'msk',msk, 'N_WF',N_WF, ...
           'NITER',NITER, 'bsur',bsur, 'S_CONV',S_CONV, 'LAM',LAM);

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
