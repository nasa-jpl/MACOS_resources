function ZW = dmg_zwfs_gauge(iTO, iMASK, iDET, opt)
%DMG_ZWFS_GAUGE  Zernike-sensor measurement factory (dimple at FocalMask).
%   ZW = dmg_zwfs_gauge(iTO, iMASK, iDET, opt) builds the flat-DM
%   reference frames and masks on the LOADED bench and returns:
%     ZW.measL(M)      frozen-reference LINEAR height map (mm) -- one
%                      masked frame; small-differential workhorse
%     ZW.steppedX(M)   rank-2 phase-stepped complex retrieval X (S2b:
%                      |c|^2 = -2 Re(c) identically, so depth steps give
%                      TWO observables/px; |Eb|^2 from the one-time
%                      flat disk-frame calibration b2cal)
%     ZW.stepdiff(X1,X0)  differential height (mm) = S_CONV *
%                      angle(X1 conj X0) * LAM/(4 pi) -- range +-pi
%     fields: msk, den, I_flat, b2cal, N_WF, ctr, dia_mm
%   opt fields: LAM, F2 (mask-leg focal, mm), R_BEAM (mm), DIA_LAMD,
%   PHI_M, PHIS (3 depths), S_CONV.  Requires zwfs_mask on the path
%   (run from zwfs_dm96/).  Extracted verbatim from zwfs_s3 @ 10cf593.
LAM = opt.LAM;  PHIS = opt.PHIS;  S_CONV = opt.S_CONV;
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

ZW = struct();
ZW.msk = msk;  ZW.den = den;  ZW.I_flat = I_flat;  ZW.b2cal = b2cal;
ZW.N_WF = N_WF;  ZW.ctr = ctr;  ZW.dia_mm = dia_mm;
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
