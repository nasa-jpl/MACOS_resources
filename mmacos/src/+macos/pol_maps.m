function pm = pol_maps(jp)
%MACOS.POL_MAPS  Polarization-aberration decomposition of a Jones pupil.
%   pm = macos.pol_maps(JP) decomposes the Jones pupil JP (the struct
%   returned by macos.jones_pupil, or any struct with fields .J
%   [N x N x 2 x 2 complex, NaN where invalid] and .mask [N x N logical])
%   into the standard polarization-aberration maps via the per-point
%   polar decomposition J = H * W (H hermitian >= 0, W unitary).
%
%   All algebra is closed-form 2x2 (Pauli basis), vectorized over the
%   pupil -- no per-point svd() calls.
%
%   Pauli ordering (polarization convention):
%     s1 = [1 0; 0 -1]   x/y      (0/90 deg linear)
%     s2 = [0 1; 1  0]   +/-45 deg linear
%     s3 = [0 -i; i 0]   circular
%
%   Returns struct of N x N maps (NaN off-mask):
%     .T        intensity transmission (s1^2+s2^2)/2, singular values s
%               -- carries the engine's common source normalization, so
%               only RATIOS across the pupil are meaningful
%     .D        diattenuation magnitude (l1-l2)/(l1+l2), l = eig(J'J)
%     .Dvec     N x N x 3 diattenuation components (Pauli axes above)
%     .ret      retardance magnitude delta in [0, pi] (radians)
%     .retvec   N x N x 3 retardance vector delta*nhat
%     .phase    global phase psi of the unitary part, wrapped [-pi/2, pi/2)
%               (psi is only defined mod pi from det W = e^{2i psi})
%     .ambiguous N x N logical: retardance within 0.2 rad of pi, where the
%               (delta, nhat) <-> (2pi-delta, -nhat) branch is unresolved --
%               flagged, never silently chosen
%     .mask     copied from input
%   Pupil statistics, mask-weighted -- the MEAN and the VARIATION are
%   reported separately and must not be conflated: uniform retardance /
%   diattenuation is a state change (and, after folds, includes the
%   system's geometric rotation), NOT an aberration; only the variation
%   drives a contrast floor or a PSI systematic.
%     .mean     .T .D .Dvec(3x1) .ret .retvec(3x1)
%     .var_rms  RMS of the per-point deviation from the mean, same fields
%
%   Basis dependence: .D and .T are invariant to the per-point output
%   basis choice in jones_pupil (singular-value invariants); .ret/.retvec
%   are NOT -- they are exactly what the double-pole basis exists to make
%   artifact-free.  Compare 'double-pole' vs 'local-sp' to see it.
%
%   See also: macos.jones_pupil.

J11 = jp.J(:,:,1,1);  J12 = jp.J(:,:,1,2);
J21 = jp.J(:,:,2,1);  J22 = jp.J(:,:,2,2);
mask = jp.mask;

% ---- hermitian product M = J'*J  (Pauli coefficients, all real) ---------
M11 = abs(J11).^2 + abs(J21).^2;
M22 = abs(J12).^2 + abs(J22).^2;
M12 = conj(J11).*J12 + conj(J21).*J22;    % M21 = conj(M12)
t0 = (M11 + M22)/2;
t1 = (M11 - M22)/2;
t2 = real(M12);
t3 = -imag(M12);
tm = sqrt(t1.^2 + t2.^2 + t3.^2);         % |t|; eigs of M = t0 +/- |t|

% ---- transmission + diattenuation (basis-invariant) ---------------------
T = t0;                                    % (s1^2+s2^2)/2
D = tm ./ max(t0, realmin);
Dvec = cat(3, t1./max(t0,realmin), t2./max(t0,realmin), t3./max(t0,realmin));

% ---- polar decomposition:  H = sqrtm(M),  W = J * H^-1 ------------------
% 2x2 psd closed form: sqrtm(M) = (M + sqrt(det M) I) / sqrt(tr M + 2 sqrt(det M))
detM = t0.^2 - tm.^2;
sq   = sqrt(max(detM, 0));
den  = sqrt(max(2*t0 + 2*sq, realmin));
H11 = (M11 + sq)./den;  H22 = (M22 + sq)./den;  H12 = M12./den;
% H^-1 (2x2): detH = sqrt(detM)
detH = max(sq, realmin);
Hi11 =  H22./detH;  Hi22 = H11./detH;  Hi12 = -H12./detH;  Hi21 = -conj(H12)./detH;
W11 = J11.*Hi11 + J12.*Hi21;
W12 = J11.*Hi12 + J12.*Hi22;
W21 = J21.*Hi11 + J22.*Hi21;
W22 = J21.*Hi12 + J22.*Hi22;

% ---- retardance from the unitary part -----------------------------------
% det W = e^{2i psi}; strip psi so W' is SU(2):
%   W' = cos(d/2) I - i sin(d/2) (nhat . sigma)
detW = W11.*W22 - W12.*W21;
psi  = angle(detW)/2;                      % psi mod pi
ph   = exp(-1i*psi);
W11p = W11.*ph;  W12p = W12.*ph;  W21p = W21.*ph;  W22p = W22.*ph;
c    = real(W11p + W22p)/2;                % cos(d/2), sign meaningful
sn1  =  imag(W22p - W11p)/2;               % sin(d/2)*n1
sn2  = -imag(W12p + W21p)/2;               % sin(d/2)*n2
sn3  =  real(W21p - W12p)/2;               % sin(d/2)*n3
snm  = sqrt(sn1.^2 + sn2.^2 + sn3.^2);     % |sin(d/2)| >= 0
% canonicalize to d in [0, pi]: if c < 0, shift psi by pi (W' -> -W')
neg  = c < 0;
c(neg)   = -c(neg);
sn1(neg) = -sn1(neg);  sn2(neg) = -sn2(neg);  sn3(neg) = -sn3(neg);
psi(neg) = psi(neg) + pi;                  % track the shift...
psi = mod(psi + pi/2, pi) - pi/2;          % ...then re-wrap to [-pi/2, pi/2)
delta = 2*atan2(snm, c);                   % in [0, pi]
snmn  = snm;  snmn(snm < 1e-300) = 1;
retvec = cat(3, delta.*sn1./snmn, delta.*sn2./snmn, delta.*sn3./snmn);
ambiguous = mask & (delta > pi - 0.2);

% ---- NaN off-mask -------------------------------------------------------
off = ~mask;
T(off) = NaN;  D(off) = NaN;  delta(off) = NaN;  psi(off) = NaN;
for k = 1:3
    a = Dvec(:,:,k);   a(off) = NaN;  Dvec(:,:,k)   = a;
    a = retvec(:,:,k); a(off) = NaN;  retvec(:,:,k) = a;
end

% ---- pupil statistics: mean and variation, separately -------------------
stat = @(x) deal(mean(x(mask)), rms_(x(mask) - mean(x(mask))));
[mT,  vT ] = stat(T);
[mD,  vD ] = stat(D);
[mR,  vR ] = stat(delta);
mDv = zeros(3,1);  vDv = zeros(3,1);  mRv = zeros(3,1);  vRv = zeros(3,1);
for k = 1:3
    a = Dvec(:,:,k);   [mDv(k), vDv(k)] = stat(a);
    a = retvec(:,:,k); [mRv(k), vRv(k)] = stat(a);
end

pm = struct('T', T, 'D', D, 'Dvec', Dvec, ...
            'ret', delta, 'retvec', retvec, 'phase', psi, ...
            'ambiguous', ambiguous, 'mask', mask, ...
            'mean',    struct('T',mT,'D',mD,'Dvec',mDv,'ret',mR,'retvec',mRv), ...
            'var_rms', struct('T',vT,'D',vD,'Dvec',vDv,'ret',vR,'retvec',vRv));
end

function r = rms_(x)
r = sqrt(mean(x.^2));
end
