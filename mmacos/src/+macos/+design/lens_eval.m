function F = lens_eval(E, P, theta_deg, opts)
%MACOS.DESIGN.LENS_EVAL  RMS wavefront error and distortion of ONE field of a loaded lens deck.
%   F = macos.design.lens_eval(E, P, theta_deg) evaluates the deck
%   macos.design.lens_emit wrote (E; macos.load_rx(E.path) first) with the
%   first-order layout P (macos.design.lens_pupils) at the semi-field
%   THETA_DEG (+y; 0 for a finite object) and returns
%
%     F.rms_waves   RMS wavefront error, waves at the deck's Wavelen (or
%                   opts.lambda_nm after a set_src_wvl), on the exit-pupil
%                   reference sphere centred on the CHIEF RAY's image point --
%                   the chief-ray reference (macos.opd_ref('chief'), the
%                   engine default) -- over every ray that reaches the sphere,
%                   piston removed, tilt and focus KEPT (the image plane is
%                   the deck's: a focus error is an error)
%     F.rms_waves_opd  the same from the engine's opd() map + opd_mask (the
%                   map omits the chief ray itself: 0.1 % lower at 741 rays)
%     F.nLaunched F.nLost   rays launched / lost before the image plane
%     F.yR F.yP F.dist_pct  real and paraxial chief heights at the image,
%                   distortion 100 (yR - yP) / yP (NaN on axis)
%     F.xp F.xp_virtual     the sphere used (FEX's, or its mirror when the
%                   exit pupil lies beyond the image plane)
%     F.W F.M       the opd() map and its validity mask
%
%   Per field: set_src_fov (infinite object; the source plane kept upstream on
%   the chief through the entrance-pupil centre) -> macos.stop(E.iStop) ->
%   macos.fex(1) -> trace to the XP Return -> trace to the image.  This is
%   challenges/iodc2010/iodc2010_eval without the challenge's pass/fail
%   rules; its gate (tIodc2010) is the engine against an independent tracer.
arguments
    E struct
    P struct
    theta_deg (1,1) double {mustBeNonnegative}
    opts.lambda_nm (1,1) double = NaN
    opts.standoff (1,1) double = NaN
end
lam_mm = E.lambda_mm;  if isfinite(opts.lambda_nm), lam_mm = opts.lambda_nm * 1e-6; end
macos.opd_ref('chief');
th = deg2rad(theta_deg);
if E.finite
    assert(theta_deg == 0, 'lens_eval: a finite-object deck is evaluated on axis only');
else
    d = [0; sin(th); cos(th)];
    sb = opts.standoff;  if isnan(sb), sb = E.standoff; end
    s_back = max(sb, P.z_ep - E.z_src);
    macos.set_src_fov('src_dir', d, 'src_pos', [0; 0; P.z_ep] - s_back * d);
end
macos.stop(E.iStop);                 % the chief through the stop centre (FEX needs a stop)
x = macos.fex(1);
ti = macos.trace(E.iImgRet);  ri = macos.get_ray_info(ti.nRays);
C_img = ri.pos(:, 1);  dz = -ri.dir(:, 1) / norm(ri.dir(:, 1));
F.xp_virtual = norm((x.vpt - x.rad * x.psi) - C_img) > 1e-3 * max(1, abs(x.rad));
if F.xp_virtual
    r = abs(x.rad);
    x.vpt = C_img - r * dz;  x.psi = dz;  x.rad = -r;
    macos.set_xp(x.vpt, x.psi, x.rad);
end
tr = macos.trace(E.iXP);
rx = macos.get_ray_info(tr.nRays);
okx = rx.ok_trace(:).' & rx.ok_pass(:).';
Wx = rx.opl(okx) - rx.opl(1);
F.rms_mm = sqrt(mean((Wx - mean(Wx)).^2));
F.rms_waves = F.rms_mm / lam_mm;
W = macos.opd();  M = macos.opd_mask();
v = W(M & isfinite(W) & abs(W) < 1e30);
F.rms_waves_opd = sqrt(mean((v - mean(v)).^2)) / lam_mm;
F.W = W;  F.M = M;
tf = macos.trace(E.iFP);
rf = macos.get_ray_info(tf.nRays);
ok = rf.ok_trace(:).' & rf.ok_pass(:).';
F.nLaunched = numel(ok);  F.nLost = nnz(~ok);
F.yR = rf.pos(2, 1);
if E.finite, F.yP = 0; else, F.yP = P.yP_unit * tan(th); end
if theta_deg > 0, F.dist_pct = 100 * (F.yR - F.yP) / F.yP; else, F.dist_pct = NaN; end
F.xp = struct('vpt', x.vpt, 'psi', x.psi, 'rad', x.rad);
F.theta_deg = theta_deg;
end
