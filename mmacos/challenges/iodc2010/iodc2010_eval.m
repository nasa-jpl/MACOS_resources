function F = iodc2010_eval(E, P, p, theta_deg, opts)
%IODC2010_EVAL  Score ONE field of the loaded green-lens deck the way the IODC did.
%   F = iodc2010_eval(E, P, p, theta_deg) evaluates the deck iodc2010_emit
%   wrote (and macos.load_rx loaded) at the semi-field THETA_DEG and returns
%
%     F.rms_waves   RMS wavefront error, waves at p.lambda_nm, over every ray that
%                   reaches the sphere (the chief included): the OPD on the
%                   exit-pupil reference sphere centred on the CHIEF RAY's
%                   image point (MACOS's chief-ray reference, FEX-placed),
%                   piston removed (population RMS about the mean), tilt and
%                   focus NOT removed -- the 2010 problem's convention,
%                   defined in the 2006 paper, Sec. 4.4.
%     F.nLaunched   rays launched over the paraxial entrance-pupil disc
%     F.nLost       rays that did not reach the image plane unclipped (surface
%                   miss, TIR, or any clear aperture incl. the stop); the field
%                   is VIGNETTED when nLost > 0 ("stop fully filled" = every
%                   launched ray passes)
%     F.yR, F.yP    real and paraxial chief-ray height at the image plane (mm)
%     F.dist_pct    100 (yR - yP) / yP   (NaN on axis)
%     F.xp          the reference sphere used: .vpt .psi .rad (FEX's, or the mirror
%                   sphere when the exit pupil lies beyond the image plane -- below)
%     F.xp_virtual  true when FEX's sphere had to be re-placed (the pupil is virtual,
%                   beyond the image plane); the OPD is then read at the near
%                   crossing of the same sphere, <= 3.5e-4 wave from the far one
%     F.pass        rms <= p.rms_limit & |dist| <= p.dist_limit & nLost == 0
%     F.reason      '' | 'rms' | 'distortion' | 'vignetting' (the first that fails)
%
%   With opts.rays = true the per-ray data the gate compares against the
%   independent tracer are returned too (F.ray: the object-space start points
%   and direction, the engine's OPL and position on the sphere per ray, the
%   chief's image point and direction).
%
%   Sequence per field (engine record notes/pupil_fex_stop.md):
%     set_src_fov (field direction; the source plane kept upstream)
%     -> macos.stop(E.iStop)   (chief aimed through the stop centre, the bundle
%                               = the paraxial-EPD disc about that chief)
%     -> macos.fex(1)          (XP sphere on the chief, radius to the image plane)
%     -> trace to the XP, opd + opd_mask  (never W ~= 0)
%     -> trace to the image plane for the ray count and the chief's height.
arguments
    E struct
    P struct
    p struct
    theta_deg (1,1) double {mustBeNonnegative}
    opts.rays (1,1) logical = false
end
th = deg2rad(theta_deg);
d  = [0; sin(th); cos(th)];
% keep the source plane upstream of everything, on the chief through the EP centre
s_back = max(p.src_standoff_mm, P.z_ep - E.z_src);
src = [0; 0; P.z_ep] - s_back * d;
macos.set_src_fov('src_dir', d, 'src_pos', src);
macos.stop(E.iStop);
x  = macos.fex(1);
% FEX's sphere has its centre at vpt - rad * psi.  When the exit pupil lies BEYOND the
% image plane (a virtual pupil: the stop inside the rear group's front focal distance, or
% ahead of a diverging front), FEX's SIGNED station-to-FP distance puts that centre on
% the far side of the station and every ray misses the sphere (the trace to the FP, which
% follows the XP Return, dies with it).  The reference sphere the metric wants is still
% the one centred on the chief ray's image point with the pupil's radius; it is re-placed
% here as the MIRROR sphere -- same centre, same radius, the vertex r BEFORE the image
% along the chief, the geometry FEX writes for a real pupil -- and the engine evaluates
% its near crossing.  The near and far crossings of one sphere differ by rho^2 / r
% (second order in the ray's transverse aberration): 2e-7 wave at D 1 mm and 3.5e-4
% wave at D 5 mm on the retrofocus trains that found this (xp_virtual, 2026-10-31); the
% engine on the mirror sphere matches the independent tracer to 2e-9 wave (the gate's
% tolerance), tIodc2010 pins both.  The chief's image point comes from the image Return,
% which precedes the XP Return in the deck (a Return reverses the chief's direction).
ti = macos.trace(E.iImgRet);  ri = macos.get_ray_info(ti.nRays);
C_img = ri.pos(:, 1);  dz = -ri.dir(:, 1) / norm(ri.dir(:, 1));
F.xp_virtual = norm((x.vpt - x.rad * x.psi) - C_img) > 1e-3 * max(1, abs(x.rad));
if F.xp_virtual
    r = abs(x.rad);
    x.vpt = C_img - r * dz;  x.psi = dz;  x.rad = -r;
    macos.set_xp(x.vpt, x.psi, x.rad);
end

tr = macos.trace(E.iXP);
% The RMS is taken over every launched ray that reaches the sphere, the chief
% (ray 1, the reference) included, from the per-ray optical path.  The
% engine's opd() map is the same quantity on the same rays EXCEPT that it
% never writes the chief (OPD loops iRay = 2..nRay; notes/opd_reference.md);
% that map's RMS is kept beside it as F.rms_waves_opd (0.1 % lower on a
% 741-ray grid, 1e-4 at 7081).  The gate (iodc2010_gate) pins both.
rx  = macos.get_ray_info(tr.nRays);
okx = rx.ok_trace(:).' & rx.ok_pass(:).';
Wx  = rx.opl(okx) - rx.opl(1);
rms_mm = sqrt(mean((Wx - mean(Wx)).^2));
F.rms_waves = rms_mm / E.lambda_mm;
F.rms_mm    = rms_mm;
F.nValid    = nnz(okx);
W  = macos.opd();  M = macos.opd_mask();
v  = W(M & isfinite(W) & abs(W) < 1e30);
F.rms_waves_opd = sqrt(mean((v - mean(v)).^2)) / E.lambda_mm;
F.nValid_opd    = numel(v);
if opts.rays
    F.ray.opl_xp = rx.opl(:).';
    F.ray.pos_xp = rx.pos;
    F.ray.ok_xp  = okx;
end

tf = macos.trace(E.iFP);
rf = macos.get_ray_info(tf.nRays);
ok = rf.ok_trace(:).' & rf.ok_pass(:).';
F.nLaunched = numel(ok);
F.nLost     = nnz(~ok);
F.yR  = rf.pos(2, 1);                 % the field is along +y
F.yP  = P.yP_unit * tan(th);
if theta_deg > 0, F.dist_pct = 100 * (F.yR - F.yP) / F.yP; else, F.dist_pct = NaN; end
F.xp  = struct('vpt', x.vpt, 'psi', x.psi, 'rad', x.rad);
F.theta_deg = theta_deg;
F.nRaysXP   = tr.nRays;  F.nRaysFP = tf.nRays;
F.rmsWFE_engine_mm = tr.rmsWFE;

F.reason = '';
if F.nLost > 0
    F.reason = 'vignetting';
elseif F.rms_waves > p.rms_limit
    F.reason = 'rms';
elseif theta_deg > 0 && abs(F.dist_pct) > p.dist_limit
    F.reason = 'distortion';
end
F.pass = isempty(F.reason);

if opts.rays
    % object-space start of every ray: the engine's crossing of element 1,
    % backed up along the (common, collimated) direction; the chief's image
    % point and final direction from the image-plane trace.
    fov = macos.get_src_fov();
    F.ray.dir_in = fov.src_dir(:) / norm(fov.src_dir);
    t1 = macos.trace(1);  r1 = macos.get_ray_info(t1.nRays);
    F.ray.pos_e1 = r1.pos;
    F.ray.ok_e1  = r1.ok_trace(:).';
    F.ray.pos_fp = rf.pos;
    F.ray.opl_fp = rf.opl(:).';
    F.ray.opl_e1 = r1.opl(:).';
    F.ray.dir_fp = rf.dir;
    F.ray.ok_fp  = ok;
    F.ray.chief_img = rf.pos(:, 1);
    F.ray.chief_dir_img = rf.dir(:, 1) / norm(rf.dir(:, 1));
end
end
