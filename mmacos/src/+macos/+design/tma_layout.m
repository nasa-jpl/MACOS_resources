function [R, t, info] = tma_layout(D, primary_fnum, system_fnum, opts)
%TMA_LAYOUT  Generic on-axis Korsch TMA first-order layout (j18 / JWST form),
%   with the intermediate focus placed where YOU want it for packaging.
%
%   [R, t, info] = macos.design.tma_layout(D, primary_fnum, system_fnum, ...)
%
%   Stage 1 -- M1+M2 Cassegrain feed.  Primary f1 = primary_fnum*D (R1=2*f1).
%   The convex secondary (magnification secondary_mag) forms a REAL intermediate
%   focus at the chosen axial position int_focus_m -- the field-stop / metrology
%   plane and the natural FOLD point.  Closed-form Cassegrain solve (z_int is the
%   focus z; M2 sits at z=-t1, the focus d_int=z_int+t1 past it):
%       t1 = (m2*f1 - z_int)/(m2 + 1)        % t1 < f1 => convex secondary
%       R2 = 2*(z_int + t1)/(m2 - 1)         % > 0 magnitude (convex by geometry)
%   A FAST feed (smaller m2) puts the intermediate focus EARLIER -- before M1 --
%   exactly like j18mono (m2~7.6 -> focus ~0.2*D before M1, where its FSM folds).
%   A slow feed (large m2) drags it behind M1 toward M3 (near-telecentric).
%
%   Stage 2 -- M3 relay.  m3 = system_fnum/(primary_fnum*m2) reimages the
%   intermediate focus to the final focus.  M3 sits m3_behind_m behind the
%   primary; R3 follows from a single 2-point linear solve for the system f/#.
%
%   Defaults are j18-like: secondary_mag=8, int_focus_m=-0.125*D (BEFORE M1),
%   m3_behind_m=0.6*D.
%
%   Inputs:
%     D            aperture diameter (m)
%     primary_fnum primary f/#  (f1 = primary_fnum*D, R1 = 2*f1)
%     system_fnum  system f/#   (EFL = system_fnum*D)
%   Options:
%     secondary_mag  Cassegrain feed magnification m2 (>1).  Default 8.
%     int_focus_m    intermediate-focus z (m).  NEGATIVE = before M1 (source
%                    side).  Default -0.125*D.
%     m3_behind_m    M3 vertex z, behind the primary (m).  Default 0.6*D.
%
%   Outputs:
%     R     [R1 R2 R3] vertex radii (magnitudes; KrElt=-|R| emitted).
%     t     [t1 t2] vertex spacings M1->M2, M2->M3.
%     info  struct: f1,R1,R2,R3,t1,t2,m2,m3,EFL,fnum,int_focus_z (target),
%           int_focus_z_check (traced), m3_z (M3 vertex z).
%
%   See also: macos.design.seidel_seed, macos.design.Telescope/add_mirror.
    arguments
        D            (1,1) double {mustBePositive}
        primary_fnum (1,1) double {mustBePositive}
        system_fnum  (1,1) double {mustBePositive}
        opts.secondary_mag (1,1) double = 8
        opts.int_focus_m   (1,1) double = NaN
        opts.m3_behind_m   (1,1) double = NaN
    end
    m2 = opts.secondary_mag;
    if m2 <= 1
        error('macos:design:tma_layout:mag', ...
            'secondary_mag must be > 1 (Cassegrain feed); got %.4g.', m2);
    end
    f1 = primary_fnum*D;  R1 = 2*f1;
    zint = opts.int_focus_m;  if isnan(zint), zint = -0.125*D; end
    m3b  = opts.m3_behind_m;  if isnan(m3b), m3b  =  0.600*D; end

    % --- Cassegrain feed: intermediate focus at z = zint ---
    t1    = (m2*f1 - zint)/(m2 + 1);
    d_int = zint + t1;                     % M2 -> intermediate focus (> 0)
    if ~(t1 > 0 && t1 < f1 && d_int > 0)
        error('macos:design:tma_layout:cass', ...
            ['Cassegrain infeasible (t1=%.4g, f1=%.4g, d_int=%.4g): adjust ', ...
             'secondary_mag / int_focus_m.'], t1, f1, d_int);
    end
    R2 = 2*d_int/(m2 - 1);
    if ~(zint < m3b)
        error('macos:design:tma_layout:order', ...
            ['intermediate focus (z=%.4g) must be BEFORE M3 (z=%.4g) -- raise ', ...
             'm3_behind_m or move int_focus_m earlier.'], zint, m3b);
    end
    t2 = m3b + t1;                         % z_M3 = -t1 + t2 = m3b

    % --- M3: R3 from the system-f/# constraint (marginal slope linear in 1/R3) ---
    um_target = -1/(2*system_fnum);
    umA = tma_marg_(R1, R2, 1e30, t1, t2, D);
    umB = tma_marg_(R1, R2, R1,   t1, t2, D);
    R3  = (umB - umA)*R1 / (um_target - umA);

    info = struct('f1',f1, 'R1',R1, 'R2',R2, 'R3',R3, 't1',t1, 't2',t2, ...
        'm2',m2, 'm3', system_fnum/(primary_fnum*m2), 'EFL', system_fnum*D, ...
        'fnum', system_fnum, 'int_focus_z', zint, ...
        'int_focus_z_check', -t1 + cassfocus_(R1, R2, t1, D), 'm3_z', m3b);
    R = [R1 R2 R3];
    t = [t1 t2];
end

% =====================================================================
function um = tma_marg_(R1, R2, R3, t1, t2, D)
%TMA_MARG_  n-flip paraxial marginal-ray final slope through M1,M2,M3.
%   EFL = -(D/2)/um.  Same recurrence as seidel_seed's seidel_trace_.
    Rk = [R1 R2 R3];  tk = [t1 t2 0];  n = 1;  y = D/2;  u = 0;
    for k = 1:3
        c = 1/Rk(k);  np = -n;  phi = (np - n)*c;
        up = (n*u - y*phi)/np;  y = y + tk(k)*up;  u = up;  n = np;
    end
    um = u;
end

function s = cassfocus_(R1, R2, t1, D)
%CASSFOCUS_  M1+M2 marginal-ray axis crossing past M2 (intermediate focus
%   distance), magnitude -- to verify the Cassegrain solve.
    n = 1;  y = D/2;  u = 0;  Rk = [R1 R2];  tk = [t1 0];
    for k = 1:2
        c = 1/Rk(k);  np = -n;  phi = (np - n)*c;
        up = (n*u - y*phi)/np;  y = y + tk(k)*up;  u = up;  n = np;
    end
    s = abs(-y/u);
end
