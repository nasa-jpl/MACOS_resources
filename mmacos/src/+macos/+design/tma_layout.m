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
%                    IGNORED (used only to seed the search) when 'telecentric'.
%     telecentric    true: place M3 so the EXIT PUPIL is at infinity -- the
%                    chief ray from the stop (M1) leaves M3 with zero slope,
%                    i.e. the stop sits at the front focus of the M2+M3 group.
%                    With R3 fixed by the system f/# at each M3 position, this
%                    is one condition on m3_behind_m, solved as a 1-D root
%                    (fzero on the paraxial chief's exit slope).  The image
%                    then feeds a telecentric slit (a Dyson spectrometer:
%                    dyson5 round 4, 2026-10-04 -- the d205 section's exit
%                    pupil 22 mm from its image put the chief 12-46 deg off
%                    the slit normal; no figure DOF can move a pupil).
%                    Default false.
%     stop           'M1' (default: the entrance aperture, what Telescope
%                    emits as ApStop) or 'M2' -- the element the chief is
%                    traced from for the telecentric condition.  MEASURED
%                    2026-10-04 (dyson5 parent, f/1.8, m2 2.5..8): with a REAL
%                    intermediate focus between M2 and M3 NO stop placement is
%                    telecentric (M3's front focus lies between that focus and
%                    M3; the chief exit slope is -11..-39 per unit field at
%                    every M3 position behind the focus).  The telecentric
%                    Korsch is the VIRTUAL-intermediate-image regime -- M3 in
%                    front of the intermediate focus, between M2 and M1 (the
%                    Cook / EMIT form) -- and there BOTH stops solve: 'M1'
%                    gives R3 0.1667 m, t2 46 mm (engine: chief directions
%                    parallel to 1e-8 rad, the stop the Telescope already
%                    emits); 'M2' gives t2 = R3/2 exactly (M2 at M3's front
%                    focus; the deck must then carry the stop at M2 --
%                    macos.stop(2) on a Telescope deck is CCMac's open item).
%
%   Outputs:
%     R     [R1 R2 R3] vertex radii (magnitudes; KrElt=-|R| emitted).
%     t     [t1 t2] vertex spacings M1->M2, M2->M3.
%     info  struct: f1,R1,R2,R3,t1,t2,m2,m3,EFL,fnum,int_focus_z (target),
%           int_focus_z_check (traced), m3_z (M3 vertex z), chief_exit_slope
%           (paraxial, per unit field slope at the stop; 0 = telecentric),
%           exit_pupil_from_m3 (m, along the exit beam; Inf = telecentric),
%           telecentric (the flag).
%
%   See also: macos.design.seidel_seed, macos.design.Telescope/add_mirror.
    arguments
        D            (1,1) double {mustBePositive}
        primary_fnum (1,1) double {mustBePositive}
        system_fnum  (1,1) double {mustBePositive}
        opts.secondary_mag (1,1) double = 8
        opts.int_focus_m   (1,1) double = NaN
        opts.m3_behind_m   (1,1) double = NaN
        opts.telecentric   (1,1) logical = false
        opts.stop          (1,:) char {mustBeMember(opts.stop, {'M1','M2'})} = 'M1'
    end
    kstop = 1 + strcmp(opts.stop, 'M2');
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
    if ~(zint < m3b) && ~opts.telecentric
        error('macos:design:tma_layout:order', ...
            ['intermediate focus (z=%.4g) must be BEFORE M3 (z=%.4g) -- raise ', ...
             'm3_behind_m or move int_focus_m earlier.'], zint, m3b);
    end
    if opts.telecentric
        % --- M3 position from the telecentric condition: the chief from the
        % stop centre leaves M3 with zero slope.  R3 follows the f/# at each
        % candidate m3b (m3_solve_), so g(m3b) is one smooth function; bracket
        % on a scan of M3 positions behind the intermediate focus, then fzero.
        g = @(z) chief_slope_(R1, R2, m3_solve_(R1, R2, t1, z + t1, D, system_fnum), t1, z + t1, kstop);
        % The scan starts right behind M2's vertex region (M3 in FRONT of the
        % intermediate focus = a VIRTUAL intermediate image for M3) and runs
        % behind it: with a REAL intermediate focus between M2 and M3, M3's
        % front focus lies between that focus and M3 and a stop at M2 can never
        % sit on it (the paraxial identity t2 = f3 has no positive solution);
        % the telecentric Korsch -- EMIT's form -- is the virtual-image regime,
        % which the non-telecentric order check above deliberately forbids.
        zs = linspace(-t1 + 0.05*D, 4*D, 320);
        gv = nan(size(zs));
        for i = 1:numel(zs)
            try, gv(i) = g(zs(i)); catch, end
        end
        i0 = find(gv(1:end-1).*gv(2:end) < 0, 1);
        if isempty(i0)
            error('macos:design:tma_layout:telecentric', ...
                ['no M3 position behind the intermediate focus makes the exit pupil ', ...
                 'telecentric for this feed (chief exit slope %.3g..%.3g over z %.3g..%.3g m); ', ...
                 'adjust secondary_mag / int_focus_m.'], min(gv), max(gv), zs(1), zs(end));
        end
        m3b = fzero(g, [zs(i0) zs(i0+1)], optimset('TolX', 1e-12));
    end
    t2 = m3b + t1;                         % z_M3 = -t1 + t2 = m3b
    R3 = m3_solve_(R1, R2, t1, t2, D, system_fnum);
    uc = chief_slope_(R1, R2, R3, t1, t2, kstop);
    xp = chief_xp_(R1, R2, R3, t1, t2, kstop);
    info = struct('f1',f1, 'R1',R1, 'R2',R2, 'R3',R3, 't1',t1, 't2',t2, ...
        'm2',m2, 'm3', system_fnum/(primary_fnum*m2), 'EFL', system_fnum*D, ...
        'fnum', system_fnum, 'int_focus_z', zint, ...
        'int_focus_z_check', -t1 + cassfocus_(R1, R2, t1, D), 'm3_z', m3b, ...
        'chief_exit_slope', uc, 'exit_pupil_from_m3', xp, 'telecentric', opts.telecentric, ...
        'stop', opts.stop);
    R = [R1 R2 R3];
    t = [t1 t2];
end

% =====================================================================
function R3 = m3_solve_(R1, R2, t1, t2, D, system_fnum)
%M3_SOLVE_  R3 from the system-f/# constraint at a given M3 position.  The
%   exit marginal slope um is LINEAR in c3 = 1/R3; the real intermediate focus
%   between M2 and M3 means the marginal ray has CROSSED the axis, flipping the
%   parity of the exit slope (the unfolded trace exposes this; the legacy n-flip
%   masked it and picked the wrong branch in the aggressive regime).  So solve
%   um = +-1/(2*system_fnum) and take the CONCAVE root (c3 > 0) -- the physical
%   Korsch tertiary that reimages the real intermediate focus.
    umA = tma_marg_(R1, R2, 1e30, t1, t2, D);    % M3 flat
    umB = tma_marg_(R1, R2, R1,   t1, t2, D);    % M3 = R1 (probe slope)
    dum_dc3 = (umB - umA)*R1;                     % d(um)/d(c3), c3 = 1/R3
    R3 = NaN;
    for s = [1 -1]
        c3 = (s/(2*system_fnum) - umA)/dum_dc3;
        if c3 > 0, R3 = 1/c3;  break; end         % concave M3
    end
    if isnan(R3) || ~isfinite(R3)
        error('macos:design:tma_layout:m3', ...
            'no concave M3 reaches f/%.2f for this feed (adjust m3_behind_m / mag).', ...
            system_fnum);
    end
end
function uc = chief_slope_(R1, R2, R3, t1, t2, kstop)
%CHIEF_SLOPE_  Unfolded paraxial CHIEF ray (y = 0 at the stop element kstop,
%   unit slope) -- its exit slope after M3.  Zero = exit pupil at infinity.
    c  = [1/R1, -1/R2, 1/R3];  tk = [t1 t2 0];  y = 0;  u = 1;
    for k = kstop:3
        u = u - 2.0*c(k)*y;  y = y + tk(k)*u;
    end
    uc = u;
end
function xp = chief_xp_(R1, R2, R3, t1, t2, kstop)
%CHIEF_XP_  Exit-pupil distance from M3 along the exit beam (the chief's axis
%   crossing after M3); Inf when telecentric.
    c  = [1/R1, -1/R2, 1/R3];  tk = [t1 t2 0];  y = 0;  u = 1;
    for k = kstop:3
        u = u - 2.0*c(k)*y;  if k < 3, y = y + tk(k)*u; end
    end
    if u == 0, xp = Inf; else, xp = -y/u; end
end
function s = tern_(c, a, b), if c, s = a; else, s = b; end, end
function um = tma_marg_(R1, R2, R3, t1, t2, D)
%TMA_MARG_  Unfolded paraxial marginal-ray final slope through M1,M2,M3.
%   EFL = -(D/2)/um.  The secondary is CONVEX by geometry (it sits before
%   the M1 focus), so its curvature is SIGNED negative -- the n-flip |radii|
%   recurrence mis-places the M3 relay of a convex-secondary reimager (the
%   same bug fixed in seidel_seed's convex path); the unfolded signed-c
%   transfer (convex = negative lens, u' = u - 2*c*y) is correct.
    c  = [1/R1, -1/R2, 1/R3];          % M1 concave, M2 CONVEX, M3 concave
    tk = [t1 t2 0];  y = D/2;  u = 0;
    for k = 1:3
        u = u - 2.0*c(k)*y;  y = y + tk(k)*u;
    end
    um = u;
end

function s = cassfocus_(R1, R2, t1, D)
%CASSFOCUS_  M1+M2 marginal-ray axis crossing past M2 (intermediate focus
%   distance), magnitude -- to verify the Cassegrain solve.  Unfolded
%   signed-c (M2 convex), like tma_marg_.
    c = [1/R1, -1/R2];  tk = [t1 0];  y = D/2;  u = 0;
    for k = 1:2
        u = u - 2.0*c(k)*y;  y = y + tk(k)*u;
    end
    s = abs(-y/u);
end
