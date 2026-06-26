function pq = pupil_quality(ep_elt, opts)
%MACOS.PUPIL_QUALITY  Exit-pupil SURFACE quality of the loaded system.
%   pq = macos.pupil_quality(EP_ELT) runs the engine XPS command (the
%   field-differential exit-pupil finder -- FEX generalized to the whole
%   ray grid), reads the per-ray crossing cloud, and fits a low-order
%   Zernike model of the pupil SURFACE in the FLAT entrance-pupil
%   coordinate.  Pure geometry, no OPD.
%
%   For an unobscured/folded system the pupil is generally NOT flat: a
%   non-zero DEFOCUS = pupil curvature, ASTIG = the pupil astigmatism
%   that a coronagraph's flat Lyot/DM conjugate must be corrected for
%   (e.g. by a near-focus field mirror).  TILT + PISTON are reference-
%   frame terms (the FEX chief-ray normal vs the surface's mean plane)
%   and are removable.
%
%   EP_ELT is the exit-pupil return surface (a Return/Reference element;
%   default nElt-1).  A stop must be set (the prescription's ApStop
%   usually suffices).  The nominal trace is restored before returning.
%
%   Returns struct pq (lengths in metres; print is in mm):
%     .vertex 3x1, .normal 3x1     pupil vertex (= FEX) + surface normal
%     .defocus                     pupil curvature (Zernike Z4)
%     .astig [astig0 astig45]      pupil astigmatism
%     .coma  [coma_x coma_y], .spherical
%     .tilt  [tilt_x tilt_y]       reference-frame (removable)
%     .sag_rms, .sag_pv            total surface sag
%     .fit_resid                   Zernike-model residual
%     .chief_vs_fex                chief crossing vs FEX (a self-check)
%     .zern, .names                raw 9-coeff fit
%     .uv (Nx2), .sag (Nx1)        flat entrance coords + per-zone sag
%     .npass, .ep_elt
%
%   See also: macos.xps, macos.get_ray_info, macos.fex.
    arguments
        ep_elt (1,1) double {mustBeInteger,mustBePositive} = macos.num_elt()-1
        opts.quiet (1,1) logical = false
    end
    nE  = macos.num_elt();
    cur = macos.get_src_fov();  cd0 = cur.src_dir(:)/norm(cur.src_dir);

    % reference vertex + normal (FEX chief ray) -- also the chief cross-check
    macos.trace(nE);  f = macos.fex(1);
    V0 = f.vpt(:);  psi0 = f.psi(:)/norm(f.psi);

    % FLAT entrance coords (u,v): transverse part of the M1 hit (sag dropped)
    [a1,a2] = perpbasis_(cd0);
    s1 = macos.trace(1);  r1 = macos.get_ray_info(s1.nRays);
    d1 = r1.pos - r1.pos(:,1);
    U = a1.'*d1;  V = a2.'*d1;  rho = hypot(U,V);  nRay = s1.nRays;

    % engine XPS -> per-ray exit-pupil crossing cloud in the ray buffer
    macos.xps(ep_elt);
    r = macos.get_ray_info(nRay);  P = r.pos;  good = r.ok_trace(:).';

    % decompose sag(u,v) against (V0,psi0); fit low-order Zernikes
    sag = sum((P - V0).*psi0, 1);
    ii  = good & isfinite(sag);
    Rm  = max(rho(ii));
    u = U(ii)/Rm;  v = V(ii)/Rm;  sg = sag(ii).';
    [c, names] = zfit_(u(:), v(:), sg);
    resid = sg - zbasis_(u(:), v(:))*c;

    macos.trace(nE);            % restore nominal grid (xps left the diff field)

    gi = @(nm) c(strcmp(names,nm));
    pq = struct( ...
      'vertex',V0, 'normal',psi0, 'chief_vs_fex',norm(P(:,1)-V0), ...
      'defocus',gi('defocus'), 'astig',[gi('astig0') gi('astig45')], ...
      'coma',[gi('coma_x') gi('coma_y')], 'spherical',gi('spherical'), ...
      'tilt',[gi('tilt_x') gi('tilt_y')], ...
      'sag_rms',rms(sg), 'sag_pv',max(sg)-min(sg), 'fit_resid',rms(resid), ...
      'zern',c(:).', 'names',{names}, 'uv',[U(ii).' V(ii).'], ...
      'sag',sag(ii).', 'npass',nnz(ii), 'ep_elt',ep_elt);

    if ~opts.quiet
        fprintf(['exit-pupil surface @ elt %d  vertex [%.3f %.3f %.3f] m\n' ...
                 '  defocus %+.3f  astig [%+.3f %+.3f]  coma [%+.3f %+.3f]' ...
                 '  sph %+.3f  (mm)\n' ...
                 '  sag %.3f mm RMS (%.3f P-V)  fit resid %.3f mm' ...
                 '  chief-vs-FEX %.1e m\n'], ep_elt, V0, ...
                 pq.defocus*1e3, pq.astig*1e3, pq.coma*1e3, pq.spherical*1e3, ...
                 pq.sag_rms*1e3, pq.sag_pv*1e3, pq.fit_resid*1e3, pq.chief_vs_fex);
    end
end

% ---- helpers -------------------------------------------------------------
function [a1,a2] = perpbasis_(n)
    n = n/norm(n);  t = [1;0;0];  if abs(n.'*t) > 0.9, t = [0;1;0]; end
    a1 = t - (n.'*t)*n;  a1 = a1/norm(a1);  a2 = cross(n,a1);
end
function B = zbasis_(u,v)            % low-order Zernike basis (unit disk)
    r2 = u.^2 + v.^2;
    B = [ones(size(u)), u, v, 2*r2-1, u.^2-v.^2, 2*u.*v, ...
         (3*r2-2).*u, (3*r2-2).*v, 6*r2.^2-6*r2+1];
end
function [c,names] = zfit_(u,v,s)
    names = {'piston','tilt_x','tilt_y','defocus','astig0','astig45', ...
             'coma_x','coma_y','spherical'};
    c = zbasis_(u,v) \ s;
end
