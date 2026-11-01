function E = camera_emit(C, path, opts)
%MACOS.DESIGN.CAMERA_EMIT  Write a Cassegrain camera (macos.design.cass_camera) as a MACOS deck.
%   E = macos.design.camera_emit(C, path) writes PATH and returns the element
%   map, ready for macos.design.lens_eval(E, E.P, theta_deg):
%     E.iObs      the secondary's shadow (Element= Obscuring at the secondary's
%                 vertex plane; 0 when C.obscuration is 0)
%     E.iM1 E.iM2 the primary (the aperture stop: E.iStop) and the secondary
%     E.iSurf     relay surfaces in light order (all relay elements, flattened)
%     E.iImgRet E.iXP E.iFP   Return at the image, exit-pupil Return, FocalPlane
%     E.P         struct('z_ep', 0, 'yP_unit', C.par.yP_unit) for lens_eval
%     E.nElt E.path E.z_img E.z_src E.lambda_mm E.D E.finite E.standoff
%
%   The deck (units mm, light +z from the source; cass_camera's geometry):
%     * A collimated source (zSource 1e22, Aperture = D) opts.standoff in front
%       of the secondary's plane; the beam fills the primary exactly, so no
%       mirror carries an aperture.  The shadow is a Circle obscuration of radius
%       C.obscuration x D/2 on a passive Obscuring plane, where the secondary
%       and its baffle physically sit.
%     * Mirrors: KcElt = K, KrElt = -R, psiElt toward the centre of curvature
%       (both -z here: the concave primary faces the sky, the convex secondary's
%       centre lies beyond it) -- the macos.design.Telescope convention.
%     * Relay surfaces exactly as macos.design.lens_emit writes them (KrElt = -R
%       with psiElt = +z; a catalog glass writes GlassElt/GlassType Sellmeier/
%       GlassCoef so a re-trace at another Wavelen carries the dispersion; a
%       numeric index writes IndRef only).
%     * The tail: Return at the image, Return at the exit pupil, FocalPlane --
%       lens_emit's.
%
%   opts.calib (empty = no CALIB block): .fields_deg (deg, +y; 0 = on axis),
%   .var_K (mirrors whose conic is free: 1 = primary, 2 = secondary; DOF 8),
%   .var_sep (true: the secondary's position free, DOF 6), .var_R (relay
%   surfaces whose radius is free, numbered over E.iSurf; DOF 7), .var_pos
%   (relay surfaces whose position is free; DOF 6), .var_fp, .max_iters, .tol,
%   .ray_grid, .fov_weights, .fex, .wavelengths_nm (OptWaveLen=).  The primary
%   is element-bound as the stop (ApStop= 0 0).  The CALIB lessons of the lens
%   templates hold (macos_f90/notes/calib.md): one element never carries both a
%   position and a radius DOF; radius solves want OptMaxItrs in the thousands.
%
%   See also macos.design.cass_camera, macos.design.lens_emit, macos.design.lens_eval.
arguments
    C struct
    path (1,:) char
    opts.nGridpts (1,1) double {mustBeInteger, mustBePositive} = 63
    opts.standoff (1,1) double {mustBePositive} = 100
    opts.calib struct = struct()
    opts.title (1,:) char = ''
end
cal = opts.calib;  do_opt = ~isempty(fieldnames(cal));
def = struct('fields_deg', 0, 'var_K', [], 'var_sep', false, 'var_R', [], 'var_pos', [], ...
    'var_fp', false, 'max_iters', 3000, 'tol', [], 'ray_grid', [], 'fov_weights', [], ...
    'fex', true, 'wavelengths_nm', []);
if do_opt
    fn = fieldnames(cal);
    for i = 1:numel(fn)
        assert(isfield(def, fn{i}), 'camera_emit: unknown calib field %s', fn{i});
        def.(fn{i}) = cal.(fn{i});
    end
    cal = def;
end
S = C.surf;  M = C.mirrors;  D = C.D;  lam = C.lambda_nm;
iR = find(strcmp({S.kind}, 'lens'));          % relay surfaces in S
NR = numel(iR);
if do_opt
    both = intersect(cal.var_R, cal.var_pos);
    if ~isempty(both)
        error('camera_emit: relay surface %d carries both a radius and a position DOF (CALIB fails; notes/calib.md)', both(1));
    end
    assert(all(ismember(cal.var_R, 1:NR)) && all(ismember(cal.var_pos, 1:NR)), 'camera_emit: relay surface index out of 1..%d', NR);
    assert(all(ismember(cal.var_K, [1 2])), 'camera_emit: var_K names mirrors 1 (primary) and 2 (secondary)');
end
z_img = C.par.z_img;
z_xp0 = C.par.z_xp;  if ~isfinite(z_xp0), z_xp0 = z_img - 1e6; end
z_xp0 = z_img - abs(z_img - z_xp0);
z_src = -M.sep - opts.standoff;
nm = @(s) sprintf('%.15E', s);

fid = fopen(path, 'w');
assert(fid > 0, 'camera_emit: cannot write %s', path);
cl = onCleanup(@() fclose(fid));
w = @(varargin) fprintf(fid, [varargin{1} '\n'], varargin{2:end});

ttl = opts.title;  if isempty(ttl), ttl = C.name; end
w('%% %s -- emitted by macos.design.cass_camera / camera_emit', ttl);
w('%% Mirrors (%s): R1 %.10g K1 %.10g, R2 %.10g K2 %.10g, separation %.10g mm; D %.6g mm, obscuration %.6g.', ...
    C.telescope.source, M.R1, M.K1, M.R2, M.K2, M.sep, D, C.obscuration);
for j = 1:size(C.relay, 1)
    L = C.relay{j, 1};
    w('%% Relay %d (%s) at z = %.10g: R [%s] t [%s]', j, L.name, C.relay{j, 2}, ...
        strjoin(arrayfun(@(r) sprintf('%.10g', r), L.R, 'UniformOutput', false), ' '), ...
        strjoin(arrayfun(@(v) sprintf('%.10g', v), L.t, 'UniformOutput', false), ' '));
end
w('%% Paraxial at %.6g nm: EFL %.10g mm, f/%.6g, image z %.10g, exit pupil z %.10g.', lam, ...
    C.par.efl, C.par.fno, z_img, C.par.z_xp);
w('%% Source Definition');
w('        ChfRayDir=  0.0E+00  0.0E+00  1.0E+00');
w('        ChfRayPos=  0.0E+00  0.0E+00  %s', nm(z_src));
w('          zSource=  1.0E+22');
w('        BaseUnits=  mm');
w('        WaveUnits=  nm');
w('           IndRef=  1.0E+00');
w('           Extinc=  0.0E+00');
if do_opt && ~isempty(cal.wavelengths_nm)
    wl = [lam, cal.wavelengths_nm(cal.wavelengths_nm ~= lam)];
    for q = 1:numel(wl), w('       OptWaveLen=  %s', nm(wl(q))); end
else
    w('          Wavelen=  %s', nm(lam));
end
w('             Flux=  1.0E+00');
w('         Aperture=  %s', nm(D));
w('         Obscratn=  0.0E+00');
w('         GridType=  Circular');
w('         nGridpts=  %d', opts.nGridpts);
w('            xGrid=  1.0E+00  0.0E+00  0.0E+00');
w('            yGrid=  0.0E+00  1.0E+00  0.0E+00');
hasObs = C.obscuration > 0;
nElt = hasObs + 2 + NR + 3;
iObs = 0;  ie = 0;
iXPe = hasObs + 2 + NR + 2;
if do_opt
    w('        OptTarget=  WFE');
    w('         OptWFElt=  %d', iXPe);
    w('       OptMaxItrs=  %d', cal.max_iters);
    if ~isempty(cal.tol), w('           OptTol=  %s', nm(cal.tol)); end
    if cal.fex, w('           OptFEX=  Yes'); else, w('           OptFEX=  No'); end
    if ~isempty(cal.ray_grid), w('       OptRayGrid=  %d', cal.ray_grid); end
    fo = cal.fields_deg(cal.fields_deg > 0);
    for th = fo(:).'
        d  = [0 sin(deg2rad(th)) cos(deg2rad(th))];
        cp = -((0 - z_src) / d(3)) * d;                   % on the source plane, aimed at the primary vertex
        w('     OptChfRayDir=  %s  %s  %s', nm(d(1)), nm(d(2)), nm(d(3)));
        w('     OptChfRayPos=  %s  %s  %s', nm(cp(1)), nm(cp(2)), nm(cp(3)));
    end
    if ~isempty(cal.fov_weights)
        assert(numel(cal.fov_weights) == 1 + numel(fo) && all(cal.fov_weights > 0), ...
            'camera_emit: fov_weights needs %d positive weights (on-axis first)', 1 + numel(fo));
        w('         OptFOVWt=  %s', strjoin(arrayfun(nm, cal.fov_weights, 'UniformOutput', false), '  '));
    else
        w('         OptFOVWt=  %s', strtrim(repmat('1 ', 1, 1 + numel(fo))));
    end
end
w('%% Element Definitions');
w('             nElt=  %d', nElt);
if hasObs
    ie = ie + 1;  iObs = ie;
    hdr_(w, ie, 'SecObs', 'Obscuring', 'Flat', -1e22, 0, [0 0 1], -M.sep, nm);
    w('           IndRef=  1.0E+00');
    w('           Extinc=  1.0E+22');
    w('            nCoat=  0');
    w('             xObs=  1.0E+00  0.0E+00  0.0E+00');
    w('             nObs=  1');
    w('          ObsType=  Circle');
    w('           ObsVec=  %s  0.0E+00  0.0E+00', nm(C.obscuration * D / 2));
    ftr_(w);
end
mir = {'M1', M.R1, M.K1, 0; 'M2', M.R2, M.K2, -M.sep};
iMir = zeros(1, 2);
for q = 1:2
    ie = ie + 1;  iMir(q) = ie;
    hdr_(w, ie, mir{q, 1}, 'Reflector', 'Conic', -mir{q, 2}, mir{q, 3}, [0 0 -1], mir{q, 4}, nm);
    w('           IndRef=  1.0E+00');
    w('           Extinc=  1.0E+22');
    if do_opt
        m = zeros(1, 8);  m(8) = any(cal.var_K == q);  m(6) = (q == 2) && cal.var_sep;
        if any(m), w('           VarElt=  %s', strjoin(arrayfun(@(v) sprintf('%d', v), m, 'UniformOutput', false), ' ')); end
        if q == 1, w('           ApStop=  0.0E+00  0.0E+00'); end
    end
    w('            nCoat=  0');
    w('             xObs=  1.0E+00  0.0E+00  0.0E+00');
    w('             nObs=  0');
    ftr_(w);
end
iSurf = zeros(1, NR);
for r = 1:NR
    s = S(iR(r));  ie = ie + 1;  iSurf(r) = ie;
    L = C.relay{s.lens, 1};  k = s.k;
    if isfinite(L.R(k))
        hdr_(w, ie, s.name, 'Refractor', 'Conic', -L.R(k), L.K(k), [0 0 1], s.z, nm);
    else
        hdr_(w, ie, s.name, 'Refractor', 'Flat', -1e22, 0, [0 0 1], s.z, nm);
    end
    if k < L.N
        [n_k, G] = macos.design.glass(L.glass{k}, lam);
    else
        n_k = 1;  G = struct('B', []);
    end
    w('           IndRef=  %s', nm(n_k));
    if ~isempty(G.B)
        w('         GlassElt=  %s', G.name);
        w('        GlassType=  Sellmeier');
        w('        GlassCoef=  %s', strjoin(arrayfun(@(v) sprintf('%.12G', v), [G.B G.C], 'UniformOutput', false), '  '));
    end
    w('           Extinc=  0.0E+00');
    if do_opt
        m = zeros(1, 8);
        m(6) = any(cal.var_pos == r);  m(7) = any(cal.var_R == r);
        assert(~(m(7) && ~isfinite(L.R(k))), 'camera_emit: relay surface %d is plano; a radius DOF needs a curved seed', r);
        if any(m), w('           VarElt=  %s', strjoin(arrayfun(@(v) sprintf('%d', v), m, 'UniformOutput', false), ' ')); end
    end
    w('            nCoat=  0');
    w('             xObs=  1.0E+00  0.0E+00  0.0E+00');
    w('             nObs=  0');
    ftr_(w);
end
tail_(w, nm, ie + 1, 'ImgReturn', 'Return', 'Flat', -1e22, z_img, 1e22, 1e22, false);
tail_(w, nm, ie + 2, 'ExitPupil', 'Return', 'Conic', -(z_img - z_xp0), z_xp0, 1e22, abs(z_img - z_xp0), false);
tail_(w, nm, ie + 3, 'Image', 'FocalPlane', 'Flat', -1e22, z_img, 0, 1e22, do_opt && cal.var_fp);
w('');
w('         nOutCord=  5');
w('             Tout=  1.0D+00  0.0D+00  0.0D+00  0.0D+00  0.0D+00  0.0D+00  0.0D+00');
w('                    0.0D+00  1.0D+00  0.0D+00  0.0D+00  0.0D+00  0.0D+00  0.0D+00');
w('                    0.0D+00  0.0D+00  0.0D+00  1.0D+00  0.0D+00  0.0D+00  0.0D+00');
w('                    0.0D+00  0.0D+00  0.0D+00  0.0D+00  1.0D+00  0.0D+00  0.0D+00');
w('                    0.0D+00  0.0D+00  0.0D+00  0.0D+00  0.0D+00  0.0D+00  1.0D+00');
assert(ie + 3 == nElt && ie + 2 == iXPe, 'camera_emit: element count');
E = struct('iObs', iObs, 'iM1', iMir(1), 'iM2', iMir(2), 'iStop', iMir(1), 'iSurf', iSurf, ...
    'iImgRet', ie + 1, 'iXP', ie + 2, 'iFP', ie + 3, 'nElt', nElt, 'path', path, ...
    'z_img', z_img, 'z_src', z_src, 'lambda_mm', lam * 1e-6, 'D', D, 'finite', false, ...
    'standoff', -z_src, 'semidia', inf(1, NR), ...
    'P', struct('z_ep', 0, 'yP_unit', C.par.yP_unit));
end

% -------------------------------------------------------------------------
function hdr_(w, ie, name, elt, srf, kr, kc, psi, z, nm)
w('             iElt=  %d', ie);
w('          EltName=  %s', name);
w('          Element=  %s', elt);
w('          Surface=  %s', srf);
w('            KrElt=  %s', nm(kr));
w('            KcElt=  %s', nm(kc));
w('           psiElt=  %s  %s  %s', nm(psi(1)), nm(psi(2)), nm(psi(3)));
w('           VptElt=  0.0E+00  0.0E+00  %s', nm(z));
w('           RptElt=  0.0E+00  0.0E+00  %s', nm(z));
end

function ftr_(w)
w('           ApType=  None');
w('         PropType=  Geometric');
w('             zElt=  1.0E+22');
w('          nECoord=  -6');
end

function tail_(w, nm, ie, name, elt, srf, kr, z, extinc, zelt, var_fp)
w('             iElt=  %d', ie);
w('          EltName=  %s', name);
w('          Element=  %s', elt);
w('          Surface=  %s', srf);
w('            KrElt=  %s', nm(kr));
w('            KcElt=  0.0E+00');
w('           psiElt=  0.0E+00  0.0E+00  1.0E+00');
w('           VptElt=  0.0E+00  0.0E+00  %s', nm(z));
w('           RptElt=  0.0E+00  0.0E+00  %s', nm(z));
w('           IndRef=  1.0E+00');
w('           Extinc=  %s', nm(extinc));
if var_fp, w('           VarElt=  0 0 0 0 0 1 0 0'); end
w('            nCoat=  0');
w('             xObs=  1.0E+00  0.0E+00  0.0E+00');
w('             nObs=  0');
w('           ApType=  None');
w('         PropType=  Geometric');
w('             zElt=  %s', nm(zelt));
w('          nECoord=  -6');
end
