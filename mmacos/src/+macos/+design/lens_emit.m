function E = lens_emit(L, P, path, opts)
%MACOS.DESIGN.LENS_EMIT  Write a lens and its stop as a MACOS evaluation deck.
%   E = macos.design.lens_emit(L, P, path) writes PATH (a .in) for the Lens L
%   with the first-order layout P (macos.design.lens_pupils) and returns the
%   element map:
%     E.iSurf(k)  element index of lens surface k
%     E.iStop     the stop plane (a passive Reference, centred on the axis)
%     E.iImgRet   the flat Return at the image plane
%     E.iXP       the exit-pupil Return sphere (FEX re-places it per field)
%     E.iFP       the FocalPlane
%     E.nElt E.path E.z_img E.z_src E.lambda_mm E.D E.finite E.semidia
%
%   The deck is the IODC 2010 evaluator's (challenges/iodc2010/iodc2010_emit,
%   where it is gated against an independent skew-ray tracer to 2e-10 wave),
%   generalized to any Lens:
%     * BaseUnits mm, WaveUnits nm, Wavelen = P.lambda_nm; light along +z,
%       surface 1's vertex at z = 0.  The lens's lengths are mm.
%     * A CODE V radius R is KrElt = -R with psiElt = (0,0,1); a conic K is
%       KcElt; plano is Surface= Flat.
%     * A catalog glass after surface k writes IndRef = n(lambda) AND
%       GlassElt= / GlassType= Sellmeier / GlassCoef= with the engine table's
%       row (macos.design.glass): the engine recomputes the index from the row
%       at every trace, so re-tracing the deck at another Wavelen
%       (macos.set_src_wvl) traces the dispersion.  A numeric medium is IndRef.
%     * The stop: Element= Reference, Surface= Flat, in the medium of its gap.
%       opts.stop_clips false (default) = no aperture there (the beam is the
%       paraxial entrance pupil); true = ApType= Circular at the stop radius.
%     * Clear apertures: ApType= Circular from L.semidia when set
%       (opts.apertures), else None.
%     * The source: collimated (zSource 1e22, Aperture = P.D) for an infinite
%       object, opts.standoff in front of the first element; a POINT source on
%       axis for a finite one (ChfRayPos just past the object point, zSource
%       negative = diverging, Aperture = P.cone, the full angle the stop
%       admits).  GridType Circular, nGridpts opts.nGridpts.
%     * The tail: Return at the image, Return at the exit pupil (the sphere
%       centred on the image point), FocalPlane -- the manual's double-pass
%       pattern; a pupil beyond the image is written as its mirror sphere
%       (same centre and radius, vertex before the image), as iodc2010_emit.
%
%   opts.calib (struct; empty = no CALIB block) writes the CALIB keywords:
%     .fields_deg  design fields (deg, +y); 0 = the header's on-axis field
%     .var_R       surfaces whose RADIUS is free (VarElt DOF 7)
%     .var_pos     surfaces whose AXIAL POSITION is free (DOF 6, PIST)
%     .var_fp      the image plane's position free (default false)
%     .max_iters .tol .ray_grid .fov_weights .fex (OptFEX, default true)
%     .wavelengths_nm  solve over these wavelengths (OptWaveLen=, P.lambda_nm
%                  first); the engine traces each with the GlassCoef rows
%   The stop is element-bound (ApStop= 0 0) so the engine re-aims the chief
%   through its centre at every trace as the surfaces move.
arguments
    L (1,1) macos.design.Lens
    P struct
    path (1,:) char
    opts.nGridpts (1,1) double = 97
    opts.standoff (1,1) double = 100
    opts.stop_clips (1,1) logical = false
    opts.apertures = []
    opts.calib struct = struct()
    opts.title (1,:) char = ''
end
N = L.N;
cal = opts.calib;  do_opt = ~isempty(fieldnames(cal));
apert = opts.apertures;  if isempty(apert), apert = ~isempty(L.semidia); end
fin = isfinite(P.object);
def = struct('fields_deg', 0, 'var_R', [], 'var_pos', [], 'var_fp', false, 'max_iters', 30, ...
    'tol', [], 'ray_grid', [], 'fov_weights', [], 'fex', true, 'wavelengths_nm', []);
if do_opt
    fn = fieldnames(cal);
    for i = 1:numel(fn)
        assert(isfield(def, fn{i}), 'lens_emit: unknown calib field %s', fn{i});
        def.(fn{i}) = cal.(fn{i});
    end
    cal = def;
end
assert(~(do_opt && fin && any(cal.fields_deg ~= 0)), 'lens_emit: a finite-object CALIB deck is on axis only');

% ---- elements in z order (the stop ahead of a surface at the same z)
zs = [L.z, P.z_s];  kind = [ones(1, N), 0];  kk = [1:N, 0];
[~, ord] = sort(zs + 1e-9 * kind);
kind = kind(ord);  kk = kk(ord);  zs = zs(ord);

z_img = P.z_img;
z_xp0 = P.z_xp;  if ~isfinite(z_xp0), z_xp0 = z_img - 1e6; end
z_xp0 = z_img - abs(z_img - z_xp0);
z_front = min([L.z, P.z_s]);
if fin
    dlt = min(1, 0.01 * P.object);
    z_src = P.z_obj + dlt;
    assert(z_src < z_front, 'lens_emit: the object is inside the lens');
else
    standoff = opts.standoff;
    if do_opt, standoff = max(standoff, z_img - z_front); end   % a solve that moves surface 1 upstream keeps its rays
    z_src = min(P.z_ep, z_front) - standoff;
end
nm = @(s) sprintf('%.15E', s);

fid = fopen(path, 'w');
assert(fid > 0, 'lens_emit: cannot write %s', path);
cl = onCleanup(@() fclose(fid));
w = @(varargin) fprintf(fid, [varargin{1} '\n'], varargin{2:end});

ttl = opts.title;  if isempty(ttl), ttl = L.name; end
w('%% %s -- emitted by macos.design.lens_emit', ttl);
w('%% Lens %s: R [%s] mm, t [%s] mm, media {%s}; stop at z = %.6g mm.', L.name, ...
    strjoin(arrayfun(@(r) sprintf('%.6g', r), L.R, 'UniformOutput', false), ' '), ...
    strjoin(arrayfun(@(v) sprintf('%.6g', v), L.t, 'UniformOutput', false), ' '), ...
    strjoin(cellfun(@medium_, L.glass, 'UniformOutput', false), ', '), P.z_s);
if ~isempty(L.source), w('%% Source of the prescription: %s', L.source); end
w('%% Light travels +z, surface 1 at z = 0; CODE V radii, KrElt = -R.');
w('%% Source Definition');
w('        ChfRayDir=  0.0E+00  0.0E+00  1.0E+00');
w('        ChfRayPos=  0.0E+00  0.0E+00  %s', nm(z_src));
if fin
    w('          zSource=  %s', nm(-dlt));
else
    w('          zSource=  1.0E+22');
end
w('        BaseUnits=  mm');
w('        WaveUnits=  nm');
w('           IndRef=  1.0E+00');
w('           Extinc=  0.0E+00');
if do_opt && ~isempty(cal.wavelengths_nm)
    % a CALIB over several wavelengths: the FIRST is the deck's Wavelen (P.lambda_nm leads)
    wl = [P.lambda_nm, cal.wavelengths_nm(cal.wavelengths_nm ~= P.lambda_nm)];
    for q = 1:numel(wl), w('       OptWaveLen=  %s', nm(wl(q))); end
else
    w('          Wavelen=  %s', nm(P.lambda_nm));
end
w('             Flux=  1.0E+00');
if fin, w('         Aperture=  %s', nm(P.cone)); else, w('         Aperture=  %s', nm(P.D)); end
w('         Obscratn=  0.0E+00');
w('         GridType=  Circular');
w('         nGridpts=  %d', opts.nGridpts);
w('            xGrid=  1.0E+00  0.0E+00  0.0E+00');
w('            yGrid=  0.0E+00  1.0E+00  0.0E+00');
nElt = N + 4;
if do_opt
    w('        OptTarget=  WFE');
    w('         OptWFElt=  %d', N + 3);
    w('       OptMaxItrs=  %d', cal.max_iters);
    if ~isempty(cal.tol), w('           OptTol=  %s', nm(cal.tol)); end
    if cal.fex, w('           OptFEX=  Yes'); else, w('           OptFEX=  No'); end
    if ~isempty(cal.ray_grid), w('       OptRayGrid=  %d', cal.ray_grid); end
    fo = cal.fields_deg(cal.fields_deg > 0);
    for th = fo(:).'
        d  = [0 sin(deg2rad(th)) cos(deg2rad(th))];
        cp = [0 0 P.z_ep] - ((P.z_ep - z_src) / d(3)) * d;       % every field starts on the source plane
        w('     OptChfRayDir=  %s  %s  %s', nm(d(1)), nm(d(2)), nm(d(3)));
        w('     OptChfRayPos=  %s  %s  %s', nm(cp(1)), nm(cp(2)), nm(cp(3)));
    end
    if ~isempty(cal.fov_weights)
        assert(numel(cal.fov_weights) == 1 + numel(fo) && all(cal.fov_weights > 0), ...
            'lens_emit: fov_weights needs %d positive weights (on-axis first)', 1 + numel(fo));
        w('         OptFOVWt=  %s', strjoin(arrayfun(nm, cal.fov_weights, 'UniformOutput', false), '  '));
    else
        w('         OptFOVWt=  %s', strtrim(repmat('1 ', 1, 1 + numel(fo))));
    end
end
w('%% Element Definitions');
w('             nElt=  %d', nElt);

iSurf = zeros(1, N);  iStop = 0;  ie = 0;  n_prev = 1;
for it = 1:numel(kind)
    ie = ie + 1;
    if kind(it) == 1
        k = kk(it);  iSurf(k) = ie;
        nmk = L.names();
        w('             iElt=  %d', ie);
        w('          EltName=  %s', nmk{k});
        w('          Element=  Refractor');
        if isfinite(L.R(k))
            w('          Surface=  Conic');
            w('            KrElt=  %s', nm(-L.R(k)));
            w('            KcElt=  %s', nm(L.K(k)));
        else
            w('          Surface=  Flat');
            w('            KrElt=  -1.0E+22');
            w('            KcElt=  0.0E+00');
        end
        w('           psiElt=  0.0E+00  0.0E+00  1.0E+00');
        w('           VptElt=  0.0E+00  0.0E+00  %s', nm(L.z(k)));
        w('           RptElt=  0.0E+00  0.0E+00  %s', nm(L.z(k)));
        if k < N
            [n_k, S] = macos.design.glass(L.glass{k}, P.lambda_nm);
        else
            n_k = 1;  S = struct('B', []);
        end
        w('           IndRef=  %s', nm(n_k));
        if ~isempty(S.B)
            w('         GlassElt=  %s', S.name);
            w('        GlassType=  Sellmeier');
            w('        GlassCoef=  %s', strjoin(arrayfun(@(v) sprintf('%.12G', v), [S.B S.C], 'UniformOutput', false), '  '));
        end
        w('           Extinc=  0.0E+00');
        if do_opt
            m = zeros(1, 8);
            m(6) = any(cal.var_pos == k);  m(7) = any(cal.var_R == k) && isfinite(L.R(k));
            assert(~(any(cal.var_R == k) && ~isfinite(L.R(k))), ...
                'lens_emit: surface %d is plano; a radius DOF needs a curved seed', k);
            if any(m), w('           VarElt=  %s', strjoin(arrayfun(@(v) sprintf('%d', v), m, 'UniformOutput', false), ' ')); end
        end
        w('            nCoat=  0');
        w('             xObs=  1.0E+00  0.0E+00  0.0E+00');
        w('             nObs=  0');
        if apert
            w('           ApType=  Circular');
            w('            ApVec=  %s  0.0E+00  0.0E+00', nm(L.semidia(k)));
        else
            w('           ApType=  None');
        end
        w('         PropType=  Geometric');
        w('             zElt=  1.0E+22');
        w('          nECoord=  -6');
        n_prev = n_k;
    else
        iStop = ie;
        w('             iElt=  %d', ie);
        w('          EltName=  Stop');
        w('          Element=  Reference');
        w('          Surface=  Flat');
        w('            KrElt=  -1.0E+22');
        w('            KcElt=  0.0E+00');
        w('           psiElt=  0.0E+00  0.0E+00  1.0E+00');
        w('           VptElt=  0.0E+00  0.0E+00  %s', nm(P.z_s));
        w('           RptElt=  0.0E+00  0.0E+00  %s', nm(P.z_s));
        w('           IndRef=  %s', nm(n_prev));
        w('           Extinc=  0.0E+00');
        w('            nCoat=  0');
        w('             xObs=  1.0E+00  0.0E+00  0.0E+00');
        w('             nObs=  0');
        if do_opt, w('           ApStop=  0.0E+00  0.0E+00'); end
        if opts.stop_clips
            w('           ApType=  Circular');
            w('            ApVec=  %s  0.0E+00  0.0E+00', nm(P.a_s));
        else
            w('           ApType=  None');
        end
        w('         PropType=  Geometric');
        w('             zElt=  1.0E+22');
        w('          nECoord=  -6');
    end
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
assert(ie + 3 == nElt, 'lens_emit: element count');
sd = L.semidia;  if ~apert, sd = inf(1, N); end
E = struct('iSurf', iSurf, 'iStop', iStop, 'iImgRet', ie + 1, 'iXP', ie + 2, 'iFP', ie + 3, ...
    'nElt', nElt, 'path', path, 'z_img', z_img, 'z_src', z_src, 'lambda_mm', P.lambda_nm * 1e-6, ...
    'D', P.D, 'finite', fin, 'semidia', sd, 'standoff', z_front - z_src);
end

% -------------------------------------------------------------------------
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
w('             nObs=  0');
w('           ApType=  None');
w('         PropType=  Geometric');
w('             zElt=  %s', nm(zelt));
w('          nECoord=  -6');
end

function s = medium_(g)
if isnumeric(g), s = sprintf('n=%.6g', g); elseif isstruct(g), s = char(g.name); else, s = char(g); end
end
