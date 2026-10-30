function E = iodc2010_emit(L, P, p, path, opt)
%IODC2010_EMIT  Write a green-lens prescription as a MACOS deck.
%   E = iodc2010_emit(L, P, p, path) writes PATH (a .in) and returns the
%   element map the evaluator needs:
%     E.iSurf(k)  element index of lens surface k
%     E.iStop     the stop plane (a passive Reference with a circular aperture)
%     E.iImgRet   the flat Return at the image plane
%     E.iXP       the exit-pupil Return sphere (FEX re-places it per field)
%     E.iFP       the FocalPlane
%     E.nElt, E.path, E.z_img, E.semidia (1xN, what was written)
%
%   Conventions (engine record: notes/surfaces.md, reference_macos_rx_emission_convention):
%     * BaseUnits mm, WaveUnits nm, Wavelen = p.lambda_nm; light travels +z,
%       surface 1 at z = 0.
%     * A CODE V radius R (centre of curvature to +z) is KrElt = -R with
%       psiElt = (0,0,1): the centre sits at VptElt - KrElt*psiElt.  Plano:
%       Surface= Flat, KrElt = -1e22.
%     * A glass gap writes IndRef = n(lambda) AND the named glass with its
%       Sellmeier row (GlassElt / GlassType / GlassCoef): the engine recomputes
%       the index from the row at every trace (tracesub.F CTRACE), so the deck
%       is the NAMED glass on any engine build, table row or not.
%     * The stop is Element= Reference, Surface= Flat, ApType= Circular with
%       the stop semi-diameter: it clips and nothing else.  It sits in the
%       medium of its gap (IndRef = that medium).
%     * Clear apertures: ApType= Circular, ApVec = [semidia 0 0] on every
%       surface (p / L.semidia; E.semidia records them).
%     * The source: collimated (zSource 1e22), Aperture = P.D (the paraxial
%       entrance-pupil DIAMETER), GridType Circular, nGridpts p.nGridpts.
%       The on-axis field is written; iodc2010_eval moves it per field with
%       macos.set_src_fov and aims the chief through the stop with
%       macos.stop(E.iStop).
%     * The tail is the manual's double-pass pattern (119 decks of the
%       corpus): Return at the image, Return at the exit pupil, FocalPlane.
arguments
    L struct
    P struct
    p struct
    path (1,:) char
    opt struct = struct()     % CALIB block (see iodc2010_search): .fields_deg .max_iters .var_surf .var_fp .tol
end
N = L.N;
do_opt = ~isempty(fieldnames(opt));
if isempty(L.semidia)
    semidia = repmat(p.semidia_default, 1, N);
else
    semidia = L.semidia;
end

% ---- element list in z order (the stop before a surface at the same z)
items = struct('kind', {}, 'z', {}, 'k', {});
for k = 1:N
    items(end+1) = struct('kind', 'surf', 'z', L.z(k), 'k', k); %#ok<AGROW>
end
items(end+1) = struct('kind', 'stop', 'z', L.stop(1), 'k', 0);
key = [items.z] + 1e-9 * strcmp({items.kind}, 'surf');   % stop first at equal z
[~, ord] = sort(key);  items = items(ord);

z_img = P.z_img;
z_xp0 = P.z_xp;  if ~isfinite(z_xp0), z_xp0 = z_img - 1e6; end

% ---- the source start: well in front of the first element
z_front = min([L.z, L.stop(1)]);
z_src   = min(P.z_ep, z_front) - p.src_standoff_mm;

lam_mm = p.lambda_nm * 1e-6;
nm = @(s) sprintf('%.15E', s);

fid = fopen(path, 'w');
assert(fid > 0, 'iodc2010_emit: cannot write %s', path);
c = onCleanup(@() fclose(fid));
w = @(varargin) fprintf(fid, [varargin{1} '\n'], varargin{2:end});

w('%% IODC 2010 "green lens" prescription: %s', L.name);
w('%% Emitted by mmacos/challenges/iodc2010/iodc2010_emit.m -- one radius magnitude %.6f mm,', L.R);
w('%% signs [%s], N-BK7 at %g nm, |EFL| target 100 mm.  Light travels +z, surface 1 at z = 0.', ...
    strjoin(arrayfun(@(s) sprintf('%+d', s), L.sgn, 'UniformOutput', false), ' '), p.lambda_nm);
w('%% Scoring convention: challenges/iodc2010/README.md (the metric is stated before the number).');
w('%% Source Definition');
w('        ChfRayDir=  0.0E+00  0.0E+00  1.0E+00');
w('        ChfRayPos=  0.0E+00  0.0E+00  %s', nm(z_src));
w('          zSource=  1.0E+22');
w('        BaseUnits=  mm');
w('        WaveUnits=  nm');
w('           IndRef=  1.0E+00');
w('           Extinc=  0.0E+00');
w('          Wavelen=  %s', nm(p.lambda_nm));
w('             Flux=  1.0E+00');
w('         Aperture=  %s', nm(P.D));
w('         Obscratn=  0.0E+00');
w('         GridType=  Circular');
w('         nGridpts=  %d', p.nGridpts);
w('            xGrid=  1.0E+00  0.0E+00  0.0E+00');
w('            yGrid=  0.0E+00  1.0E+00  0.0E+00');
if do_opt
    % CALIB: a WFE target at the XP Return, OptFEX on (the stop is declared
    % ELEMENT-bound in the stop block below, so the engine re-aims the chief
    % through the stop's centre at every trace as the surfaces move -- a fixed
    % object-space ApStop point would drift off the stop as the solve moves
    % the lens), the off-axis fields as chief directions (field 1 = the
    % header's on-axis ChfRayDir), unit weights sized 1 + n_off.
    w('        OptTarget=  WFE');
    w('         OptWFElt=  %d', numel(items) + 2);
    w('       OptMaxItrs=  %d', opt.max_iters);
    if isfield(opt, 'tol') && ~isempty(opt.tol), w('           OptTol=  %s', nm(opt.tol)); end
    w('           OptFEX=  Yes');
    fo = opt.fields_deg(opt.fields_deg > 0);
    for th = fo(:).'
        d  = [0 sin(deg2rad(th)) cos(deg2rad(th))];
        cp = [0 0 P.z_ep] - p.src_standoff_mm * d;
        w('     OptChfRayDir=  %s  %s  %s', nm(d(1)), nm(d(2)), nm(d(3)));
        w('     OptChfRayPos=  %s  %s  %s', nm(cp(1)), nm(cp(2)), nm(cp(3)));
    end
    w('         OptFOVWt=  %s', strtrim(repmat('1 ', 1, 1 + numel(fo))));
end
w('%% Element Definitions');
nElt = numel(items) + 3;
w('             nElt=  %d', nElt);

iSurf = zeros(1, N);  iStop = 0;  ie = 0;
n_prev = 1;
for it = 1:numel(items)
    ie = ie + 1;
    switch items(it).kind
        case 'surf'
            k = items(it).k;  iSurf(k) = ie;
            w('             iElt=  %d', ie);
            w('          EltName=  S%d', k);
            w('          Element=  Refractor');
            if L.sgn(k) == 0
                w('          Surface=  Flat');
                w('            KrElt=  -1.0E+22');
            else
                w('          Surface=  Conic');
                w('            KrElt=  %s', nm(-L.sgn(k) * L.R));
            end
            w('            KcElt=  0.0E+00');
            w('           psiElt=  0.0E+00  0.0E+00  1.0E+00');
            w('           VptElt=  0.0E+00  0.0E+00  %s', nm(L.z(k)));
            w('           RptElt=  0.0E+00  0.0E+00  %s', nm(L.z(k)));
            [n_k, S] = iodc2010_glass(L.glass{k}, p.lambda_nm);
            w('           IndRef=  %s', nm(n_k));
            if ~isempty(S.B)
                w('         GlassElt=  %s', S.name);
                w('        GlassType=  Sellmeier');
                w('        GlassCoef=  %s', strjoin(arrayfun(@(v) sprintf('%.12G', v), [S.B S.C], 'UniformOutput', false), '  '));
            end
            w('           Extinc=  0.0E+00');
            if do_opt && any(opt.var_surf == k)
                w('           VarElt=  0 0 0 0 0 1 0 0');     % PIST: the axial position
            end
            w('            nCoat=  0');
            w('             xObs=  1.0E+00  0.0E+00  0.0E+00');
            w('             nObs=  0');
            if isfinite(semidia(k))
                w('           ApType=  Circular');
                w('            ApVec=  %s  0.0E+00  0.0E+00', nm(semidia(k)));
            else
                w('           ApType=  None');          % the sizing pass: nothing clips but the stop
            end
            w('         PropType=  Geometric');
            w('             zElt=  1.0E+22');
            w('          nECoord=  -6');
            n_prev = n_k;
        case 'stop'
            iStop = ie;
            w('             iElt=  %d', ie);
            w('          EltName=  Stop');
            w('          Element=  Reference');
            w('          Surface=  Flat');
            w('            KrElt=  -1.0E+22');
            w('            KcElt=  0.0E+00');
            w('           psiElt=  0.0E+00  0.0E+00  1.0E+00');
            w('           VptElt=  0.0E+00  0.0E+00  %s', nm(L.stop(1)));
            w('           RptElt=  0.0E+00  0.0E+00  %s', nm(L.stop(1)));
            w('           IndRef=  %s', nm(n_prev));
            w('           Extinc=  0.0E+00');
            w('            nCoat=  0');
            w('             xObs=  1.0E+00  0.0E+00  0.0E+00');
            w('             nObs=  0');
            if do_opt
                w('           ApStop=  0.0E+00  0.0E+00');   % element-bound stop: re-aimed every trace
            end
            w('           ApType=  Circular');
            w('            ApVec=  %s  0.0E+00  0.0E+00', nm(L.stop(2)));
            w('         PropType=  Geometric');
            w('             zElt=  1.0E+22');
            w('          nECoord=  -6');
    end
end

% ---- Return at the image plane
ie = ie + 1;  iImgRet = ie;
w('             iElt=  %d', ie);
w('          EltName=  ImgReturn');
w('          Element=  Return');
w('          Surface=  Flat');
w('            KrElt=  -1.0E+22');
w('            KcElt=  0.0E+00');
w('           psiElt=  0.0E+00  0.0E+00  1.0E+00');
w('           VptElt=  0.0E+00  0.0E+00  %s', nm(z_img));
w('           RptElt=  0.0E+00  0.0E+00  %s', nm(z_img));
w('           IndRef=  1.0E+00');
w('           Extinc=  1.0E+22');
w('            nCoat=  0');
w('             nObs=  0');
w('           ApType=  None');
w('         PropType=  Geometric');
w('             zElt=  1.0E+22');
w('          nECoord=  -6');

% ---- Return at the exit pupil (FEX re-places vertex, normal and radius)
ie = ie + 1;  iXP = ie;
w('             iElt=  %d', ie);
w('          EltName=  ExitPupil');
w('          Element=  Return');
w('          Surface=  Conic');
w('            KrElt=  %s', nm(-(z_img - z_xp0)));
w('            KcElt=  0.0E+00');
w('           psiElt=  0.0E+00  0.0E+00  1.0E+00');
w('           VptElt=  0.0E+00  0.0E+00  %s', nm(z_xp0));
w('           RptElt=  0.0E+00  0.0E+00  %s', nm(z_xp0));
w('           IndRef=  1.0E+00');
w('           Extinc=  1.0E+22');
w('            nCoat=  0');
w('             nObs=  0');
w('           ApType=  None');
w('         PropType=  Geometric');
w('             zElt=  %s', nm(abs(z_img - z_xp0)));
w('          nECoord=  -6');

% ---- the image plane
ie = ie + 1;  iFP = ie;
w('             iElt=  %d', ie);
w('          EltName=  Image');
w('          Element=  FocalPlane');
w('          Surface=  Flat');
w('            KrElt=  -1.0E+22');
w('            KcElt=  0.0E+00');
w('           psiElt=  0.0E+00  0.0E+00  1.0E+00');
w('           VptElt=  0.0E+00  0.0E+00  %s', nm(z_img));
w('           RptElt=  0.0E+00  0.0E+00  %s', nm(z_img));
w('           IndRef=  1.0E+00');
w('           Extinc=  0.0E+00');
if do_opt && isfield(opt, 'var_fp') && opt.var_fp
    w('           VarElt=  0 0 0 0 0 1 0 0');             % the image distance
end
w('            nCoat=  0');
w('             nObs=  0');
w('           ApType=  None');
w('         PropType=  Geometric');
w('             zElt=  1.0E+22');
w('          nECoord=  -6');
w('');
w('         nOutCord=  5');
w('             Tout=  1.0D+00  0.0D+00  0.0D+00  0.0D+00  0.0D+00  0.0D+00  0.0D+00');
w('                    0.0D+00  1.0D+00  0.0D+00  0.0D+00  0.0D+00  0.0D+00  0.0D+00');
w('                    0.0D+00  0.0D+00  0.0D+00  1.0D+00  0.0D+00  0.0D+00  0.0D+00');
w('                    0.0D+00  0.0D+00  0.0D+00  0.0D+00  1.0D+00  0.0D+00  0.0D+00');
w('                    0.0D+00  0.0D+00  0.0D+00  0.0D+00  0.0D+00  0.0D+00  1.0D+00');
assert(ie == nElt, 'iodc2010_emit: element count mismatch');

E = struct('iSurf', iSurf, 'iStop', iStop, 'iImgRet', iImgRet, 'iXP', iXP, 'iFP', iFP, ...
    'nElt', nElt, 'path', path, 'z_img', z_img, 'z_src', z_src, 'semidia', semidia, ...
    'lambda_mm', lam_mm, 'D', P.D);
end
