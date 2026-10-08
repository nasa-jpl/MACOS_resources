function T = tls_dyson_chief_tilt(P, M, opts)
%TLS_DYSON_CHIEF_TILT  The join's own diagnostic: the spectrometer ALONE fed with the telescope's chief-angle pattern.
%
%   T = TLS_DYSON_CHIEF_TILT(P, M) scores the spectrometer of P.e2e (default
%   the dyson5 3k Dyson of record, 'size:F:240') ALONE with spectrometer_score
%   -- the record's own path: spectrometer_geom -> spectrometer_rx (apertures,
%   5 mm margin) -> load -> score -- with every slit point's input CHIEF tilted
%   by the telescope's measured chief angle at the slit.  M is a TLS_MEASURE
%   record (its fields_deg, chief_x_mrad = along the slit / cross-track,
%   chief_y_mrad = across the slit / along-track); a slit point at x takes the
%   angles of the field theta = atan(x/f), interpolated.  The tilt is imposed by
%   wrapping the chain's G.aim (the direction spectrometer_score launches each
%   slit point's chief with, through the grating vertex): the whole cone tilts
%   with it -- what a telescope with that telecentric error delivers.
%
%   Legs (opts.legs, default all five): 'base' (no tilt: must reproduce the
%   record's row), 'along' (along-track only), 'cross' (cross-track only),
%   'along2' (the along-track pattern x2: linearity), 'both'.
%   T.rows: leg, scale, smile, keystone, CRF, SRF, EE (px).  T.base_gate:
%   |base - record| per metric.
%
%   Why it exists: the e2e smile of the telescope of record (0.14 px) is not
%   the telescope's centroid bow (0.04 px) and the Dyson alone is 0.005 px; the
%   telescope's along-track chief angle is even in field (smile's shape).  This
%   attributes the smile to the chief angle -- or rules it out -- without a solve.
%
%   See also TLS_E2E, TLS_MEASURE.
arguments
    P struct
    M struct
    opts.legs (1,:) cell = {'base', 'along', 'cross', 'along2', 'both'}
end
here = fileparts(mfilename('fullpath'));
ddir = fullfile(here, '..', '..', '..', 'challenges', 'dyson5');
addpath(ddir);  addpath(fullfile(here, '..', '..', '..', 'design', 'src'));
e = struct('tel_dyson', 'size:F:240');
if isfield(P, 'e2e') && ~isempty(P.e2e) && isfield(P.e2e, 'tel_dyson'), e.tel_dyson = P.e2e.tel_dyson; end
tk = strsplit(e.tel_dyson, ':');  Z = load(fullfile(ddir, 'dyson5_size.mat'));  rr = Z.OUT.rows;
k = find(strcmp(string({rr.family}), tk{2}) & abs([rr.r_mm] - str2double(tk{3})) < 1e-9 & strcmp(string({rr.variant}), 'solve'), 1);
row = rr(k);  Pd = dyson5_params();
G = spectrometer_geom('dyson', row.P);
file = [tempname '_tlsdyson.in'];  cln = onCleanup(@() delete_if_(file));
macos.init(Pd.model);
Mx = spectrometer_rx(G, file, 'ngridpts', Pd.ngridpts, 'name', 'tls_dyson', 'apertures', true, 'margin', 5e-3);
% the chief-angle pattern vs field (rad), even/odd as measured
th = deg2rad(M.fields_deg(:));  ax = M.chief_x_mrad(:)*1e-3;  ay = M.chief_y_mrad(:)*1e-3;
cx = @(x) interp1(th, ax, atan(x/P.f_m), 'pchip', 'extrap');
cy = @(x) interp1(th, ay, atan(x/P.f_m), 'pchip', 'extrap');
legs = struct('base', [0 0], 'along', [0 1], 'cross', [1 0], 'along2', [0 2], 'both', [1 1]);
T = struct('rows', [], 'record', struct('smile', row.smile, 'keystone', row.keystone, 'CRF', row.CRF, 'SRF', row.SRF, 'EE', row.EE), ...
           'dyson', e.tel_dyson, 'pattern', struct('fields_deg', M.fields_deg, 'chief_x_mrad', M.chief_x_mrad, 'chief_y_mrad', M.chief_y_mrad));
for L = opts.legs
    sc = legs.(L{1});
    Gt = G;
    if any(sc)
        Gt.aim = @(slit, lam) tilt_(G.aim(slit, lam), sc(1)*cx(slit(1) - G.slit(1)), sc(2)*cy(slit(1) - G.slit(1)));
    end
    macos.load_rx(file);
    Pk = row.P;  for f = {'npix', 'pixel_m', 'band_m', 'slit_px', 'blaze_m', 'qe'}, if ~isfield(Pk, f{1}) && isfield(Pd, f{1}), Pk.(f{1}) = Pd.(f{1}); end, end
    R = spectrometer_score(Gt, Mx, Pk, 'nx', Pd.score_nx, 'nlam', Pd.score_nlam, 'quiet', true);
    T.rows = [T.rows, struct('leg', L{1}, 'scale', sc, 'smile', R.smile_max, 'keystone', R.keystone_max, 'CRF', R.crf_max, ...
                             'SRF', R.srf_max, 'EE', R.ee_min, 'smile_px', R.smile_px)];
end
b = T.rows(strcmp({T.rows.leg}, 'base'));
if ~isempty(b)
    T.base_gate = struct('smile', abs(b.smile - row.smile), 'keystone', abs(b.keystone - row.keystone), 'CRF', abs(b.CRF - row.CRF), 'SRF', abs(b.SRF - row.SRF));
end
end

function d = tilt_(d0, ax, ay)
% tilt the launch direction by ax (along the slit, x) and ay (across it, y) in the slit frame of the chain (slit along x)
d0 = d0(:)/norm(d0);  ex = [1; 0; 0] - d0(1)*d0;  ex = ex/norm(ex);  ey = cross(d0, ex);
d = d0 + tan(ax)*ex + tan(ay)*ey;  d = d/norm(d);
end

function delete_if_(f), if exist(f, 'file'), delete(f); end, end
