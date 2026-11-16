function M = tls_measure(P, G, deck, opts)
%TLS_MEASURE  The first-order numbers of a long-slit telescope deck, from the ENGINE.
%
%   M = TLS_MEASURE(P, G, DECK) loads DECK (emitted by TLS_SECTION or a later
%   stage), aims the chief through the element stop for every strip field
%   (P.nfield across +-P.strip_half_deg, along global x) and reads, per field:
%     pass       fraction of the source rays that reach the slit unblocked
%     chief_deg  angle of the chief to the slit normal (telecentricity), and
%                its components chief_x_mrad (along the slit, cross-track) and
%                chief_y_mrad (across the slit, along-track: the dispersion plane)
%     spread_deg angle of the chief to the CENTRE field's chief
%     x_m, y_m   chief hit on the slit in the slit frame (x = the slit
%                axis = global x; y = across the slit, in the fold plane)
%     bow_um     y of the chief hit minus the line through the end fields
%                (the image of the strip is a curved line by this much)
%     fno_x/y    working F/# of the cone at the slit, 1/(2 sin u), u = half
%                the angular extent of the passing rays about the chief, in
%                the slit-axis (x) and the across-slit (y) section
%     rms_um     rms spot radius on the slit as placed (no refocus)
%     bf_um      rms spot radius at the field's own best focus
%   and, over all fields: plate_local (dx/dtan(theta) about the centre, m),
%   plate_edge (x at the edge / tan(edge), m), the footprint of the passing
%   rays on each mirror (x / y full extents in the section frame, m, and the
%   enclosing radius about the pole), the chief hit at the stop element vs
%   its pole (m: 0 = the stop is where the deck says), and the clearance
%   table (TLS_CLEARANCE).
%
%   M = TLS_MEASURE(..., 'fields_deg', F) measures at the strip angles F.
%
%   See also TLS_SECTION, TLS_CLEARANCE, TMA_LONGSLIT_RUN.
arguments
    P struct
    G struct
    deck (1,:) char
    opts.fields_deg (1,:) double = linspace(-P.strip_half_deg, P.strip_half_deg, P.nfield)
    opts.clearance (1,1) logical = true
end
% The stop is the deck's ELEMENT stop (ApStop= in the stop element's block),
% applied at LOAD.  macos.stop cannot re-aim it per field: the api's
% stop_info_set refuses a stop element >= nElt-2 (every 4-element telescope's
% M2).  So each field is a COPY of the deck with its ChfRayDir, loaded, and
% the load-time STOP aims that field's chief through the pole.
txt0 = fileread(deck);
assert(~isempty(regexp(txt0, '(?m)^\s*ApStop=', 'once')), 'tls_measure: %s carries no element stop', deck);
tmp = [tempname '_tls.in'];  cln = onCleanup(@() delete_if_(tmp));
macos.load_rx(deck);
nE = macos.num_elt();  nM = nE - 1;
fd = opts.fields_deg;  nf = numel(fd);  [~, ic] = min(abs(fd));
sl = G.slit;  ez = -sl.normal(:);  ex = [1; 0; 0];  ey = cross(ez, ex);
M = struct('fields_deg', fd, 'pass', nan(1, nf), 'chief_deg', nan(1, nf), 'spread_deg', nan(1, nf), ...
           'x_m', nan(1, nf), 'y_m', nan(1, nf), 'bow_um', nan(1, nf), 'fno_x', nan(1, nf), 'fno_y', nan(1, nf), ...
           'rms_um', nan(1, nf), 'bf_um', nan(1, nf), 'stop_miss_m', nan(1, nf), 'fwhm_x_px', nan(1, nf), 'fwhm_y_px', nan(1, nf), ...
           'eip', nan(1, nf));
cdir = nan(3, nf);  paths = cell(1, nf);  foot = cell(1, nM);
for q = 1:nf
    th = deg2rad(fd(q));
    d = [sin(th); 0; cos(th)];
    txt = regexprep(txt0, '(?m)^(\s*ChfRayDir=).*$', sprintf('$1  %.16E  %.16E  %.16E', d), 'dotexceptnewline');
    assert(numel(txt) > 0.9*numel(txt0), 'tls_measure: the ChfRayDir substitution damaged the deck');
    fid = fopen(tmp, 'w');  fwrite(fid, txt);  fclose(fid);
    macos.load_rx(tmp);
    Pk = cell(1, nE);
    for k = 1:nE
        s = macos.trace(k);  ri = macos.get_ray_info(s.nRays);
        Pk{k} = ri;
    end
    ri = Pk{nE};
    ok = ri.ok_trace & ri.ok_pass;
    M.pass(q) = mean(ok);
    c = ri.dir(:, 1)/norm(ri.dir(:, 1));  cdir(:, q) = c;
    M.chief_deg(q) = acosd(min(1, abs(c.'*ez)));
    M.chief_x_mrad(q) = 1e3*atan2(c.'*ex, c.'*ez);   % component ALONG the slit (cross-track)
    M.chief_y_mrad(q) = 1e3*atan2(c.'*ey, c.'*ez);   % component ACROSS the slit (along-track, the dispersion plane)
    r0 = ri.pos(:, 1) - sl.point(:);
    M.x_m(q) = r0.'*ex;  M.y_m(q) = r0.'*ey;
    % the cone about the chief, in the slit frame
    Dd = ri.dir(:, ok)./vecnorm(ri.dir(:, ok));
    ax = atan2(Dd.'*ex, Dd.'*ez) - atan2(c.'*ex, c.'*ez);
    ay = atan2(Dd.'*ey, Dd.'*ez) - atan2(c.'*ey, c.'*ez);
    M.fno_x(q) = 1/(2*sin((max(ax) - min(ax))/2));
    M.fno_y(q) = 1/(2*sin((max(ay) - min(ay))/2));
    % spots: as placed, and at the field's best focus (least-squares point of the rays)
    Pp = ri.pos(:, ok);
    M.rms_um(q) = sqrt(mean(sum((Pp - mean(Pp, 2)).^2, 1)))*1e6;
    cen = mean(Pp, 2) - sl.point(:);  M.cy_m(q) = cen.'*ey;  M.cx_m(q) = cen.'*ex;   % the spot CENTROID on the slit
    % FWHM per axis, the record's scorer (spectrometer_score_fwhm: rays (x) 1 px (x) Airy LSF at score_lambda_m), px
    px = P.pixel_m;  a_px = P.score_lambda_m*(P.f_m/P.D_m)/px;
    u = (ex.'*(Pp - mean(Pp, 2)))/px;  v = (ey.'*(Pp - mean(Pp, 2)))/px;
    M.fwhm_x_px(q) = spectrometer_score_fwhm(u, 0, a_px);  M.fwhm_y_px(q) = spectrometer_score_fwhm(v, 0, a_px);
    M.eip(q) = mean(abs(u) <= 0.5 & abs(v) <= 0.5);     % geometric energy in a pixel about the centroid
    A = zeros(3);  b = zeros(3, 1);
    for i = 1:size(Pp, 2), Q = eye(3) - Dd(:, i)*Dd(:, i)';  A = A + Q;  b = b + Q*Pp(:, i); end
    Qp = Pp - A\b;  qd = sum(Qp.*Dd, 1);  Tt = Qp - Dd.*qd;
    M.bf_um(q) = sqrt(mean(sum(Tt.^2, 1)))*1e6;
    % stop check: the chief at the stop element vs that element's pole
    rs = Pk{G.stop_elt};  M.stop_miss_m(q) = norm(rs.pos(:, 1) - G.m(G.stop_elt).pole);
    % footprints on the mirrors (section frame about the pole) + the paths for clearance
    for k = 1:nM
        rk = Pk{k};  okk = rk.ok_trace & rk.ok_pass;  Fr = G.m(k).frame;
        foot{k} = [foot{k}, Fr(:, 1:2).'*(rk.pos(:, okk) - G.m(k).pole)];
    end
    okp = ok;  pts = zeros(3, nnz(okp), nE + 1);
    pts(:, :, 1) = Pk{1}.pos(:, okp) - 0.6*d;          % the incoming leg starts upstream of M1
    for k = 1:nE, pts(:, :, k + 1) = Pk{k}.pos(:, okp); end
    paths{q} = pts;
end
for q = 1:nf, M.spread_deg(q) = acosd(min(1, cdir(:, q).'*cdir(:, ic))); end
% the along-track (fold-plane) focal length: a 0.05 deg along-track probe at the centre and the edge field
dyd = 0.05;  fy = nan(1, 2);  qq = [ic nf];
for j = 1:2
    p2 = zeros(3, 2);
    for a = 0:1
        sx = sind(fd(qq(j)));  sy = sind(a*dyd);  dd = [sx; sy; sqrt(1 - sx^2 - sy^2)];
        txt = regexprep(txt0, '(?m)^(\s*ChfRayDir=).*$', sprintf('$1  %.16E  %.16E  %.16E', dd), 'dotexceptnewline');
        fid = fopen(tmp, 'w');  fwrite(fid, txt);  fclose(fid);  macos.load_rx(tmp);
        s = macos.trace(nE);  ri = macos.get_ray_info(s.nRays);  p2(:, a + 1) = ri.pos(:, 1);
    end
    fy(j) = abs(ey.'*(p2(:, 2) - p2(:, 1)))/tand(dyd);
end
M.plate_y_centre = fy(1);  M.plate_y_edge = fy(2);
% the bow: departure of the chief hits from the line through the end fields
p1 = [M.x_m(1); M.y_m(1)];  p2 = [M.x_m(end); M.y_m(end)];  u = (p2 - p1)/norm(p2 - p1);  nrm = [-u(2); u(1)];
for q = 1:nf, M.bow_um(q) = ([M.x_m(q); M.y_m(q)] - p1).'*nrm*1e6; end
% the CENTROID bow across the slit, vs the centre field -- what the spectrometer's smile reads (spectrometer_score scores
% each field's centroid); coma moves the centroid off the chief, so the chief bow alone under-reads it
M.cbow_um = (M.cy_m - M.cy_m(ic))*1e6;
% plate scale
tq = tan(deg2rad(fd));
j = find(abs(fd) > 0 & abs(fd) <= max(abs(fd))/(nf - 1)*2 + 1e-9);
M.plate_local = polyfit(tq([j ic]), M.x_m([j ic]), 1);  M.plate_local = abs(M.plate_local(1));
M.plate_edge = abs((M.x_m(end) - M.x_m(1))/(tq(end) - tq(1)));
% footprints
M.foot = struct('name', {G.m.name}, 'x_m', 0, 'y_m', 0, 'r_m', 0, 'cx_m', 0, 'cy_m', 0);
for k = 1:nM
    f = foot{k};
    M.foot(k).x_m = max(f(1, :)) - min(f(1, :));  M.foot(k).y_m = max(f(2, :)) - min(f(2, :));
    M.foot(k).cx_m = (max(f(1, :)) + min(f(1, :)))/2;  M.foot(k).cy_m = (max(f(2, :)) + min(f(2, :)))/2;
    M.foot(k).r_m = max(vecnorm(f));
end
M.paths = paths;
if opts.clearance, M.clear = tls_clearance(G, M, P); end
end

function delete_if_(f), if exist(f, 'file'), delete(f); end, end
