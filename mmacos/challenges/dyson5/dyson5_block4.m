function S = dyson5_block4(deck, opts)
%DYSON5_BLOCK4  Four 1.5k modules (telescope + Dyson) filling the 6000-pixel swath: a layout sketch with the conflict check
%   and the size / volume / mass of the block.  Dave's ask, 2026-10-06.
%   S = dyson5_block4()                    the 1.5k record end to end (dyson5_t5f_GM_1k5_bAs_e2e.in), four modules canted
%                                          +-2.35 / +-7.05 deg across the track, side by side along the slit (cross-track) axis
%   S = dyson5_block4(deck, 'pitch_m', p)  a chosen module pitch (default: the module's cross-track extent + gap_m)
%   The module is the e2e deck AS TRACED: every body is read from the engine's own ray hits (footprint per element, with the
%   aperture + mount margins the clearance gate uses), the beams are the engine's rays, and each module is the deck moved
%   RIGIDLY (every point and direction keyword, the TElt/Tout frames) -- so the module's internal clearances are the record's
%   (+39 mm telescope, +0.38 mm overall) by construction and this tool checks only MODULE vs MODULE: body against body, and body
%   against another module's incoming sky beam (a cylinder of the entrance aperture's diameter on the sky side of its aperture).
%   Figure: macos.view_rx of the four moved decks in one axes (3-D and the cross-track view), <tag>_views.png; record <tag>.txt.
%   Mass is OPTICS ONLY (block from the size record; grating substrate, mirrors as 10 mm blanks sized to footprint + margin);
%   no structure, detector, electronics or baffles -- stated on the record.
    arguments
        deck (1,:) char = 'dyson5_t5f_GM_1k5_bAs_e2e.in'
        opts.cant_deg (1,:) double = []                         % each module's cross-track pointing; [] = four strips abutting with overlap_px of overlap
        opts.overlap_px (1,1) double = 10                       % pixels of overlap between neighbouring strips when the cants are derived
        opts.npix (1,1) double = 1500                           % cross-track pixels per module
        opts.alt_m (1,1) double = 550e3                         % for the ground scale on the swath figure
        opts.pitch_m (1,1) double = NaN
        opts.gap_m (1,1) double = 0.020                         % body-to-body gap between neighbours when the pitch is derived
        opts.margin_ap_m (1,1) double = 0.005                   % aperture margin on a footprint (the clearance gate's)
        opts.margin_mount_m (1,1) double = 0.010                % mount margin beyond the aperture (the clearance gate's)
        opts.blank_m (1,1) double = 0.010                       % mirror / grating blank thickness
        opts.rho_zerodur (1,1) double = 2530
        opts.rho_silica (1,1) double = 2200
        opts.block_kg (1,1) double = 1.6                        % dyson5_size.txt row D: the edged 130 mm silica block
        opts.model (1,1) double = 128
        opts.tag (1,:) char = 'dyson5_block4'
        opts.quiet (1,1) logical = false
    end
    here = fileparts(mfilename('fullpath'));
    if ~isfile(deck), deck = fullfile(here, deck); end
    txt = fileread(deck);
    d = vec_(txt, 'ChfRayDir');  d = d/norm(d);              % the sky direction, light travels along +d
    p0 = vec_(txt, 'ChfRayPos');  ap = vec_(txt, 'Aperture');  Dap = ap(1);
    xhat = [1; 0; 0];                                          % the slit / cross-track direction of the e2e deck (the strip's x)
    a = cross(xhat, d);  a = a/norm(a);                        % the cant axis: normal to cross-track and the line of sight
    % the strip each module admits, from the t5f record of the same deck (admit_half_rad); cants derived so the strips overlap
    rec = regexprep(deck, '_e2e\.in$', '.mat');  ta = NaN;
    if isfile(rec), L = load(rec);  if isfield(L, 'S') && isfield(L.S, 'admit_half_rad'), ta = L.S.admit_half_rad*180/pi; end, end
    assert(~isnan(ta), 'dyson5 block4: no t5f record beside %s (admit_half_rad)', deck);
    ppd = opts.npix/(2*ta);                                     % pixels per degree across the track
    if isempty(opts.cant_deg)
        cp = 2*ta - opts.overlap_px/ppd;  nM0 = 4;  opts.cant_deg = ((1:nM0) - (nM0 + 1)/2)*cp;
    end
    nM = numel(opts.cant_deg);
    macos.init(opts.model);
    % ---- the module as traced: per-element footprint centres and radii from the engine's hits
    macos.load_rx(deck);  nE = macos.num_elt();
    macos.ray_hist('on');  s = macos.trace(nE);  H = macos.ray_hist(s.nRays);
    names = regexp(txt, '(?m)^\s*EltName=\s*(\S+)', 'tokens');  names = cellfun(@(c) c{1}, names, 'uni', 0);
    types = arrayfun(@(k) macos.get_elt_info(k).type, 1:nE, 'uni', 0);
    fc = zeros(3, nE);  fr = zeros(1, nE);  allp = zeros(3, 0);
    for k = 1:nE
        okk = H.ok(:, k + 1);  P = H.P(:, okk, k + 1);          % slot k+1 = element k, the rays that reached it
        fc(:, k) = mean(P, 2);  fr(k) = max(vecnorm(P - fc(:, k))) + opts.margin_ap_m;  allp = [allp P]; %#ok<AGROW>
    end
    sub = allp(:, 1:max(1, floor(size(allp, 2)/1500)):end);   % ~1500 surface samples for the conflict check
    % the module's own extent along x (cross-track), from every hit + the mount margin
    xext = [min(allp(1, :)), max(allp(1, :))] + [-1 1]*(opts.margin_ap_m + opts.margin_mount_m);
    wmod = diff(xext);
    half_strip = 2.35*pi/180;                                   % the strip's half-field: the sky beams fan by this across the track
    mg = opts.margin_ap_m + opts.margin_mount_m;
    iM1 = find(strcmp(types, 'Reflector'), 1);
    chk = @(pp) conflicts_(pp, nM, opts.cant_deg, a, xhat, p0, d, Dap, sub, fc(:, iM1), mg, half_strip);
    pitch = opts.pitch_m;
    if isnan(pitch)   % the smallest pitch (5 mm steps) at which every module-vs-module margin is at least the gap
        pitch = wmod;
        while true
            pr_ = chk(pitch);  if min(pr_(:, 3)) >= opts.gap_m && min(pr_(:, 4)) >= opts.gap_m, break; end
            pitch = pitch + 0.005;  assert(pitch < 2, 'dyson5 block4: no pitch under 2 m clears');
        end
    end
    pair = chk(pitch);
    % ---- optics mass, one module
    isMir = strcmp(types, 'Reflector');  isGr = strcmp(types, 'Grating');
    rb = fr + opts.margin_mount_m;
    m_mir = sum(pi*rb(isMir).^2*opts.blank_m*opts.rho_zerodur);
    m_gr = sum(pi*rb(isGr).^2*opts.blank_m*opts.rho_silica);
    m_mod = opts.block_kg + m_mir + m_gr;
    % ---- the four moved decks
    tmp = cell(1, nM);  R = cell(1, nM);  T = cell(1, nM);  Bc = cell(1, nM);
    for k = 1:nM
        th = opts.cant_deg(k)*pi/180;  K = [0 -a(3) a(2); a(3) 0 -a(1); -a(2) a(1) 0];
        Rk = eye(3) + sin(th)*K + (1 - cos(th))*K*K;
        tk = (k - (nM + 1)/2)*pitch*xhat + (p0 - Rk*p0);     % rotate about the source point, then step across the track
        R{k} = Rk;  T{k} = tk;
        tmp{k} = [tempname '_block4_' num2str(k) '.in'];
        fid = fopen(tmp{k}, 'w');  fprintf(fid, '%s', transform_deck_(txt, Rk, tk));  fclose(fid);
        Bc{k} = Rk*fc + tk;                                    % body centres (the envelope)
    end
    % ---- the block's envelope: every body's centre +- blank radius, over the four modules
    allc = [Bc{:}];  allr = repmat(rb, 1, nM);
    lo = min(allc - allr, [], 2);  hi = max(allc + allr, [], 2);  box = hi - lo;
    % ---- the figure: the engine's own renders of the four moved decks in one axes
    f = figure('Visible', 'off', 'Position', [40 40 1600 800], 'Color', 'w');
    ax1 = subplot(1, 2, 1);  ax2 = subplot(1, 2, 2);  hold(ax1, 'on');  hold(ax2, 'on');
    [az, el] = view_along_(a);
    for k = 1:nM
        macos.load_rx(tmp{k});
        macos.view_rx('ax', ax1, 'nrings', 1, 'nspokes', 4, 'labels', false, 'title', '');
        macos.view_rx('ax', ax2, 'nrings', 1, 'nspokes', 4, 'labels', false, 'title', '', 'view', [az el]);
    end
    for ax = [ax1 ax2], axis(ax, 'equal');  axis(ax, 'tight');  grid(ax, 'on'); end
    title(ax1, sprintf('four 1.5k modules, cants %s deg, pitch %.0f mm: as traced by the engine', mat2str(round(opts.cant_deg, 2)), pitch*1e3), 'Interpreter', 'none');
    title(ax2, 'looking along the cant axis: cross-track across the page; light enters from below', 'Interpreter', 'none');
    out = fullfile(here, [opts.tag '_views.png']);  print(f, out, '-dpng', '-r130');  close(f);
    cellfun(@delete, tmp);
    % ---- the swath: four strips across the track, their overlaps, the ground scale
    f2 = figure('Visible', 'off', 'Position', [40 40 1400 760], 'Color', 'w');
    axs = subplot(2, 1, 1, 'Parent', f2);  hold(axs, 'on');  axd = subplot(2, 1, 2, 'Parent', f2);  hold(axd, 'on');
    cols = lines(nM);  edges = zeros(nM, 2);
    for k = 1:nM
        th = opts.cant_deg(k);  edges(k, :) = [th - ta, th + ta];
        patch(axs, [th-ta th+ta th+ta th-ta], [0 0 1 1] + (k - 1)*0, cols(k, :), 'FaceAlpha', 0.25, 'EdgeColor', cols(k, :), 'LineWidth', 1.2);
        text(axs, th, 0.5, sprintf('module %d\n%d px\n%+.3f to %+.3f deg', k, opts.npix, th - ta, th + ta), 'HorizontalAlignment', 'center', 'FontSize', 10);
    end
    ovl = zeros(1, nM - 1);
    for k = 1:nM - 1
        o = edges(k, 2) - edges(k+1, 1);  ovl(k) = o*ppd;
        patch(axs, [edges(k+1, 1) edges(k, 2) edges(k, 2) edges(k+1, 1)], [0 0 1 1], [0.2 0.2 0.2], 'FaceAlpha', 0.6, 'EdgeColor', 'none');
        text(axs, (edges(k, 2) + edges(k+1, 1))/2, 1.08, sprintf('%.0f px', ovl(k)), 'HorizontalAlignment', 'center', 'FontSize', 9);
    end
    tot = edges(end, 2) - edges(1, 1);  upx = nM*opts.npix - sum(ovl);
    xlabel(axs, 'cross-track angle (deg)');  set(axs, 'YTick', []);  ylim(axs, [-0.35 1.25]);  xlim(axs, [edges(1, 1) - 0.3, edges(end, 2) + 0.3]);
    gk = @(th) 2*opts.alt_m*tand(th)/2e3;   % km on the ground from nadir
    for th = ceil(edges(1,1)):floor(edges(end,2))
        text(axs, th, -0.18, sprintf('%.0f km', gk(th)), 'HorizontalAlignment', 'center', 'FontSize', 8, 'Color', [0.3 0.3 0.3]);
    end
    ylim(axs, [-0.25 1.25]);  axs.YColor = 'none';  axs.Box = 'off';   % (box is the envelope variable)
    title(axs, {sprintf('the four strips on the ground: %d px each over %.3f deg (the slit admits +-%.3f deg), %.0f px of overlap at each join; along-track offset between strips 0 (the four lines of sight lie in one cross-track plane)', ...
          opts.npix, 2*ta, ta, mean(ovl)), sprintf('ground distance from nadir at %.0f km altitude (gray); swath %.2f deg = %.0f km, %d unique pixels, %.1f m per pixel at nadir', ...
          opts.alt_m/1e3, tot, 2*opts.alt_m*tand(tot/2)/1e3, round(upx), 2*opts.alt_m*tand(ta)/opts.npix)}, 'FontSize', 10, 'Interpreter', 'none');
    % the sweep: one ground line per frame, pushed along the track by the spacecraft's motion
    annotation(f2, 'arrow', [0.05 0.05], [0.64 0.84], 'Color', [0.2 0.2 0.2], 'LineWidth', 1.5);
    annotation(f2, 'textbox', [0.003 0.46 0.10 0.16], 'String', {'along-track sweep', '(spacecraft motion);', 'each frame is one', 'ground line'}, ...
               'EdgeColor', 'none', 'FontSize', 8, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'top');
    % ---- the focal planes: each module's 1500 x 500 px detector under its strip, the spectral axis from the engine's dispersion
    RE = L.S.e2e;  im = ceil(numel(RE.xs)/2);  V = RE.V(im, :);  U = RE.U(:, ceil(numel(RE.lams)/2));
    [~, i380] = min(RE.lams);  [~, i2500] = max(RE.lams);  nsp = 500;
    vlo = min(V);  vhi = max(V);  sgn = sign(V(i2500) - V(i380));                 % +1: long wavelengths toward +v
    for k = 1:nM
        x0 = edges(k, 1);  x1 = edges(k, 2);
        patch(axd, [x0 x1 x1 x0], [-nsp/2 -nsp/2 nsp/2 nsp/2], cols(k, :), 'FaceAlpha', 0.15, 'EdgeColor', cols(k, :), 'LineWidth', 1.2);
        for j = 1:numel(RE.lams)         % the engine's centroid rows (the slit's image per wavelength), smile exaggerated by nothing: as scored
            plot(axd, linspace(x0, x1, numel(RE.xs)), RE.V(:, j) - mean(V)*0, '-', 'Color', cols(k, :)*0.6, 'LineWidth', 0.8);
        end
        text(axd, (x0 + x1)/2, 0, sprintf('module %d detector\n%d spatial x %d spectral px', k, opts.npix, nsp), 'HorizontalAlignment', 'center', 'FontSize', 9, 'BackgroundColor', 'w');
    end
    text(axd, edges(1, 1), V(i380), sprintf(' %.0f nm', RE.lams(i380)*1e9), 'HorizontalAlignment', 'left', 'VerticalAlignment', 'bottom', 'FontSize', 9);
    text(axd, edges(1, 1), V(i2500), sprintf(' %.0f nm', RE.lams(i2500)*1e9), 'HorizontalAlignment', 'left', 'VerticalAlignment', 'top', 'FontSize', 9);
    text(axd, edges(end, 2), V(i380), sprintf('%.0f nm ', RE.lams(i380)*1e9), 'HorizontalAlignment', 'right', 'VerticalAlignment', 'bottom', 'FontSize', 9);
    text(axd, edges(end, 2), V(i2500), sprintf('%.0f nm ', RE.lams(i2500)*1e9), 'HorizontalAlignment', 'right', 'VerticalAlignment', 'top', 'FontSize', 9);
    ylabel(axd, 'spectral pixel (v)');  xlabel(axd, 'cross-track angle of the slit image (deg); spatial pixel u runs the other way (the image is inverted)');
    ylim(axd, [-nsp/2 - 60, nsp/2 + 60]);  xlim(axd, [edges(1, 1) - 0.3, edges(end, 2) + 0.3]);  axd.Box = 'off';
    title(axd, sprintf('the four focal planes under their strips: the lines are the slit''s image at the seven scored wavelengths (the engine''s centroids, %.2f px/nm); %s', ...
          abs(V(i2500) - V(i380))/((RE.lams(i2500) - RE.lams(i380))*1e9), tern_(sgn > 0, 'wavelength increases toward +v', 'wavelength increases toward -v')), 'FontSize', 10, 'Interpreter', 'none');
    out2 = fullfile(here, [opts.tag '_swath.png']);  print(f2, out2, '-dpng', '-r130');  close(f2);
    S_swath = struct('edges_deg', edges, 'overlap_px', ovl, 'swath_deg', tot, 'unique_px', upx, 'px_per_deg', ppd, 'admit_half_deg', ta, 'fig', out2);
    % ---- the record
    S = struct('deck', deck, 'cant_deg', opts.cant_deg, 'pitch_m', pitch, 'module_x_extent_m', wmod, 'box_m', box(:)', ...
               'volume_L', prod(box)*1e3, 'mass_optics_kg', m_mod*nM, 'mass_module_kg', m_mod, 'mass_mirrors_kg', m_mir, ...
               'mass_grating_kg', m_gr, 'pairs', pair, 'footprint_r_m', fr, 'names', {names}, 'fig', out, 'swath', S_swath);
    if ~opts.quiet
        fid = fopen(fullfile(here, [opts.tag '.txt']), 'w');  pr = @(varargin) dp_(fid, varargin{:});
        pr('dyson5 block4 -- four 1.5k modules across the 6000-pixel swath (%s)\n', datestr(now, 'yyyy-mm-dd HH:MM'));
        pr('CONVENTIONS: module = %s as traced; cants %s deg about the axis normal to cross-track and the line of sight; modules side by side\n', deck, mat2str(round(opts.cant_deg, 3)));
        pr('  along the cross-track (slit) axis at pitch %.1f mm (the smallest 5 mm step with every margin >= the %.0f mm gap; module extent %.1f mm);\n  bodies = engine surface hits + %.0f mm aperture + %.0f mm mount; sky beams %.0f mm wide, fanning by the strip''s +-%.2f deg;\n', ...
           pitch*1e3, opts.gap_m*1e3, wmod*1e3, opts.margin_ap_m*1e3, opts.margin_mount_m*1e3, Dap*1e3, half_strip*180/pi);
        pr('  mass = OPTICS ONLY: block %.1f kg (dyson5_size.txt row D), mirrors and grating as %.0f mm blanks (Zerodur %d, silica %d kg/m3); no structure, detector, electronics, baffles.\n\n', ...
           opts.block_kg, opts.blank_m*1e3, opts.rho_zerodur, opts.rho_silica);
        pr('FOOTPRINTS (engine hits, radius + aperture margin, mm):\n');
        for k = 1:nE, pr('  %-16s %-12s r %6.1f  blank r %6.1f\n', names{k}, types{k}, fr(k)*1e3, rb(k)*1e3); end
        pr('\nONE MODULE: optics %.2f kg (block %.2f, mirrors %.2f, grating %.2f); cross-track extent %.0f mm\n', m_mod, opts.block_kg, m_mir, m_gr, wmod*1e3);
        pr('THE BLOCK OF FOUR: envelope %.0f x %.0f x %.0f mm (x cross-track, y, z of the deck frame) = %.0f L; optics %.1f kg\n', box*1e3, prod(box)*1e3, m_mod*nM);
        pr('CONFLICTS (module vs module, m; negative = conflict):\n');
        pr('  worst body-body margin %+.3f m, worst body-vs-sky-beam margin %+.3f m\n', min(pair(:, 3)), min(pair(:, 4)));
        for r = 1:size(pair, 1), pr('    %d vs %d: body-body %+.3f  body-in-beam %+.3f\n', pair(r, :)); end
        pr('SWATH: each module admits +-%.3f deg (%d px, %.1f px/deg); cants chosen for %.0f px of overlap; strip edges (deg):\n', ta, opts.npix, ppd, opts.overlap_px);
        for k = 1:nM, pr('    module %d  %+.3f .. %+.3f\n', k, edges(k, :)); end
        pr('  swath %.3f deg = %.1f km at %.0f km; %d unique pixels; along-track offset between strips 0 (one cross-track plane of lines of sight)\n', tot, 2*opts.alt_m*tand(tot/2)/1e3, opts.alt_m/1e3, round(upx));
        pr('FIGURES %s, %s\n', out, out2);
        fclose(fid);  save(fullfile(here, [opts.tag '.mat']), 'S');
    end
end

function pair = conflicts_(pitch, nM, cant, a, xhat, p0, d, Dap, sub, apc0, mg, half)
%CONFLICTS_  Module-vs-module margins at a pitch: [i j body-body body-in-beam] in m, from the moved surface samples.
%   body-body: the smallest distance between two modules' surface samples minus the aperture + mount margin on each;
%   body-in-beam: module j's samples on the sky side of module i's entrance aperture, their distance from i's beam axis minus
%   the beam's radius there (Dap/2 widened by the strip's half-field fan) and the margin.
    K = [0 -a(3) a(2); a(3) 0 -a(1); -a(2) a(1) 0];
    Q = cell(1, nM);  D = cell(1, nM);  C = zeros(3, nM);
    for k = 1:nM
        th = cant(k)*pi/180;  Rk = eye(3) + sin(th)*K + (1 - cos(th))*K*K;  tk = (k - (nM + 1)/2)*pitch*xhat + (p0 - Rk*p0);
        Q{k} = Rk*sub + tk;  D{k} = Rk*d;  C(:, k) = Rk*apc0 + tk;
    end
    pair = zeros(0, 4);
    for i = 1:nM
        for j = 1:nM
            if j == i, continue; end
            dm = Inf;
            for c = 1:size(Q{i}, 2)
                dm = min(dm, min(vecnorm(Q{j} - Q{i}(:, c))));
            end
            mbb = dm - 2*mg;
            q = Q{j} - C(:, i);  along = D{i}'*q;  perp = vecnorm(q - D{i}*along);  up = along < 0;
            if any(up), mbeam = min(perp(up) - (Dap/2 + abs(along(up))*tan(half)) - mg); else, mbeam = Inf; end
            pair(end+1, :) = [i j mbb mbeam]; %#ok<AGROW>
        end
    end
end

function out = transform_deck_(txt, Rm, tr)
%TRANSFORM_DECK_  Rigid move of a whole deck: p -> Rm p + tr on every point keyword, d -> Rm d on every direction keyword,
%   the TElt frame rows rotated (after dyson5_t5f's helper, every element kept; Tout is left alone).
    pts = {'VptElt', 'RptElt', 'pMon', 'pFF', 'pData', 'ChfRayPos', 'ApStop', 'StopPos'};
    dirs = {'psiElt', 'xMon', 'yMon', 'zMon', 'xFF', 'yFF', 'zFF', 'xData', 'yData', 'zData', 'ChfRayDir', 'xGrid', 'yGrid', 'xObs', 'h1HOE'};
    L = splitlines(string(txt));  inFrame = 0;  o = strings(numel(L), 1);
    for i = 1:numel(L)
        s = L(i);  key = regexp(char(s), '^\s*(\w+)=', 'tokens', 'once');
        if ~isempty(key), key = key{1}; else, key = ''; end
        if strcmp(key, 'TElt'), inFrame = 6; end                % Tout here is the nOutCord x 7 output selector, not a frame
        if inFrame > 0
            v = sscanf(strrep(char(regexprep(s, '^\s*\w+=', '')), 'D', 'E'), '%f')';
            r = 7 - inFrame;  b = 1 + 3*(r > 3);  v(b:b+2) = v(b:b+2)*Rm';
            lead = regexp(char(s), '^\s*\w+=', 'match', 'once');  if isempty(lead), lead = '                   '; end
            s = string(lead) + sprintf('  %.16E', v);  inFrame = inFrame - 1;
        elseif any(strcmp(key, pts))
            v = vec_(char(s), key);  s = sprintf('%18s=  %.16E  %.16E  %.16E', key, Rm*v + tr);
        elseif any(strcmp(key, dirs))
            v = vec_(char(s), key);  s = sprintf('%18s=  %.16E  %.16E  %.16E', key, Rm*v);
        end
        o(i) = s;
    end
    out = char(strjoin(o, newline));
end

function [az, el] = view_along_(a)
%VIEW_ALONG_  MATLAB view angles looking along the unit vector a (camera at -a... i.e. the line of sight is a).
    el = asind(a(3));  az = atan2d(a(1), -a(2));
end

function t = tern_(c, a, b), if c, t = a; else, t = b; end, end
function v = vec_(t, key), m = regexp(t, ['(?m)^\s*' key '=\s*([^\n]*)'], 'tokens', 'once');  v = sscanf(strrep(m{1}, 'D', 'E'), '%f'); end
function dp_(fid, varargin), fprintf(fid, varargin{:});  fprintf(varargin{:}); end
