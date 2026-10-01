function T = dyson5_trade(tag)
%DYSON5_TRADE  One trade table across the ladder records (fixed + free radius).
%   T = dyson5_trade(tag) reads <tag>_s3.mat (block radius held) and
%   <tag>_s3free.mat (radius free, the "size alone" column) and writes
%   <tag>_s3_trade.txt + <tag>_s3_trade.png: per rung the ENGINE keystone,
%   smile, CRF FWHM, SRF FWHM, ensquared energy, and the instrument's
%   length (slit plane to grating vertex), grating footprint (diameter of
%   the traced hits over slit centre + ends x band edges), element count and
%   glass volume (block cap + meniscus, from the chain geometry).  One table,
%   one comparison (BRIEF_to_dyson5 addendum 4).
    here = fileparts(mfilename('fullpath'));
    run(fullfile(here, '..', '..', 'mmacos_setup.m'));
    recs = {'_s3', 'r held'; '_s3free', 'r free'; '_s4', 'native'};
    rows = {};
    for q = 1:size(recs, 1)
        fn = [tag recs{q,1} '.mat'];
        if ~isfile(fn), continue; end
        S = load(fn);  L = S.S;
        for k = 1:numel(L.rung)
            r = L.rung(k);  G = spectrometer_geom('dyson', r.P);  Re = r.engine;
            geo = geom_(G);
            rows(end+1, :) = {recs{q,2}, r.name, Re.keystone_max, Re.smile_max, Re.crf_max, Re.srf_max, Re.ee_min, ...
                              geo.length_mm, geo.footprint_mm, geo.n_elt, geo.glass_cm3, G.r*1e3, G.Rg*1e3, ...
                              geo.block_diam_mm, geo.block_thick_mm, geo.grating_diam_mm, geo.men_diam_mm};  %#ok<AGROW>
        end
    end
    T = cell2table(rows, 'VariableNames', {'radius', 'rung', 'keystone_px', 'smile_px', 'CRF_px', 'SRF_px', ...
                   'EE_1px', 'length_mm', 'grating_footprint_mm', 'n_elements', 'glass_cm3', 'r_mm', 'Rg_mm', ...
                   'block_diam_mm', 'block_thick_mm', 'grating_diam_mm', 'meniscus_diam_mm'});
    fid = fopen([tag '_s3_trade.txt'], 'w');
    fprintf(fid, 'dyson5 trade table (%s) -- engine rows; length = slit plane to grating vertex; footprint = diameter of the\n', datestr(now, 'yyyy-mm-dd HH:MM'));
    fprintf(fid, '  grating hits over slit centre + ends x band edges (chief + 8 marginals); glass = block spherical cap + meniscus.\n');
    fprintf(fid, '%-7s %-56s %9s %8s %7s %7s %6s %8s %8s %5s %8s %6s %7s %8s %8s %8s %8s\n', 'radius', 'rung', 'keystone', 'smile', 'CRF', 'SRF', 'EE', 'length', 'footpr.', 'nElt', 'glass', 'r', 'R_g', 'blockD', 'blockT', 'gratD', 'menD');
    fprintf(fid, '%-7s %-56s %9s %8s %7s %7s %6s %8s %8s %5s %8s %6s %7s %8s %8s %8s %8s\n', '', '', 'px', 'px', 'px', 'px', '1px', 'mm', 'mm', '', 'cm^3', 'mm', 'mm', 'mm', 'mm', 'mm', 'mm');
    for i = 1:height(T)
        fprintf(fid, '%-7s %-56s %9.4f %8.4f %7.3f %7.3f %6.3f %8.1f %8.1f %5d %8.0f %6.0f %7.0f %8.1f %8.1f %8.1f %8.1f\n', T.radius{i}, T.rung{i}, ...
            T.keystone_px(i), T.smile_px(i), T.CRF_px(i), T.SRF_px(i), T.EE_1px(i), T.length_mm(i), T.grating_footprint_mm(i), ...
            T.n_elements(i), T.glass_cm3(i), T.r_mm(i), T.Rg_mm(i), T.block_diam_mm(i), T.block_thick_mm(i), T.grating_diam_mm(i), T.meniscus_diam_mm(i));
    end
    fprintf(fid, 'sizes: block/grating/meniscus diameters = 2 x (footprint radius + %.0f mm aperture margin) about the footprint centre;\n', 5);
    fprintf(fid, '  block thickness = sphere vertex to flat face on the axis; length = slit plane to grating vertex.\n');
    fclose(fid);
    % ---- figure: distortion (log) and blur/energy per rung, both records
    f = figure('Visible', 'off', 'Position', [40 40 1200 400], 'Color', 'w');
    lab = cellfun(@(a, b) sprintf('%s, %s', regexprep(b, ' .*', ''), a), T.radius, T.rung, 'uni', 0);
    subplot(1, 3, 1);  bar([T.keystone_px, T.smile_px]);  set(gca, 'YScale', 'log', 'XTickLabel', lab, 'FontSize', 8, 'XTickLabelRotation', 35);
    yline(0.1, '--', 'spec 0.1 px');  yline(0.01, ':', 'paper 1 %');  legend({'keystone max', 'smile max'}, 'Location', 'southwest');  ylabel('px');  grid on;  title('distortion (engine)');
    subplot(1, 3, 2);  bar([T.CRF_px, T.SRF_px]);  set(gca, 'XTickLabel', lab, 'FontSize', 8, 'XTickLabelRotation', 35);  yline(1.5, '--', 'spec CRF 1.5');  yline(2, ':', '2-px slit floor');
    legend({'CRF FWHM', 'SRF FWHM'}, 'Location', 'northeast');  ylabel('px');  grid on;  title('response functions (engine)');
    subplot(1, 3, 3);  bar(T.EE_1px);  set(gca, 'XTickLabel', lab, 'FontSize', 8, 'XTickLabelRotation', 35);  yline(0.75, '--', 'paper > 0.75');  ylabel('min ensquared (1 px)');  grid on;  title('ensquared energy (engine, geometric)');
    print(f, [tag '_s3_trade.png'], '-dpng', '-r130');  close(f);
    fprintf('dyson5 trade: wrote %s_s3_trade.{txt,png}\n', tag);
end

function g = geom_(G)
    g.length_mm = (G.surf(G.iG).vpt(3) - G.slit(3))*1e3;
    W = G.P.npix(1)*G.P.pixel_m;  lams = [G.P.band_m(1) G.src.lambda_c G.P.band_m(2)];
    ang = linspace(0, 2*pi, 9);  ang(end) = [];  H = [];
    for xs = [-W/2 0 W/2]
        slit = G.slit + [xs;0;0];
        for lam = lams
            d0 = G.aim(slit, lam);  ez = d0;  ex = cross([0;1;0], ez);  ex = ex/norm(ex);  ey = cross(ez, ex);
            dirs = [d0, cos(G.src.u)*d0 + sin(G.src.u)*(ex*cos(ang) + ey*sin(ang))];
            for k = 1:size(dirs, 2)
                [pts, ~, ok] = G.trace(slit, dirs(:,k), lam);  if ok, H(:, end+1) = pts(:, G.iG); end  %#ok<AGROW>
            end
        end
    end
    g.footprint_mm = 2*max(vecnorm(H(1:2,:) - mean(H(1:2,:), 2)))*1e3;
    g.n_elt = numel(G.surf) - 1;              % the FPA plane is the detector, not an optic
    r = G.r;  dz = G.surf(1).C(3) - G.surf(2).C(3);  h = r - dz;   % spherical cap beyond the flat face
    vol = pi*h^2*(3*r - h)/3;
    im = find(strcmp({G.surf.name}, 'MenA_out'));
    if ~isempty(im)
        % meniscus: footprint-limited slab of thickness t (approximation)
        t = G.surf(im+1).vpt(3) - G.surf(im).vpt(3);  vol = vol + pi*(g.footprint_mm*1e-3/2)^2*t;
    end
    g.glass_cm3 = vol*1e6;
    % element sizes from the declared apertures (footprint + margin)
    F = G.footprints();  m = 5e-3;
    g.block_diam_mm = 2*(F(2).radius + m)*1e3;
    g.block_thick_mm = (G.surf(2).vpt(3) - G.surf(1).C(3))*1e3;
    g.grating_diam_mm = 2*(F(G.iG).radius + m)*1e3;
    im = find(strcmp({G.surf.name}, 'MenA_out'));
    if isempty(im), g.men_diam_mm = 0; else, g.men_diam_mm = 2*(F(im).radius + m)*1e3; end
end
