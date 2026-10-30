function dyson5_size_fig(OUT, fname)
%DYSON5_SIZE_FIG  The block-size trade figure (BRIEF_ccmac_dyson_size, rounds 1+2).
%
%   DYSON5_SIZE_FIG() loads dyson5_size.mat and writes dyson5_size.png.
%   DYSON5_SIZE_FIG(OUT, fname) uses a passed result and output name.
%
%   Three panels against block radius, one curve per family: CRF (px),
%   ensquared energy (1 px), and edged rod mass (kg).  LINE STYLE shows the
%   rung -- solid = R4 (meniscus), dashed = R3 (no meniscus).  A FILLED marker
%   meets the spec (CRF < 1.5, SRF < 2.1, smile/keystone < 0.1, no bound); an
%   open marker fails it; a filled marker ringed in black also MATCHES R4
%   (CRF <= 1.33, EE >= 0.76).  R4-of-record values are the solid grey lines
%   (CRF 1.327, EE 0.759); the dashed grey lines are the closure / spec bars.
%   Only the primary 'solve' rows are drawn (the bound-scaled re-runs stay in
%   the table).  Single-point families (G) are drawn as a lone marker.
    here = fileparts(mfilename('fullpath'));
    if nargin < 1 || isempty(OUT)
        S = load(fullfile(here, 'dyson5_size.mat'));  OUT = S.OUT;
    end
    if nargin < 2 || isempty(fname), fname = fullfile(here, 'dyson5_size.png'); end
    T = OUT.table;
    T = T(strcmp(T.variant, 'solve'), :);

    fams = unique(T.family, 'stable');
    cols = lines(max(7, numel(fams)));

    f = figure('Visible', 'off', 'Position', [40 40 1560 480], 'Color', 'w');
    axl  = {'CRF (px, system LSF \otimes pixel)', 'ensquared energy (1 px, geometric)', 'edged rod mass (kg)'};
    flds = {'CRF', 'EE', 'edged_kg'};
    recs = [OUT.record.CRF, OUT.record.EE, NaN];
    specbar = [1.5, NaN, NaN];                           % the SPEC CRF bar
    r4bar   = [1.33, 0.76, NaN];                         % the match-R4 bar

    for p = 1:3
        subplot(1, 3, p);  hold on;  hleg = [];  lnames = {};
        for i = 1:numel(fams)
            sel = strcmp(T.family, fams{i});
            r   = T.r_mm(sel);  y = T.(flds{p})(sel);
            sp  = T.meets_spec(sel);  r4 = T.matches_R4(sel);  ru = T.rung(find(sel,1));
            good = ~isnan(y);
            ls = '--';  if ru >= 4, ls = '-'; end          % R3 dashed, R4 solid
            if nnz(good) >= 2
                hp = plot(r(good), y(good), ls, 'Color', cols(i,:), 'LineWidth', 1.3);
            else
                hp = plot(NaN, NaN, ls, 'Color', cols(i,:), 'LineWidth', 1.3);   % legend proxy for a single point
            end
            % markers: filled = meets spec, open = fails; black ring = matches R4
            plot(r(sp & good),  y(sp & good),  'o', 'Color', cols(i,:), 'MarkerFaceColor', cols(i,:), 'MarkerSize', 6);
            plot(r(~sp & good), y(~sp & good), 'o', 'Color', cols(i,:), 'MarkerFaceColor', 'w', 'MarkerSize', 6);
            plot(r(r4 & good),  y(r4 & good),  'o', 'Color', 'k', 'MarkerFaceColor', 'none', 'MarkerSize', 9, 'LineWidth', 1);
            hleg(end+1) = hp;  lnames{end+1} = sprintf('%s (R%d %s)', fams{i}, ru, glass_of_(T, fams{i}));  %#ok<AGROW>
        end
        if ~isnan(recs(p)) && ~isnan(r4bar(p)) && isnan(specbar(p)) && abs(recs(p)-r4bar(p)) <= 0.01
            yline(recs(p), '-', sprintf('R4 record / match %.3g', recs(p)), 'Color', [.45 .45 .45]);   % the two coincide
        else
            if ~isnan(recs(p)),    yline(recs(p),    '-',  sprintf('R4 record %.3g', recs(p)),  'Color', [.45 .45 .45]); end
            if ~isnan(specbar(p)), yline(specbar(p), '--', sprintf('spec %.3g', specbar(p)),     'Color', [.45 .45 .45]); end
            if ~isnan(r4bar(p)) && isnan(specbar(p)), yline(r4bar(p), '--', sprintf('match-R4 %.3g', r4bar(p)), 'Color', [.45 .45 .45]); end
        end
        grid on;  xlabel('block radius (mm)');  ylabel(axl{p});  set(gca, 'XDir', 'reverse');
        if p == 1, legend(hleg, lnames, 'Location', 'northwest', 'FontSize', 7); end
        titles = {'response (solid=R4, dashed=R3; filled=meets spec, ring=matches R4)', 'ensquared energy', 'mass of the edged block'};
        title(titles{p}, 'FontSize', 9);
    end
    sgtitle('dyson5: how small can the Dyson block be?  R3 (no meniscus) vs R4 -- continuation walks, engine scores', 'FontSize', 11);
    print(f, fname, '-dpng', '-r130');  close(f);
    fprintf('dyson5_size_fig: wrote %s\n', fname);
end

function g = glass_of_(T, fam)
    i = find(strcmp(T.family, fam), 1);  gl = T.glass{i};
    if strcmpi(gl, 'Silica'), g = 'SiO2'; else, g = gl; end
end
