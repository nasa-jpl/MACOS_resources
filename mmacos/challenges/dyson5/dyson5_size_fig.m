function dyson5_size_fig(OUT, fname)
%DYSON5_SIZE_FIG  The block-size trade figure (BRIEF_ccmac_dyson_size deliverable 2).
%
%   DYSON5_SIZE_FIG() loads dyson5_size.mat and writes dyson5_size.png.
%   DYSON5_SIZE_FIG(OUT, fname) uses a passed result and output name.
%
%   Three panels against block radius, one curve per family (A silica/54 mm,
%   B CaF2/54 mm, C silica/27 mm, C CaF2/27 mm): CRF (px), ensquared energy
%   (1 px), and edged rod mass (kg).  The R4-of-record values are horizontal
%   reference lines (CRF 1.327, EE 0.759); the closure bar (CRF 1.33, EE 0.76)
%   is the dashed line.  A filled marker closes (matches R4); an open marker
%   fails or sits on a bound.  Only the primary 'solve' rows are drawn (the
%   bound-scaled re-runs stay in the table).
    here = fileparts(mfilename('fullpath'));
    if nargin < 1 || isempty(OUT)
        S = load(fullfile(here, 'dyson5_size.mat'));  OUT = S.OUT;
    end
    if nargin < 2 || isempty(fname), fname = fullfile(here, 'dyson5_size.png'); end
    T = OUT.table;
    T = T(strcmp(T.variant, 'solve'), :);                 % primary rows only

    fams = unique(T.family, 'stable');
    cols = lines(max(4, numel(fams)));
    lab  = containers.Map({'A', 'B', 'C-silica', 'C-CaF2'}, ...
                          {'A silica 54 mm', 'B CaF2 54 mm', 'C silica 27 mm', 'C CaF2 27 mm'});

    f = figure('Visible', 'off', 'Position', [40 40 1500 460], 'Color', 'w');

    axl = {'CRF (px, system LSF \otimes pixel)', 'ensquared energy (1 px, geometric)', 'edged rod mass (kg)'};
    flds = {'CRF', 'EE', 'edged_kg'};
    recs = [OUT.record.CRF, OUT.record.EE, NaN];
    bars = [1.33, 0.76, NaN];                             % the closure bar
    for p = 1:3
        subplot(1, 3, p);  hold on;  hleg = [];  lnames = {};
        for i = 1:numel(fams)
            sel = strcmp(T.family, fams{i});
            r   = T.r_mm(sel);  y = T.(flds{p})(sel);  cl = T.closes(sel);
            good = ~isnan(y);
            hp = plot(r(good), y(good), '-', 'Color', cols(i,:), 'LineWidth', 1.4);
            % closed = filled, failed/on-bound = open
            plot(r(cl & good),  y(cl & good),  'o', 'Color', cols(i,:), 'MarkerFaceColor', cols(i,:), 'MarkerSize', 6);
            plot(r(~cl & good), y(~cl & good), 'o', 'Color', cols(i,:), 'MarkerFaceColor', 'w', 'MarkerSize', 6);
            hleg(end+1) = hp;  %#ok<AGROW>
            if isKey(lab, fams{i}), lnames{end+1} = lab(fams{i}); else, lnames{end+1} = fams{i}; end  %#ok<AGROW>
        end
        if ~isnan(recs(p)) && ~isnan(bars(p)) && abs(recs(p) - bars(p)) <= 0.01
            yline(recs(p), '-', sprintf('R4 record / closure %.3g', recs(p)), 'Color', [.4 .4 .4]);   % the two coincide (closure = R4 rounded)
        else
            if ~isnan(recs(p)), yline(recs(p), '-',  sprintf('R4 record %.3g', recs(p)), 'Color', [.4 .4 .4]); end
            if ~isnan(bars(p)), yline(bars(p), '--', sprintf('closure %.3g', bars(p)), 'Color', [.4 .4 .4]); end
        end
        grid on;  xlabel('block radius (mm)');  ylabel(axl{p});  set(gca, 'XDir', 'reverse');
        if p == 1, legend(hleg, lnames, 'Location', 'northwest', 'FontSize', 8); end
        if p == 1, title('response (filled = matches R4)'); elseif p == 2, title('ensquared energy'); else, title('mass of the edged block'); end
    end
    sgtitle('dyson5: how small can the Dyson block be at R4 performance?  (continuation walks, engine scores)', 'FontSize', 11);
    print(f, fname, '-dpng', '-r130');  close(f);
    fprintf('dyson5_size_fig: wrote %s\n', fname);
end
