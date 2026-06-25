function fig = plot_dw_channels(out, ttl, here, pngname)
%PLOT_DW_CHANNELS  One subplot per channel: each channel's MULTI-FIELD dW.
%   fig = plot_dw_channels(OUT, TTL) renders, for every column of the
%   multi-field Jacobian OUT.dwdxall, that channel's wavefront sensitivity
%   reconstructed onto the tiled field canvas (macos.v2m on OUT.indxall) --
%   so each subplot shows the sensitivity at ALL field points at once.
%
%   Generic across every dw_d*_multi supervisor: dw_dx_multi,
%   dw_dz_zernike_multi, dw_dsurf_multi and dw_dgrid_multi all expose the
%   same canonical fields (dwdxall / indxall / channel_names), with channel
%   names of the form "Elt N <suffix>" (suffix = Rx.. / MonZern4 / Kr / Grid2).
%
%   Layout: rows = element id (sorted), cols = that element's channels.
%   Source channels (name not starting "Elt") fall into a row id 0.
%
%   OUT       struct from any macos.dw_d*_multi call.
%   TTL       figure title / sgtitle string.
%   HERE      (optional) directory to write PNGNAME into.
%   PNGNAME   (optional) filename; when given, the figure is printed there.
%
%   See also: macos.dw_dgrid_multi, macos.dw_dz_zernike_multi,
%             macos.dw_dsurf_multi, macos.dw_dx_multi, macos.v2m.

J     = out.dwdxall;          % canonical multi-field Jacobian (alias in all 4)
indx  = out.indxall;
names = out.channel_names;
nchan = size(J, 2);

% --- parse "Elt N <suffix>" -> element id + short suffix label ----------
elt = zeros(nchan, 1);
suf = cell(nchan, 1);
for k = 1:nchan
    t = regexp(strtrim(char(names{k})), '^Elt\s+(\d+)\s+(.*)$', 'tokens', 'once');
    if isempty(t)
        elt(k) = 0;  suf{k} = char(names{k});      % source / unparsed
    else
        elt(k) = str2double(t{1});  suf{k} = strtrim(t{2});
    end
end
ue   = unique(elt);                                 % sorted element ids
nrow = numel(ue);
ncol = max(arrayfun(@(e) sum(elt == e), ue));       % widest element row

fig = figure('Name', ttl, 'Position', ...
    [40 40 min(170*ncol + 140, 1850) min(150*nrow + 130, 1150)]);
for r = 1:nrow
    ks = find(elt == ue(r));
    for c = 1:numel(ks)
        subplot(nrow, ncol, (r-1)*ncol + c);
        M = macos.v2m(J(:, ks(c)), indx);
        M(M == 0) = NaN;                            % mask outside the pupils
        h = imagesc(M);  set(h, 'AlphaData', ~isnan(M));
        axis image off;  set(gca, 'Color', 'w');
        if ue(r) == 0
            ttl_k = suf{ks(c)};                     % source channel
        else
            ttl_k = sprintf('E%d %s', ue(r), suf{ks(c)});
        end
        title(ttl_k, 'FontSize', 7, 'Interpreter', 'none');
    end
end
colormap(parula);
sgtitle(ttl, 'Interpreter', 'none');

if nargin >= 4 && ~isempty(pngname)
    if nargin < 3 || isempty(here), here = pwd; end
    print(fig, fullfile(here, pngname), '-dpng', '-r140');
    fprintf('wrote %s\n', fullfile(here, pngname));
end
end
