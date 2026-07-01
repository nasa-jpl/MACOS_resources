function plot_dw_per_element(out, fieldmode, here, prefix)
%PLOT_DW_PER_ELEMENT  One page per element: that element's dW/d(DOF) channels.
%   plot_dw_per_element(OUT, FIELDMODE, HERE, PREFIX) writes one PNG per
%   element, with one subplot per channel (DOF / mode / parameter) on that
%   element.  This is the "single page per element" companion to
%   plot_dw_channels (the all-channels overview), and is GENERIC across every
%   dw_d*_multi supervisor:
%     dw_dx_multi    -> per optic   x {Rx Ry Rz Tx Ty Tz}
%     dw_dz_*_multi  -> per optic   x Zernike modes
%     dw_dsurf_multi -> per optic   x {Kr, Kc}
%     dw_dgrid_multi -> per segment x grid-influence modes
%
%   FIELDMODE = 'center' : the CENTER-field dW (single-field image).
%             = 'multi'  : the MULTI-field tiled canvas.
%
%   Display matches plot_dw_channels: parula, zeros masked white, each subplot
%   auto-scaled -- no thresholding / caxis band-aids.
%
%   OUT     struct from any macos.dw_d*_multi call.  Uses the canonical fields
%           dwdxall / indxall / iElt / channel_names / field_names, plus (for
%           'center') per_field_w_nom_2d and the per-field cell
%           per_field_dwd{x,z,s,g} (auto-detected).
%   PREFIX  filename stem; writes <PREFIX>_elt<N>_<FIELDMODE>.png per element
%           (channels with no element id -- e.g. source -- go to <PREFIX>_src_*).
%
%   See also: plot_dw_channels, plot_opd_canvas, macos.dw_dgrid_multi.

if strcmpi(fieldmode, 'center')
    ctr = find(strcmp(out.field_names, 'C'), 1);  if isempty(ctr), ctr = 1; end
    [~, idx] = macos.m2v(out.per_field_w_nom_2d{ctr});
    src = local_per_field(out, ctr);              % per-field cell (name varies)
    tag = 'center field';
elseif strcmpi(fieldmode, 'multi')
    src = out.dwdxall;
    idx = out.indxall;
    tag = sprintf('%d fields', numel(out.field_names));
else
    error('plot_dw_per_element:mode', 'FIELDMODE must be ''center'' or ''multi''');
end

ue = unique(out.iElt(:)).';
for ie = ue
    cols = find(out.iElt == ie);
    nm = numel(cols);  nc = ceil(sqrt(nm));  nr = ceil(nm / nc);
    if ie == 0, etag = 'src';  ttl_e = 'source';
    else,       etag = sprintf('elt%d', ie);  ttl_e = sprintf('element %d', ie);
    end
    f = figure('Visible', 'off', 'Position', [40 40 1400 950]);
    for j = 1:nm
        c = cols(j);
        M = macos.v2m(src(:, c), idx);
        M(M == 0) = NaN;                          % mask outside the pupils (white)
        subplot(nr, nc, j);
        h = imagesc(M);  set(h, 'AlphaData', ~isnan(M));
        axis image off;  set(gca, 'Color', 'w');  colorbar;  colormap(parula);
        title(strtrim(char(out.channel_names{c})), 'FontSize', 8, 'Interpreter', 'none');
    end
    sgtitle(sprintf('%s -- dW, %s, %s', prefix, ttl_e, tag), 'Interpreter', 'none');
    png = sprintf('%s_%s_%s.png', prefix, etag, fieldmode);
    print(f, fullfile(here, png), '-dpng', '-r140');
    close(f);
    fprintf('wrote %s\n', png);
end
end

% -----------------------------------------------------------------------------
function pf = local_per_field(out, k)
% Return the k-th single-field dW block; the cell field name is supervisor-
% specific (per_field_dwdx/dwdz/dwds/dwdg), so detect it.
cand = {'per_field_dwdx', 'per_field_dwdz', 'per_field_dwds', 'per_field_dwdg'};
for i = 1:numel(cand)
    if isfield(out, cand{i}), pf = out.(cand{i}){k};  return;  end
end
error('plot_dw_per_element:nofield', ...
      'no per_field_dwd{x,z,s,g} cell in OUT -- cannot build center-field view');
end
