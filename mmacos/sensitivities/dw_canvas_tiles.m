function tiles = dw_canvas_tiles(out)
%DW_CANVAS_TILES  [rows cols] of FIELD TILES in a harvest's canvas.
%   The multi-field Jacobian lives on a tiled canvas: one tile per field
%   point, and with a CONFIGURATION axis one outer tile per configuration
%   holding that configuration's whole field canvas (macos.config_canvas).
%   The tile count is what a canvas PANEL has to be sized by -- the jwst
%   zoom fixture's 5 configurations x 5 fields is a 9 x 9 tile canvas, so
%   a panel drawn at a single map's size shows each field at a ninth of
%   it.
%
%   Derived from the canvas size over one field's map, so it needs no
%   knowledge of the tiling rule and follows 'orient','xy' (which
%   transposes both).  Returns [1 1] when the harvest carries no
%   per-field map or the ratio is not integral.
%
%   See also: dw_page_layout, macos.config_canvas.

tiles = [1 1];
if ~isfield(out, 'per_field_w_nom_2d') || isempty(out.per_field_w_nom_2d)
    return
end
m = size(out.per_field_w_nom_2d{1});
t = out.indxall.size ./ m;
if all(abs(t - round(t)) < 1e-9) && all(round(t) >= 1)
    tiles = round(t);
end
end
