function idx = per_field_indx(out, k)
%PER_FIELD_INDX  m2v bookkeeping for ONE field of a dw_d*_multi harvest, in
%   the orientation the harvest was delivered in.
%   idx = per_field_indx(OUT, K) returns the index struct (.i/.j/.size) that
%   scatters the rows of OUT.per_field_dwd{x,z,s,g}{K} onto OUT's
%   per_field_w_nom_2d{K} grid.
%
%   Why this exists (Luis round 4, 2026-09-10): the per-field Jacobian
%   rows are built in the RAW orientation's m2v order and never reordered;
%   'orient','xy' transposes per_field_w_nom_2d and remaps indx/indxall
%   (apply_opd_convention) but the per-field cell carries no index of its
%   own.  Rebuilding one with m2v on the TRANSPOSED nominal map enumerates
%   the pixels in a different order, so the rows land on the wrong pixels
%   and a single-segment poke smears into diagonal streaks -- the
%   "residual" seen on the per-element centre-field pages under orient xy.
%   The right index is m2v of the map in RAW orientation, remapped with the
%   same rule apply_opd_convention uses (swap subscripts, flip size).
W = out.per_field_w_nom_2d{k};
if isfield(out, 'opd_orient') && strcmp(out.opd_orient, 'xy')
    [~, idx] = macos.m2v(W.');                       % the order the rows were built in
    idx = struct('i', idx.j, 'j', idx.i, 'size', fliplr(idx.size));   % onto the xy grid
else
    [~, idx] = macos.m2v(W);
end
end
