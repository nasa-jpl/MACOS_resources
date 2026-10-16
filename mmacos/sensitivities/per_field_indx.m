function idx = per_field_indx(out, ic, k)
%PER_FIELD_INDX  m2v bookkeeping for ONE field of a dw_d*_multi harvest, in
%   the orientation the harvest was delivered in.
%   idx = per_field_indx(OUT, IC, K) returns the index struct (.i/.j/.size)
%   that scatters the rows of OUT.per_field_dwd{x,z,s,g}{IC,K} onto OUT's
%   per_field_w_nom_2d{IC,K} grid -- configuration IC, field K.
%   idx = per_field_indx(OUT, K) is the two-argument form: configuration 1,
%   field K.  With a CONFIGURATION axis the per-field cells are Nc x Nf, so
%   they must be indexed 2-D: a linear {K} lands on (configuration K, field
%   1), which is the centre field only by the accident that 'C' is listed
%   first.  Rays can be lost differently per configuration, so the index
%   genuinely belongs to the block, not to the field alone.
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
if nargin < 3, k = ic;  ic = 1;  end
C = out.per_field_w_nom_2d;
if isvector(C), W = C{k};  else, W = C{ic, k};  end
if isfield(out, 'opd_orient') && strcmp(out.opd_orient, 'xy')
    [~, idx] = macos.m2v(W.');                       % the order the rows were built in
    idx = struct('i', idx.j, 'j', idx.i, 'size', fliplr(idx.size));   % onto the xy grid
else
    [~, idx] = macos.m2v(W);
end
end
