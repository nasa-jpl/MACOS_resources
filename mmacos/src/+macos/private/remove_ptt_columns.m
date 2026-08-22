function [J, coef] = remove_ptt_columns(J, indx, groups)
%REMOVE_PTT_COLUMNS  Project piston + tip + tilt out of each Jacobian column.
%   J = remove_ptt_columns(J, INDX) fits and subtracts a plane -- piston and
%   two tilts, the basis [1, x, y] -- from every column of the wavefront
%   Jacobian J (Nw x Ns), over the aperture coordinates the rows sit on.
%   Each column is one sensitivity response (dW / d parameter); removing its
%   piston + tip + tilt models the part that is NORMALLY ALIGNED OUT during
%   assembly (a Kr/Kc error re-focuses and re-points, and the alignment step
%   takes the global piston and pointing back out), leaving the higher-order
%   figure that survives alignment -- the quantity a sensitivity budget wants.
%
%   Row coordinates come from INDX (the m2v bookkeeping struct, fields .i /
%   .j / .size): row k sits at canvas pixel (INDX.i(k), INDX.j(k)), so x =
%   j, y = i, mean-removed and scaled to O(1) for a well-conditioned fit.
%   Rows that are NaN in a column (a ray that missed for that perturbation)
%   are excluded from that column's fit and left NaN.
%
%   J = remove_ptt_columns(J, INDX, GROUPS) removes PTT over SUB-APERTURES
%   instead of the whole pupil: GROUPS is an Nw x 1 label per row (e.g. the
%   element footprint each ray belongs to), and the plane is fit and removed
%   INDEPENDENTLY within each group.  Use it for a per-optic footprint
%   removal on a segmented pupil (each segment aligned in its own mount);
%   omit it (or pass []) for a single global exit-pupil plane, which is what
%   a full-beam optic (an SM/TM whose footprint IS the pupil) reduces to.
%
%   Returns the PTT-removed J and, optionally, COEF (3 x Ns, or a struct per
%   group) -- the [piston; tiltx; tilty] removed from each column, for report.
%
%   See also: macos.m2v, dw_dsurf, optic_footprints.

if nargin < 3, groups = []; end
Nw = size(J, 1);
assert(numel(indx.i) == Nw && numel(indx.j) == Nw, ...
    'remove_ptt_columns: indx.i/.j (%d/%d) must match J rows (%d)', ...
    numel(indx.i), numel(indx.j), Nw);

% aperture coordinates, mean-removed + scaled so [1 x y] is well conditioned
x = double(indx.j(:));  y = double(indx.i(:));
sc = max([1, max(abs(x - mean(x))), max(abs(y - mean(y)))]);
x = (x - mean(x)) / sc;  y = (y - mean(y)) / sc;

if isempty(groups)
    glab = ones(Nw, 1);            % one global plane
else
    glab = groups(:);
    assert(numel(glab) == Nw, 'remove_ptt_columns: groups must be Nw x 1');
end
ug = unique(glab(~isnan(glab)));

coef = zeros(3, size(J, 2));
for g = ug(:).'
    gm = (glab == g);
    Xg = [ones(nnz(gm),1), x(gm), y(gm)];
    for s = 1:size(J, 2)
        col = J(gm, s);
        ok = isfinite(col);
        if nnz(ok) < 3, continue; end        % too few points to fit a plane
        A = Xg(ok, :);
        c = A \ col(ok);                      % LS piston + 2 tilts
        col(ok) = col(ok) - A * c;
        J(gm, s) = col;
        if numel(ug) == 1, coef(:, s) = c; end
    end
end
end
