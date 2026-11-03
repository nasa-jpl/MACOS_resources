function c = iodc2010_curv(L, k)
%IODC2010_CURV  Signed curvature of a green-lens surface (1/mm; 0 = plano).
%   c = iodc2010_curv(L) returns 1xN; c = iodc2010_curv(L, k) one surface.
%   A green lens: sgn / R.  A RELAXED lens (iodc2010_relax, L.Rk present):
%   1 / Rk, 0 where Rk is Inf.  Every reader of the prescription's shape
%   (iodc2010_paraxial, iodc2010_emit, iodc2010_seq) goes through here so the
%   two representations cannot disagree.
if isfield(L, 'Rk') && ~isempty(L.Rk)
    c = 1 ./ L.Rk;  c(~isfinite(L.Rk)) = 0;
else
    c = L.sgn / L.R;
end
if nargin > 1, c = c(k); end
end
