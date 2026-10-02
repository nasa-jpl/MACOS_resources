function F = oi_fieldset(P, offset_deg, n)
%OI_FIELDSET  n x n field grid over the box (CODE V XAN/YAN, deg).
%
%   F = OI_FIELDSET(P, OFFSET_DEG, N) returns the N^2 x 2 [XAN YAN] list
%   covering the full P.box_deg box centred at YAN = OFFSET_DEG.  N = 1
%   returns the box centre alone.  The solve set uses N = P.nsolve, the
%   dense report map N = P.map_n -- solve set != scoring set, always.
%   N = [NX NY] (opt-in) gives NX across XAN by NY across YAN -- for a
%   thin strip box (a push-broom slit) where an N x N grid would spend
%   N rows on a fraction of a degree.  Scalar N is the record path.
%
%   See also OFFSET_IMAGER_PARAMS, OI_SCORE.

    if isscalar(n) && n == 1
        F = [0, offset_deg];
        return
    end
    if isscalar(n), n = [n n]; end
    xg = linspace(-P.box_deg(1)/2, P.box_deg(1)/2, n(1));
    yg = offset_deg + linspace(-P.box_deg(2)/2, P.box_deg(2)/2, n(2));
    if n(1) == 1, xg = 0; end
    if n(2) == 1, yg = offset_deg; end
    [XG, YG] = meshgrid(xg, yg);
    F = [XG(:), YG(:)];
end
