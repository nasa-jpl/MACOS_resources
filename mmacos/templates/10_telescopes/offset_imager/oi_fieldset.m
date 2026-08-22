function F = oi_fieldset(P, offset_deg, n)
%OI_FIELDSET  n x n field grid over the box (CODE V XAN/YAN, deg).
%
%   F = OI_FIELDSET(P, OFFSET_DEG, N) returns the N^2 x 2 [XAN YAN] list
%   covering the full P.box_deg box centred at YAN = OFFSET_DEG.  N = 1
%   returns the box centre alone.  The solve set uses N = P.nsolve, the
%   dense report map N = P.map_n -- solve set != scoring set, always.
%
%   See also OFFSET_IMAGER_PARAMS, OI_SCORE.

    if n == 1
        F = [0, offset_deg];
        return
    end
    xg = linspace(-P.box_deg(1)/2, P.box_deg(1)/2, n);
    yg = offset_deg + linspace(-P.box_deg(2)/2, P.box_deg(2)/2, n);
    [XG, YG] = meshgrid(xg, yg);
    F = [XG(:), YG(:)];
end
