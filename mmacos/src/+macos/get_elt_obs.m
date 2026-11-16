function obs = get_elt_obs(k)
%MACOS.GET_ELT_OBS  Obscuration declarations of element K (loaded Rx).
%   obs = macos.get_elt_obs(K) returns the element's declared
%   obscurations (a perforated primary's central hole, a coronagraph
%   mask, a spider blade, ...):
%     .n      declared count (engine nObs; may exceed the returned N)
%     .type   1xN ObsType codes as stored (+/-1 Circle/NegCircle,
%             +/-2 Rectangle, +/-3 Ellipse, +/-4 Triangle, ... --
%             positive blocks, negative transmits)
%     .vec    6xN ObsVec as stored (Circle: radius, xc, yc in the
%             element's xObs frame)
%
%   See also: macos.get_elt_info, macos.view_rx.
arguments
    k (1,1) double {mustBeInteger, mustBePositive}
end
mo = 10;                                   % buffer; engine max is mObs
[n, ty, vc] = mmacos('elt_obs_get', ...
    zeros(mo,1), zeros(6,mo), double(k), double(mo));
nn = min(n, mo);
obs = struct('n', n, 'type', ty(1:nn).', 'vec', vc(:, 1:nn));
end
