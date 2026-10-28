function out = obs(opt)
%MACOS.OBS  Query or set the ray-trace obscuration option (OBS).
%   OPT = macos.obs() returns the option in force (api obs_opt_get):
%   'all' (every ray), 'positive' (unobscured rays only: the engine
%   default) or 'negative' (obscured rays only).
%   macos.obs(OPT) sets it (OBS).  Persists until changed or a load_rx,
%   which resets it to POSITIVE and says so (Dave 2026-10-28).  Twin of
%   pymacos obs_set (Dave 2026-10-16).
%
%   Raw equivalents: mmacos('obs_opt_get'), mmacos('obs_set', CODE),
%   CODE 0 = all, 1 = positive, 2 = negative.
arguments
    opt (1,:) char {mustBeMember(opt, {'all','positive','negative',''})} = ''
end
names = {'all', 'positive', 'negative'};
if isempty(opt)
    out = names{mmacos('obs_opt_get') + 1};
    return
end
mmacos('obs_set', double(find(strcmp(names, opt)) - 1));
end
