% e2e_pie_init.m -- populate the GMI demo workspace for the e2e_pie
% 7-segment pie-PM three-mirror telescope (templates/80_end_to_end/e2e).
%
% Replaces the retired GMI param initializer: unpacks the slimmed
% regression initializer regression/lib/init_e2e_pie.m into the loose
% workspace variables the legacy test_gmi.m demos expect
% (param, mprb, mpzern, mpgrid, InfFcnZern, InfFcnGrid).  Point the
% demos at this script instead of the old interactive initializer.

here = fileparts(mfilename('fullpath'));
addpath(fullfile(here, 'regression', 'lib'));

[param, prb0, pzern0, pgrid0, InfFcnZern, InfFcnGrid] = init_e2e_pie();

% Channel-vector sizes (init_e2e_pie computes these internally; recompute
% them here for the loose-variable demo interface).
numseg = 7;                              % 7 pie PM segments
mprb   = size(param.rbSrf, 1) * 6;       % 6-DOF per rigid-body optic
mpzern = numseg * param.mzern;           % Zernike modes per segment
mpgrid = param.mgrid^2 * numseg;         % grid nodes per segment
mpdm   = 90 * numseg;                    % (DM channel, unused by these demos)
