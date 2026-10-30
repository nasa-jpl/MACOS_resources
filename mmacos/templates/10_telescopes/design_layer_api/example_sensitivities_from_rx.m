% example_sensitivities_from_rx.m
% -------------------------------------------------------------------
% Sprint 2A-i headline example: import a prescription and produce a
% sensitivity table — the design layer's first deliverable and the
% Thrust A headline recipe (PLAN.md §1.3) at the package level.
%
% Flow:  System.from_rx  ->  describe  ->  sensitivities  ->  table
%
% The same flow runs on a CodeV/Zemax-converted Rx unchanged — that is
% the expected dominant entry point (PLAN_DESIGN_LAYER §1.0).  Here we
% use e5hex1.in (rigid-body + Zernike elements) so both sensitivity
% families appear.
%
% Run:  matlab -batch "run('.../example_sensitivities_from_rx.m')"
% Ends in exit(0) (batch-mode rule, mmacos/CLAUDE.md).

addpath(fullfile(fileparts(mfilename('fullpath')), '..', '..', '..', 'src'));   % +macos on path

here = fileparts(mfilename('fullpath'));
if isempty(here), here = pwd; end
rx = fullfile(here, '..', '..', '50_sensitivities', 'e5hex1', 'e5hex1.in');

% 1. Import (engine readback — no MATLAB text parser).
s = macos.design.System.from_rx(rx, 'model_size', 128);
s.describe();

% 2. Sensitivities — harvests the bitwise-verified Phase 7 drivers.
sens = s.sensitivities();

% 3. Report.  The two Jacobians share the same Nw wavefront rows, so a
%    user who wants one matrix just concatenates them.
fprintf('\n=== Sensitivity table: %s ===\n', rx);
fprintf('  wavefront samples (Nw) : %d\n', size(sens.rigid.dwdx, 1));
fprintf('  rigid-body channels    : %d\n', size(sens.rigid.dwdx, 2));
fprintf('  Zernike channels       : %d\n', size(sens.zern.dwdz, 2));

J = [sens.rigid.dwdx, sens.zern.dwdz];          % combined Nw x Ncol
names = [sens.rigid.channel_names(:); sens.zern.channel_names(:)];
rms_per_channel = sqrt(mean(J.^2, 1));           % RMS OPD response per DOF

fprintf('\n  per-channel RMS OPD response (m per SI perturbation):\n');
[~, ord] = sort(rms_per_channel, 'descend');
for ii = 1:min(8, numel(ord))
    j = ord(ii);
    fprintf('   %-28s  %.3e\n', names{j}, rms_per_channel(j));
end
fprintf('   ... (%d channels total)\n', numel(rms_per_channel));

exit(0)
