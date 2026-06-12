% example_align_from_rx.m
% -------------------------------------------------------------------
% Sprint 2A-i, the full story for the manual: import a prescription,
% inspect its sensitivities, then ALIGN it — declare design variables
% and let the optimizer drive a misaligned element back to minimum WFE.
%
%   from_rx  ->  sensitivities  ->  vary  ->  evaluate  ->  optimize
%
% The same flow runs unchanged on a CodeV/Zemax-converted Rx — the
% expected dominant entry point (PLAN_DESIGN_LAYER §1.0).  e5hex1.in is
% used here so it is self-contained.
%
% Run:  matlab -batch "run('.../example_align_from_rx.m')"     (ends exit(0))

addpath(fullfile(fileparts(mfilename('fullpath')), '..', '..'));   % +macos on path

here = fileparts(mfilename('fullpath'));
if isempty(here), here = pwd; end
rx = fullfile(here, '..', 'sensitivities', 'e5hex1', 'e5hex1.in');

%% 1. Import (engine readback — no MATLAB text parser).
s = macos.design.System.from_rx(rx, 'model_size', 128);
fprintf('=== Imported %s ===\n', rx);
fprintf('  elements: %d\n', s.n_elt());

%% 2. Sensitivities — which DOFs move the wavefront most?
sens = s.sensitivities('families', {'rigid'});
rms  = sqrt(mean(sens.rigid.dwdx.^2, 1));
[~, ord] = sort(rms, 'descend');
fprintf('\n  top rigid-body sensitivities (RMS OPD per SI perturbation):\n');
for ii = 1:min(5, numel(ord))
    fprintf('   %-22s %.3e\n', sens.rigid.channel_names{ord(ii)}, rms(ord(ii)));
end

%% 3. Declare a design variable — element 2 despace (local frame).
ELT = 2;
s.vary(ELT, 'despace', 'bounds', [-1 1], 'unit', 'mm');

%% 4. Evaluate: nominal vs a deliberate 0.1 mm misalignment.
m_nom = s.evaluate(0).merit;
m_off = s.evaluate(0.10).merit;
fprintf('\n  WFE nominal           : %.6e (WaveUnits)\n', m_nom);
fprintf('  WFE +0.1 mm despace   : %.6e\n', m_off);

%% 5. Optimize from the misaligned start back to minimum WFE.
res = s.optimize('x0', 0.10, 'MaxIter', 30);
fprintf('\n  optimize() from the 0.1 mm disturbance:\n');
fprintf('   merit0 (start)       : %.6e\n', res.merit0);
fprintf('   merit_opt (end)      : %.6e\n', res.merit_opt);
fprintf('   recovered despace    : % .4e mm  (target ~0)\n', res.x_opt);
fprintf('   exitflag             : %d\n', res.exitflag);

exit(0)
