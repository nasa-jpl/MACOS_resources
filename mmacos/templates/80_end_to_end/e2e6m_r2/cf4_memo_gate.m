function cf4_memo_gate()
%CF4_MEMO_GATE  The brief's S4 gate: set_lambda mask memoization is lambda-correct.
%
%   A stale memo is silent and plausible: the mask cache could return the
%   wrong lambda's mask and every downstream number stays credible.  The
%   memo lives in cf_chain closure state, so it is observed END TO END
%   through the runner: a memo-hit revisit must reproduce the cold-build
%   field BIT-EXACTLY, and distinct lambdas must differ.
%   Measured at gate time (2026-08-27): revisit 0.0, cross-lambda 2.5e-2.
%
%   See also CF_CHAIN, CF4_PHYSICS.
cd(fileparts(mfilename('fullpath')));
run('../../../mmacos_setup.m');
addpath('../../30_instruments/bench_ctb');
P = e2e6m_r2_params();
C1 = load('cf1_run.mat');
FC = struct();
for k = 1:numel(C1.OUT.F), FC.(C1.OUT.F(k).key) = C1.OUT.F(k); end
% the hard family exercises the lambda-dependent FPM rebuild path
ch = cf_chain('rx', 'r1_seg_dm.in', 'model_size', P.dj.model, ...
              'prolate_iter', P.co.prolate_iter, ...
              'circ_stop_frac', P.cf.circ_stop_frac, FC.hard.cfg{:});
ch.set_lambda(0.90);  E09a = ch.run();   % cold build at 0.90
ch.set_lambda(1.10);  E11  = ch.run();   % cold build at 1.10
ch.set_lambda(0.90);  E09b = ch.run();   % MEMO HIT at 0.90
d_self  = max(abs(E09a(:) - E09b(:)));
d_cross = max(abs(E09a(:) - E11(:)));
fprintf('MEMOGATE: memo-hit revisit differs by %.3e (must be 0); cross-lambda %.3e (must be >0)\n', ...
        d_self, d_cross);
assert(d_self == 0 && d_cross > 1e-12);
ch.set_lambda(1.0);
fprintf('MEMOGATE PASS\n');
end
