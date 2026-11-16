function st = calib_stats()
%MACOS.CALIB_STATS  Iteration count and final merit of the last CALIB.
%   st = macos.calib_stats() returns a struct:
%     .n_iter  (double)  LM iterations the last CALIB took; -1 when none ran
%                        (no CALIB yet, or it stopped before the solve); the
%                        linear (SVD) path reports its count
%     .merit   (double)  the final LM merit -- the chisq the solve minimises,
%                        in target units squared summed over fields and
%                        wavelengths; -1 when not available (linear path,
%                        or a failure before any accepted step)
%     .stop_reason (char) 'converged' (two flat accepted steps), 'floor'
%                        (ten rejected trials: the round-off floor), 'cap'
%                        (the iteration cap), 'failed', 'stalled' (no
%                        trial step lowered the merit and none failed: the
%                        start is stationary -- e.g. already optimal; it
%                        counts as converged), 'none'
%     .n_rejected (double) LM trials rejected (the damping grew); -1 none
%     .n_failed  (double)  of those, trials whose evaluation FAILED -- a trace
%                        lost the chief ray; such a trial is REJECTED (a
%                        smaller step follows), not an abort (PLAN 3.4)
%   macos.calib returns the same fields; this reads them after a FAILED
%   run, which macos.calib reports as an error (the iterations it completed,
%   the merit at its last accepted step).  PLAN_CONSOLIDATION 3.3.
%
%   See also: macos.calib.

[n, m, why, nrej, nfail] = mmacos('calib_stats_get');
names = {'converged', 'floor', 'cap', 'failed', 'stalled'};
reason = 'none';
if why >= 1 && why <= 5, reason = names{why}; end
st = struct('n_iter', double(n), 'merit', double(m), 'stop_reason', reason, ...
            'n_rejected', double(nrej), 'n_failed', double(nfail));
end
