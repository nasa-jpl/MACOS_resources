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
%                        (the iteration cap), 'failed', or 'none'
%   macos.calib returns the same fields; this reads them after a FAILED
%   run, which macos.calib reports as an error (the iterations it completed,
%   the merit at its last accepted step).  PLAN_CONSOLIDATION 3.3.
%
%   See also: macos.calib.

[n, m, why] = mmacos('calib_stats_get');
names = {'converged', 'floor', 'cap', 'failed'};
reason = 'none';
if why >= 1 && why <= 4, reason = names{why}; end
st = struct('n_iter', double(n), 'merit', double(m), 'stop_reason', reason);
end
