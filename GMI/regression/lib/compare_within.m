function [pass, max_diff, msg] = compare_within(a, b, tol, label)
% COMPARE_WITHIN  Hard absolute-tolerance equality check.
%
%   [pass, max_diff, msg] = compare_within(a, b, tol, label)
%
% Inputs:
%   a, b   -- arrays of equal shape (numeric, real)
%   tol    -- max allowed |a - b| (absolute)
%   label  -- short string for the failure message
%
% Returns:
%   pass     -- true if max(|a - b|) <= tol
%   max_diff -- max(|a - b|) (the actual measurement)
%   msg      -- one-line description suitable for printing
%
% Hard-absolute rather than relative because the reference values are
% generated from the same build (bootstrap_reference) and should match
% bit-for-bit modulo a small floor.  If the floor needs raising for a
% specific test, pass a larger tol -- don't switch to relative.

    if ~isequal(size(a), size(b))
        pass = false;
        max_diff = NaN;
        msg = sprintf('[%s] shape mismatch: %s vs %s', label, ...
                      mat2str(size(a)), mat2str(size(b)));
        return
    end

    d = abs(a(:) - b(:));
    max_diff = max(d);
    if isempty(max_diff)
        max_diff = 0;
    end

    pass = max_diff <= tol;
    if pass
        msg = sprintf('[%s] PASS  max|a-b| = %.3e (tol %.3e)', ...
                      label, max_diff, tol);
    else
        msg = sprintf('[%s] FAIL  max|a-b| = %.3e > tol %.3e', ...
                      label, max_diff, tol);
    end
end
