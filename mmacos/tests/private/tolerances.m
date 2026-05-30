function t = tolerances()
%TOLERANCES  Shared (abs, rel) tolerance constants for the mmacos suite.
%   Mirrors pymacos tests/test_settings.py:_Tol.  Use these as the
%   default tolerance class for assertions so the whole suite has a
%   consistent precision contract — individual tests can override
%   locally when a specific computation needs tighter or looser bounds.
%
%   Returned struct fields:
%       .P    [abs, rel] position tolerance      (1e-10, 1e-10)
%       .r    [abs, rel] direction tolerance     (1e-13, 1e-13)
%       .L    [abs, rel] path-length tolerance   (1e-11, 1e-11)
%       .v    [abs, rel] generic value tolerance (1e-15, 1e-15)
%       .eps  double-precision unit (2*eps(1))   ≈ 4.44e-16
%
%   Convention: assertions use the FIELD that names the physical
%   quantity being compared (positions, directions, path lengths,
%   numeric values).  Example:
%       tol = tolerances();
%       testCase.verifyEqual(vpt_actual, vpt_expected, ...
%           'AbsTol', tol.P(1), 'RelTol', tol.P(2));
t.P   = [1e-10, 1e-10];
t.r   = [1e-13, 1e-13];
t.L   = [1e-11, 1e-11];
t.v   = [1e-15, 1e-15];
t.eps = 2 * eps(1.0);
end
