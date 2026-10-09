function s = normalize_zero_signs(s)
%NORMALIZE_ZERO_SIGNS  Strip the sign off negative-zero numeric literals.
%   Fortran WRITE prints a zero as -0.0E+00 or 0.0E+00 depending on the
%   IEEE sign bit, which is FP-round-off dependent (e.g. a -1e-18 that
%   rounds to zero keeps its sign) and physically inconsequential -- but
%   it randomly breaks byte-identity regressions.  Filter both sides of
%   a file comparison through this before verifyEqual.
    s = regexprep(s, '-(?=0(\.0+)?E\+00(?![0-9]))', '');
end
