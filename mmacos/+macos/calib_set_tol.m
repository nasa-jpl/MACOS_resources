function calib_set_tol(tol)
%MACOS.CALIB_SET_TOL  Set CALIB convergence tolerance (dopt_tol).
t = double(tol);
if t <= 0
    error('macos:calib_set_tol:bad', 'tol must be > 0 (got %g)', t);
end
mmacos('calib_set_tol', t);
end
