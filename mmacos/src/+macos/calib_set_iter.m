function calib_set_iter(n_iter)
%MACOS.CALIB_SET_ITER  Set CALIB iteration cap (nitrs_dopt).
n = double(n_iter);
if n < 1
    error('macos:calib_set_iter:bad', 'n_iter must be >= 1 (got %g)', n);
end
mmacos('calib_set_iter', n);
end
