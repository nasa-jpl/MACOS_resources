function calib_clear_var_elts()
%MACOS.CALIB_CLEAR_VAR_ELTS  Wipe all CALIB variable-element state.
%   Use before defining a fresh variable-element list with
%   macos.calib_set_var_elt; otherwise subsequent set calls accumulate
%   on top of whatever the prescription's VarDOF= keywords already set.
mmacos('calib_clear_var_elts');
end
