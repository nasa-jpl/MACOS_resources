function result = calib()
%MACOS.CALIB  Run the macos design optimizer (CALIB).
%   result = macos.calib() invokes the SMACOS CALIB command using the
%   current optimization configuration (variable elements, FOVs,
%   wavelengths, target, iteration cap, tolerance).  The config can be
%   baked into the Rx file (OptTarget=, OptMxItrs=, VarDOF=, etc.) or
%   set programmatically via the macos.calib_set_* family.
%
%   Returns a struct with fields:
%     .converged    (logical)  true if rtn_flag == 0
%     .rtn_flag     (int)      0 = converged, nonzero = failure
%     .n_fov        (int)      # field-of-view points used
%     .n_wavelength (int)      # wavelengths used
%     .old_wfe      (n_fov x n_wl double)  RMS WFE before optim
%     .new_wfe      (n_fov x n_wl double)  RMS WFE after optim
%
%   The WFE arrays are zero for non-WFE_TARGET runs (BEAM / SPOT / OPL
%   targets fill different state vars not yet surfaced -- Phase 1b.2).
%
%   Example workflow (programmatic config):
%     m = macos.Session(256);
%     m.load_rx('cass_design.in');
%     m.calib_clear_var_elts();
%     m.calib_set_var_elt(7, 'TIP', 'TILT');
%     m.calib_set_iter(50);
%     m.calib_set_tol(1e-10);
%     m.calib_set_target('WFE');
%     result = m.calib();
%
%   See also: macos.calib_set_var_elt, macos.calib_clear_var_elts,
%             macos.calib_set_iter, macos.calib_set_tol,
%             macos.calib_set_target.

[rtn_flag, n_fov, n_wl, old_wfe, new_wfe] = mmacos('calib_run');

result = struct( ...
    'converged',    rtn_flag == 0, ...
    'rtn_flag',     double(rtn_flag), ...
    'n_fov',        double(n_fov), ...
    'n_wavelength', double(n_wl), ...
    'old_wfe',      old_wfe(1:n_fov, 1:n_wl), ...
    'new_wfe',      new_wfe(1:n_fov, 1:n_wl));
end
