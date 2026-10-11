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
%     .n_iter       (int)      LM iterations taken (-1: no LM run)
%     .merit        (double)   final LM merit, the chisq the solve minimises
%                              (-1: none)
%     .stop_reason  (char)     'converged' / 'floor' / 'cap' / 'failed' /
%                              'none' -- see macos.calib_stats
%
%   The WFE arrays are zero for non-WFE_TARGET runs (BEAM / SPOT
%   targets fill different state vars not yet surfaced -- Phase 1b.2).
%   LM stops when two consecutive accepted steps each change the rms
%   merit (sqrt(chisq/rows), target units) by no more than the tolerance
%   (absolute, OptTol / calib_set_tol) AND lower chisq by no more than
%   1e-6 of it, after ten consecutive
%   rejected trials (the round-off floor), or at the iteration cap.  A
%   failed run raises an error; macos.calib_stats then reads its count.
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
st = macos.calib_stats();

result = struct( ...
    'converged',    rtn_flag == 0, ...
    'rtn_flag',     double(rtn_flag), ...
    'n_fov',        double(n_fov), ...
    'n_wavelength', double(n_wl), ...
    'old_wfe',      old_wfe(1:n_fov, 1:n_wl), ...
    'new_wfe',      new_wfe(1:n_fov, 1:n_wl), ...
    'n_iter',       st.n_iter, ...
    'merit',        st.merit, ...
    'stop_reason',  st.stop_reason);
end
