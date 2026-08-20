function fpa = oi_apply_fpa(X)
%OI_APPLY_FPA  Apply the FPA refit deltas to the closure's base pose.
%
%   FPA = OI_APPLY_FPA(X) returns X.fpa with the stage refit deltas
%   X.fpa_refit = [dz_m, tilt_deg] applied: dz slides the plane along its
%   own normal (focus), tilt rotates the normal about global x through
%   the vertex (the FPA tilt freedom Mike's rungs allow).  Absent or
%   zero deltas return the base pose unchanged.
%
%   See also OI_CLOSE, OI_SOLVE.

    fpa = X.fpa;
    if ~isfield(X,'fpa_refit') || all(X.fpa_refit == 0), return; end
    dz = X.fpa_refit(1);  th = X.fpa_refit(2);
    n = fpa.psi(:)/norm(fpa.psi);
    fpa.Vpt = fpa.Vpt(:) + dz*n;
    Rx = [1 0 0; 0 cosd(th) -sind(th); 0 sind(th) cosd(th)];
    fpa.psi = Rx*n;
end
