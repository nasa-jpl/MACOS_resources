function out = strict_rung(t, Frel, lam_nm)
%STRICT_RUNG  Dave's strict WFE metric (2026-07-30), computed by the
%   sanctioned engine loop: for each field, set the field, run FEX to TIE
%   the exit-pupil reference sphere to that field's chief-ray intercept on
%   the fixed detector, read macos.opd() at the exit pupil, take RMS.
%
%   Dave (2026-07-30): "FEX does not optimize WFE -- it FINDS the pupil
%   associated with each field point, tying it to the ray intercept on the
%   fixed FP surface."  So FEX-per-field IS the ruled metric: per-field
%   reference sphere centered at the chief-ray detector intercept, piston-
%   only removal (macos.opd() is already piston-removed, SUBROUTINE OPD
%   tracesub.F:226), focus kept, no 2*rho^2-1 fit -> no f/0.86 artifact.
%   The image-displacement tilt is removed geometrically by the chief tie
%   (the sphere axis follows that field's chief ray), leaving focus + astig
%   + the tilt-like residual coma injects.
%
%   Requires t to have an exit pupil (add_pupil) and >3 elements.  The
%   system stop is set at M1 so FEX can aim the chief ray.

    pu  = t.spec.pupil;  iEP = pu.ep_elt;
    nF  = size(Frel,1);
    rms = nan(nF,1);  nk = zeros(nF,1);
    for j = 1:nF
        t.trace_at_field(Frel(j,:));     % set + emit this field
        macos.stop(1);                    % chief through M1 (system stop)
        try
            macos.fex(1);                 % TIE EP sphere to this field's chief intercept
        catch
            % FEX unavailable -> leave the fixed sphere (records via NaN guard)
        end
        macos.trace(iEP);                 % OPD on that field's chief-tied sphere
        W = macos.opd();  v = W(isfinite(W) & W~=0 & abs(W)<1e30);
        nk(j) = numel(v);
        if numel(v) >= 8, rms(j) = std(v) * lam_nm; end   % opd() already piston-removed
    end
    t.trace_at_field([]);
    out = struct('rms',rms,'max',max(rms),'avg',mean(rms(isfinite(rms))),'n',nk);
end
