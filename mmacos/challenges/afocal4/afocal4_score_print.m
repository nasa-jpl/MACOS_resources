function afocal4_score_print(P, S, label)
%AFOCAL4_SCORE_PRINT  One S4 score as a target-versus-achieved block.
%
%   Every line carries the metric, its target and the RATIO, because the
%   ratio is what the merit sees and the raw number is what goes in the
%   report.  A term inside its target is marked; the worst ratio is the
%   headline, since the S4 result is a PAIR (image quality AND pupil) and a
%   design that meets five targets and misses one has not met the pair.
%
%   See also AFOCAL4_SCORE.

    if nargin < 3, label = ''; end
    T = P.targets;
    if ~S.ok
        fprintf('  %-14s SCORE FAILED: %s\n', label, S.err);   return;
    end
    if ~isempty(label), fprintf('  %s\n', label); end
    row = @(n,v,t,u) fprintf('    %-22s %10.4g %-4s  target %8.4g   %6.2fx %s\n', ...
                             n, v, u, t, v/t, mark_(v/t));
    row('WFE rung 2 (max)',   S.wfe_max_nm,  T.wfe_rung2_nm,  'nm');
    if isfield(S,'pupil_scored') && ~S.pupil_scored
        fprintf(['    %-22s %10.4g nm   rung 1 (piston only) %.4g nm\n' ...
                 '    WFE-ONLY score -- the pupil ladder was not measured, so ' ...
                 'this is a diagnostic, not a result.\n'], ...
                'rung 3 (+power)', S.wfe_rung3_max_nm, S.wfe_rung1_max_nm);
        if isfield(S,'wfe_grid_max_nm')
            fprintf('    %-22s %10.4g nm    (solve-set max %.4g nm)\n', ...
                    'WFE uniform grid max', S.wfe_grid_max_nm, S.wfe_max_nm);
        end
        return;
    end
    row('pupil blur rms',     S.blur_um,     T.blur_um,       'um');
    row('breathing (chief-N)',S.breathe_pct, T.breathe_pct,   '%');
    row('wander (refit)',     S.wander_um,   T.wander_um,     'um');
    row('surface vs sag',     S.surf_pv_mm,  T.surface_pv_mm, 'mm');
    row('M error at centre',  S.mag_pct,     T.mag_pct,       '%');
    fprintf(['    %-22s %10.4f x   (placed-plane wander %.1f um; refit ' ...
             'shift %+.2f mm, tilt %+.3f deg)\n'], ...
            'M at box centre', S.mag_centre_chief, S.wander_placed_um, ...
            S.pose.shift_mm, S.pose.tilt_deg);
    if isfield(S,'wfe_grid_max_nm')
        fprintf('    %-22s %10.4g nm    (solve-set max %.4g nm)\n', ...
                'WFE uniform grid max', S.wfe_grid_max_nm, S.wfe_max_nm);
    end
    fprintf('    %-22s %10.3f     merit %.4f\n', ...
            'WORST normalised miss', S.worst, S.merit);
    % THE ANCHORING RESIDUAL IS A VALIDITY CHECK ON THE PUPIL NUMBERS, not a
    % metric.  pupil_map regrids each field's exit-side data onto common
    % entrance-surface nodes; when a design scrambles part of the beam the
    % two interpolants disagree, and a residual comparable to the blur means
    % the blur is a number about nothing.  Measured at 0.1 um on every sound
    % design here and at 84 mm on the failed 220 mm trade solve -- which is
    % how that solve was caught.
    if isfinite(S.anchor_resid_um) && S.anchor_resid_um > 0.1*S.blur_um
        fprintf(['    ANCHORING RESIDUAL %.3g um is %.0f%% of the blur -- the ' ...
                 'pupil numbers above are NOT valid.\n'], ...
                S.anchor_resid_um, 100*S.anchor_resid_um/max(S.blur_um,eps));
    end
end

function s = mark_(r)
    if r <= 1, s = '  MET'; else, s = ''; end
end
