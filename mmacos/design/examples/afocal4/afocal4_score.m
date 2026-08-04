function S = afocal4_score(P, deck, opts)
%AFOCAL4_SCORE  The S4 merit: image quality AND the pupil ladder, on a deck.
%
%   S = AFOCAL4_SCORE(P, DECK) scores a committed prescription against the
%   six S4 targets and returns both the METRICS and the residual vector the
%   joint solve minimises.  It is the only place a number is turned into a
%   merit, so the solve, the ladder, the trade curve and the tests all score
%   identically by construction.
%
%   THE SIX TERMS (targets in P.targets, weights in P.weights):
%     wfe       afocal ladder RUNG 2 (piston + per-field tip/tilt -- the rung
%               that matches his CODE V field maps, rodgers2 PACKET), max
%               over the field set, nm.  Entered PER FIELD, so the solve sees
%               which field is failing rather than a max whose gradient is
%               one field's.
%     blur      pupil_map (1): per-node cone-convergence waist, rms, um.
%     breathe   pupil_map (3): CHIEF-NORMAL per-field magnification
%               half-range as a percentage of the box-centre value.  Chief-
%               normal because a footprint read on the placed plane carries
%               that plane's own 1/cos obliquity, which is a frame term and
%               not a pupil-imaging defect (rodgers2 PACKET section 4).
%     wander    pupil_map (4) at the REFIT plane, um rms.  See below.
%     surf_pv   pupil_map (2): max residual of the convergence surface
%               against the IDEAL IMAGE of the primary's own sag, mm.  Never
%               charge a fast primary's faithfully-imaged sag as pupil
%               curvature.
%     mag       |M_centre_chief / 30 - 1| as a percentage.  NOT the same
%               statement as the afocal condition, which the inner closure
%               makes an identity: this is the PUPIL magnification at the box
%               centre, and on a perfect afocal the two coincide.  The gap is
%               pupil aberration.
%
%   THE WANDER IS SCORED AT THE REFIT PLANE (S4 brief; the S2 convention).
%   A cold stop is a mechanical part with a pose, and its pose is chosen
%   after the optics -- Rodgers tunes his coldstop's DAR tilt exactly this
%   way.  So the operational number is the wander at the best plane (station
%   + tilt), and .wander_placed_um is reported beside it as what the AS-EMITTED
%   plane would deliver.  Both come from the same pupil_map call.
%
%   THE RESIDUALS ARE LOGARITHMIC, AND THAT IS AN EARNED RULE.
%     r_i = w_i * max(0, log(m_i / (floor * t_i)))
%   The alternative -- residuals linear in m_i/t_i, which is what "normalize
%   every target to 1" reads as -- was tried first and does not work here: an
%   unoptimised four-mirror layout misses the WFE target by ~200x and the
%   pupil targets by 6-7x, so in a sum of squares the WFE term carries 99.9%
%   of the merit and the solve simply walks the field mirror's power back
%   toward zero, i.e. back to the three-mirror it was built to improve on.
%   In the log domain a 200x miss is 5.3 and a 7x miss is 1.9, the gradient
%   is scale-free, and every term keeps a vote.  The floor (P.merit_floor,
%   default 0.5) stops a term earning credit once it is twice inside its
%   target, so freedom goes to whatever is still missing.
%   The comparison is measured, not asserted: afocal4_ladder('merit_ab').
%
%   THE PUPIL CONVENTION IS SELECTABLE, AND THE DEFAULT IS UNCHANGED.
%   Dave's 2026-08-04 ruling is that the pupil object for a COLDSTOP metric
%   is the RIM of the primary, not its pole, and that what the coldstop
%   cares about is the EDGE of the pupil image rather than its average.
%   Both are opt-in here and neither moves an existing number:
%     'anchor' 'surface' (default) | 'rim'   -- passed to PUPIL_MAP
%     'zone'   'full' (default) | 'rim'      -- 'rim' scores the blur and
%              wander terms on the outer 'rim_zone' annulus instead of the
%              whole aperture
%   A P carrying P.pupil.anchor / P.pupil.zone / P.pupil.rim_zone supplies
%   the defaults, which is how a SOLVE selects them: AFOCAL4_SOLVE hands
%   this function P and nothing else, so a convention that could only be
%   passed as a name-value could never be optimised against.
%
%   Name-value:
%     'fields'  K x 2 box-relative field offsets, rad (default P.Fsolve --
%               HIS 3x3 solve set).  Score with a uniform grid; quote both.
%     'nodes'   pupil_map lattice (default P.solve.nodes_score)
%     'grid'    also score the WFE on this uniform n x n grid (0 = skip)
%     'anchor'  pupil anchor (above)
%     'zone'    which part of the pupil the blur/wander terms score (above)
%     'quiet'   (true)
%
%   Returns S with the metrics above, .norm (each metric over its target),
%   .worst (the largest of those -- the "worst normalized miss"), .resid
%   (the weighted log residual vector for lsqnonlin), .merit (sum of its
%   squares), .pm (the full pupil_map), .L (the K x 3 afocal ladder, m),
%   .pose (the refit interface plane), and .ok.
%
%   See also AFOCAL4_BUILD, AFOCAL4_SOLVE, PUPIL_MAP, AFOCAL_LADDER_DECK.

    arguments
        P (1,1) struct
        deck (1,:) char
        opts.fields (:,2) double = P.Fsolve
        opts.nodes  (1,1) double = P.solve.nodes_score
        opts.grid   (1,1) double = 0
        opts.pupil  (1,1) logical = true
        opts.anchor (1,:) char = ''
        opts.zone   (1,:) char = ''
        opts.quiet  (1,1) logical = true
    end

    [anch, zone, zfrac] = pupil_cfg_(P, opts);
    T = P.targets;
    S = struct('deck',deck, 'fields',opts.fields, 'ok',false, ...
               'L',[], 'pm',[], 'err','');

    try
        [L, li] = afocal_ladder_deck(deck, opts.fields, 'init',false, ...
                                     'lambda',P.lambda);
        if opts.pupil
            pm = pupil_map(deck, opts.fields, 'nodes',opts.nodes, 'init',false, ...
                           'anchor',anch, 'rim_zone',zfrac);
        else
            % WFE-ONLY mode, for the diagnostic that asks what the wavefront
            % floor of an operating point IS before any pupil term is allowed
            % to trade against it.  Halves the cost per evaluation.  A score
            % taken this way is NOT a result -- .pupil_scored says so.
            pm = [];
        end
    catch ME
        % A design the solver walked into that will not trace or whose cones
        % will not converge is not a small number -- it is no number.  Return
        % a large finite residual so lsqnonlin backs out instead of dying.
        S.err = ME.message;
        S.resid = repmat(20, size(opts.fields,1) + 5*opts.pupil, 1);
        S.merit = sum(S.resid.^2);   S.worst = Inf;
        return;
    end

    S.ok = true;   S.L = L;   S.pm = pm;   S.ladder_info = li;
    S.pupil_scored = opts.pupil;
    w2  = L(:,2)*1e9;                       % rung 2, nm, per field
    % A field that did not trace comes back NaN.  It is not "no error" and it
    % is not zero -- it is a design that loses a corner of its own field box,
    % so it enters the merit as a large miss and is COUNTED, never dropped.
    S.n_fields_lost = nnz(~isfinite(w2));
    w2(~isfinite(w2)) = 1e5;
    S.wfe_nm            = w2(:).';
    S.wfe_max_nm        = max(w2);
    S.wfe_avg_nm        = mean(w2);
    S.wfe_rung1_max_nm  = max(L(:,1))*1e9;
    S.wfe_rung3_max_nm  = max(L(:,3))*1e9;

    if ~opts.pupil
        % THE TWO MODES MUST CARRY THE SAME FIELD SET.  A ladder or a trade
%       sweep stores S structs from both, and MATLAB's `arr(k) = s` fails
%       on dissimilar structures only when REACHED -- i.e. after the
%       expensive part is already spent (RESULTS rule 7).  Any field added
%       to the pupil branch belongs here too.
        blank = {'blur_um','blur_max_um','breathe_pct','wander_um', ...
                 'wander_max_um','wander_placed_um','surf_pv_mm', ...
                 'mag_centre_chief','mag_pct','distortion_pct','anchor_resid_um', ...
                 'blur_full_um','wander_full_um','blur_rim_um', ...
                 'blur_rim_max_um','wander_rim_um','wander_rim_placed_um'};
        for q = blank, S.(q{1}) = NaN; end
        S.pupil_anchor = anch;   S.pupil_zone = zone;
        S.pose = struct('shift_mm',NaN,'tilt_deg',NaN,'Vpt',[NaN NaN NaN], ...
                        'psi',[NaN NaN NaN]);
        S.norm = struct('wfe', S.wfe_max_nm/T.wfe_rung2_nm);
        S.worst = S.norm.wfe;
        K = numel(w2);
        S.resid = (P.weights.wfe/sqrt(K)) * lg_(w2/T.wfe_rung2_nm, P.merit_floor);
        S.merit = sum(S.resid.^2);
        S.terms = {'wfe'};
        if opts.grid > 0, S = grid_score_(P, deck, opts.grid, S); end
        if ~opts.quiet, afocal4_score_print(P, S); end
        return;
    end

    c   = pm.map.mag_per_field_chief;
    magc = pm.map.mag_centre_chief;
    S.pupil_anchor      = anch;
    S.pupil_zone        = zone;
    S.blur_um           = pm.blur.rms*1e6;
    S.blur_max_um       = pm.blur.max*1e6;
    S.breathe_pct       = 100*(max(c) - min(c))/2/magc;
    S.wander_um         = pm.best_plane.rms*1e6;      % REFIT plane
    S.wander_max_um     = pm.best_plane.max*1e6;
    S.wander_placed_um  = pm.wander.rms*1e6;          % as emitted
    % the FULL-APERTURE numbers stay in the struct under every convention,
    % so a rim-scored solve is still quotable against the curve it is being
    % compared with; only which of them the RESIDUAL sees changes.
    S.blur_full_um      = pm.blur.rms*1e6;
    S.wander_full_um    = pm.best_plane.rms*1e6;
    S.blur_rim_um       = pm.rim_zone.blur_rms*1e6;
    S.blur_rim_max_um   = pm.rim_zone.blur_max*1e6;
    S.wander_rim_um     = pm.rim_zone.wander_best_rms*1e6;
    S.wander_rim_placed_um = pm.rim_zone.wander_rms*1e6;
    if strcmp(zone,'rim')
        S.blur_um       = S.blur_rim_um;
        S.blur_max_um   = S.blur_rim_max_um;
        S.wander_um     = S.wander_rim_um;
        S.wander_max_um = pm.rim_zone.wander_best_max*1e6;
    end
    S.surf_pv_mm        = pm.surface.ideal.resid_max*1e3;
    S.mag_centre_chief  = magc;
    S.mag_pct           = 100*abs(magc/T.mag - 1);
    S.distortion_pct    = 100*pm.map.distortion_frac_max;
    S.anchor_resid_um   = pm.anchor.resid_max*1e6;
    S.pose = struct('shift_mm', pm.best_plane.shift*1e3, ...
                    'tilt_deg', pm.best_plane.tilt_deg, ...
                    'Vpt', pm.best_plane.Vpt, 'psi', pm.best_plane.psi);

    if opts.grid > 0, S = grid_score_(P, deck, opts.grid, S); end

    % ---- normalised misses, then the log residual ------------------------
    S.norm = struct('wfe', S.wfe_max_nm/T.wfe_rung2_nm, ...
                    'blur', S.blur_um/T.blur_um, ...
                    'breathe', S.breathe_pct/T.breathe_pct, ...
                    'wander', S.wander_um/T.wander_um, ...
                    'surf_pv', S.surf_pv_mm/T.surface_pv_mm, ...
                    'mag', S.mag_pct/T.mag_pct);
    S.worst = max(cell2mat(struct2cell(S.norm)));

    W = P.weights;   fl = P.merit_floor;
    K = numel(w2);
    r = [ (W.wfe/sqrt(K)) * lg_(w2/T.wfe_rung2_nm, fl); ...
          W.blur    * lg_(S.norm.blur,    fl); ...
          W.breathe * lg_(S.norm.breathe, fl); ...
          W.wander  * lg_(S.norm.wander,  fl); ...
          W.surf_pv * lg_(S.norm.surf_pv, fl); ...
          W.mag     * lg_(S.norm.mag,     fl) ];
    r(~isfinite(r)) = 20;
    S.resid = r;
    S.merit = sum(r.^2);
    S.terms = {'wfe','blur','breathe','wander','surf_pv','mag'};

    if ~opts.quiet, afocal4_score_print(P, S); end
end

% =====================================================================
function S = grid_score_(P, deck, n, S)
%GRID_SCORE_  Score the WFE again on the UNIFORM grid.  Solve set is not
%   scoring set: his 3x3 is a third corners, which biased the rodgers1 area
%   average by 8% at an identical max.  Quote both, always.
    Fg = macos.design.field_grid(P.fov_half_deg*60, n, 'units','arcmin');
    Lg = afocal_ladder_deck(deck, Fg, 'init',false, 'lambda',P.lambda);
    g2 = Lg(:,2)*1e9;
    S.grid_fields     = Fg;
    S.L_grid          = Lg;
    S.wfe_grid_max_nm = max(g2);
    S.wfe_grid_avg_nm = mean(g2(isfinite(g2)));
end

function [anch, zone, zfrac] = pupil_cfg_(P, opts)
%PUPIL_CFG_  Which pupil convention this score is taken under.  Name-value
%   beats P.pupil beats the study default, and the study default is the
%   surface anchor over the whole aperture -- i.e. every number already in
%   RESULTS.md re-derives unchanged from a P that has no .pupil field at all.
    anch = 'surface';   zone = 'full';   zfrac = 0.10;
    if isfield(P,'pupil')
        if isfield(P.pupil,'anchor'),   anch  = P.pupil.anchor;   end
        if isfield(P.pupil,'zone'),     zone  = P.pupil.zone;     end
        if isfield(P.pupil,'rim_zone'), zfrac = P.pupil.rim_zone; end
    end
    if ~isempty(opts.anchor), anch = opts.anchor; end
    if ~isempty(opts.zone),   zone = opts.zone;   end
    if ~any(strcmp(zone,{'full','rim'}))
        error('macos:design:afocal4_score:zone', ...
              'pupil zone must be "full" or "rim" (got "%s")', zone);
    end
end

function y = lg_(ratio, floor_frac)
%LG_  The log-domain miss: how many e-folds a metric is above the level at
%   which it stops mattering.  Zero once inside floor_frac x target.
    y = max(0, log(max(ratio, realmin) / floor_frac));
    y = y(:);
end
