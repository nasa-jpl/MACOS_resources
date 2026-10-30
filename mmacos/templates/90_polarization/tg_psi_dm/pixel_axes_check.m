function chk = pixel_axes_check(names, scores, spec, tol)
%PIXEL_AXES_CHECK  Check a SPECIFIED pixel-axis convention against all eight.
%   CHK = PIXEL_AXES_CHECK(NAMES, SCORES, SPEC, TOL) takes the eight
%   candidate conventions (NAMES, 1x8 cell, in register_to_dm's order) and
%   the truth-vs-recovery correlation each one scores (SCORES, 1x8), and
%   checks the convention the user specified (SPEC, one of NAMES).  The
%   specified convention is always the one used: the eight scores are a
%   CHECK, not a chooser (Dave 2026-10-29, PLAN 4a: coordinates are the
%   user's to specify; an unspecified one gets a clear consistent default).
%
%   It prints the table, and:
%     - WARNS (never errors), id 'tg_psi_dm:pixelAxes', when another
%       convention beats the specified one by more than TOL;
%     - names a TIE when the specified convention's 180-degree partner
%       (both axes reversed) scores within TOL of it: the injected pattern
%       is then centrosymmetric about the beam and cannot resolve the
%       orientation, so the measurement neither confirms nor refutes SPEC.
%
%   CHK fields: name (= SPEC, the convention used), i (its index), score,
%   best_name / best_score (the top scorer), beaten (logical), margin
%   (best_score - score), partner / partner_score, tie (logical), within
%   (every other convention scoring within TOL of SPEC), tol.
%
%   TOL, measured 2026-10-29 on the shipped run (checker pokes): the default
%   and its partner differ by 1e-16, the two transposes sit 2.2e-9 below,
%   a wrong-sign convention scores -0.997 (a 2.0 gap).  1e-6 sits above the
%   round-off and the transpose spread and far below a real mismatch.
%
%   See also: example_tg_psi_dm (register_to_dm).
    arguments
        names  (1,8) cell
        scores (1,8) double
        spec   (1,:) char
        tol    (1,1) double {mustBePositive}
    end
    i = find(strcmp(names, spec), 1);
    if isempty(i)
        error('tg_psi_dm:pixelAxesName', ...
            'pixel axes ''%s'' is not one of: %s', spec, strjoin(names, ' | '));
    end
    % the 180-degree partner reverses both axes: x=+col,y=+row <-> x=-col,y=-row
    partner = [4 3 2 1 8 7 6 5];
    [bs, bi] = max(scores);
    chk = struct('name', spec, 'i', i, 'score', scores(i), ...
        'best_name', names{bi}, 'best_score', bs, ...
        'margin', bs - scores(i), 'beaten', bs - scores(i) > tol, ...
        'partner', names{partner(i)}, 'partner_score', scores(partner(i)), ...
        'tie', abs(scores(partner(i)) - scores(i)) <= tol, 'tol', tol);
    near = setdiff(find(abs(scores - scores(i)) <= tol), i);
    chk.within = names(near);                  % every convention SPEC cannot be told from
    fprintf('  pixel-axis check (score = truth-vs-recovery corr. before refinement):\n');
    for k = 1:8
        tag = '';
        if k == i, tag = '  <- specified'; elseif k == partner(i), tag = '  <- its 180-deg partner'; end
        fprintf('    %-14s %+.6f%s\n', names{k}, scores(k), tag);
    end
    if chk.beaten
        warning('tg_psi_dm:pixelAxes', ...
            ['pixel axes %s score %.6f, but %s scores %.6f (%.2e higher, tolerance %.1e): ' ...
             'the specified convention may be wrong'], spec, chk.score, chk.best_name, ...
             chk.best_score, chk.margin, tol);
    end
    if chk.tie
        fprintf(['  TIE: %s and its 180-deg partner %s score within %.1e -- the injected ' ...
                 'pattern is centrosymmetric and cannot resolve the orientation; ' ...
                 'the specified convention is used, not confirmed\n'], spec, chk.partner, tol);
    end
    if numel(chk.within) > 1 || (numel(chk.within) == 1 && ~chk.tie)
        fprintf('  also within %.1e of %s: %s\n', tol, spec, strjoin(chk.within, ', '));
    end
end
