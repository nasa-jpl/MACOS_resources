function S = stage_score(deck, opts)
%STAGE_SCORE  A design stage's headline: the full reference ladder on a
%   UNIFORM field grid, with the Strehl that goes with each rung.
%
%   S = STAGE_SCORE(DECK, 'lambda', lam, 'fov_half_deg', h) scores the
%   committed prescription DECK over an n x n uniform grid spanning the
%   used box (+-h in both axes about the deck's OWN chief ray) and returns
%   every rung of STRICT_LADDER_DECK plus its Strehl, the diffraction-limit
%   verdict, and a formatted table for the stage report.
%
%   SOLVE SET != SCORING SET.  Pass the solve field set as 'solve_fields'
%   and the table gains a second block scored on it.  The two differ, and
%   the difference is a property of the SAMPLING, not of the optics: an
%   edge-weighted solve set reports a higher AVERAGE at an identical max
%   (rodgers1 DENSE_FIELD_CHECK measured ~8% on a 15-point quincunx vs a
%   uniform 9x9).  Quote statistics from the uniform grid; quote the solve
%   set only to show what the optimizer actually saw.
%
%   NAME THE RUNG.  Nothing here picks a single number and calls it "the"
%   WFE.  All four rungs are tabled every time, and 'rung' only chooses
%   which one the pass/fail verdict reads.  Rung 4 (best focus + LS
%   tip/tilt) is the convention external field-map RMS numbers are
%   consistent with; rung 2 (centroid) is the primary reference for this
%   project's own reporting.
%
%   Name-value:
%     'lambda'        design wavelength, m (REQUIRED for waves + Strehl)
%     'fov_half_deg'  half-field of the used box (required unless
%                     'fields' is given)
%     'n'             uniform grid density (default 9 -> 81 points)
%     'fields'        explicit (:,2) scoring offsets, rad -- supersedes
%                     fov_half_deg/n
%     'solve_fields'  (:,2) the solve set, scored as a second block
%     'rung'          which rung the verdict reads (default 4)
%     'dl_waves'      RMS bar in waves (default 1/14, Marechal-class)
%     'strehl_min'    Strehl bar (default 0.8)
%     'title'         heading for the formatted block
%     'quiet'         suppress printing (default false)
%
%   Returns S with, per block (.uniform and .solve):
%     .L (K x 4) rung RMS in metres, .waves (K x 4), .strehl (K x 4),
%     .max_m .avg_m .max_waves .avg_waves .strehl_min (1 x 4), .K
%   plus .rung .dl_waves .strehl_min_bar .ok_rms .ok_strehl .ok
%   .lambda .fields .text
%
%   See also STRICT_LADDER_DECK, STRICT_RUNGS, macos.design.field_grid.

    arguments
        deck (1,:) char
        opts.lambda       (1,1) double {mustBePositive}
        opts.fov_half_deg (1,1) double = NaN
        opts.n            (1,1) double = 9
        opts.fields       (:,2) double = zeros(0,2)
        opts.solve_fields (:,2) double = zeros(0,2)
        opts.rung         (1,1) double {mustBeInteger} = 4
        opts.dl_waves     (1,1) double = 1/14
        opts.strehl_min   (1,1) double = 0.80
        opts.title        (1,:) char = 'FIELD SCORE'
        opts.quiet        (1,1) logical = false
    end

    F = opts.fields;
    if isempty(F)
        if isnan(opts.fov_half_deg)
            error('macos:design:stage_score:fields', ...
                  'give either ''fields'' or ''fov_half_deg''.');
        end
        F = macos.design.field_grid(opts.fov_half_deg*60, opts.n, 'units','arcmin');
    end

    S = struct('lambda',opts.lambda, 'rung',opts.rung, ...
               'dl_waves',opts.dl_waves, 'strehl_min_bar',opts.strehl_min, ...
               'deck',deck, 'title',opts.title);
    S.uniform = block_(deck, F, opts.lambda);
    S.fields  = F;
    S.solve   = struct([]);
    if ~isempty(opts.solve_fields)
        S.solve = block_(deck, opts.solve_fields, opts.lambda);
    end

    r = opts.rung;
    S.ok_rms    = S.uniform.max_waves(r) <= opts.dl_waves;
    S.ok_strehl = S.uniform.strehl_min(r) >= opts.strehl_min;
    S.ok        = S.ok_rms && S.ok_strehl;

    % ---- format --------------------------------------------------------
    nm = {'strict-chief','strict-centroid','+best focus','+LS tip/tilt'};
    L = {};
    L{end+1} = sprintf('---------------- %s ----------------', S.title);
    L{end+1} = sprintf('  lambda %g nm | used box scored on a UNIFORM %d-point grid', ...
                       opts.lambda*1e9, S.uniform.K);
    L{end+1} = sprintf('  %-16s %10s %10s %10s %10s %10s', ...
                       'rung','max nm','avg nm','max waves','avg waves','min Strehl');
    for k = 1:4
        L{end+1} = sprintf('  %-16s %10.3f %10.3f %10.4f %10.4f %10.4f%s', ...
            nm{k}, S.uniform.max_m(k)*1e9, S.uniform.avg_m(k)*1e9, ...
            S.uniform.max_waves(k), S.uniform.avg_waves(k), ...
            S.uniform.strehl_min(k), tern_(k==r,'   <- verdict','')); %#ok<AGROW>
    end
    if ~isempty(fieldnames(S.solve))
        L{end+1} = sprintf(['  the SOLVE set (%d points) for comparison -- ' ...
                            'sampling, not optics:'], S.solve.K);
        for k = 1:4
            L{end+1} = sprintf('  %-16s %10.3f %10.3f', nm{k}, ...
                S.solve.max_m(k)*1e9, S.solve.avg_m(k)*1e9); %#ok<AGROW>
        end
    end
    L{end+1} = sprintf(['  VERDICT at "%s": RMS %.3f nm vs %.3f nm bar -> %s;  ' ...
                        'Strehl %.4f vs %.2f -> %s'], nm{r}, ...
                       S.uniform.max_m(r)*1e9, opts.dl_waves*opts.lambda*1e9, ...
                       tern_(S.ok_rms,'PASS','FAIL'), ...
                       S.uniform.strehl_min(r), opts.strehl_min, ...
                       tern_(S.ok_strehl,'PASS','FAIL'));
    L{end+1} = repmat('-', 1, 72);
    S.text = sprintf('%s\n', L{:});
    if ~opts.quiet, fprintf('%s', S.text); end
end

% ---------------------------------------------------------------------
function b = block_(deck, F, lam)
    [L, info] = strict_ladder_deck(deck, F, 'lambda', lam);
    ok = all(isfinite(L), 2);
    b = struct();
    b.L = L;  b.strehl = info.strehl;  b.waves = L/lam;
    b.K = nnz(ok);  b.K_requested = size(F,1);  b.info = info;
    if b.K == 0
        [b.max_m, b.avg_m, b.max_waves, b.avg_waves, b.strehl_min] = deal(nan(1,4));
        return;
    end
    b.max_m      = max(L(ok,:), [], 1);
    b.avg_m      = mean(L(ok,:), 1);
    b.max_waves  = b.max_m/lam;
    b.avg_waves  = b.avg_m/lam;
    Sm = info.strehl(ok,:);
    if all(isnan(Sm(:))), b.strehl_min = nan(1,4);
    else,                 b.strehl_min = min(Sm, [], 1);
    end
end

function s = tern_(c,a,b), if c, s = a; else, s = b; end, end
