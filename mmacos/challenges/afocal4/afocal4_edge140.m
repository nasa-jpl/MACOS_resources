function R = afocal4_edge140(opts)
%AFOCAL4_EDGE140  The 3-vs-4 mirror fork at 140 mm, priced at the pupil EDGE.
%
%   R = AFOCAL4_EDGE140() closes the one qualifier the 2026-09-02 edge-only
%   audit left standing.  S4c re-scored the fork under the rim ZONE at
%   343 mm; the deck's headline pair (469 -> 157 um) is at the 140 mm
%   operating point.  Comparing a rim number from one standoff against a
%   full-aperture number from another is exactly the cross-condition error
%   this slice corrected in RESULTS.md:51, so the matched table is measured
%   here rather than argued.
%
%   The matched pair, both scored over the SAME offset field box with
%   tilt/decentre freedom, at the SAME 140 mm interface standoff:
%     3 mirrors  rodgers2_S4_tiltdec.in   (RESULTS/deck: blur 469, wander 557)
%     4 mirrors  afocal4_r4_tiltdec.in    (RESULTS section 2 rung 4: 152.9 / 156.4)
%
%   THREE METRICS, and they are not interchangeable:
%     blur          per-node cone waist -- how sharply the edge images
%     wander        per-node scatter across field -- edge smear, which mixes
%                   the edge's translation with its breathing
%     edge centroid the rim AS A BODY: how far the whole pupil edge
%                   TRANSLATES on the stop across the field box (new here;
%                   PUPIL_MAP .rim_zone.centroid)
%
%   Every length is reported in um AND as a fraction of the EXIT PUPIL
%   RADIUS, which is the radiometric currency: a stop undersized by a
%   fraction f to guarantee it masks the edge costs 1-(1-f)^2 of throughput.
%   RADIUS, not diameter -- the decks normalise against diameter, which is
%   the same defect reported at half its size.
%
%   Name-value:
%     'nodes'  pupil_map lattice (default P.solve.nodes_score)
%     'zone'   rim-zone width as a fraction of pupil radius (0.10)
%     'save'   write afocal4_edge140.mat (true)
%
%   See also PUPIL_MAP, AFOCAL4_FORK, AFOCAL4_SCORE.

    arguments
        opts.nodes (1,1) double  = 0
        opts.zone  (1,1) double  = 0.10
        opts.save  (1,1) logical = true
    end
    here = fileparts(mfilename('fullpath'));
    P    = afocal4_params();
    if opts.nodes <= 0, opts.nodes = P.solve.nodes_score; end
    macos.init(P.model_size);
    R = struct('P',P, 'zone',opts.zone, 'nodes',opts.nodes);

    D = {
      '3-mir, tilt/dec (the 469)', fullfile(fileparts(here),'rodgers2','rodgers2_S4_tiltdec.in')
      '3-mir, re-solved',          fullfile(fileparts(here),'rodgers2','rodgers2_S3_newconics.in')
      '3-mir, his parent',         fullfile(here,'afocal4_parent3.in')
      '4-mir, rung 4 (the 157)',   fullfile(here,'afocal4_r4_tiltdec.in')
      '4-mir, rung 3',             fullfile(here,'afocal4_r3_resolve.in')
      };
    keep = cellfun(@(f) isfile(f), D(:,2));
    for i = find(~keep(:).')
        fprintf('  (%s missing -- skipped)\n', D{i,2});
    end
    D = D(keep,:);

    fprintf('\n============================================================\n');
    fprintf('  THE 140 mm FORK AT THE PUPIL EDGE  (zone = outer %.0f%%)\n', ...
            100*opts.zone);
    fprintf('============================================================\n\n');

    % Field list must match the assignment below EXACTLY -- MATLAB's
    % arr(k)=s fails on dissimilar structs only when REACHED (RESULTS rule 7).
    R.rows = struct('name',{},'deck',{},'Rex_mm',{}, ...
                    'blur_full_um',{},'blur_rim_um',{}, ...
                    'wander_full_um',{},'wander_rim_um',{},'wander_placed_um',{}, ...
                    'cen_rms_um',{},'cen_max_um',{},'cen_placed_um',{}, ...
                    'blur_rim_pctR',{},'cen_rms_pctR',{},'throughput_pct',{});

    for i = 1:size(D,1)
        o = pupil_map(D{i,2}, P.Fsolve, 'nodes',opts.nodes, 'init',false, ...
                      'rim_zone',opts.zone);
        Rex = max(vecnorm(o.w(:,o.good)));      % exit pupil radius, m
        % FRAME.  The record quotes wander on the REFIT plane
        % (afocal4_score: S.wander_um = pm.best_plane.rms) -- the analogue of
        % a coldstop you are allowed to position and tilt.  The deck's own
        % placed plane is carried beside it.  They differ by ~6x on these
        % decks; quoting one under the other's name reverses the verdict.
        c   = o.rim_zone.centroid_best;         % refit frame, to match
        cp  = o.rim_zone.centroid;              % placed frame, as emitted
        % the leakage statement: undersize by the edge error, pay 1-(1-f)^2
        f   = o.rim_zone.blur_rms/Rex;
        R.rows(end+1) = struct( ...
            'name',D{i,1}, 'deck',D{i,2}, 'Rex_mm',Rex*1e3, ...
            'blur_full_um',   o.blur.rms*1e6, ...
            'blur_rim_um',    o.rim_zone.blur_rms*1e6, ...
            'wander_full_um', o.best_plane.rms*1e6, ...
            'wander_rim_um',  o.rim_zone.wander_best_rms*1e6, ...
            'wander_placed_um', o.wander.rms*1e6, ...
            'cen_rms_um',     c.rms*1e6, 'cen_max_um', c.max*1e6, ...
            'cen_placed_um',  cp.rms*1e6, ...
            'blur_rim_pctR',  100*f, ...
            'cen_rms_pctR',   100*c.frac_rms, ...
            'throughput_pct', 100*(1-(1-f)^2)); %#ok<AGROW>
    end

    T = R.rows;
    fprintf('%-27s %8s %8s %8s %8s %9s %9s\n', 'design', 'blur', 'blur', ...
            'wander', 'wander', 'edge cen', 'edge cen');
    fprintf('%-27s %8s %8s %8s %8s %9s %9s\n', '', 'full um', 'RIM um', ...
            'full um', 'RIM um', 'rms um', 'max um');
    fprintf('%s\n', repmat('-',1,84));
    for i = 1:numel(T)
        fprintf('%-27s %8.1f %8.1f %8.1f %8.1f %9.1f %9.1f\n', T(i).name, ...
            T(i).blur_full_um, T(i).blur_rim_um, T(i).wander_full_um, ...
            T(i).wander_rim_um, T(i).cen_rms_um, T(i).cen_max_um);
    end

    % ---- identity check: does this reproduce the committed record? ---------
    % Non-vacuity for the whole table.  If these four do not land, the frame
    % or the field set is wrong and nothing below is quotable.
    ref = {'3-mir, tilt/dec (the 469)', 469.0, 557.0
           '4-mir, rung 4 (the 157)',   152.9, 156.4};
    fprintf('\n  IDENTITY CHECK vs the committed record (blur / wander, refit frame)\n');
    R.identity = true;
    for i = 1:size(ref,1)
        k = find(strcmp({R.rows.name}, ref{i,1}), 1);
        if isempty(k), continue; end
        db = 100*(R.rows(k).blur_full_um/ref{i,2} - 1);
        dw = 100*(R.rows(k).wander_full_um/ref{i,3} - 1);
        ok = abs(db) < 1.0 && abs(dw) < 1.0;
        R.identity = R.identity && ok;
        verdict = {'FAIL','ok'};
        fprintf('    %-27s blur %7.1f vs %6.1f (%+.2f%%)   wander %7.1f vs %6.1f (%+.2f%%)  %s\n', ...
            ref{i,1}, R.rows(k).blur_full_um, ref{i,2}, db, ...
            R.rows(k).wander_full_um, ref{i,3}, dw, verdict{ok+1});
    end

    fprintf('\n  IN THE RADIOMETRIC CURRENCY (fraction of exit pupil RADIUS)\n');
    fprintf('%-27s %9s %11s %11s %14s\n', 'design', 'Rex mm', 'rim blur %R', ...
            'edge cen %R', 'throughput cost');
    fprintf('%s\n', repmat('-',1,78));
    for i = 1:numel(T)
        fprintf('%-27s %9.3f %10.2f%% %10.2f%% %13.1f%%\n', T(i).name, ...
            T(i).Rex_mm, T(i).blur_rim_pctR, T(i).cen_rms_pctR, ...
            T(i).throughput_pct);
    end

    % ---- the fork, stated ---------------------------------------------------
    i3 = find(strcmp({T.name},'3-mir, tilt/dec (the 469)'),1);
    i4 = find(strcmp({T.name},'4-mir, rung 4 (the 157)'),1);
    if ~isempty(i3) && ~isempty(i4)
        R.ratio_full = T(i3).blur_full_um / T(i4).blur_full_um;
        R.ratio_rim  = T(i3).blur_rim_um  / T(i4).blur_rim_um;
        R.ratio_cen  = T(i3).cen_rms_um   / T(i4).cen_rms_um;
        R.dthrough   = T(i3).throughput_pct - T(i4).throughput_pct;
        fprintf(['\n  WHAT THE FOURTH MIRROR BUYS AT 140 mm\n' ...
                 '    on the full aperture   %.2fx\n' ...
                 '    at the RIM             %.2fx\n' ...
                 '    edge centroid          %.2fx\n' ...
                 '    throughput             %.1f%% -> %.1f%%  ' ...
                 '(%.1f points)\n'], ...
            R.ratio_full, R.ratio_rim, R.ratio_cen, ...
            T(i3).throughput_pct, T(i4).throughput_pct, R.dthrough);
    end

    if opts.save
        matf = fullfile(here,'afocal4_edge140.mat');
        save(matf,'R');
        fprintf('\n  saved %s\n', matf);
    end
end
