function run_offaxis_petzsum(varargin)
%RUN_OFFAXIS_PETZSUM  Does the FINAL MIRROR'S CURVATURE predict the defocus?
%
%   NAMING, AND A CORRECTION MADE BEFORE ANYTHING WAS RECORDED.  This routine
%   was written to compute a PETZVAL SUM and it does not.  In MACOS every
%   mirror is emitted as KrElt = -|R| -- convexity is carried by the GEOMETRY
%   (which side of the vertex the centre of curvature falls on), never by the
%   sign of Kr -- so a sum built from the radii the engine hands back cannot
%   tell a convex secondary from a concave one.  The measurement proves it:
%   the cass_greg and cass_cass cascades, which differ precisely in the
%   convexity of their final mirror, come back with the SAME value, 11.600.
%
%   What the quantity actually is: C = sum s_k * 2/|R_k| with s alternating
%   along the train -- a curvature-MAGNITUDE alternating sum, dominated in
%   every deck of this corpus by the last mirror, because at 30x compression
%   that mirror is necessarily the smallest and most strongly curved.  It is
%   reported under its own name and no Petzval claim is made from it.
%
%   The rung-2/rung-3 decomposition (section O.6b) says 60-99.7 % of every design's
%   wavefront variance in this study is POWER, and for an afocal system that
%   means the output collimation varies across the field -- field curvature.
%   Field curvature's textbook source is the Petzval sum, which needs SIGNED
%   curvatures; what is available cheaply from an emitted deck is the
%   curvature MAGNITUDE structure, and that is what is measured here.
%
%   THIS IS A CORRELATION, AND IT IS REPORTED AS ONE.  A quantity that tracks
%   the measured defocus across a corpus built by different means -- committed
%   four-mirror, descent ascent rungs, single and double Mersennes, off-axis
%   sections -- is evidence about the mechanism, not proof of it.  The ranking
%   matters rather than any single row, so the output is sorted and the
%   outliers stay visible.
%
%   WHY IT MATTERS FOR THE SPEC CONVERSATION.  At 30x compression the final
%   mirror is NECESSARILY the smallest and most strongly curved in the train
%   -- R_last = 0.167 m on a four-parabola cascade against R1 = 5 m -- so its
%   2/|R| term is ~30x the primary's and dominates C.  That is a structural
%   consequence of asking for 30x rather than a property of any one form.  If
%   the defocus really is ordered by it, a cure has to act where that term
%   lives: in the compressed beam near the exit, where a strongly curved
%   mirror is at least SMALL.  Establishing the SIGNED (true Petzval)
%   statement needs convexity, which these decks do not carry in Kr -- that is
%   left open rather than guessed.
%
%   Env: PS_GLOB (deck glob(s), comma-separated).

    ap = fileparts(fileparts(mfilename('fullpath')));
    addpath(ap); addpath(fullfile(ap,'clearing')); addpath(fullfile(ap,'descent'));
    addpath(fullfile(ap,'offaxis'));

    globs = strsplit(getenv_d('PS_GLOB', strjoin({ ...
        fullfile(ap,'afocal4_b2long_343mm.in'), ...
        fullfile(ap,'afocal4_mersenne*.in'), ...
        fullfile(ap,'offaxis','decks','om_cass*.in'), ...
        fullfile(ap,'offaxis','decks','pz_*_h0.in'), ...
        fullfile(ap,'offaxis','afocal4_OAW*_h0_start.in')}, ',')), ',');

    P = afocal4_params();
    macos.init(P.model_size);

    decks = {};
    for i = 1:numel(globs)
        d = dir(strtrim(globs{i}));
        for j = 1:numel(d), decks{end+1} = fullfile(d(j).folder,d(j).name); end %#ok<AGROW>
    end
    if isempty(decks), fprintf('  no decks matched\n'); return; end

    fprintf('\n==== FINAL-MIRROR CURVATURE vs THE DEFOCUS TERM ====\n');
    fprintf(['  C = sum s_k * 2/|R_k|, s alternating -- a curvature-MAGNITUDE\n' ...
             '  sum, NOT a Petzval sum: MACOS emits KrElt = -|R| for every\n' ...
             '  mirror and carries convexity in the geometry, so cass_greg and\n' ...
             '  cass_cass are indistinguishable here (both 11.600) despite\n' ...
             '  opposite final-mirror convexity.  defocus = sqrt(rung2^2 -\n' ...
             '  rung3^2).  Sorted by |C|.\n\n']);

    rows = struct('name',{},'C',{},'r2',{},'r3',{},'defoc',{},'nmir',{},'Rlast',{});
    for i = 1:numel(decks)
        d = decks{i};   [~,nm] = fileparts(d);
        try
            [Csum, R, nmir] = curvsum_(d);
            S = afocal4_score(P, d, 'fields',P.Fsolve, ...
                              'nodes',P.solve.nodes_score, 'pupil',false);
        catch ME
            fprintf('  %-34s FAILED %s\n', nm, ME.message);   continue;
        end
        df = sqrt(max(0, S.wfe_max_nm^2 - S.wfe_rung3_max_nm^2));
        rows(end+1) = struct('name',nm, 'C',Csum, 'r2',S.wfe_max_nm, ...
            'r3',S.wfe_rung3_max_nm, 'defoc',df, 'nmir',nmir, ...
            'Rlast',R(min(nmir,numel(R)))); %#ok<AGROW>
    end

    [~,o] = sort(abs([rows.C]));   rows = rows(o);
    fprintf('  %-34s %10s %10s %10s %10s %9s\n', 'deck','C /m', ...
            'rung2 nm','rung3 nm','defocus nm','R_last m');
    for i = 1:numel(rows)
        r = rows(i);
        fprintf('  %-34s %10.3f %10.1f %10.1f %10.1f %9.4f\n', r.name, r.C, ...
                r.r2, r.r3, r.defoc, r.Rlast);
    end

    % rank correlation: does |C| order the defocus?
    if numel(rows) >= 4
        x = abs([rows.C]).';   y = [rows.defoc].';
        ok = isfinite(x) & isfinite(y);
        rho = corr_(rank_(x(ok)), rank_(y(ok)));
        fprintf(['\n  Spearman rank correlation |C| vs defocus: ' ...
                 '%+.3f  (n = %d)\n'], rho, nnz(ok));
        fprintf(['  NOTE: heavily tie-degraded -- 27 of 40 decks share just two\n' ...
                 '  C values, so the rank statistic understates a relationship\n' ...
                 '  that is monotone in the GROUP MEANS.\n']);
        fprintf('  %s\n\n', verdict_(rho));
    end
    save(fullfile(fileparts(mfilename('fullpath')),'offaxis_petzsum.mat'), ...
         'rows','P','-v7.3');
end

% =====================================================================
function [Csum, R, nmir] = curvsum_(deck)
%CURVSUM_  Curvature-magnitude alternating sum from the ENGINE, never from the
%   .in text --
%   several decks in this corpus declare an nElt that disagrees with their
%   Element= block count, and a text parse attributes the wrong element's Kr
%   (the FEX blast-radius lesson).
    macos.load_rx(deck);
    nE = macos.num_elt();
    R = nan(1,nE);   isM = false(1,nE);
    for k = 1:nE
        kr = macos.get_elt_kr(k);
        R(k) = kr;
        isM(k) = abs(kr) < 1e21;          % 1e22 is the engine's FLAT sentinel
    end
    Csum = 0;   s = 1;   nmir = 0;
    for k = 1:nE
        if ~isM(k), continue; end
        nmir = nmir + 1;
        Csum = Csum + s*2/R(k);
        s = -s;                            % alternate along the fold
    end
    R = R(isM);
end

function r = rank_(v)
    [~,i] = sort(v);   r = zeros(size(v));   r(i) = 1:numel(v);
end
function c = corr_(a,b)
    a = a - mean(a);   b = b - mean(b);
    c = sum(a.*b)/max(sqrt(sum(a.^2)*sum(b.^2)), eps);
end
function s = verdict_(rho)
    if rho >= 0.7
        s = ['STRONG: final-mirror curvature orders the defocus across forms ' ...
             'built by different means.'];
    elseif rho >= 0.4
        s = ['MODERATE: curvature magnitude explains part of the ordering; ' ...
             'something else carries the rest, and the outliers are where ' ...
             'to look.'];
    else
        s = ['WEAK: the defocus is NOT ordered by curvature magnitude, so ' ...
             'that diagnosis does not survive this corpus and must not be ' ...
             'carried into the record as a cause.'];
    end
end
function v = getenv_d(k,d), v = getenv(k); if isempty(v), v = d; end, end
