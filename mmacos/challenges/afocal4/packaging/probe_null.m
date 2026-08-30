function T = probe_null()
%PROBE_NULL  Is AFOCAL4_SCORE invariant under the fold isometry?
%
%   The four-fold route scored EXACTLY the parent at one lateral step and
%   674 nm away at another, on decks that are isometric copies of the same
%   design and whose main trace agrees to ten digits.  Something in the
%   scoring is placement-dependent.  This scores the same route at a range
%   of lateral steps and prints the pose the metric chose, so the sensitive
%   quantity can be named instead of guessed at.

    here = fileparts(mfilename('fullpath'));
    up   = fileparts(here);
    addpath(here);  addpath(up);
    P = afocal4_params();
    macos.init(256);
    src = fullfile(up, 'afocal4_b2long_343mm.in');
    tmp = fullfile(here, 'probe_tmp.in');

    S0 = afocal4_score(P, src, 'nodes',P.solve.nodes_score, 'grid',P.grid_n, ...
                       'quiet',true);
    fprintf('\n  %8s %11s %9s %10s %10s %11s %11s %10s\n', 'x_step', ...
            'WFE nm','blur um','breathe %','wander um','anchor um','tilt deg','shift mm');
    row = @(tag,S) fprintf('  %8s %11.2f %9.2f %10.4f %10.2f %11.4f %11.4f %10.4f\n', ...
        tag, S.wfe_max_nm, S.blur_um, S.breathe_pct, S.wander_um, ...
        S.anchor_resid_um, S.pose.tilt_deg, norm(S.pose.shift_mm));
    row('parent', S0);

    T = struct('x_step',{},'S',{});
    for x = [0.110 0.125 0.150 0.175 0.200]
        macos.load_rx(src);
        f = pack_route(src, 'init',false, 'x_step',x, 'quiet',true);
        pack_fold(src, f, tmp, 'quiet',true);
        S = afocal4_score(P, tmp, 'nodes',P.solve.nodes_score, 'grid',P.grid_n, ...
                          'quiet',true);
        row(sprintf('%.3f',x), S);
        T(end+1) = struct('x_step',x, 'S',S); %#ok<AGROW>
    end

    % ---- and the same question with ONE fold, at a range of stations ----
    fprintf('\n  one fold after the last mirror, station swept:\n');
    L = pack_legs(src, 'quiet', true);
    nM = L.nElt - 1;
    for d = [0.05 0.10 0.15 0.20 0.25]
        macos.load_rx(src);
        f1 = struct('name','Fold','after',nM,'dist',d,'to',[1 0 0]);
        pack_fold(src, f1, tmp, 'quiet',true);
        S = afocal4_score(P, tmp, 'nodes',P.solve.nodes_score, 'grid',P.grid_n, ...
                          'quiet',true);
        row(sprintf('d=%.2f',d), S);
    end
    if isfile(tmp), delete(tmp); end
end
