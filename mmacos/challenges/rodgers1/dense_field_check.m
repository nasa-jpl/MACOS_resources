function D = dense_field_check(opts)
%DENSE_FIELD_CHECK  Is the residual on the AVERAGE a field-sampling artifact?
%
%   The strict metric reproduces Rodgers' reported MAX to 0.3% or better on
%   all three of his designs (PACKET.md Addendum 10, re-run natively after
%   the ColSource pupil fix).  The AVERAGE still sits high -- S3 reads
%   50.19 nm against his 46.38, +8.2%.  This asks the cheapest question
%   that could explain it: are we averaging over the same field set he is?
%
%   We score on the 15 XAN/YAN points his .seq files define.  That set is a
%   HALF box on a QUINCUNX -- 3 y-points at XAN = 0, 0.05, 0.1 and 2 at
%   XAN = 0.025, 0.075, plus 2 interior y-points on the XAN = 0 column
%   (rodgers_seq.m).  It is a solve sampling, chosen to constrain an
%   optimizer, and it is weighted toward the box EDGE.  CODE V's field-map
%   statistics are computed on CODE V's own map grid.  If WFE grows toward
%   the edge -- which on a biased box it does -- then an edge-weighted set
%   reports a HIGHER average for the same design, at the same max.
%
%   So: re-score the SAME committed decks with the SAME metric and the SAME
%   rung, changing ONLY the field sampling, and watch the average.  The max
%   is the control: it must not move (a denser grid can only find the same
%   or a slightly worse worst point).
%
%   Grids compared:
%     seq       his 15 quincunx points (the incumbent)
%     halfNxN   uniform (N x N) over the HALF box, XAN in [0,0.1]
%     fullNxN   uniform (N x N) over the FULL box, XAN in [-0.1,0.1]
%
%   The full box is the honest comparator if his map is over the whole
%   field; the design is symmetric about the y-z plane, so the full-box
%   average equals the half-box average up to sampling, and disagreement
%   between the two is itself a sampling diagnostic.
%
%   Name-value:
%     'decks' cellstr (default the three committed .seq-truth decks)
%     'N'     grid density, default 9 (-> 81 points per grid)
%     'save'  write rodgers1_dense_field.mat (default true)
%
%   NOTE this changes only the SCORING field set.  No deck is re-solved.

    arguments
        opts.decks (1,:) cell   = {}
        opts.N     (1,1) double = 9
        opts.save  (1,1) logical = true
    end
    here = fileparts(mfilename('fullpath'));
    root = fileparts(fileparts(here));
    run(fullfile(root,'mmacos_setup.m'));
    addpath(here);

    P = rodgers_common('seq');
    decks = opts.decks;
    if isempty(decks)
        decks = { fullfile(here,'rodgers1_seq_rodgersS2.in')
                  fullfile(here,'rodgers1_seq_rodgersS3.in')
                  fullfile(here,'rodgers1_seq_rodgersS4.in') };
    end
    tags = {'rodgersS2','rodgersS3','rodgersS4'};
    gts  = { P.gt.s2_box, P.gt.s3_box, P.gt.s4_box };

    N = opts.N;
    h = P.seq.fov_half_deg;                     % 0.1 deg
    u = linspace(0, h, N);   v = linspace(-h, h, N);
    [UX,UY] = meshgrid(u,v);  halfG = deg2rad([UX(:), UY(:)]);
    w = linspace(-h, h, N);
    [WX,WY] = meshgrid(w,v);  fullG = deg2rad([WX(:), WY(:)]);

    grids = { 'seq(15)',            P.seq.Frel
              sprintf('half%dx%d',N,N), halfG
              sprintf('full%dx%d',N,N), fullG };

    macos.init(P.model_size);
    D = struct('N',N,'grids',{grids(:,1)},'stage',struct([]));

    fprintf('\n############ FIELD-SAMPLING CHECK (metric and decks unchanged) ############\n');
    for c = 1:numel(decks)
        g = gts{c}*1e3;
        fprintf('\n  === %s   (CODE V reported: max %.2f  avg %.2f nm) ===\n', tags{c}, g(2), g(3));
        fprintf('  %-12s %6s | %8s %8s | %8s %8s\n', ...
                'field set','K','max nm','x his','avg nm','x his');
        D(c).tag = tags{c};  D(c).gt_nm = g;
        for k = 1:size(grids,1)
            L = strict_ladder_deck(decks{c}, grids{k,2});
            a = L(:,4)*1e9;                      % bestfoc + LS tip/tilt rung
            a = a(isfinite(a));
            fprintf('  %-12s %6d | %8.2f %8.3f | %8.2f %8.3f\n', ...
                    grids{k,1}, numel(a), max(a), max(a)/g(2), mean(a), mean(a)/g(3));
            D(c).set(k).name = grids{k,1};
            D(c).set(k).K    = numel(a);
            D(c).set(k).max_nm = max(a);
            D(c).set(k).avg_nm = mean(a);
        end
    end

    fprintf(['\n  READING.  If the average falls toward his on the uniform grids while\n' ...
             '  the max holds, the residual on the average was the SAMPLING, not the\n' ...
             '  optics.  If it does not move, the average is a real disagreement and\n' ...
             '  the next suspect is his map''s own weighting.\n']);

    if opts.save
        out = fullfile(here,'rodgers1_dense_field.mat');
        save(out,'D');  fprintf('\n  saved %s\n', out);
    end
end

% =====================================================================
% The per-field ladder and the deck helpers it needed now live in
% design/src (STRICT_LADDER_DECK / STRICT_RUNGS) -- hoisted 2026-08-01 so
% the e2e2 flow and this diagnostic score through ONE kernel.
