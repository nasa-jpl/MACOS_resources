function OUT = r1_shroud_union(over)
%R1_SHROUD_UNION  Both instruments in the shroud, round-2 train.
%
%   The imager leg is optically UNCHANGED from round 1 (the deployable
%   pick-off sits 0.15 m after OAP1; DM1 starts at 0.25 m, so the two
%   never contest the same space) -- its committed round-1 deck and its
%   PSF/WFE record stand.  What round 2 changes is the CORONAGRAPH leg,
%   so the union shroud gate is re-measured: the DM-bearing train
%   unioned with round 1's imager configuration.
%
%   See also R1_BACKEND, ../e2e6m/s3_imager, shroud_deck.

    arguments
        over struct = struct()
    end
    here = fileparts(mfilename('fullpath'));
    run(fullfile(here,'..','..','..','mmacos_setup.m'));
    P = e2e6m_r2_params(over);
    addpath(fullfile(here,'..','..','..','design','src'));

    coro = fullfile(P.outdir, 'r1_seg_full.in');
    imgr = fullfile(P.r1dir, 's3_imager_full.in');
    assert(isfile(coro), 'r1_shroud_union: run r1_backend first');
    assert(isfile(imgr), 'r1_shroud_union: round-1 imager deck missing');

    L = {};
    L = say_(L, '==================== e2e6m R1 -- two-instrument shroud union');
    sh = shroud_deck(coro, P, 'extra', imgr, ...
            'labels', {'coronagraph leg (DM-bearing)', 'imager leg (round 1)'}, ...
            'png', fullfile(P.outdir, 'r1_union_shroud.png'));
    L = say_(L, 'union: %.3f m against the %.1f m gate  [%s]', ...
             sh.D, P.shroud_D_m, gate_(sh.D <= P.shroud_D_m));
    for k = 1:numel(sh.per_deck)
        L = say_(L, '  %-34s D %.3f m  (%d hardware elts)', ...
                 sh.per_deck(k).rx, sh.per_deck(k).D, sh.per_deck(k).n_hw);
    end
    txt = strjoin(L, newline);
    fid = fopen(fullfile(P.outdir,'r1_union_report.txt'),'w');
    fprintf(fid,'%s\n',txt);  fclose(fid);
    OUT = struct('P',P, 'sh',sh, 'text',txt);
    save(fullfile(P.outdir,'r1_union_run.mat'),'OUT');
end

function L = say_(L, varargin)
    s = sprintf(varargin{:});  L{end+1} = s;  fprintf('%s\n', s);
end
function s = gate_(ok), if ok, s = 'PASS'; else, s = 'FAIL'; end, end
