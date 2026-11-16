function fpath = save_coro_workspace(tag, ws, keep)
%SAVE_CORO_WORKSPACE  Persist a coro-experiment workspace for resume.
%   FPATH = SAVE_CORO_WORKSPACE(TAG, WS) saves struct WS (the full
%   workspace of a batch run -- including the heavy 1024x1024 intensity
%   arrays and DM states) to
%       templates/30_instruments/coro_experiments/results/<TAG>_<timestamp>.mat
%   so post-eval/dialog analysis (new metrics, plots) can resume from
%   disk instead of re-tracing the multi-minute model-1024 propagation.
%
%   DISK GUARD: after saving, prunes to the most recent KEEP files
%   matching <TAG>_*.mat (default 2).  These arrays are tens of MB each
%   and fill the disk fast.  This deterministic keep-last-K is the
%   primary protection; clean_results.sh is the scheduled age-based
%   catch-all.
    arguments
        tag  (1,:) char
        ws   struct
        keep (1,1) double = 2
    end

    here = fileparts(mfilename('fullpath'));
    rdir = fullfile(here, 'results');
    if ~exist(rdir, 'dir'), mkdir(rdir); end

    ts    = datestr(now, 'yyyymmdd_HHMMSS'); %#ok<TNOW1,DATST>
    fpath = fullfile(rdir, sprintf('%s_%s.mat', tag, ts));
    save(fpath, 'ws', '-v7.3');              % v7.3 compresses big arrays
    fprintf('[save] workspace -> %s\n', fpath);

    % Prune to the most-recent KEEP files for this tag.
    d = dir(fullfile(rdir, [tag '_*.mat']));
    if numel(d) > keep
        [~, ord] = sort([d.datenum], 'descend');
        for i = keep+1:numel(d)
            old = fullfile(rdir, d(ord(i)).name);
            delete(old);
            fprintf('[save] pruned old workspace %s\n', d(ord(i)).name);
        end
    end
end
