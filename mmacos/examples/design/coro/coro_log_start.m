function c = coro_log_start(tag)
%CORO_LOG_START  Start a stable, tail-able diary log for a coro run.
%   C = CORO_LOG_START(TAG) begins capturing all console output to
%       examples/design/coro/results/<TAG>.log
%   (overwritten each run), so a run stream can be watched live with
%       tail -f .../coro/results/<TAG>.log
%   at ~zero load (no parallel MATLAB / no license-seat contention).
%
%   Hold the returned onCleanup object C in the caller's workspace; it
%   turns the diary off automatically when the caller returns or errors:
%       cleanup = coro_log_start('E1_darkzone_contrast');   %#ok<NASGU>
%
%   The timestamped *.mat workspace (save_coro_workspace) is the
%   resume artifact; this .log is the human-watchable stream.  Both are
%   .gitignore'd local state.
    here = fileparts(mfilename('fullpath'));
    rdir = fullfile(here, 'results');
    if ~exist(rdir, 'dir'), mkdir(rdir); end
    logf = fullfile(rdir, [tag '.log']);
    if exist(logf, 'file'), delete(logf); end
    diary(logf);
    fprintf('[log] streaming to %s  (tail -f to watch)\n', logf);
    c = onCleanup(@() diary('off'));
end
