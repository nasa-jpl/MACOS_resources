function lines = messages(clear)
%MACOS.MESSAGES  The engine's messages since the last read.
%   lines = macos.messages() returns the engine's kept messages as an n×1 string array, oldest
%   first, trailing blanks trimmed, and empties the buffer.  macos.messages(false) reads without
%   emptying it.
%
%   The load path's notes, warnings and refusals, CALIB's cause lines and the SMACOS loop's
%   'Unknown command' print to Fortran unit 6, which evalc never sees and which reaches no
%   MATLAB-visible stream on macOS (PLAN_CONSOLIDATION 1.2 (ee)); the engine also keeps each line
%   in a buffer, and this reads it.  The buffer holds the last 500 lines; when older ones were
%   dropped the first entry says how many.  Works with no Rx loaded (a refused load's messages are
%   the point).  pymacos: messages().
%
%   See also: macos.load_rx, macos.calib.
arguments
    clear (1,1) logical = true
end
% Cmd 'messages_count' plhs: nLines_, nChars_, nDropped_.  'messages_get' prhs: nChars.
[~, n_chars, n_dropped] = mmacos('messages_count');
lines = strings(0, 1);
if n_dropped > 0
    lines(end+1, 1) = sprintf('(%d earlier messages dropped)', n_dropped);
end
if n_chars > 0
    codes = mmacos('messages_get', double(n_chars));
    txt = char(double(codes(:)'));
    parts = split(string(txt), newline);
    lines = [lines; parts(1:end-1)];
end
if clear
    mmacos('messages_clear');
end
end
