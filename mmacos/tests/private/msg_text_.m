function t = msg_text_(out)
%MSG_TEXT_  The engine messages a nested `matlab -batch` child echoed, as one string.
%   The engine's notes, warnings and refusals are Fortran unit-6 text, which a nested MATLAB's
%   output does not carry on macOS (PLAN_CONSOLIDATION 1.2 (ee)).  The child echoes the engine's
%   message buffer (MSG_ECHO_'s statement; lines tagged 'MSG|') through MATLAB's own fprintf, which does
%   arrive; this keeps only those lines, so a count is the same on Linux (where the untagged unit-6
%   copy also arrives) and on the Mac.
tok = regexp(out, '(?m)^MSG\|([^\r\n]*)', 'tokens');
t = string(strjoin(cellfun(@(c) c{1}, tok, 'UniformOutput', false), newline));
end
