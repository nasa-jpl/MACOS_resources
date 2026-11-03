function s = msg_echo_()
%MSG_ECHO_  The statement a nested `matlab -batch` child script runs to echo the engine's messages.
%   Write it into the child script with fprintf(fid, '%s\n', msg_echo_()).  It prints each line of
%   macos.messages() (which then empties the buffer) tagged 'MSG|' through MATLAB's own fprintf --
%   the Fortran unit-6 copy does not reach a nested MATLAB's output on macOS (PLAN_CONSOLIDATION
%   1.2 (ee)).  The parent reads the tagged lines back with MSG_TEXT_.
s = 'fprintf(''MSG|%s\n'', macos.messages());';
end
