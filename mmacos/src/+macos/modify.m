function modify()
%MACOS.MODIFY  Reset ray-trace-dependent state (the SMACOS MODIFY cmd).
%   Call after any direct prescription mutation (perturb, set_elt_*)
%   before re-running a trace, so cached intermediates don't carry
%   stale geometry.
mmacos('modified_rx');
end
