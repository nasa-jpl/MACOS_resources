function tf = has_rx()
%MACOS.HAS_RX  True iff a prescription is loaded.
%   Detects via `num_elt > 0`; if init hasn't run yet, mmacos errors
%   internally and we return false.
try
    tf = macos.num_elt() > 0;
catch
    tf = false;
end
end
