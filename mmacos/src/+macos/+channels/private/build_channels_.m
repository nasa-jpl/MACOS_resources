function chans = build_channels_(session, targets, rx_modes, modes_per_elt, factory)
%BUILD_CHANNELS_  Materialize a list of channels for (elt, mode) pairs.
%   chans = build_channels_(SESSION, TARGETS, RX_MODES, MODES_PER_ELT, FACTORY)
%   returns a cell row vector of channel handles.  For each element in
%   TARGETS, the active mode list is resolved in order:
%
%       1) MODES_PER_ELT(iElt) if provided and present, else
%       2) RX_MODES(iElt) (parsed from the prescription) if present, else
%       3) [1] (single-mode default).
%
%   FACTORY is a function handle (session, iElt, mode) -> channel that
%   selects the kind (MonZern / FFZern / Zern).

if isempty(targets)
    chans = {};
    return;
end

chans = cell(0, 1);
for k = 1:numel(targets)
    iElt = double(targets(k));
    modes = resolve_modes_(iElt, rx_modes, modes_per_elt);
    if isempty(modes)
        continue;
    end
    for jj = 1:numel(modes)
        chans{end+1, 1} = factory(session, iElt, double(modes(jj))); %#ok<AGROW>
    end
end
end


function modes = resolve_modes_(iElt, rx_modes, modes_per_elt)
key32 = int32(iElt);
if ~isempty(modes_per_elt) && isa(modes_per_elt, 'containers.Map') ...
        && isKey(modes_per_elt, key32)
    modes = modes_per_elt(key32);
    modes = modes(:);
    return;
end
if isa(rx_modes, 'containers.Map') && isKey(rx_modes, key32)
    modes = rx_modes(key32);
    modes = modes(:);
    return;
end
modes = 1;
end
